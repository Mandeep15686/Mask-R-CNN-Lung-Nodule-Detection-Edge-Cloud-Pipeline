"""MGDA objective-weight and Trainer integration tests."""
import json
import hashlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from training.mgda import (
    MGDASolver, collect_flat_gradients, compute_mgda_weights,
    frank_wolfe_min_norm, min_norm_simplex_weights, solve_mgda,
)
from training.trainer import Trainer, MGDATrainer, _estimate_sgd_step_norm, build_trainer
from training.losses import CompositeMaskLoss


def test_min_norm_solution_stays_on_simplex_and_resolves_conflicting_gradients():
    gram = torch.tensor([[4.0, -4.0], [-4.0, 4.0]])
    weights = min_norm_simplex_weights(gram)
    assert torch.isfinite(weights).all()
    assert torch.all(weights >= 0)
    assert weights.sum().item() == pytest.approx(1.0)
    assert weights.tolist() == pytest.approx([0.5, 0.5], abs=1e-5)
    assert (weights @ gram.to(dtype=weights.dtype) @ weights).item() < 1e-8


def test_min_norm_is_stable_for_zero_and_large_gram_matrices():
    zero = min_norm_simplex_weights(torch.zeros((5, 5)))
    assert zero.tolist() == pytest.approx([0.2] * 5)

    large = min_norm_simplex_weights(torch.eye(5, dtype=torch.float64) * 1e12)
    assert torch.isfinite(large).all()
    assert torch.all(large >= 0)
    assert large.sum().item() == pytest.approx(1.0)


def test_mgda_gradients_are_computed_and_combined_loss_can_backpropagate():
    parameter = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    objectives = {
        "left": (parameter[0] - 1).square(),
        "right": (parameter[0] + 1).square(),
        "other": (parameter[1] - 2).square(),
    }
    weights, norms = compute_mgda_weights(objectives, [parameter])
    assert set(weights) == set(objectives)
    assert set(norms) == set(objectives)
    assert all(torch.isfinite(torch.tensor(value)) and value >= 0 for value in norms.values())
    assert all(0 <= value <= 1 for value in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0)
    diagnostics = solve_mgda(objectives, [parameter]).diagnostics
    assert diagnostics["mgda_shared_parameter_count"] == 1
    assert diagnostics["mgda_shared_parameter_numel"] == 2
    assert "mgda_gram_left_right" in diagnostics
    assert "mgda_cosine_left_right" in diagnostics

    combined = sum(weights[name] * objective for name, objective in objectives.items())
    combined.backward()
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()


def test_unit_gradient_profile_applies_the_same_normalized_gradients_it_solves():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    objectives = {"large": 10.0 * parameter, "small": -parameter}

    raw = solve_mgda(objectives, [parameter], gradient_normalization="raw")
    unit = solve_mgda(objectives, [parameter], gradient_normalization="unit")

    assert raw.weights["large"] == pytest.approx(1 / 11, abs=1e-6)
    assert raw.weights["small"] == pytest.approx(10 / 11, abs=1e-6)
    assert unit.weights == pytest.approx({"large": 0.5, "small": 0.5}, abs=1e-6)
    assert unit.diagnostics["mgda_cosine_large_small"] == pytest.approx(-1.0)
    assert unit.applied_coefficients["large"] == pytest.approx(0.05)
    assert unit.applied_coefficients["small"] == pytest.approx(0.5)
    assert unit.diagnostics["mgda_normalized_grad_norm_large"] == pytest.approx(1.0)
    assert unit.diagnostics["mgda_normalized_grad_norm_small"] == pytest.approx(1.0)

    combined = sum(
        unit.applied_coefficients[name] * objective
        for name, objective in objectives.items()
    )
    grad = torch.autograd.grad(combined, parameter)[0]
    assert grad.item() == pytest.approx(0.0, abs=1e-6)


def test_zero_gradient_objective_is_reported_without_forcing_a_weight():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    result = solve_mgda(
        {"active": (parameter - 1).square(), "locally_constant": parameter * 0.0},
        [parameter],
        gradient_normalization="unit",
    )
    assert result.gradient_norms["locally_constant"] == 0.0
    assert result.weights["locally_constant"] == 0.0
    assert sum(result.weights.values()) == pytest.approx(1.0)


def test_near_zero_gradient_normalization_is_finite_and_explicitly_inactive():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    result = solve_mgda(
        {
            "small_gradient": 1e-14 * parameter,
            "regular_gradient": (parameter - 1.0).square(),
        },
        [parameter],
        gradient_normalization="unit",
    )
    assert result.weights["small_gradient"] == 0.0
    assert result.applied_coefficients["small_gradient"] == 0.0
    assert all(torch.isfinite(torch.tensor(v)) for v in result.applied_coefficients.values())


def test_unit_gradient_combination_updates_parameters_with_existing_optimizer():
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    objectives = {"first": parameter.square(), "second": 3.0 * parameter.square()}
    solution = solve_mgda(objectives, [parameter], gradient_normalization="unit")
    combined = sum(
        solution.applied_coefficients[name] * objective
        for name, objective in objectives.items()
    )
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    before = parameter.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    combined.backward()
    optimizer.step()
    assert torch.isfinite(parameter).all()
    assert not torch.equal(before, parameter.detach())


def test_sgd_step_norm_estimate_tracks_lr_momentum_and_weight_decay():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD(
        [parameter], lr=0.1, momentum=0.9, weight_decay=0.2
    )
    for gradient in (0.5, 0.3):
        parameter.grad = torch.tensor(gradient)
        before = parameter.detach().clone()
        estimated = _estimate_sgd_step_norm(optimizer)
        optimizer.step()
        actual = torch.linalg.vector_norm(parameter.detach() - before).item()
        assert estimated == pytest.approx(actual, rel=1e-6, abs=1e-7)
        optimizer.zero_grad(set_to_none=True)


class _TinyMGDAModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_mask_head = torch.nn.Parameter(torch.tensor([0.8, -0.6]))
        self.detector_parameter = torch.nn.Parameter(torch.tensor(0.7))
        self.composite_mask_loss = SimpleNamespace(last_components=None)

    def forward(self, images, targets=None):
        first, second = self.shared_mask_head
        components = {
            "bce": (first - 0.1).square(),
            "dice": (second + 0.1).square(),
            "boundary": (first + second).square() + 0.1,
            "focal": (first - second).square(),
            "hd": (first + second - 0.4).square(),
        }
        self.composite_mask_loss.last_components = components
        mask_total = sum(components.values())
        detector_loss = (self.detector_parameter - 0.2).square()
        return {"loss_mask": mask_total, "loss_classifier": detector_loss}


def test_mgda_training_epoch_updates_model_and_logs_objectives_weights_and_norms():
    model = _TinyMGDAModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer.__new__(Trainer)
    trainer.training_strategy = "mgda"
    trainer.mgda_gradient_normalization = "raw"
    trainer.mgda_parameter_scope = "mask_head"
    trainer.model = model
    trainer.mgda_task_names = ("bce", "dice", "boundary", "focal", "hd")
    trainer.mgda_shared_parameters = [model.shared_mask_head]
    trainer.mgda_weights = {name: 0.2 for name in trainer.mgda_task_names}
    trainer.optimizer = optimizer
    trainer.device = torch.device("cpu")
    trainer.train_loader = [{
        "image": torch.ones((1, 3, 8, 8)),
        "mask": torch.ones((1, 8, 8)),
        "label": torch.ones(1, dtype=torch.int64),
    }]
    before = model.detector_parameter.detach().clone()

    metrics = trainer._train_epoch(1)

    # The toy mask objectives can have a zero MGDA compromise direction; the
    # detector objective still verifies the shared optimizer step occurs.
    assert not torch.equal(before, model.detector_parameter.detach())
    assert metrics["mgda_active_batches"] == 1
    assert sum(metrics[f"mgda_weight_{name}"] for name in trainer.mgda_task_names) == pytest.approx(1.0)
    for name in trainer.mgda_task_names:
        assert torch.isfinite(torch.tensor(metrics[f"mgda_loss_{name}"]))
        assert torch.isfinite(torch.tensor(metrics[f"mgda_weight_{name}"]))
        assert torch.isfinite(torch.tensor(metrics[f"mgda_grad_norm_{name}"]))
        assert torch.isfinite(torch.tensor(metrics[f"mgda_coefficient_{name}"]))
    assert metrics["mgda_shared_parameter_count"] == 1
    assert metrics["mgda_shared_parameter_numel"] == 2
    assert metrics["mgda_learning_rate"] == pytest.approx(0.1)
    assert metrics["mgda_lr_times_postclip_grad_norm"] == pytest.approx(
        metrics["mgda_learning_rate"] * metrics["mgda_optimizer_grad_norm_post_clip"]
    )
    assert metrics["mgda_optimizer_grad_norm_pre_clip"] >= metrics["mgda_optimizer_grad_norm_post_clip"]
    assert metrics["mgda_optimizer_step_norm_estimate"] == pytest.approx(0.1, abs=1e-5)


def test_real_mask_rcnn_mgda_step_backpropagates_all_objectives():
    from training.model import LungNoduleMaskRCNN

    torch.manual_seed(7)
    all_losses = {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "delta": 1.0, "epsilon": 1.0}
    model = LungNoduleMaskRCNN(
        pretrained_backbone=False,
        mask_loss_cfg=all_losses,
        min_size=64,
        max_size=64,
        box_detections_per_img=10,
    )
    model.composite_mask_loss.capture_components = True
    model.train()
    image = torch.rand((3, 64, 64))
    mask = torch.zeros((1, 64, 64), dtype=torch.uint8)
    mask[:, 20:44, 20:44] = 1
    target = {
        "boxes": torch.tensor([[20.0, 20.0, 44.0, 44.0]]),
        "labels": torch.tensor([1], dtype=torch.int64),
        "masks": mask,
    }
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    losses = model([image], [target])
    components = model.composite_mask_loss.last_components
    assert components is not None
    assert set(components) == {"bce", "dice", "boundary", "focal", "hd"}

    shared = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and ("roi_heads.mask_head." in name or "roi_heads.mask_predictor." in name)
    ]
    weights, _ = compute_mgda_weights(components, shared)
    weighted_mask = sum(weights[name] * components[name] for name in components)
    detector = sum(value for name, value in losses.items() if name != "loss_mask")
    total = detector + weighted_mask
    assert torch.isfinite(total)
    total.backward()
    assert any(parameter.grad is not None for parameter in shared)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    optimizer.step()


class _TinyConfiguredModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = nn.Module()
        self.model.backbone = nn.Module()
        self.model.backbone.body = nn.Linear(2, 2)
        self.model.backbone.fpn = nn.Linear(2, 2)
        self.model.roi_heads = nn.Module()
        self.model.roi_heads.mask_head = nn.Linear(2, 2)
        self.model.roi_heads.mask_predictor = nn.Linear(2, 1)
        self.composite_mask_loss = CompositeMaskLoss.from_config(config["mask_loss"])

    def get_loss_description(self):
        return self.composite_mask_loss.describe()


def test_trainer_loads_shared_initial_state_strictly_and_keeps_optimizer_fresh(tmp_path, monkeypatch):
    import training.trainer as trainer_module

    mask_cfg = {"alpha": 1.0}
    initial_model = _TinyConfiguredModel({"mask_loss": mask_cfg})
    initial_path = tmp_path / "shared_initial.pth"
    torch.save(initial_model.state_dict(), initial_path)
    digest = hashlib.sha256(initial_path.read_bytes()).hexdigest()
    split = {"train_subsets": [1], "validation_subset": 0}
    loader = SimpleNamespace(dataset=SimpleNamespace(split_info=split))
    monkeypatch.setattr(trainer_module, "build_model", lambda config: _TinyConfiguredModel(config))
    monkeypatch.setattr(trainer_module, "build_dataloaders", lambda **kwargs: (loader, loader))

    trainer = Trainer({
        "training_strategy": "fixed", "mask_loss": mask_cfg,
        "data_root": "unused", "epochs": 2, "warmup_epochs": 1,
        "batch_size": 1, "seed": 20261121,
        "initial_state_path": str(initial_path), "initial_state_sha256": digest,
    }, str(tmp_path / "run"))
    assert trainer.config["initial_state_sha256"] == digest
    for key, tensor in initial_model.state_dict().items():
        assert torch.equal(tensor, trainer.model.state_dict()[key].cpu())
    assert trainer.optimizer.state == {}


@pytest.mark.parametrize("scope", ["mask_head", "shared_mask_path"])
def test_trainer_initializes_mgda_with_all_five_objectives(tmp_path, monkeypatch, scope):
    import training.trainer as trainer_module

    captured = {}
    model = None

    def build_model(config):
        nonlocal model
        captured.update(config)
        model = _TinyConfiguredModel(config)
        return model

    dataset = SimpleNamespace(split_info={"train_subsets": [1], "validation_subset": 0})
    loaders = SimpleNamespace(dataset=dataset)
    monkeypatch.setattr(trainer_module, "build_model", build_model)
    monkeypatch.setattr(trainer_module, "build_dataloaders", lambda **kwargs: (loaders, loaders))

    trainer = Trainer({
        "training_strategy": "mgda",
        "mgda_parameter_scope": scope,
        "mask_loss": {"alpha": 0.3, "beta": 0.5, "gamma": 0.5},
        "data_root": "unused-by-stub",
        "epochs": 2,
        "warmup_epochs": 1,
        "batch_size": 1,
    }, str(tmp_path / "mgda-init"))

    assert trainer.training_strategy == "mgda"
    assert captured["mask_loss"] == {
        "alpha": 1.0, "beta": 1.0, "gamma": 1.0, "delta": 1.0, "epsilon": 1.0,
    }
    assert model.composite_mask_loss.capture_components is True
    expected_parameters = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and (
            "roi_heads.mask_head." in name
            or "roi_heads.mask_predictor." in name
            or (scope == "shared_mask_path" and (
                "backbone.body." in name or "backbone.fpn." in name
            ))
        )
    ]
    assert [id(parameter) for parameter in trainer.mgda_shared_parameters] == [
        id(parameter) for parameter in expected_parameters
    ]


def test_experiment_runner_exposes_single_mgda_run_without_overwriting(tmp_path, monkeypatch):
    import training.trainer as trainer_module
    from training.experiment import ExperimentRunner

    captured = {}

    class FakeTrainer:
        def __init__(self, config, run_dir):
            captured.update(config)
            self.run_dir = run_dir

        def train(self):
            from pathlib import Path
            run_dir = Path(self.run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "summary.json").write_text("{}", encoding="utf-8")
            return {
                "config_id": captured["config_id"],
                "training_strategy": captured["training_strategy"],
                "best_metrics": {},
                "final_metrics": {},
            }

    monkeypatch.setattr(trainer_module, "Trainer", FakeTrainer)
    runner = ExperimentRunner({
        "epochs": 10,
        "baseline_dir": str(tmp_path / "absent-baseline"),
    }, str(tmp_path / "runs"))
    result = runner.run_mgda()

    assert result["config_id"] == "MGDA-all5-ep10"
    assert captured["training_strategy"] == "mgda"
    assert captured["mask_loss"] == {
        "alpha": 1.0, "beta": 1.0, "gamma": 1.0, "delta": 1.0, "epsilon": 1.0,
    }
    with pytest.raises(FileExistsError, match="already exists"):
        runner.run_mgda()


def test_experiment_runner_creates_distinct_scale_normalized_profile(tmp_path, monkeypatch):
    import training.trainer as trainer_module
    from training.experiment import ExperimentRunner

    captured = {}

    class FakeTrainer:
        def __init__(self, config, run_dir):
            captured.update(config)
            self.run_dir = run_dir

        def train(self):
            from pathlib import Path
            run_dir = Path(self.run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "summary.json").write_text("{}", encoding="utf-8")
            return {"config_id": captured["config_id"], "best_metrics": {}, "final_metrics": {}}

    monkeypatch.setattr(trainer_module, "Trainer", FakeTrainer)
    runner = ExperimentRunner({
        "epochs": 2,
        "baseline_dir": str(tmp_path / "absent-baseline"),
    }, str(tmp_path / "runs"))
    result = runner.run_mgda(profile="unit_shared_path")

    assert result["config_id"] == "MGDA-unit_shared_path-all5-ep2"
    assert captured["mgda_gradient_normalization"] == "unit"
    assert captured["mgda_parameter_scope"] == "shared_mask_path"


def test_frank_wolfe_solves_conflicting_task_gradients_on_the_simplex():
    gradients = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0]])
    weights, norm_sq, _ = frank_wolfe_min_norm(gradients)
    assert weights.sum().item() == pytest.approx(1.0)
    assert torch.all(weights >= 0)
    assert norm_sq < 1e-8


def test_frank_wolfe_solution_is_invariant_to_gradient_scale():
    gradients = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0]])
    reference, _, _ = frank_wolfe_min_norm(gradients)
    tiny, tiny_norm_sq, _ = frank_wolfe_min_norm(gradients * 1e-8)
    huge, huge_norm_sq, _ = frank_wolfe_min_norm(gradients * 1e8)
    assert tiny.tolist() == pytest.approx(reference.tolist(), abs=1e-8)
    assert huge.tolist() == pytest.approx(reference.tolist(), abs=1e-8)
    assert tiny_norm_sq == pytest.approx(0.0, abs=1e-28)
    assert huge_norm_sq == pytest.approx(0.0, abs=1e-8)


def test_collect_flat_gradients_uses_same_parameters_without_mutating_grad():
    shared = nn.Parameter(torch.tensor([2.0, 3.0]))
    unused = nn.Parameter(torch.tensor([4.0]))
    shared.grad = torch.tensor([9.0, 8.0])
    names, gradients = collect_flat_gradients(
        {"x": shared[0].square(), "y": shared[1] * 3.0}, [shared, unused]
    )
    assert names == ["x", "y"]
    assert gradients.shape == (2, 3)
    assert gradients[0].tolist() == pytest.approx([4.0, 0.0, 0.0])
    assert gradients[1].tolist() == pytest.approx([0.0, 3.0, 0.0])
    assert shared.grad.tolist() == [9.0, 8.0]
    assert unused.grad is None


def test_taskwise_loss_helpers_follow_configured_objectives():
    loss = CompositeMaskLoss.from_config({"alpha": 1.0, "beta": 1.0, "gamma": 0.0})
    pred = torch.randn((2, 8, 8), requires_grad=True)
    target = torch.zeros_like(pred)
    assert loss.active_tasks() == ["bce", "dice"]
    all_tasks = loss.forward_all_tasks(pred, target)
    assert set(all_tasks) == {"bce", "dice"}
    assert torch.equal(all_tasks["dice"], loss.forward_single_task(pred, target, "dice"))
    with pytest.raises(ValueError, match="Unknown mask-loss task"):
        loss.forward_single_task(pred, target, "unknown")


def test_mgda_solver_uses_fixed_selected_parameter_vector_and_raw_coefficients():
    model = _TinyConfiguredModel({"mask_loss": {"alpha": 1.0}})
    solver = MGDASolver(model, param_scope="mask_head")
    head_params = [p for n, p in model.named_parameters() if "roi_heads.mask_head." in n or "roi_heads.mask_predictor." in n]
    assert [id(p) for p in solver.parameters] == [id(p) for p in head_params]
    parameter = solver.parameters[0]
    loss_a = parameter.square().sum()
    loss_b = (parameter - 1.0).square().sum()
    result = solver.solve({"a": loss_a, "b": loss_b})
    assert result.lambdas.sum().item() == pytest.approx(1.0)
    assert result.applied_coefficients == pytest.approx({"a": result.lambdas[0].item(), "b": result.lambdas[1].item()})
    assert result.diagnostics["mgda_shared_parameter_count"] == len(solver.parameters)


def test_phase4_config_generation_has_18_unique_subset_scope_configs():
    from training.experiment import generate_phase4_configs

    configs = generate_phase4_configs()
    ids = [config["config_id"] for config in configs]
    assert len(configs) == 18
    assert len(ids) == len(set(ids))
    assert {config["mgda_param_scope"] for config in configs} == {"mask_head", "full"}
    for config in configs:
        active = {key for key, value in config["mask_loss"].items() if value}
        assert len(active) == len(config["mgda_tasks"])
        assert config["use_mgda"] is True


def test_single_training_run_preserves_existing_aggregate_results(tmp_path, monkeypatch):
    from training.experiment import ExperimentRunner
    import training.trainer as trainer_module

    aggregate = tmp_path / "all_results.json"
    aggregate.write_text(json.dumps([{"config_id": "prior-run", "best_dice": 0.4}]), encoding="utf-8")

    class FakeTrainer:
        def __init__(self, config, run_dir):
            self.config = config

        def train(self):
            return {"config_id": self.config["config_id"], "best_dice": 0.5}

    monkeypatch.setattr(trainer_module, "build_trainer", lambda config, run_dir: FakeTrainer(config, run_dir))
    runner = ExperimentRunner({"epochs": 1}, str(tmp_path))
    runner._run_single({"config_id": "new-run"})
    saved = json.loads(aggregate.read_text(encoding="utf-8"))
    assert {row["config_id"] for row in saved} == {"prior-run", "new-run"}


def test_build_trainer_keeps_fixed_path_and_routes_explicit_mgda(monkeypatch):
    import training.trainer as trainer_module

    class StubTrainer:
        def __init__(self, config, run_dir):
            self.config = config
            self.run_dir = run_dir

    monkeypatch.setattr(trainer_module, "Trainer", StubTrainer)
    monkeypatch.setattr(trainer_module, "MGDATrainer", StubTrainer)
    assert isinstance(build_trainer({}, "fixed"), StubTrainer)
    assert isinstance(build_trainer({"use_mgda": True}, "mgda"), StubTrainer)


def test_mgda_capture_training_step_combines_tasks_and_updates_shared_parameters():
    class CapturingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.roi_heads = nn.Module()
            self.roi_heads.mask_head = nn.Linear(1, 1, bias=False)
            nn.init.constant_(self.roi_heads.mask_head.weight, 0.2)
            self.composite_mask_loss = CompositeMaskLoss.from_config(
                {"alpha": 1.0, "beta": 1.0}
            )

        def forward_mgda(self, images, targets, cache):
            logits = (self.roi_heads.mask_head.weight * torch.ones(1, 1, 8, 8)).squeeze(1)
            cache["logits"] = logits
            cache["targets"] = torch.zeros_like(logits)
            return {"loss_mask": logits.sum() * 0.0, "loss_classifier": self.roi_heads.mask_head.weight.square().sum()}

    model = CapturingModel()
    trainer = MGDATrainer.__new__(MGDATrainer)
    trainer.config = {"mgda_log_conflict": True}
    trainer.model = model
    trainer.mgda_solver = MGDASolver(model, "mask_head")
    trainer.lambda_history = __import__("training.mgda", fromlist=["LambdaHistory"]).LambdaHistory()
    loss_dict, objectives = trainer._forward_train([], [])
    assert set(objectives) == {"bce", "dice"}
    weights, _, coefficients, _ = trainer._solve_mgda(objectives)
    combined = loss_dict["loss_classifier"] + sum(coefficients[key] * value for key, value in objectives.items())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.roi_heads.mask_head.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    combined.backward()
    optimizer.step()
    assert not torch.equal(before, model.roi_heads.mask_head.weight.detach())
    assert sum(weights.values()) == pytest.approx(1.0)
    assert trainer.lambda_history.epoch_averages()


def test_real_mask_rcnn_mgda_capture_forward_and_single_backward():
    from training.model import LungNoduleMaskRCNN

    torch.manual_seed(19)
    model = LungNoduleMaskRCNN(
        pretrained_backbone=False,
        mask_loss_cfg={"alpha": 1.0, "beta": 1.0},
        min_size=64,
        max_size=64,
        box_detections_per_img=8,
    )
    model.train()
    image = torch.rand((3, 64, 64))
    mask = torch.zeros((1, 64, 64), dtype=torch.uint8)
    mask[:, 18:42, 18:42] = 1
    target = {
        "boxes": torch.tensor([[18.0, 18.0, 42.0, 42.0]]),
        "labels": torch.tensor([1], dtype=torch.int64),
        "masks": mask,
    }
    cache = {}
    losses = model.forward_mgda([image], [target], cache)
    assert set(cache) == {"logits", "targets"}
    objectives = model.composite_mask_loss.forward_all_tasks(cache["logits"], cache["targets"])
    solver = MGDASolver(model, param_scope="mask_head")
    result = solver.solve(objectives)
    total = sum(value for key, value in losses.items() if key != "loss_mask")
    total = total + sum(
        result.applied_coefficients[name] * objectives[name] for name in objectives
    )
    assert torch.isfinite(total)
    total.backward()
    assert any(parameter.grad is not None for parameter in solver.parameters)

    # A negative-only target may have no positive ROI. It must still return
    # ordinary detector losses and must not leak captured logits across calls.
    negative_target = {
        "boxes": torch.zeros((0, 4)),
        "labels": torch.zeros((0,), dtype=torch.int64),
        "masks": torch.zeros((0, 64, 64), dtype=torch.uint8),
    }
    negative_cache = {}
    negative_losses = model.forward_mgda([image], [negative_target], negative_cache)
    assert "loss_classifier" in negative_losses
    assert negative_cache == {}
