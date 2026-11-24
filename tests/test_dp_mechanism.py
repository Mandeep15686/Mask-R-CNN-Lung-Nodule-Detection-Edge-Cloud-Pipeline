import math

import pytest
import torch

from edge.dp import BasicCompositionAccountant, gaussian_feature_release


def test_gaussian_release_clips_joint_feature_set_and_reports_bound():
    torch.manual_seed(4)
    accountant = BasicCompositionAccountant(epsilon_budget=0.5, delta_budget=1e-5)
    released, report = gaussian_feature_release(
        {"layer1": torch.full((1, 4, 2, 2), 3.0), "layer2": torch.full((1, 2, 2, 2), 4.0)},
        epsilon=0.5,
        delta=1e-5,
        clip_norm=2.0,
        accountant=accountant,
    )
    assert set(released) == {"layer1", "layer2"}
    assert all(tensor.shape[0] == 1 and torch.isfinite(tensor).all() for tensor in released.values())
    assert report["sensitivity_bound"] == 4.0
    assert report["noise_sigma"] == pytest.approx(4.0 * math.sqrt(2 * math.log(1.25 / 1e-5)) / 0.5)
    assert accountant.epsilon_spent == pytest.approx(0.5)
    assert accountant.delta_spent == pytest.approx(1e-5)


def test_basic_composition_refuses_release_past_both_budgets():
    accountant = BasicCompositionAccountant(epsilon_budget=0.4, delta_budget=2e-5)
    accountant.reserve(0.2, 1e-5)
    accountant.reserve(0.2, 1e-5)
    with pytest.raises(RuntimeError, match="exceed budget"):
        accountant.reserve(0.01, 1e-6)
    assert accountant.releases == 2


def test_persistent_ledger_keeps_spend_across_accountant_restart(tmp_path):
    ledger = tmp_path / "edge-dp.json"
    first = BasicCompositionAccountant(0.5, 1e-5, ledger_path=ledger)
    first.reserve(0.5, 1e-5)

    restarted = BasicCompositionAccountant(0.5, 1e-5, ledger_path=ledger)
    assert restarted.epsilon_spent == pytest.approx(0.5)
    assert restarted.delta_spent == pytest.approx(1e-5)
    assert restarted.releases == 1
    with pytest.raises(RuntimeError, match="exceed budget"):
        restarted.reserve(0.01, 1e-6)


def test_persistent_ledger_rejects_budget_changes_and_corruption(tmp_path):
    ledger = tmp_path / "edge-dp.json"
    BasicCompositionAccountant(0.5, 1e-5, ledger_path=ledger)
    with pytest.raises(RuntimeError, match="budgets differ"):
        BasicCompositionAccountant(0.6, 1e-5, ledger_path=ledger)

    ledger.write_text("not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unreadable"):
        BasicCompositionAccountant(0.5, 1e-5, ledger_path=ledger)


def test_release_rejects_multi_patient_batches_and_invalid_epsilon():
    accountant = BasicCompositionAccountant(epsilon_budget=1.0, delta_budget=1e-5)
    with pytest.raises(ValueError, match="exactly one patient"):
        gaussian_feature_release(
            {"layer1": torch.ones(2, 4, 2, 2)}, epsilon=1.0, delta=1e-5,
            clip_norm=1.0, accountant=accountant,
        )
    with pytest.raises(ValueError, match="epsilon must be in"):
        gaussian_feature_release(
            {"layer1": torch.ones(1, 4, 2, 2)}, epsilon=1.01, delta=1e-5,
            clip_norm=1.0, accountant=accountant,
        )


def test_dp_utility_helper_uses_actual_split_continuation():
    from scripts.cn5_differential_privacy import _measure_dice_from_features

    class FakeInference:
        calls = []

        def predict_from_edge_features(self, features, split_layer, scan_uid, **metadata):
            self.calls.append((features, split_layer, scan_uid, metadata))
            return []

    engine = FakeInference()
    score = _measure_dice_from_features(
        engine,
        [{"layer1": torch.ones(1, 1, 1, 1)}],
        [torch.zeros((4, 4), dtype=torch.uint8).numpy()],
        "layer1",
        "checkpoint-test",
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    )

    assert score == pytest.approx(1.0)
    _, split_layer, scan_uid, metadata = engine.calls[0]
    assert (split_layer, scan_uid) == ("layer1", "cn5-dp-evaluation")
    assert metadata["expected_checkpoint_sha256"] == "checkpoint-test"
