"""
Tests for experiment configuration generation and runner logic.
Run: pytest tests/test_experiment_config.py -v
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pytest
import importlib.util

# Load experiment module directly to avoid torch dependency via __init__.py
def load_module(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, rel_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

experiment = load_module("experiment", "training/experiment.py")


# ---------------------------------------------------------------------------
# Phase 1 config structure
# ---------------------------------------------------------------------------
class TestPhase1Configs:
    def test_count(self):
        assert len(experiment.PHASE1_CONFIGS) == 6

    def test_all_have_config_id(self):
        for c in experiment.PHASE1_CONFIGS:
            assert "config_id" in c
            assert len(c["config_id"]) > 0

    def test_all_have_mask_loss(self):
        for c in experiment.PHASE1_CONFIGS:
            assert "mask_loss" in c

    def test_each_uses_exactly_one_loss(self):
        """Each Phase 1 config should activate exactly one loss."""
        for c in experiment.PHASE1_CONFIGS:
            ml    = c["mask_loss"]
            keys  = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
            active = [v for k, v in ml.items() if k in keys and v > 0]
            assert len(active) == 1, (
                f"{c['config_id']} should use exactly 1 loss, "
                f"got {len(active)}: {ml}"
            )

    def test_weights_are_exactly_one(self):
        for c in experiment.PHASE1_CONFIGS:
            ml = c["mask_loss"]
            for k, v in ml.items():
                assert v in (0.0, 1.0), f"{c['config_id']}.{k} = {v}, expected 0 or 1"

    def test_unique_config_ids(self):
        ids = [c["config_id"] for c in experiment.PHASE1_CONFIGS]
        assert len(ids) == len(set(ids))

    def test_covers_all_six_losses(self):
        """All configured loss types, including Tversky, appear in Phase 1."""
        active_keys = set()
        for c in experiment.PHASE1_CONFIGS:
            ml = c["mask_loss"]
            for k, v in ml.items():
                if v > 0:
                    active_keys.add(k)
        assert active_keys == {"alpha", "beta", "gamma", "delta", "epsilon", "zeta"}


# ---------------------------------------------------------------------------
# Phase 2 config structure
# ---------------------------------------------------------------------------
class TestPhase2Configs:
    @pytest.fixture(scope="class")
    def p2(self):
        return experiment.generate_phase2_configs()

    def test_count_124(self, p2):
        """5³ = 125 total, minus the all-zero config = 124."""
        assert len(p2) == 124

    def test_all_have_required_keys(self, p2):
        for c in p2:
            assert "config_id"  in c
            assert "mask_loss"  in c
            ml = c["mask_loss"]
            for k in ("alpha", "beta", "gamma", "delta", "epsilon"):
                assert k in ml, f"Missing key {k} in {c['config_id']}"

    def test_no_all_zero_config(self, p2):
        for c in p2:
            ml = c["mask_loss"]
            assert any(v > 0 for v in ml.values()), (
                f"All-zero config found: {c['config_id']}"
            )

    def test_delta_epsilon_always_zero_in_phase2(self, p2):
        """Phase 2 grid is only over alpha/beta/gamma; delta/epsilon should be 0."""
        for c in p2:
            ml = c["mask_loss"]
            assert ml["delta"]   == 0.0
            assert ml["epsilon"] == 0.0

    def test_weights_in_grid_values(self, p2):
        GRID = {0.0, 0.3, 0.5, 0.7, 1.0}
        for c in p2:
            ml = c["mask_loss"]
            for k in ("alpha", "beta", "gamma"):
                assert ml[k] in GRID, (
                    f"{c['config_id']}.{k} = {ml[k]} not in {GRID}"
                )

    def test_unique_config_ids(self, p2):
        ids = [c["config_id"] for c in p2]
        assert len(ids) == len(set(ids))

    def test_unique_weight_combos(self, p2):
        combos = [(c["mask_loss"]["alpha"],
                   c["mask_loss"]["beta"],
                   c["mask_loss"]["gamma"]) for c in p2]
        assert len(combos) == len(set(combos))

    def test_all_zero_triple_absent(self, p2):
        combos = [(c["mask_loss"]["alpha"],
                   c["mask_loss"]["beta"],
                   c["mask_loss"]["gamma"]) for c in p2]
        assert (0.0, 0.0, 0.0) not in combos

    def test_extreme_configs_present(self, p2):
        """(1,0,0), (0,1,0), (0,0,1), (1,1,1) should all be in Phase 2."""
        combos = {(c["mask_loss"]["alpha"],
                   c["mask_loss"]["beta"],
                   c["mask_loss"]["gamma"]) for c in p2}
        for expected in [(1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0),(1.0,1.0,1.0)]:
            assert expected in combos, f"Expected combo {expected} not found"


# ---------------------------------------------------------------------------
# Base config
# ---------------------------------------------------------------------------
class TestBaseConfig:
    def test_has_required_keys(self):
        bc = experiment.BASE_CONFIG
        for k in ("num_classes","pretrained_backbone","batch_size","lr","epochs","fold","data_root"):
            assert k in bc, f"BASE_CONFIG missing key: {k}"

    def test_num_classes_2(self):
        assert experiment.BASE_CONFIG["num_classes"] == 2

    def test_lr_positive(self):
        assert experiment.BASE_CONFIG["lr"] > 0


# ---------------------------------------------------------------------------
# Config JSON files
# ---------------------------------------------------------------------------
class TestConfigFiles:
    def test_phase1_json_valid(self):
        with open("configs/phase1_ablations.json") as f:
            cfg = json.load(f)
        assert "experiments" in cfg
        assert len(cfg["experiments"]) == 5    # 5 in JSON (BCE,Dice,Boundary,Focal,HD95)
        for exp in cfg["experiments"]:
            assert "config_id"    in exp
            assert "loss_weights" in exp
            lw = exp["loss_weights"]
            assert all(k in lw for k in ("alpha","beta","gamma","delta","epsilon"))

    def test_phase2_json_valid(self):
        with open("configs/phase2_grid.json") as f:
            cfg = json.load(f)
        assert "experiments" in cfg
        assert len(cfg["experiments"]) >= 10
        for exp in cfg["experiments"]:
            lw = exp["loss_weights"]
            # At least one weight must be non-zero
            assert any(v > 0 for v in lw.values())

    def test_phase3_json_valid(self):
        with open("configs/phase3_bayesian.json") as f:
            cfg = json.load(f)
        assert "n_trials" in cfg
        assert cfg["n_trials"] > 0
        assert "search_space" in cfg
        for k in ("alpha","beta","gamma","delta","epsilon"):
            assert k in cfg["search_space"]

    def test_phase2_json_no_zero_weights(self):
        with open("configs/phase2_grid.json") as f:
            cfg = json.load(f)
        for exp in cfg["experiments"]:
            lw = exp["loss_weights"]
            assert any(v > 0 for v in lw.values()), (
                f"All-zero weights in {exp['config_id']}"
            )
