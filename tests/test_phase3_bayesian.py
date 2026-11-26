"""
Tests for Phase 3 Bayesian Optimisation (training/phase3_bayesian.py).

Strategy
--------
All tests use a fake ``Phase3TrialTrainer`` that returns controlled summaries
without touching the filesystem for model weights, datasets, or GPU.  Optuna
uses an in-process (non-persistent) SQLite URL backed by a tmp_path, so tests
run quickly and leave no global state.

Test categories
---------------
1. Pure-math helpers: normalise_weights, compute_objective
2. CSV persistence helpers
3. Full integration via run_phase3_bayesian (with mocked trainer)
4. Resume / idempotency behaviour
5. Pruning propagation
6. Interaction with ExperimentRunner.run_phase3
7. Backward-compat of run_bayesian_optimisation wrapper in experiment.py
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from training.phase3_bayesian import (
    _CSV_FIELDNAMES,
    _append_trial_csv,
    _init_csv,
    _phase3_study_lock,
    _rebuild_csv_from_study,
    _update_best_json,
    compute_objective,
    normalise_weights,
    run_phase3_bayesian,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def base_cfg() -> dict:
    """Minimal base config — data_root is a placeholder; it is never read."""
    return {
        "data_root":    "/nonexistent/data",
        "epochs":       30,
        "batch_size":   4,
        "lr":           0.005,
        "momentum":     0.9,
        "weight_decay": 1e-4,
        "warmup_epochs": 2,
        "num_workers":  0,
        "fold":         0,
        "num_classes":  2,
        "patch_size":   64,
        "training_strategy": "fixed",
    }


def _study_dir(output_dir: Path, study_name: str) -> Path:
    return output_dir / "phase3" / "studies" / study_name


@pytest.fixture()
def _fake_summary_factory():
    """Return a factory for fake trainer summaries with configurable metrics."""
    def factory(
        config_id: str = "p3_trial_0000",
        dice:  float = 0.60,
        hd95:  float = 15.0,
        bdf1:  float = 0.55,
        epoch: int   = 10,
    ) -> dict:
        metrics = {
            "val_dice":        dice,
            "val_hd95":        hd95,
            "val_boundary_f1": bdf1,
        }
        return {
            "config_id":     config_id,
            "best_epoch":    epoch,
            "best_metrics":  metrics,
            "final_metrics": metrics,
        }
    return factory


class _FakePhase3Trainer:
    """
    Drop-in replacement for Phase3TrialTrainer that never touches the
    filesystem for model weights / datasets.
    """
    _calls: list[dict] = []

    def __init__(
        self,
        config: dict,
        run_dir: str,
        optuna_trial: Any = None,
        dice:  float = 0.60,
        hd95:  float = 15.0,
        bdf1:  float = 0.55,
    ) -> None:
        self.config       = config
        self.run_dir      = Path(run_dir)
        self._trial       = optuna_trial
        self._dice        = dice
        self._hd95        = hd95
        self._bdf1        = bdf1
        self.__class__._calls.append(deepcopy(config))

    def train(self) -> dict:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        epochs = int(self.config.get("epochs", 10))
        metrics = {
            "val_dice":        self._dice,
            "val_hd95":        self._hd95,
            "val_boundary_f1": self._bdf1,
        }
        summary = {
            "config_id":     self.config.get("config_id", "unknown"),
            "best_epoch":    epochs,
            "best_metrics":  metrics,
            "final_metrics": metrics,
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        (self.run_dir / "history.json").write_text(
            json.dumps([{"epoch": e} for e in range(1, epochs + 1)]),
            encoding="utf-8",
        )
        (self.run_dir / "latest_model.pth").write_bytes(b"fake-weights")
        return summary


@pytest.fixture(autouse=True)
def _reset_fake_trainer():
    """Clear call log before each test."""
    _FakePhase3Trainer._calls = []
    yield
    _FakePhase3Trainer._calls = []


def _make_runner(monkeypatch, dice=0.60, hd95=15.0, bdf1=0.55):
    """Patch Phase3TrialTrainer and return the patched class."""
    import training.phase3_bayesian as p3_mod

    class _Configured(_FakePhase3Trainer):
        def __init__(self, config, run_dir, optuna_trial=None):
            super().__init__(
                config, run_dir, optuna_trial=optuna_trial,
                dice=dice, hd95=hd95, bdf1=bdf1,
            )

    monkeypatch.setattr(p3_mod, "Phase3TrialTrainer", _Configured)
    return _Configured


# ===========================================================================
# 1. Pure-math helpers
# ===========================================================================

class TestNormaliseWeights:
    def test_sums_to_one_for_typical_values(self):
        raw = {"alpha": 0.8, "beta": 0.6, "gamma": 0.4,
               "delta": 0.2, "epsilon": 0.1}
        norm = normalise_weights(raw)
        assert abs(sum(norm.values()) - 1.0) < 1e-9

    def test_proportions_preserved(self):
        raw = {"alpha": 2.0, "beta": 1.0, "gamma": 1.0,
               "delta": 0.0, "epsilon": 0.0}
        norm = normalise_weights(raw)
        assert abs(norm["alpha"] - 0.5) < 1e-9
        assert abs(norm["beta"]  - 0.25) < 1e-9

    def test_all_zeros_returns_uniform(self):
        raw = {"alpha": 0.0, "beta": 0.0, "gamma": 0.0,
               "delta": 0.0, "epsilon": 0.0}
        norm = normalise_weights(raw)
        assert abs(sum(norm.values()) - 1.0) < 1e-9
        expected = 0.2
        for v in norm.values():
            assert abs(v - expected) < 1e-9

    def test_single_nonzero_becomes_one(self):
        raw = {"alpha": 0.0, "beta": 1.0, "gamma": 0.0,
               "delta": 0.0, "epsilon": 0.0}
        norm = normalise_weights(raw)
        assert abs(norm["beta"] - 1.0) < 1e-9
        assert all(abs(norm[k]) < 1e-9 for k in norm if k != "beta")

    def test_returns_all_five_keys(self):
        raw = {"alpha": 0.5, "beta": 0.5, "gamma": 0.0,
               "delta": 0.0, "epsilon": 0.0}
        norm = normalise_weights(raw)
        assert set(norm.keys()) == {"alpha", "beta", "gamma", "delta", "epsilon"}

    def test_values_non_negative(self):
        raw = {"alpha": 0.3, "beta": 0.7, "gamma": 0.1,
               "delta": 0.05, "epsilon": 0.05}
        norm = normalise_weights(raw)
        assert all(v >= 0.0 for v in norm.values())


class TestComputeObjective:
    def test_perfect_segmentation(self):
        """dice=1 hd95=0 bdf1=1 → score = 0.5+0.3+0.2 = 1.0."""
        score = compute_objective(
            {"val_dice": 1.0, "val_hd95": 0.0, "val_boundary_f1": 1.0}
        )
        assert abs(score - 1.0) < 1e-9

    def test_zero_segmentation(self):
        """dice=0 hd95→worst bdf1=0 → score near 0."""
        score = compute_objective(
            {"val_dice": 0.0, "val_hd95": 999.0, "val_boundary_f1": 0.0}
        )
        assert score < 0.01

    def test_weights_sum_correctly(self):
        """Score = 0.5×0.8 + 0.3×(1-0.15) + 0.2×0.7 = 0.4+0.255+0.14 = 0.795."""
        score = compute_objective(
            {"val_dice": 0.8, "val_hd95": 15.0, "val_boundary_f1": 0.7}
        )
        expected = 0.5 * 0.8 + 0.3 * (1.0 - 15.0 / 100.0) + 0.2 * 0.7
        assert abs(score - expected) < 1e-9

    def test_missing_dice_defaults_to_zero(self):
        score = compute_objective({"val_hd95": 10.0, "val_boundary_f1": 0.5})
        expected = 0.3 * (1.0 - 10.0 / 100.0) + 0.2 * 0.5
        assert abs(score - expected) < 1e-9

    def test_none_values_treated_as_worst_case(self):
        score_none = compute_objective(
            {"val_dice": None, "val_hd95": None, "val_boundary_f1": None}
        )
        score_worst = compute_objective(
            {"val_dice": 0.0, "val_hd95": 999.0, "val_boundary_f1": 0.0}
        )
        assert abs(score_none - score_worst) < 1e-9

    def test_nan_treated_as_worst_case(self):
        score = compute_objective(
            {"val_dice": float("nan"), "val_hd95": float("nan"),
             "val_boundary_f1": float("nan")}
        )
        assert np.isfinite(score)
        assert score < 0.01

    def test_hd95_clipped_at_100(self):
        """hd95 of 200 and 999 should both give the same contribution."""
        score_200 = compute_objective(
            {"val_dice": 0.5, "val_hd95": 200.0, "val_boundary_f1": 0.5}
        )
        score_999 = compute_objective(
            {"val_dice": 0.5, "val_hd95": 999.0, "val_boundary_f1": 0.5}
        )
        assert abs(score_200 - score_999) < 1e-9

    def test_returns_finite_float(self):
        score = compute_objective(
            {"val_dice": 0.7, "val_hd95": 20.0, "val_boundary_f1": 0.6}
        )
        assert isinstance(score, float)
        assert np.isfinite(score)


# ===========================================================================
# 2. CSV persistence helpers
# ===========================================================================

class TestCSVHelpers:
    def test_init_csv_creates_file_with_header(self, tmp_path):
        csv_path = tmp_path / "trials.csv"
        _init_csv(csv_path)
        assert csv_path.exists()
        with csv_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            assert set(reader.fieldnames or []) == set(_CSV_FIELDNAMES)

    def test_init_csv_does_not_overwrite_existing(self, tmp_path):
        csv_path = tmp_path / "trials.csv"
        csv_path.write_text("existing", encoding="utf-8")
        _init_csv(csv_path)
        assert csv_path.read_text(encoding="utf-8") == "existing"

    def test_append_trial_csv_adds_row(self, tmp_path):
        csv_path = tmp_path / "trials.csv"
        _init_csv(csv_path)
        _append_trial_csv(
            csv_path,
            trial_number=0, state="COMPLETE", elapsed_s=42.5,
            raw_weights={"alpha": 0.8, "beta": 0.6, "gamma": 0.4,
                         "delta": 0.2, "epsilon": 0.1},
            norm_weights={"alpha": 0.38, "beta": 0.29, "gamma": 0.19,
                          "delta": 0.10, "epsilon": 0.05},
            extra_hp={"bce_pos_weight": 1.5, "lr": 0.001},
            score=0.75, dice=0.70, hd95=12.0, bdf1=0.65,
            best_epoch=8, pruned=False,
        )
        with csv_path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert rows[0]["trial_number"] == "0"
        assert rows[0]["state"] == "COMPLETE"
        assert rows[0]["pruned"] == "False"

    def test_multiple_appends_preserve_all_rows(self, tmp_path):
        csv_path = tmp_path / "trials.csv"
        _init_csv(csv_path)
        for i in range(5):
            _append_trial_csv(
                csv_path,
                trial_number=i, state="COMPLETE", elapsed_s=10.0,
                raw_weights={"alpha": 0.5, "beta": 0.5, "gamma": 0.0,
                             "delta": 0.0, "epsilon": 0.0},
                norm_weights={"alpha": 0.5, "beta": 0.5, "gamma": 0.0,
                              "delta": 0.0, "epsilon": 0.0},
                extra_hp={"bce_pos_weight": 1.0, "lr": 0.005},
                score=float(i) * 0.1, dice=0.5, hd95=20.0, bdf1=0.4,
                best_epoch=10, pruned=False,
            )
        with csv_path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 5
        assert [int(r["trial_number"]) for r in rows] == list(range(5))


# ===========================================================================
# 3. Integration via run_phase3_bayesian (mocked trainer)
# ===========================================================================

class TestRunPhase3Integration:
    def test_creates_phase3_subdirectory(self, tmp_path, monkeypatch, base_cfg):
        _make_runner(monkeypatch)
        run_phase3_bayesian(
            base_cfg, str(tmp_path), n_trials=2, epochs_per_trial=5,
            study_name="test_study_create"
        )
        assert _study_dir(tmp_path, "test_study_create").is_dir()

    def test_phase3_does_not_touch_other_dirs(self, tmp_path, monkeypatch, base_cfg):
        # Pre-create sibling directories that simulate Phase 1/2/MGDA output
        (tmp_path / "C-BCE").mkdir()
        (tmp_path / "MGDA-all5").mkdir()
        (tmp_path / "G-a1.0-b0.5-g0.0").mkdir()
        _make_runner(monkeypatch)
        run_phase3_bayesian(
            base_cfg, str(tmp_path), n_trials=1, epochs_per_trial=5,
            study_name="test_study_isolation"
        )
        # Sibling dirs must still exist and be empty (untouched)
        assert (tmp_path / "C-BCE").exists()
        assert (tmp_path / "MGDA-all5").exists()
        assert list((tmp_path / "C-BCE").iterdir()) == []

    def test_sqlite_db_created(self, tmp_path, monkeypatch, base_cfg):
        _make_runner(monkeypatch)
        run_phase3_bayesian(
            base_cfg, str(tmp_path), n_trials=2, epochs_per_trial=5,
            study_name="test_study_db"
        )
        assert _study_dir(tmp_path, "test_study_db").joinpath("optuna_study.db").is_file()

    def test_trials_csv_created_with_header(self, tmp_path, monkeypatch, base_cfg):
        _make_runner(monkeypatch)
        run_phase3_bayesian(
            base_cfg, str(tmp_path), n_trials=2, epochs_per_trial=5,
            study_name="test_study_csv"
        )
        csv_path = _study_dir(tmp_path, "test_study_csv") / "trials.csv"
        assert csv_path.is_file()
        with csv_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            assert "trial_number" in (reader.fieldnames or [])
            assert "score" in (reader.fieldnames or [])

    def test_csv_has_correct_number_of_rows(self, tmp_path, monkeypatch, base_cfg):
        _make_runner(monkeypatch)
        run_phase3_bayesian(
            base_cfg, str(tmp_path), n_trials=3, epochs_per_trial=5,
            study_name="test_study_rows"
        )
        with (_study_dir(tmp_path, "test_study_rows") / "trials.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 3

    def test_best_json_created(self, tmp_path, monkeypatch, base_cfg):
        _make_runner(monkeypatch)
        run_phase3_bayesian(
            base_cfg, str(tmp_path), n_trials=2, epochs_per_trial=5,
            study_name="test_study_best"
        )
        best_json = _study_dir(tmp_path, "test_study_best") / "phase3_best.json"
        assert best_json.is_file()
        data = json.loads(best_json.read_text(encoding="utf-8"))
        assert "best_trial" in data
        assert "best_score" in data
        assert "best_norm_weights" in data

    def test_best_json_norm_weights_sum_to_one(self, tmp_path, monkeypatch, base_cfg):
        _make_runner(monkeypatch)
        run_phase3_bayesian(
            base_cfg, str(tmp_path), n_trials=3, epochs_per_trial=5,
            study_name="test_study_norm"
        )
        data = json.loads(
            (_study_dir(tmp_path, "test_study_norm") / "phase3_best.json").read_text(encoding="utf-8")
        )
        weights = data["best_norm_weights"]
        total = sum(weights[k] for k in
                    ["alpha", "beta", "gamma", "delta", "epsilon"])
        assert abs(total - 1.0) < 1e-6

    def test_trial_dirs_created(self, tmp_path, monkeypatch, base_cfg):
        _make_runner(monkeypatch)
        run_phase3_bayesian(
            base_cfg, str(tmp_path), n_trials=3, epochs_per_trial=5,
            study_name="test_study_dirs"
        )
        trial_dirs = sorted(_study_dir(tmp_path, "test_study_dirs").glob("trial_*"))
        assert len(trial_dirs) == 3
        assert all(d.name.startswith("trial_") for d in trial_dirs)

    def test_trial_dirs_contain_summary_json(self, tmp_path, monkeypatch, base_cfg):
        _make_runner(monkeypatch)
        run_phase3_bayesian(
            base_cfg, str(tmp_path), n_trials=2, epochs_per_trial=5,
            study_name="test_study_summary"
        )
        for trial_dir in _study_dir(tmp_path, "test_study_summary").glob("trial_*"):
            assert (trial_dir / "summary.json").is_file()

    def test_trainer_called_once_per_trial(self, tmp_path, monkeypatch, base_cfg):
        _make_runner(monkeypatch)
        run_phase3_bayesian(
            base_cfg, str(tmp_path), n_trials=4, epochs_per_trial=5,
            study_name="test_study_calls"
        )
        assert len(_FakePhase3Trainer._calls) == 4

    def test_config_id_uses_trial_number(self, tmp_path, monkeypatch, base_cfg):
        _make_runner(monkeypatch)
        run_phase3_bayesian(
            base_cfg, str(tmp_path), n_trials=3, epochs_per_trial=5,
            study_name="test_study_id"
        )
        config_ids = [c["config_id"] for c in _FakePhase3Trainer._calls]
        for cid in config_ids:
            assert cid.startswith("p3_trial_")

    def test_epochs_per_trial_respected(self, tmp_path, monkeypatch, base_cfg):
        _make_runner(monkeypatch)
        run_phase3_bayesian(
            base_cfg, str(tmp_path), n_trials=2, epochs_per_trial=7,
            study_name="test_study_epochs"
        )
        for call_cfg in _FakePhase3Trainer._calls:
            assert call_cfg["epochs"] == 7

    def test_base_cfg_epochs_not_used_for_trials(self, tmp_path, monkeypatch, base_cfg):
        """Base config has 30 epochs; each trial must use epochs_per_trial=10."""
        base_cfg_copy = {**base_cfg, "epochs": 30}
        _make_runner(monkeypatch)
        run_phase3_bayesian(
            base_cfg_copy, str(tmp_path), n_trials=2, epochs_per_trial=10,
            study_name="test_study_base_epochs"
        )
        for call_cfg in _FakePhase3Trainer._calls:
            assert call_cfg["epochs"] == 10

    def test_norm_weights_in_mask_loss_sum_to_one(self, tmp_path, monkeypatch, base_cfg):
        _make_runner(monkeypatch)
        run_phase3_bayesian(
            base_cfg, str(tmp_path), n_trials=3, epochs_per_trial=5,
            study_name="test_study_weights_sum"
        )
        for call_cfg in _FakePhase3Trainer._calls:
            ml = call_cfg["mask_loss"]
            total = sum(ml[k] for k in ["alpha", "beta", "gamma", "delta", "epsilon"])
            assert abs(total - 1.0) < 1e-6, (
                f"Expected normalised weights to sum to 1; got {total}"
            )

    def test_search_space_contains_only_the_five_loss_weights(
        self, tmp_path, monkeypatch, base_cfg
    ):
        _make_runner(monkeypatch)
        run_phase3_bayesian(
            base_cfg, str(tmp_path), n_trials=1, epochs_per_trial=5,
            study_name="test_weight_only_search",
        )
        import optuna
        db_path = _study_dir(tmp_path, "test_weight_only_search") / "optuna_study.db"
        storage_url = f"sqlite:///{db_path.as_posix()}"
        storage_url = storage_url.replace("{", "{{").replace("}", "}}")
        study = optuna.load_study(
            study_name="test_weight_only_search",
            storage=storage_url,
        )
        assert set(study.best_trial.params) == {
            "alpha", "beta", "gamma", "delta", "epsilon"
        }

    def test_bce_pos_weight_remains_fixed_to_base_config(self, tmp_path, monkeypatch, base_cfg):
        _make_runner(monkeypatch)
        run_phase3_bayesian(
            base_cfg, str(tmp_path), n_trials=2, epochs_per_trial=5,
            study_name="test_study_posw"
        )
        for call_cfg in _FakePhase3Trainer._calls:
            assert "bce_pos_weight" in call_cfg["mask_loss"]
            assert call_cfg["mask_loss"]["bce_pos_weight"] == 1.0

    def test_lr_is_fixed_to_base_config(self, tmp_path, monkeypatch, base_cfg):
        _make_runner(monkeypatch)
        run_phase3_bayesian(
            base_cfg, str(tmp_path), n_trials=3, epochs_per_trial=5,
            study_name="test_study_lr"
        )
        for call_cfg in _FakePhase3Trainer._calls:
            assert call_cfg["lr"] == base_cfg["lr"]

    def test_return_value_has_required_keys(self, tmp_path, monkeypatch, base_cfg):
        _make_runner(monkeypatch)
        result = run_phase3_bayesian(
            base_cfg, str(tmp_path), n_trials=2, epochs_per_trial=5,
            study_name="test_study_return"
        )
        assert "best_trial" in result
        assert "best_score" in result
        assert "best_norm_weights" in result

    def test_csv_state_column_is_complete(self, tmp_path, monkeypatch, base_cfg):
        _make_runner(monkeypatch)
        run_phase3_bayesian(
            base_cfg, str(tmp_path), n_trials=3, epochs_per_trial=5,
            study_name="test_study_state"
        )
        with (_study_dir(tmp_path, "test_study_state") / "trials.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for row in rows:
            assert row["state"] in {"COMPLETE", "PRUNED", "FAIL"}


# ===========================================================================
# 4. Resume / idempotency
# ===========================================================================

class TestResume:
    def test_study_lock_rejects_a_second_writer(self, tmp_path):
        lock_path = tmp_path / "phase3" / ".study.lock"
        with _phase3_study_lock(lock_path):
            with pytest.raises(RuntimeError, match="already holds"):
                with _phase3_study_lock(lock_path):
                    pass

    def test_study_lock_rejects_a_second_process(self, tmp_path):
        lock_path = tmp_path / "phase3" / ".study.lock"
        child = (
            "from pathlib import Path; "
            "from training.phase3_bayesian import _phase3_study_lock; "
            f"lock_path=Path({str(lock_path)!r}); "
            "exec('with _phase3_study_lock(lock_path):\\n    pass')"
        )
        with _phase3_study_lock(lock_path):
            result = subprocess.run(
                [sys.executable, "-c", child],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
                timeout=30,
            )
        assert result.returncode != 0
        assert "already holds" in result.stderr

    def test_failed_attempt_does_not_consume_completed_trial_target(
        self, tmp_path, monkeypatch, base_cfg
    ):
        import training.phase3_bayesian as p3_mod

        class _FailOnceTrainer(_FakePhase3Trainer):
            attempts = 0

            def train(self):
                self.__class__.attempts += 1
                if self.__class__.attempts == 1:
                    raise RuntimeError("synthetic interrupted trial")
                return super().train()

        monkeypatch.setattr(p3_mod, "Phase3TrialTrainer", _FailOnceTrainer)
        with pytest.raises(RuntimeError, match="synthetic interrupted trial"):
            run_phase3_bayesian(
                base_cfg, str(tmp_path), n_trials=1, epochs_per_trial=5,
                study_name="test_resume_failed_attempt",
            )

        result = run_phase3_bayesian(
            base_cfg, str(tmp_path), n_trials=1, epochs_per_trial=5,
            study_name="test_resume_failed_attempt",
        )
        assert result["best_trial"] == 1
        assert _FailOnceTrainer.attempts == 2

    def test_second_call_runs_no_new_trials_when_target_met(
        self, tmp_path, monkeypatch, base_cfg
    ):
        """Running with n_trials=3 twice should call trainer exactly 3 times."""
        _make_runner(monkeypatch)
        run_phase3_bayesian(
            base_cfg, str(tmp_path), n_trials=3, epochs_per_trial=5,
            study_name="test_resume_exact"
        )
        # Reset call log to detect second run
        _FakePhase3Trainer._calls = []
        run_phase3_bayesian(
            base_cfg, str(tmp_path), n_trials=3, epochs_per_trial=5,
            study_name="test_resume_exact"
        )
        assert len(_FakePhase3Trainer._calls) == 0

    def test_increasing_n_trials_runs_additional_trials(
        self, tmp_path, monkeypatch, base_cfg
    ):
        """First run: 2 trials; second run with n_trials=5: 3 more."""
        _make_runner(monkeypatch)
        run_phase3_bayesian(
            base_cfg, str(tmp_path), n_trials=2, epochs_per_trial=5,
            study_name="test_resume_increase"
        )
        _FakePhase3Trainer._calls = []
        run_phase3_bayesian(
            base_cfg, str(tmp_path), n_trials=5, epochs_per_trial=5,
            study_name="test_resume_increase"
        )
        assert len(_FakePhase3Trainer._calls) == 3

    def test_csv_rebuilt_when_missing_on_resume(
        self, tmp_path, monkeypatch, base_cfg
    ):
        """Delete trials.csv; second call should rebuild it from SQLite."""
        _make_runner(monkeypatch)
        run_phase3_bayesian(
            base_cfg, str(tmp_path), n_trials=3, epochs_per_trial=5,
            study_name="test_resume_csv"
        )
        (_study_dir(tmp_path, "test_resume_csv") / "trials.csv").unlink()
        _FakePhase3Trainer._calls = []
        run_phase3_bayesian(
            base_cfg, str(tmp_path), n_trials=3, epochs_per_trial=5,
            study_name="test_resume_csv"
        )
        assert (_study_dir(tmp_path, "test_resume_csv") / "trials.csv").is_file()

    def test_best_json_updated_after_additional_trials(
        self, tmp_path, monkeypatch, base_cfg
    ):
        _make_runner(monkeypatch, dice=0.5)
        run_phase3_bayesian(
            base_cfg, str(tmp_path), n_trials=2, epochs_per_trial=5,
            study_name="test_resume_best"
        )
        old_best = json.loads(
            (_study_dir(tmp_path, "test_resume_best") / "phase3_best.json").read_text(encoding="utf-8")
        )

        # Second call with better metrics
        _make_runner(monkeypatch, dice=0.9, hd95=5.0, bdf1=0.85)
        run_phase3_bayesian(
            base_cfg, str(tmp_path), n_trials=4, epochs_per_trial=5,
            study_name="test_resume_best"
        )
        new_best = json.loads(
            (_study_dir(tmp_path, "test_resume_best") / "phase3_best.json").read_text(encoding="utf-8")
        )
        # Score should be higher with better metrics
        assert new_best["best_score"] >= old_best["best_score"]


# ===========================================================================
# 5. Pruning propagation
# ===========================================================================

class TestPruning:
    def test_trial_pruned_exception_does_not_crash_study(
        self, tmp_path, monkeypatch, base_cfg
    ):
        """A TrialPruned mid-epoch must not kill the study; study continues."""
        import optuna
        import training.phase3_bayesian as p3_mod

        pruned_once = {"done": False}

        class _PruningTrainer:
            _calls: list = []

            def __init__(self, config, run_dir, optuna_trial=None):
                self.config = config
                self.run_dir = Path(run_dir)
                self._trial = optuna_trial
                self.__class__._calls.append(config)

            def train(self) -> dict:
                self.run_dir.mkdir(parents=True, exist_ok=True)
                # First call: raise TrialPruned; subsequent calls succeed
                if not pruned_once["done"]:
                    pruned_once["done"] = True
                    raise optuna.exceptions.TrialPruned()
                metrics = {"val_dice": 0.5, "val_hd95": 20.0,
                           "val_boundary_f1": 0.4}
                summary = {
                    "config_id":    self.config.get("config_id"),
                    "best_epoch":   5,
                    "best_metrics": metrics, "final_metrics": metrics,
                }
                (self.run_dir / "summary.json").write_text(
                    json.dumps(summary), encoding="utf-8"
                )
                (self.run_dir / "history.json").write_text(
                    json.dumps([{"epoch": e} for e in range(1, 6)]),
                    encoding="utf-8",
                )
                (self.run_dir / "latest_model.pth").write_bytes(b"fake")
                return summary

        monkeypatch.setattr(p3_mod, "Phase3TrialTrainer", _PruningTrainer)

        result = run_phase3_bayesian(
            base_cfg, str(tmp_path), n_trials=3, epochs_per_trial=5,
            study_name="test_pruning"
        )
        # The pruned attempt is retried so the target reaches 3 completions.
        assert "best_trial" in result
        assert len(_PruningTrainer._calls) == 4


# ===========================================================================
# 6. ExperimentRunner.run_phase3 integration
# ===========================================================================

class TestExperimentRunnerPhase3:
    def test_runner_delegates_to_new_module(self, tmp_path, monkeypatch, base_cfg):
        from training.experiment import ExperimentRunner
        import training.phase3_bayesian as p3_mod

        called: list[dict] = []

        def _fake_run_phase3(base_cfg, output_dir, n_trials, epochs_per_trial,
                             seed, study_name):
            called.append({
                "n_trials": n_trials,
                "epochs_per_trial": epochs_per_trial,
                "study_name": study_name,
            })
            return {"best_trial": 0, "best_score": 0.5,
                    "best_norm_weights": {}}

        monkeypatch.setattr(p3_mod, "run_phase3_bayesian", _fake_run_phase3)
        # ExperimentRunner imports run_bayesian_optimisation from experiment.py
        # which now delegates to run_phase3_bayesian in phase3_bayesian.py
        import training.experiment as exp_mod
        monkeypatch.setattr(exp_mod, "run_bayesian_optimisation",
                            lambda **kw: _fake_run_phase3(**kw))

        runner = ExperimentRunner(base_cfg, str(tmp_path))
        runner.run_phase3(n_trials=5, epochs_per_trial=8,
                          study_name="runner_test")

    def test_runner_phase3_returns_dict(self, tmp_path, monkeypatch, base_cfg):
        from training.experiment import ExperimentRunner
        import training.experiment as exp_mod

        monkeypatch.setattr(
            exp_mod, "run_bayesian_optimisation",
            lambda **_kw: {"best_trial": 0, "best_score": 0.7,
                           "best_norm_weights": {"alpha": 1.0, "beta": 0.0,
                                                 "gamma": 0.0, "delta": 0.0,
                                                 "epsilon": 0.0}},
        )
        runner = ExperimentRunner(base_cfg, str(tmp_path))
        result = runner.run_phase3(n_trials=2)
        assert isinstance(result, dict)
        assert "best_trial" in result


# ===========================================================================
# 7. Backward compatibility of run_bayesian_optimisation wrapper
# ===========================================================================

class TestBackwardCompatWrapper:
    def test_wrapper_signature_accepts_old_kwargs(
        self, tmp_path, monkeypatch, base_cfg
    ):
        """run_bayesian_optimisation must accept n_jobs without crashing."""
        import training.phase3_bayesian as p3_mod

        monkeypatch.setattr(
            p3_mod, "run_phase3_bayesian",
            lambda **_: {"best_trial": 0, "best_score": 0.5,
                         "best_norm_weights": {}},
        )
        from training.experiment import run_bayesian_optimisation
        # n_jobs should be accepted (and silently ignored)
        result = run_bayesian_optimisation(
            base_cfg=base_cfg,
            output_dir=str(tmp_path),
            n_trials=2,
            n_jobs=4,  # old caller passes this
        )
        assert "best_trial" in result

    def test_wrapper_passes_seed_and_study_name(
        self, tmp_path, monkeypatch, base_cfg
    ):
        import training.phase3_bayesian as p3_mod

        received: list[dict] = []

        def _capture(**kw):
            received.append(kw)
            return {"best_trial": 0, "best_score": 0.5, "best_norm_weights": {}}

        monkeypatch.setattr(p3_mod, "run_phase3_bayesian", _capture)
        from training.experiment import run_bayesian_optimisation
        run_bayesian_optimisation(
            base_cfg=base_cfg,
            output_dir=str(tmp_path),
            n_trials=5,
            seed=99,
            study_name="custom_name",
        )
        assert received[0]["seed"] == 99
        assert received[0]["study_name"] == "custom_name"


# ===========================================================================
# 8. CompositeMaskLoss.from_config accepts bce_pos_weight
# ===========================================================================

class TestLossesFromConfig:
    def test_from_config_accepts_bce_pos_weight(self):
        from training.losses import CompositeMaskLoss
        loss = CompositeMaskLoss.from_config(
            {"alpha": 0.5, "beta": 0.5, "bce_pos_weight": 2.5}
        )
        assert loss.bce.pos_weight.item() == pytest.approx(2.5, rel=1e-5)

    def test_from_config_default_bce_pos_weight_is_one(self):
        from training.losses import CompositeMaskLoss
        loss = CompositeMaskLoss.from_config({"alpha": 1.0})
        assert loss.bce.pos_weight.item() == pytest.approx(1.0, rel=1e-5)

    def test_from_config_bce_pos_weight_does_not_break_existing_configs(self):
        """Configs that do NOT contain bce_pos_weight still build correctly."""
        from training.losses import CompositeMaskLoss
        # Typical Phase 1 / Phase 2 config (no bce_pos_weight key)
        loss = CompositeMaskLoss.from_config(
            {"alpha": 0.3, "beta": 0.7, "gamma": 0.0, "delta": 0.0, "epsilon": 0.0}
        )
        assert loss.alpha == pytest.approx(0.3, rel=1e-5)
        assert loss.beta  == pytest.approx(0.7, rel=1e-5)


# ===========================================================================
# 9. normalise_weights preserves Phase 1/2 existing weight structure
# ===========================================================================

class TestNormaliseWeightsCompatibility:
    @pytest.mark.parametrize("alpha,beta,gamma,delta,epsilon", [
        (1.0, 0.0, 0.0, 0.0, 0.0),   # C-BCE
        (0.0, 1.0, 0.0, 0.0, 0.0),   # C-DICE
        (0.3, 0.5, 0.2, 0.0, 0.0),   # typical Phase 2 config
        (0.3, 0.5, 0.1, 0.05, 0.05), # all five
    ])
    def test_normalised_weights_sum_to_one(
        self, alpha, beta, gamma, delta, epsilon
    ):
        raw = {"alpha": alpha, "beta": beta, "gamma": gamma,
               "delta": delta, "epsilon": epsilon}
        norm = normalise_weights(raw)
        assert abs(sum(norm.values()) - 1.0) < 1e-9


# ===========================================================================
# 10. Script argument parsing (dry-run)
# ===========================================================================

class TestCLIScript:
    def test_dry_run_prints_config_and_exits_without_training(
        self, tmp_path, monkeypatch, capsys
    ):
        """--dry-run should print config and return without calling trainer."""
        import training.phase3_bayesian as p3_mod

        called = {"n": 0}

        def _capture(**_):
            called["n"] += 1
            return {}

        monkeypatch.setattr(p3_mod, "run_phase3_bayesian", _capture)
        from scripts.run_phase3_bayesian import main

        main([
            "--data-root", "/fake/data",
            "--out", str(tmp_path),
            "--n-trials", "5",
            "--dry-run",
        ])

        captured = capsys.readouterr()
        assert "data_root" in captured.out
        assert called["n"] == 0   # trainer never called

    def test_main_builds_expected_config_shape(
        self, tmp_path, monkeypatch
    ):
        import training.phase3_bayesian as p3_mod

        received: list[dict] = []

        def _capture(base_cfg, **_):
            received.append(deepcopy(base_cfg))
            return {"best_trial": 0, "best_score": 0.5, "best_norm_weights": {}}

        monkeypatch.setattr(p3_mod, "run_phase3_bayesian", _capture)
        from scripts.run_phase3_bayesian import main

        main([
            "--data-root", "/fake/data",
            "--out", str(tmp_path),
            "--n-trials", "2",
            "--epochs-per-trial", "5",
            "--batch-size", "8",
        ])

        assert len(received) == 1
        cfg = received[0]
        assert cfg["batch_size"] == 8
        assert cfg["training_strategy"] == "fixed"
        assert cfg["data_root"].replace("\\", "/").endswith("/fake/data")
