"""
Experiment runner for loss weight optimisation.
Phase 1: Single-loss ablations
Phase 2: Grid search (alpha, beta, gamma)
Phase 3: Bayesian optimisation via Optuna
"""
import os
import json
import logging
import itertools
import random
import re
import math
from pathlib import Path
from copy import deepcopy
from typing import Optional

logger = logging.getLogger(__name__)


def _reject_json_constant(value: str):
    raise ValueError(f"Invalid JSON constant: {value}")


# ---------------------------------------------------------------------------
# Base config shared by all experiments
# ---------------------------------------------------------------------------
BASE_CONFIG = {
    "training_strategy": "fixed",
    "num_classes": 2,
    "pretrained_backbone": True,
    "trainable_backbone_layers": 3,
    "min_size": 64,
    "max_size": 256,
    "patch_size": 64,
    "batch_size": 4,
    "lr": 0.005,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
    "epochs": 30,
    "fold": 0,
    "train_subsets": None,
    "validation_subset": None,
    "test_subset": None,
    "target_spacing_mm": (1.0, 1.0, 1.0),
    "num_workers": 2,
    "data_root": "./data/luna16",
    # Set this to the output of preprocess_cache.py to eliminate SSD bottleneck:
    # "cache_dir": "./data/patches_cache",
    "cache_dir": None,
}


# ---------------------------------------------------------------------------
# Phase 1: Single-loss ablations
# ---------------------------------------------------------------------------
PHASE1_CONFIGS = [
    {"config_id": "C-BCE",      "mask_loss": {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0, "epsilon": 0.0}},
    {"config_id": "C-DICE",     "mask_loss": {"alpha": 0.0, "beta": 1.0, "gamma": 0.0, "delta": 0.0, "epsilon": 0.0}},
    {"config_id": "C-BDRY",     "mask_loss": {"alpha": 0.0, "beta": 0.0, "gamma": 1.0, "delta": 0.0, "epsilon": 0.0}},
    {"config_id": "C-FOCAL",    "mask_loss": {"alpha": 0.0, "beta": 0.0, "gamma": 0.0, "delta": 1.0, "epsilon": 0.0}},
    {"config_id": "C-HD95",     "mask_loss": {"alpha": 0.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0, "epsilon": 1.0}},
]


# ---------------------------------------------------------------------------
# Phase 2: Grid search over (alpha, beta, gamma)
# ---------------------------------------------------------------------------
GRID_VALUES = [0.0, 0.3, 0.5, 0.7, 1.0]


def select_phase2_configs(max_configs: Optional[int] = None) -> list[dict]:
    """Return the deterministic Phase 2 selection (seed 42 when sampling)."""
    configs = generate_phase2_configs()
    if max_configs is not None:
        if max_configs < 1:
            raise ValueError("max_configs must be at least 1")
        if max_configs < len(configs):
            configs = random.Random(42).sample(configs, k=max_configs)
    return configs


def generate_phase2_configs() -> list[dict]:
    configs = []
    for alpha, beta, gamma in itertools.product(GRID_VALUES, repeat=3):
        # Skip all-zero
        if alpha == 0.0 and beta == 0.0 and gamma == 0.0:
            continue
        config_id = f"G-a{alpha:.1f}-b{beta:.1f}-g{gamma:.1f}".replace(".", "")
        configs.append({
            "config_id": config_id,
            "mask_loss": {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "delta": 0.0,
                "epsilon": 0.0,
            },
        })
    return configs


PHASE4_TASK_SUBSETS = (
    ("bce", "dice"),
    ("bce", "boundary"),
    ("dice", "boundary"),
    ("bce", "focal"),
    ("bce", "dice", "boundary"),
    ("bce", "dice", "focal"),
    ("bce", "dice", "hd"),
    ("bce", "dice", "boundary", "focal"),
    ("bce", "dice", "boundary", "focal", "hd"),
)
_MGDA_TASK_CONFIG_WEIGHTS = {
    "bce": "alpha", "dice": "beta", "boundary": "gamma",
    "focal": "delta", "hd": "epsilon",
}
_MGDA_TASK_CODES = {"bce": "B", "dice": "D", "boundary": "R", "focal": "F", "hd": "H"}


def generate_phase4_configs(max_iter: int = 200, tol: float = 1e-8) -> list[dict]:
    """Generate the deterministic 18-config MGDA task/scope screening set."""
    configs = []
    for tasks in PHASE4_TASK_SUBSETS:
        code = "".join(_MGDA_TASK_CODES[task] for task in tasks)
        weights = {
            weight: float(task in tasks)
            for task, weight in _MGDA_TASK_CONFIG_WEIGHTS.items()
        }
        for scope in ("mask_head", "full"):
            configs.append({
                "config_id": f"MGDA-{code}-{scope}",
                "training_strategy": "mgda",
                "use_mgda": True,
                "mgda_use_configured_tasks": True,
                "mgda_tasks": list(tasks),
                "mgda_param_scope": scope,
                "mgda_max_iter": int(max_iter),
                "mgda_tol": float(tol),
                "mask_loss": weights,
            })
    return configs


# ---------------------------------------------------------------------------
# Phase 3: Bayesian Optimisation via Optuna
# ---------------------------------------------------------------------------
def run_bayesian_optimisation(
    base_cfg: dict,
    output_dir: str,
    n_trials: int = 30,
    n_jobs: int = 1,
    epochs_per_trial: int = 10,
    seed: int = 42,
    study_name: str = "lung_nodule_loss_weights_v1",
) -> dict:
    """Backward-compatible entry point for the persistent Phase 3 study.

    ``n_jobs`` remains accepted for older callers, but trials run sequentially
    to share the single training GPU safely.
    """
    from .phase3_bayesian import run_phase3_bayesian

    return run_phase3_bayesian(
        base_cfg=base_cfg,
        output_dir=output_dir,
        n_trials=n_trials,
        epochs_per_trial=epochs_per_trial,
        seed=seed,
        study_name=study_name,
    )


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------
class ExperimentRunner:
    """
    Runs Phase 1, 2, or 3 experiments sequentially,
    collecting results into a summary CSV/JSON.
    """

    def __init__(self, base_config: dict, output_dir: str):
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[dict] = []

    def _upsert_result(self, result: dict) -> None:
        """Keep one result per config ID while retaining prior result order."""
        config_id = result.get("config_id")
        if config_id is None:
            self.results.append(result)
            return

        replaced = False
        merged = []
        for existing in self.results:
            if existing.get("config_id") == config_id:
                if not replaced:
                    merged.append(result)
                    replaced = True
            else:
                merged.append(existing)
        if not replaced:
            merged.append(result)
        self.results = merged

    def _load_prior_results(self) -> None:
        """Load aggregate results so resume never drops earlier completions."""
        results_path = self.output_dir / "all_results.json"
        if not results_path.exists():
            return
        try:
            with results_path.open(encoding="utf-8") as f:
                saved_results = json.load(f, parse_constant=_reject_json_constant)
        except (OSError, ValueError) as exc:
            logger.warning("Could not load prior results from %s: %s", results_path, exc)
            return
        if not isinstance(saved_results, list):
            logger.warning("Ignoring non-list prior results in %s", results_path)
            return
        for result in saved_results:
            if isinstance(result, dict) and isinstance(result.get("config_id"), str):
                self._upsert_result(result)

    def _completed_summary(self, config_id: str, requested_epochs: int) -> Optional[dict]:
        """Return a summary only when all required completion artifacts validate."""
        run_dir = self.output_dir / config_id
        summary_path = run_dir / "summary.json"
        history_path = run_dir / "history.json"
        checkpoint_path = run_dir / "latest_model.pth"
        try:
            with summary_path.open(encoding="utf-8") as f:
                summary = json.load(f, parse_constant=_reject_json_constant)
            if not isinstance(summary, dict) or summary.get("config_id") != config_id:
                logger.info("Resume: %s has a missing or mismatched summary; rerunning", config_id)
                return None
            with history_path.open(encoding="utf-8") as f:
                history = json.load(f, parse_constant=_reject_json_constant)
        except (OSError, ValueError) as exc:
            logger.info("Resume: %s has invalid or incomplete artifacts (%s); rerunning", config_id, exc)
            return None

        if not isinstance(history, list) or len(history) < requested_epochs:
            logger.info("Resume: %s has fewer than %d completed epochs; rerunning", config_id, requested_epochs)
            return None
        completed_epochs = [
            entry.get("epoch") for entry in history[:requested_epochs]
            if isinstance(entry, dict)
        ]
        if completed_epochs != list(range(1, requested_epochs + 1)):
            logger.info("Resume: %s history does not contain sequential completed epochs; rerunning", config_id)
            return None
        if not checkpoint_path.is_file():
            logger.info("Resume: %s is missing latest_model.pth; rerunning", config_id)
            return None
        return summary

    def _run_single(self, cfg_override: dict) -> dict:
        from .trainer import build_trainer
        # A normal run may reuse an output directory that already contains
        # completed experiments. Load their aggregate before the first new
        # result is saved so `_run_single` cannot replace it with just this run.
        if not self.results:
            self._load_prior_results()
        cfg = {**deepcopy(self.base_config), **cfg_override}
        run_dir = self.output_dir / cfg["config_id"]
        trainer = build_trainer(cfg, str(run_dir))
        summary = trainer.train()
        self._upsert_result(summary)
        self._save_results()
        return summary

    def _save_results(self):
        with open(self.output_dir / "all_results.json", "w") as f:
            json.dump(self.results, f, indent=2, allow_nan=False)
        # Also save CSV
        import csv
        if self.results:
            keys = list(dict.fromkeys(
                key
                for result in self.results
                for key, value in result.items()
                if not isinstance(value, dict)
            ))
            with open(self.output_dir / "all_results.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                for r in self.results:
                    flat = {k: v for k, v in r.items() if not isinstance(v, dict)}
                    writer.writerow(flat)

    def run_phase1(self):
        logger.info("=== PHASE 1: Single-loss ablations ===")
        for cfg_override in PHASE1_CONFIGS:
            logger.info(f"Running {cfg_override['config_id']}...")
            self._run_single(cfg_override)
        logger.info("Phase 1 complete.")

    def run_mgda(
        self,
        config_id: Optional[str] = None,
        profile: str = "raw_mask_head",
    ) -> dict:
        """Run one named MGDA profile without changing the fixed-weight path.

        Profiles compare the original raw-gradient approximation with a broader
        shared-path scope and a scale-normalized variant. Each profile gets its
        own output directory unless an explicit config ID is supplied.
        """
        profiles = {
            "raw_mask_head": ("raw", "mask_head"),
            "raw_shared_path": ("raw", "shared_mask_path"),
            "unit_shared_path": ("unit", "shared_mask_path"),
        }
        if profile not in profiles:
            raise ValueError(f"Unknown MGDA profile {profile!r}; choose from {sorted(profiles)}")
        epochs = int(self.base_config.get("epochs", BASE_CONFIG["epochs"]))
        normalization, parameter_scope = profiles[profile]
        default_id = (
            f"MGDA-all5-ep{epochs}"
            if profile == "raw_mask_head"
            else f"MGDA-{profile}-all5-ep{epochs}"
        )
        experiment_id = config_id or default_id
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", experiment_id):
            raise ValueError("MGDA config_id may contain only letters, numbers, '.', '_' and '-'")
        run_dir = self.output_dir / experiment_id
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"MGDA output already exists at {run_dir}; choose a new --config-id or --out"
            )
        config = {
            "config_id": experiment_id,
            "training_strategy": "mgda",
            "mgda_profile": profile,
            "mgda_gradient_normalization": normalization,
            "mgda_parameter_scope": parameter_scope,
            # These unit entries enable all five differentiable components;
            # MGDA computes the effective simplex weights per batch.
            "mask_loss": {
                "alpha": 1.0,
                "beta": 1.0,
                "gamma": 1.0,
                "delta": 1.0,
                "epsilon": 1.0,
            },
        }
        logger.info(
            "=== MGDA profile=%s normalization=%s scope=%s (%d epochs) ===",
            profile, normalization, parameter_scope, epochs,
        )
        summary = self._run_single(config)
        comparison = self._compare_mgda_with_baseline(summary, run_dir)
        if comparison is not None:
            summary["baseline_comparison"] = comparison
            with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, allow_nan=False)
            self._upsert_result(summary)
            self._save_results()
        return summary

    def _compare_mgda_with_baseline(self, summary: dict, run_dir: Path) -> Optional[dict]:
        """Compare measured best-epoch metrics with the local G-all5 baseline."""
        baseline_dir = Path(self.base_config.get("baseline_dir", "experiments/G-all5-ep10"))
        if not baseline_dir.is_absolute():
            baseline_dir = Path.cwd() / baseline_dir
        summary_path = baseline_dir / "summary.json"
        if not summary_path.is_file():
            logger.warning("MGDA completed; no baseline summary found at %s", summary_path)
            return None

        try:
            with summary_path.open(encoding="utf-8") as f:
                baseline = json.load(f, parse_constant=_reject_json_constant)
            with (baseline_dir / "config.json").open(encoding="utf-8") as f:
                baseline_config = json.load(f, parse_constant=_reject_json_constant)
        except (OSError, ValueError) as exc:
            logger.warning("Could not validate baseline artifacts: %s", exc)
            return None

        baseline_metrics = baseline.get("best_metrics") or baseline.get("final_metrics") or {}
        mgda_metrics = summary.get("best_metrics") or summary.get("final_metrics") or {}
        metric_keys = ("val_dice", "val_hd95", "val_boundary_f1")
        if any(
            not isinstance(metrics.get(key), (float, int))
            or not math.isfinite(float(metrics[key]))
            for metrics in (baseline_metrics, mgda_metrics)
            for key in metric_keys
        ):
            logger.warning("MGDA/baseline comparison skipped: one or more metrics are unavailable")
            return None

        comparison = {
            "baseline_config_id": baseline.get("config_id", "G-all5-ep10"),
            "mgda_config_id": summary.get("config_id"),
            "metric_source": "best validation epoch in each run",
            "baseline_validation_split_metadata": (
                "UID-level split metadata was not recorded in this legacy baseline"
            ),
            "baseline_config": baseline_config,
            "baseline_metrics": {key: float(baseline_metrics[key]) for key in metric_keys},
            "mgda_metrics": {key: float(mgda_metrics[key]) for key in metric_keys},
            "delta_mgda_minus_baseline": {
                key: float(mgda_metrics[key]) - float(baseline_metrics[key])
                for key in metric_keys
            },
            "interpretation": (
                "Validation comparison only; it is not an independent test result. "
                "Confirm the split and preprocessing match before making a research claim."
            ),
        }
        path = run_dir / "baseline_comparison.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, allow_nan=False)
        logger.info("Wrote measured baseline comparison to %s", path)
        return comparison

    def run_phase2(self, max_configs: Optional[int] = None, resume: bool = False):
        logger.info("=== PHASE 2: Grid search (alpha, beta, gamma) ===")
        configs = select_phase2_configs(max_configs)
        if resume:
            self._load_prior_results()
        logger.info(f"Running {len(configs)} configurations...")
        requested_epochs = int(self.base_config.get("epochs", BASE_CONFIG["epochs"]))
        if requested_epochs < 1:
            raise ValueError("epochs must be at least 1")
        for i, cfg_override in enumerate(configs):
            config_id = cfg_override["config_id"]
            logger.info(f"[{i+1}/{len(configs)}] {config_id}")
            if resume:
                completed = self._completed_summary(config_id, requested_epochs)
                if completed is not None:
                    self._upsert_result(completed)
                    self._save_results()
                    logger.info("Resume: skipping completed configuration %s", config_id)
                    continue
            self._run_single(cfg_override)
        if resume:
            self._save_results()
        logger.info("Phase 2 complete.")

    def run_phase3(
        self,
        n_trials: int = 30,
        epochs_per_trial: int = 10,
        seed: int = 42,
        study_name: str = "lung_nodule_loss_weights_v1",
    ) -> dict:
        logger.info(
            "=== PHASE 3: Bayesian Optimisation | trials=%d epochs/trial=%d ===",
            n_trials, epochs_per_trial,
        )
        result = run_bayesian_optimisation(
            base_cfg=self.base_config,
            output_dir=str(self.output_dir),
            n_trials=n_trials,
            epochs_per_trial=epochs_per_trial,
            seed=seed,
            study_name=study_name,
        )
        logger.info("Phase 3 best: %s", result)
        return result

    def run_phase4(
        self, max_iter: int = 200, tol: float = 1e-8,
        resume: bool = False, log_conflict: bool = True,
    ) -> list[dict]:
        """Run the deterministic MGDA objective-subset and parameter-scope study."""
        configs = generate_phase4_configs(max_iter=max_iter, tol=tol)
        if resume:
            self._load_prior_results()
        requested_epochs = int(self.base_config.get("epochs", BASE_CONFIG["epochs"]))
        if requested_epochs < 1:
            raise ValueError("epochs must be at least 1")
        completed_results = []
        for index, cfg in enumerate(configs, start=1):
            logger.info("[Phase 4 %d/%d] %s", index, len(configs), cfg["config_id"])
            cfg = {**cfg, "mgda_log_conflict": bool(log_conflict)}
            if resume:
                completed = self._completed_summary(cfg["config_id"], requested_epochs)
                if completed is not None:
                    self._upsert_result(completed)
                    completed_results.append(completed)
                    self._save_results()
                    continue
            completed_results.append(self._run_single(cfg))
        self._save_results()
        return completed_results

    def run_mgda_vs_fixed_comparison(
        self, max_iter: int = 200, tol: float = 1e-8, resume: bool = False
    ) -> list[dict]:
        """Run matched equal-weight fixed and MGDA runs for each task subset."""
        if resume:
            self._load_prior_results()
        results = []
        epochs = int(self.base_config.get("epochs", BASE_CONFIG["epochs"]))
        for tasks in PHASE4_TASK_SUBSETS:
            code = "".join(_MGDA_TASK_CODES[task] for task in tasks)
            mask_loss = {
                weight: float(task in tasks)
                for task, weight in _MGDA_TASK_CONFIG_WEIGHTS.items()
            }
            pair = (
                {
                    "config_id": f"FIXED-{code}", "training_strategy": "fixed",
                    "use_mgda": False, "mask_loss": mask_loss,
                },
                {
                    "config_id": f"MGDA-{code}-mask_head",
                    "training_strategy": "mgda", "use_mgda": True,
                    "mgda_use_configured_tasks": True, "mgda_tasks": list(tasks),
                    "mgda_param_scope": "mask_head", "mgda_max_iter": max_iter,
                    "mgda_tol": tol, "mask_loss": mask_loss,
                },
            )
            for cfg in pair:
                if resume:
                    completed = self._completed_summary(cfg["config_id"], epochs)
                    if completed is not None:
                        self._upsert_result(completed)
                        results.append(completed)
                        continue
                results.append(self._run_single(cfg))
        self._save_results()
        return results

    def get_pareto_frontier(self) -> list[dict]:
        """Extract Pareto-optimal configs (Dice vs HD95)."""
        if not self.results:
            return []
        pareto = []
        for r in self.results:
            m = r.get("final_metrics", {})
            dice = m.get("val_dice")
            dice = 0.0 if dice is None else dice
            hd95 = m.get("val_hd95")
            hd95 = 999.0 if hd95 is None else hd95
            dominated = False
            for other in self.results:
                om = other.get("final_metrics", {})
                o_dice = om.get("val_dice")
                o_dice = 0.0 if o_dice is None else o_dice
                o_hd95 = om.get("val_hd95")
                o_hd95 = 999.0 if o_hd95 is None else o_hd95
                if o_dice >= dice and o_hd95 <= hd95 and other is not r:
                    dominated = True
                    break
            if not dominated:
                pareto.append(r)
        return pareto
