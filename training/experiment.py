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
from pathlib import Path
from copy import deepcopy
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base config shared by all experiments
# ---------------------------------------------------------------------------
BASE_CONFIG = {
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
    {"config_id": "C-TVK",      "mask_loss": {"alpha": 0.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0, "epsilon": 0.0, "zeta": 1.0}},
    {"config_id": "C-HD95",     "mask_loss": {"alpha": 0.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0, "epsilon": 1.0}},
]


# ---------------------------------------------------------------------------
# Phase 2: Grid search over (alpha, beta, gamma)
# ---------------------------------------------------------------------------
GRID_VALUES = [0.0, 0.3, 0.5, 0.7, 1.0]

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


# ---------------------------------------------------------------------------
# Phase 3: Bayesian Optimisation via Optuna
# ---------------------------------------------------------------------------
def run_bayesian_optimisation(
    base_cfg: dict,
    output_dir: str,
    n_trials: int = 100,
    n_jobs: int = 1,
) -> dict:
    """Run Optuna TPE search over all 5 loss weights."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("Install optuna: pip install optuna")

    from .trainer import Trainer

    def objective(trial: "optuna.Trial") -> float:
        alpha   = trial.suggest_float("alpha",   0.0, 1.0)
        beta    = trial.suggest_float("beta",    0.0, 1.0)
        gamma   = trial.suggest_float("gamma",   0.0, 1.0)
        delta   = trial.suggest_float("delta",   0.0, 0.5)
        epsilon = trial.suggest_float("epsilon", 0.0, 0.3)

        if alpha + beta + gamma + delta + epsilon < 0.01:
            return 0.0

        cfg = deepcopy(base_cfg)
        cfg["config_id"] = f"optuna_trial_{trial.number}"
        cfg["mask_loss"] = {
            "alpha": alpha, "beta": beta,
            "gamma": gamma, "delta": delta, "epsilon": epsilon,
        }
        # Shorter training for search
        cfg["epochs"] = max(10, base_cfg.get("epochs", 30) // 3)

        run_dir = Path(output_dir) / "phase3" / f"trial_{trial.number:04d}"
        trainer = Trainer(cfg, str(run_dir))
        summary = trainer.train()

        # Composite objective: 50% Dice + 30% (1 - HD95_norm) + 20% BoundaryF1
        metrics = summary.get("final_metrics", {})
        dice = metrics.get("val_dice")
        dice = 0.0 if dice is None else dice
        hd95 = metrics.get("val_hd95")
        hd95 = 999.0 if hd95 is None else min(hd95, 100.0)
        bdf1 = metrics.get("val_boundary_f1")
        bdf1 = 0.0 if bdf1 is None else bdf1
        score = 0.5 * dice + 0.3 * (1.0 - hd95 / 100.0) + 0.2 * bdf1

        trial.set_user_attr("dice", dice)
        trial.set_user_attr("hd95", hd95)
        trial.set_user_attr("boundary_f1", bdf1)
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="lung_nodule_loss_opt",
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    best = study.best_trial
    result = {
        "best_trial": best.number,
        "best_score": best.value,
        "best_params": best.params,
        "best_dice": best.user_attrs.get("dice"),
        "best_hd95": best.user_attrs.get("hd95"),
        "best_boundary_f1": best.user_attrs.get("boundary_f1"),
    }
    with open(Path(output_dir) / "phase3_best.json", "w") as f:
        json.dump(result, f, indent=2, allow_nan=False)

    return result


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

    def _run_single(self, cfg_override: dict) -> dict:
        from .trainer import Trainer
        cfg = {**deepcopy(self.base_config), **cfg_override}
        run_dir = self.output_dir / cfg["config_id"]
        trainer = Trainer(cfg, str(run_dir))
        summary = trainer.train()
        self.results.append(summary)
        self._save_results()
        return summary

    def _save_results(self):
        with open(self.output_dir / "all_results.json", "w") as f:
            json.dump(self.results, f, indent=2, allow_nan=False)
        # Also save CSV
        import csv
        if self.results:
            keys = list(self.results[0].keys())
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

    def run_phase2(self, max_configs: Optional[int] = None):
        logger.info("=== PHASE 2: Grid search (alpha, beta, gamma) ===")
        configs = generate_phase2_configs()
        if max_configs is not None:
            if max_configs < 1:
                raise ValueError("max_configs must be at least 1")
            if max_configs < len(configs):
                configs = random.Random(42).sample(configs, k=max_configs)
        logger.info(f"Running {len(configs)} configurations...")
        for i, cfg_override in enumerate(configs):
            logger.info(f"[{i+1}/{len(configs)}] {cfg_override['config_id']}")
            self._run_single(cfg_override)
        logger.info("Phase 2 complete.")

    def run_phase3(self, n_trials: int = 100):
        logger.info("=== PHASE 3: Bayesian Optimisation ===")
        result = run_bayesian_optimisation(
            base_cfg=self.base_config,
            output_dir=str(self.output_dir),
            n_trials=n_trials,
        )
        logger.info(f"Phase 3 best: {result}")
        return result

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
