#!/usr/bin/env python3
"""
Phase 3 — Single-Objective Bayesian Optimisation CLI
=====================================================

Runs (or resumes) an Optuna TPE search over five normalised loss weights plus
two secondary hyperparameters on a single RTX 4050 6 GB GPU.

Objective = 0.5 × Dice  +  0.3 × (1 − hd95/100)  +  0.2 × BoundaryF1

Weights are suggested in raw form and L1-normalised to sum=1 before the model
sees them.  All Phase 1 / 2 / MGDA outputs are untouched.

Output layout (all in --out/phase3/studies/<study-name>/)
--------------------------------------
  optuna_study.db  — SQLite study (authoritative; enables resume)
  trials.csv       — one row per trial (rebuilt from DB on resume)
  phase3_best.json — best normalised weights + metrics (updated per trial)
  trial_NNNN/      — per-trial Trainer artefacts (config, history, checkpoint)

PowerShell execution commands
------------------------------
# First run (30 trials × 10 epochs):
python scripts/run_phase3_bayesian.py `
    --data-root ./data/luna16 `
    --out ./experiments `
    --n-trials 30 `
    --epochs-per-trial 10

# Resume after interruption (re-run the SAME command):
python scripts/run_phase3_bayesian.py `
    --data-root ./data/luna16 `
    --out ./experiments `
    --n-trials 30 `
    --epochs-per-trial 10

# More trials starting from existing study:
python scripts/run_phase3_bayesian.py `
    --data-root ./data/luna16 `
    --out ./experiments `
    --n-trials 50 `
    --epochs-per-trial 10

# With cached pre-processed patches (recommended for speed):
python scripts/run_phase3_bayesian.py `
    --data-root ./data/luna16 `
    --cache-dir  ./data/patches_cache `
    --out ./experiments `
    --n-trials 30

# Custom seed / study name (starts a fresh study):
python scripts/run_phase3_bayesian.py `
    --data-root ./data/luna16 `
    --out ./experiments `
    --n-trials 30 `
    --seed 123 `
    --study-name my_study_v2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
import torch

# ---------------------------------------------------------------------------
# Make sure the repo root is on sys.path so `training` is importable whether
# the script is run from the repo root or from the scripts/ directory.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.experiment import BASE_CONFIG
import training.phase3_bayesian as phase3_bayesian


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_phase3_bayesian",
        description="Phase 3: Single-Objective Bayesian Optimisation via Optuna TPE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- data ---------------------------------------------------------------
    data = parser.add_argument_group("Data")
    data.add_argument(
        "--data-root", required=True,
        help="Path to LUNA16 raw data root (parent of subset0…subset9).",
    )
    data.add_argument(
        "--cache-dir", default=None,
        help="Pre-processed patch cache directory (output of preprocess_cache.py). "
             "Recommended: eliminates SSD bottleneck and significantly speeds up trials.",
    )
    data.add_argument(
        "--fold", type=int, default=0,
        help="Cross-validation fold index (0-based).",
    )
    data.add_argument(
        "--batch-size", type=int, default=4,
        help="Training batch size per trial.",
    )
    data.add_argument(
        "--num-workers", type=int, default=2,
        help="DataLoader worker processes.",
    )

    # --- optimisation -------------------------------------------------------
    opt = parser.add_argument_group("Optimisation")
    opt.add_argument(
        "--n-trials", type=int, default=30,
        help="TOTAL target number of trials.  Already-completed trials count "
             "towards this — rerunning is idempotent and resumes safely.",
    )
    opt.add_argument(
        "--epochs-per-trial", type=int, default=10,
        help="Training epochs per trial (full training uses 30).",
    )
    opt.add_argument(
        "--seed", type=int, default=42,
        help="TPE sampler seed for reproducibility.",
    )
    opt.add_argument(
        "--study-name", default="lung_nodule_loss_weights_v1",
        help="Optuna study name.  Changing it starts a fresh study from scratch.",
    )

    # --- model / training ---------------------------------------------------
    model = parser.add_argument_group("Model / training")
    model.add_argument(
        "--trainable-backbone-layers", type=int, default=3,
        help="Number of trainable ResNet backbone layers.",
    )
    model.add_argument(
        "--warmup-epochs", type=int, default=BASE_CONFIG["warmup_epochs"],
        help="LR warmup epochs per trial.",
    )
    model.add_argument(
        "--lr", type=float, default=BASE_CONFIG["lr"],
        help="Fixed optimizer learning rate shared by all weight trials.",
    )
    model.add_argument(
        "--patch-size", type=int, default=64,
        help="2-D patch size extracted from LUNA16 CT volumes.",
    )

    # --- output -------------------------------------------------------------
    out = parser.add_argument_group("Output")
    out.add_argument(
        "--out", default="./experiments",
        help="Root output directory. Phase 3 artefacts are isolated by study "
             "under <out>/phase3/studies/. "
             "Phase 1/2/MGDA results are NOT modified.",
    )
    out.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    out.add_argument(
        "--dry-run", action="store_true",
        help="Print the resolved base config and exit without running any trials.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("phase3")

    # ---- Build base config (inherits from experiment BASE_CONFIG) ----------
    base_cfg: dict = {
        **BASE_CONFIG,
        "data_root":               str(Path(args.data_root).expanduser()),
        "cache_dir":               (
            str(Path(args.cache_dir).expanduser()) if args.cache_dir else None
        ),
        "fold":                    args.fold,
        "batch_size":              args.batch_size,
        "lr":                      args.lr,
        "num_workers":             args.num_workers,
        "trainable_backbone_layers": args.trainable_backbone_layers,
        "warmup_epochs":           args.warmup_epochs,
        "patch_size":              args.patch_size,
        # NOTE: epochs and mask_loss are overridden per trial
        "training_strategy":       "fixed",
        "require_cuda":            True,
    }

    if args.dry_run:
        print("=== Dry-run: resolved base config ===")
        print(json.dumps(base_cfg, indent=2, default=str))
        print(f"\nWould run {args.n_trials} trial(s) × {args.epochs_per_trial} epochs")
        print(f"Output dir : {args.out}/phase3/studies/{args.study_name}/")
        print(f"Study name : {args.study_name}")
        print(f"Seed       : {args.seed}")
        return

    if not torch.cuda.is_available():
        parser.error(
            "Phase 3 requires CUDA for the requested RTX 4050 run; "
            "torch.cuda.is_available() is False."
        )

    # ---- Run / resume Phase 3 -----------------------------------------------
    logger.info(
        "Starting Phase 3 | n_trials=%d | epochs_per_trial=%d | out=%s",
        args.n_trials, args.epochs_per_trial, args.out,
    )

    result = phase3_bayesian.run_phase3_bayesian(
        base_cfg=base_cfg,
        output_dir=args.out,
        n_trials=args.n_trials,
        epochs_per_trial=args.epochs_per_trial,
        seed=args.seed,
        study_name=args.study_name,
    )

    print("\n" + "=" * 60)
    print("Phase 3 complete.")
    print(json.dumps(result, indent=2, default=str))
    print("=" * 60)
    logger.info("Phase 3 finished.  Best result: %s", result)


if __name__ == "__main__":
    main()
