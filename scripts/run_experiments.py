#!/usr/bin/env python3
"""
CLI entry-point to run Phase 1, 2, or 3 loss-weight experiments.

Usage:
  python scripts/run_experiments.py --phase 1 --data ./data/luna16 --cache-dir ./data/patches_cache_v2 --out ./experiments
  python scripts/run_experiments.py --phase 2 --data ./data/luna16 --cache-dir ./data/patches_cache_v2 --out ./experiments --max-configs 20
  python scripts/run_experiments.py --phase 3 --data ./data/luna16 --cache-dir ./data/patches_cache_v2 --out ./experiments --n-trials 50
"""
import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Run Mask R-CNN loss-weight experiments")
    parser.add_argument("--phase",       type=int,   default=1,            help="1=ablation, 2=grid, 3=bayesian")
    parser.add_argument("--data",        type=str,   default="./data/luna16", help="LUNA16 data root")
    parser.add_argument("--cache-dir",   type=str,   default=None,         help="Pre-extracted LUNA16 patch cache directory")
    parser.add_argument("--out",         type=str,   default="./experiments", help="Output directory")
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch-size",  type=int,   default=4)
    parser.add_argument("--fold",        type=int,   default=0)
    parser.add_argument("--train-subsets", type=int, nargs="+", default=None,
                        help="Explicit training subset IDs")
    parser.add_argument("--validation-subset", type=int, default=None,
                        help="Explicit validation subset ID")
    parser.add_argument("--test-subset", type=int, default=None,
                        help="Reserved independent test subset ID; excluded from training")
    parser.add_argument("--lr",          type=float, default=0.005)
    parser.add_argument("--max-configs", type=int,   default=None,         help="Phase 2: max configs to run")
    parser.add_argument("--resume", action="store_true",
                        help="Phase 2: skip configurations with complete validated artifacts")
    parser.add_argument("--n-trials",    type=int,   default=100,          help="Phase 3: Optuna trials")
    args = parser.parse_args()

    from training.experiment import ExperimentRunner, BASE_CONFIG
    base_cfg = {
        **BASE_CONFIG,
        "data_root":  args.data,
        "cache_dir":  args.cache_dir,
        "epochs":     args.epochs,
        "batch_size": args.batch_size,
        "fold":       args.fold,
        "train_subsets": args.train_subsets,
        "validation_subset": args.validation_subset,
        "test_subset": args.test_subset,
        "lr":         args.lr,
    }

    runner = ExperimentRunner(base_cfg, args.out)

    if args.phase == 1:
        runner.run_phase1()
    elif args.phase == 2:
        runner.run_phase2(max_configs=args.max_configs, resume=args.resume)
    elif args.phase == 3:
        result = runner.run_phase3(n_trials=args.n_trials)
        print("\n=== Phase 3 Best Result ===")
        import json
        print(json.dumps(result, indent=2))
    else:
        print(f"Unknown phase: {args.phase}", file=sys.stderr)
        sys.exit(1)

    # Print Pareto frontier
    pareto = runner.get_pareto_frontier()
    print(f"\n=== Pareto Frontier ({len(pareto)} configs) ===")
    for r in pareto:
        m = r.get("final_metrics", {})
        print(f"  {r['config_id']:30s}  Dice={m.get('val_dice',0):.4f}  HD95={m.get('val_hd95',999):.2f}")


if __name__ == "__main__":
    main()
