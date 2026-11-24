#!/usr/bin/env python3
"""
CN3 · Federated Learning across 10 simulated hospital silos.

Each of LUNA16's 10 subsets (subset0 … subset9) represents one hospital.
FedAvg aggregation is run for the requested number of rounds.

Usage
-----
  python scripts/run_federated_10hospitals.py \\
      --data  ./data/luna16 \\
      --out   ./experiments/federated_10h \\
      --rounds 50 \\
      --strategy fedavg

Requirements
------------
  pip install flwr torch torchvision
  (all other deps come from the project's training/ package)
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FL-10H] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

NUM_HOSPITALS = 10


def main():
    parser = argparse.ArgumentParser(
        description="CN3: Federated Learning across 10 LUNA16 hospital silos"
    )
    parser.add_argument(
        "--data", default="./data/luna16",
        help="Path to LUNA16 root (must contain subset0 … subset9)"
    )
    parser.add_argument(
        "--out", default="./experiments/federated_10h",
        help="Output directory for fl_history.json and fl_summary.json"
    )
    parser.add_argument(
        "--rounds", type=int, default=50,
        help="Number of federated aggregation rounds (default: 50)"
    )
    parser.add_argument(
        "--strategy", choices=["fedavg", "fedprox"], default="fedavg",
        help="Aggregation strategy (default: fedavg)"
    )
    parser.add_argument(
        "--local-epochs", type=int, default=1,
        help="Local training epochs per client per round (default: 1)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Per-client batch size (default: 4)"
    )
    parser.add_argument(
        "--patch-size", type=int, default=64,
        help="Input patch size in pixels (default: 64)"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.3,
        help="Composite mask-loss α (BCE weight, default: 0.3)"
    )
    parser.add_argument(
        "--beta", type=float, default=0.5,
        help="Composite mask-loss β (Dice weight, default: 0.5)"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.5,
        help="Composite mask-loss γ (Focal weight, default: 0.5)"
    )
    args = parser.parse_args()

    data_root = Path(args.data).expanduser().resolve()
    if not data_root.is_dir():
        logger.error("Data directory not found: %s", data_root)
        sys.exit(1)

    # Verify all 10 subsets are present
    missing = [
        f"subset{i}" for i in range(NUM_HOSPITALS)
        if not (data_root / f"subset{i}").is_dir()
    ]
    if missing:
        logger.error(
            "Missing LUNA16 subsets under %s: %s\n"
            "Download from https://luna16.grand-challenge.org/ and extract all 10 subsets.",
            data_root, ", ".join(missing),
        )
        sys.exit(1)

    logger.info(
        "Starting 10-hospital federated simulation: %d rounds · %s · %d local epochs",
        args.rounds, args.strategy, args.local_epochs,
    )

    config = {
        "mask_loss": {
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
        },
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "local_epochs": args.local_epochs,
        "validation_fraction": 0.2,
        "split_seed": 42,
    }

    from training.federated import run_simulation

    result = run_simulation(
        data_root=str(data_root),
        num_clients=NUM_HOSPITALS,
        num_rounds=args.rounds,
        strategy_name=args.strategy,
        config=config,
        output_dir=args.out,
    )

    best_dice = result.get("best_dice", 0)
    logger.info(
        "10-hospital FL complete — best Dice across rounds: %.4f",
        best_dice,
    )
    logger.info("Results written to: %s", args.out)

    # Write an explicitly labeled simulation record. This is not evidence of
    # deployment across independent hospitals.
    evidence = {
        "evidence_type": "single-machine_federated_simulation",
        "real_hospital_deployment": False,
        "num_hospitals": NUM_HOSPITALS,
        "subsets_used": [f"subset{i}" for i in range(NUM_HOSPITALS)],
        "num_rounds": args.rounds,
        "strategy": args.strategy,
        "config": config,
        "best_dice": best_dice,
        "history_path": str(Path(args.out) / "fl_history.json"),
        "summary_path": str(Path(args.out) / "fl_summary.json"),
    }
    evidence_path = Path(args.out) / "10hospital_simulation_evidence.json"
    os.makedirs(args.out, exist_ok=True)
    with open(evidence_path, "w") as fh:
        json.dump(evidence, fh, indent=2)
    logger.info("Deployment evidence written to: %s", evidence_path)


if __name__ == "__main__":
    main()
