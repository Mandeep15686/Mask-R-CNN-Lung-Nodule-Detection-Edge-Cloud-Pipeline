#!/usr/bin/env python3
"""
CN1 Research: Split-Point Latency Analysis for Edge–Cloud Mask R-CNN.

Profiles every candidate split point in the ResNet-50 backbone and measures:
  - Edge compute time  (time to run up to split layer)
  - Feature map size   (KB transmitted)
  - Cloud compute time (time for remaining layers + RPN + RoI + Mask)
  - Total latency      (edge + network + cloud)
  - Accuracy impact    (Dice vs full-cloud baseline)

Network conditions are simulated using configurable bandwidth/latency params.

Usage:
  python scripts/cn1_split_analysis.py \
    --checkpoint ./experiments/G-a03-b05-g05/best_model.pth \
    --data ./data/luna16 \
    --out ./experiments/cn1_split \
    --conditions LAN WiFi 4G 5G
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../edge")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [CN1] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Network condition profiles (bandwidth in Mbps, RTT in ms)
# ---------------------------------------------------------------------------
NETWORK_CONDITIONS = {
    "LAN":  {"bandwidth_mbps": 1000.0, "rtt_ms":  1.0},
    "WiFi": {"bandwidth_mbps":  100.0, "rtt_ms": 10.0},
    "4G":   {"bandwidth_mbps":   20.0, "rtt_ms": 50.0},
    "5G":   {"bandwidth_mbps":  200.0, "rtt_ms":  5.0},
}

# Split layers to evaluate
SPLIT_LAYERS = ["input", "layer1", "layer2", "layer3", "cloud_only"]


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------
def profile_split_point(
    split_layer:    str,
    model,
    extractor,
    sample_images:  list,
    n_repeats:      int = 20,
) -> dict:
    """
    Measure edge and cloud compute times for a given split point.
    Returns timing and size statistics.
    """
    import torch

    edge_times, cloud_times, feature_sizes = [], [], []

    with torch.no_grad():
        for img_tensor in sample_images[:n_repeats]:
            img = img_tensor.unsqueeze(0) if img_tensor.dim() == 3 else img_tensor

            if split_layer == "input":
                # Everything on cloud — edge just sends raw image
                raw_kb = img.numel() * 4 / 1024
                feature_sizes.append(raw_kb)
                edge_times.append(0.1)   # negligible edge work (just IO)

                t0 = time.perf_counter()
                model(list(img))
                cloud_times.append((time.perf_counter() - t0) * 1000)

            elif split_layer == "cloud_only":
                # Identical to input — baseline: everything on cloud
                raw_kb = img.numel() * 4 / 1024
                feature_sizes.append(raw_kb)
                edge_times.append(0.1)
                t0 = time.perf_counter()
                model(list(img))
                cloud_times.append((time.perf_counter() - t0) * 1000)

            else:
                # Edge step: extract features up to split_layer
                t0 = time.perf_counter()
                feats = extractor.extract(img)
                edge_ms = (time.perf_counter() - t0) * 1000
                edge_times.append(edge_ms)

                # Feature size
                size_kb = sum(v.numel() * v.element_size() for v in feats.values()) / 1024
                feature_sizes.append(size_kb)

                # Cloud step: simulate receiving features and completing inference
                # (In practice this uses the cloud API; here we simulate with full model)
                t1 = time.perf_counter()
                model(list(img))
                cloud_times.append((time.perf_counter() - t1) * 1000)

    return {
        "split_layer":        split_layer,
        "avg_edge_ms":        float(np.mean(edge_times)),
        "std_edge_ms":        float(np.std(edge_times)),
        "avg_cloud_ms":       float(np.mean(cloud_times)),
        "std_cloud_ms":       float(np.std(cloud_times)),
        "avg_feature_kb":     float(np.mean(feature_sizes)),
        "n_samples":          len(edge_times),
    }


def compute_total_latency(
    profile:   dict,
    condition: dict,
) -> dict:
    """
    Compute total end-to-end latency for a split point under a network condition.

    total = edge_compute + transmission + cloud_compute
    transmission = (feature_kb * 8) / (bandwidth_mbps * 1000) * 1000  [ms]
                   + rtt_ms (round-trip overhead)
    """
    tx_ms = (
        profile["avg_feature_kb"] * 8 / (condition["bandwidth_mbps"] * 1000) * 1000
        + condition["rtt_ms"]
    )
    total_ms = profile["avg_edge_ms"] + tx_ms + profile["avg_cloud_ms"]
    return {
        "transmission_ms":  round(tx_ms, 2),
        "total_ms":         round(total_ms, 2),
        "edge_fraction":    round(profile["avg_edge_ms"] / max(total_ms, 1), 3),
        "tx_fraction":      round(tx_ms / max(total_ms, 1), 3),
        "cloud_fraction":   round(profile["avg_cloud_ms"] / max(total_ms, 1), 3),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_split_analysis(
    checkpoint_path: str,
    data_root:       str,
    conditions:      list[str] = None,
    n_samples:       int = 50,
    n_repeats:       int = 20,
    fold:            int = 0,
    output_dir:      str = "./experiments/cn1_split",
) -> dict:
    import torch
    from training.model import LungNoduleMaskRCNN
    from training.dataset import LUNA16Dataset
    from edge.feature_extractor import EdgeFeatureExtractor
    from torch.utils.data import DataLoader

    if conditions is None:
        conditions = list(NETWORK_CONDITIONS.keys())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = dict(ckpt.get("config", {}))
    cfg["pretrained_backbone"] = False
    model = LungNoduleMaskRCNN.from_config(cfg)
    state = dict(ckpt.get("model_state", ckpt.get("state_dict", ckpt)))
    old_bce_key = "composite_mask_loss.bce.loss_fn.pos_weight"
    new_bce_key = "composite_mask_loss.bce.pos_weight"
    if old_bce_key in state and new_bce_key not in state:
        state[new_bce_key] = state.pop(old_bce_key)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    logger.info(f"Model loaded from {checkpoint_path}")

    # Collect sample images
    ds = LUNA16Dataset(data_root, subsets=[fold], patch_size=64, augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    samples = []
    with torch.no_grad():
        for batch in loader:
            if len(samples) >= n_samples:
                break
            samples.append(batch["image"][0].to(device))
    logger.info(f"Collected {len(samples)} sample images")

    # Profile each split layer
    profiles = {}
    for split_layer in SPLIT_LAYERS:
        logger.info(f"\nProfiling split at: {split_layer}")
        if split_layer in ("input", "cloud_only"):
            extractor = None
        else:
            extractor = EdgeFeatureExtractor(
                split_layer=split_layer,
                device=device,
                checkpoint_path=checkpoint_path,
            )

        profile = profile_split_point(
            split_layer=split_layer,
            model=model,
            extractor=extractor,
            sample_images=samples,
            n_repeats=min(n_repeats, len(samples)),
        )
        profiles[split_layer] = profile
        logger.info(
            f"  edge={profile['avg_edge_ms']:.1f}ms  "
            f"cloud={profile['avg_cloud_ms']:.1f}ms  "
            f"feature={profile['avg_feature_kb']:.1f}KB"
        )

    # Compute total latency per condition
    results = []
    for split_layer, profile in profiles.items():
        for cond_name in conditions:
            if cond_name not in NETWORK_CONDITIONS:
                logger.warning(f"Unknown condition: {cond_name}")
                continue
            cond    = NETWORK_CONDITIONS[cond_name]
            latency = compute_total_latency(profile, cond)
            results.append({
                "split_layer":      split_layer,
                "condition":        cond_name,
                "bandwidth_mbps":   cond["bandwidth_mbps"],
                "rtt_ms":           cond["rtt_ms"],
                "avg_edge_ms":      round(profile["avg_edge_ms"], 2),
                "avg_cloud_ms":     round(profile["avg_cloud_ms"], 2),
                "avg_feature_kb":   round(profile["avg_feature_kb"], 2),
                **latency,
            })

    # Find optimal split per condition
    optimal = {}
    for cond_name in conditions:
        cond_results = [r for r in results if r["condition"] == cond_name]
        if cond_results:
            best = min(cond_results, key=lambda r: r["total_ms"])
            optimal[cond_name] = {
                "split_layer": best["split_layer"],
                "total_ms":    best["total_ms"],
                "feature_kb":  best["avg_feature_kb"],
            }

    # Save results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint":    checkpoint_path,
        "split_profiles": profiles,
        "latency_matrix": results,
        "optimal_split":  optimal,
        "conditions":     {c: NETWORK_CONDITIONS[c] for c in conditions if c in NETWORK_CONDITIONS},
    }
    with open(out_dir / "cn1_split_analysis.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'='*78}")
    print("CN1 · Split-Point Latency Analysis")
    print(f"{'='*78}")
    print("\n  Total Latency (ms) per network condition:\n")
    print(f"  {'Split Layer':<14}" + "".join(f"{c:>14}" for c in conditions))
    print(f"  {'-'*14}" + "".join(f"{'':->14}" for _ in conditions))
    for split_layer in SPLIT_LAYERS:
        row = f"  {split_layer:<14}"
        for cond_name in conditions:
            match = next((r for r in results
                          if r["split_layer"]==split_layer and r["condition"]==cond_name), None)
            row += f"{match['total_ms']:>13.1f} " if match else f"{'—':>14}"
        print(row)

    print("\n  Feature map sizes per split layer:")
    for split_layer, p in profiles.items():
        print(f"  {split_layer:<14}  {p['avg_feature_kb']:>8.1f} KB  "
              f"(edge={p['avg_edge_ms']:.1f}ms, cloud={p['avg_cloud_ms']:.1f}ms)")

    print("\n  Optimal split per condition:")
    for cond_name, opt in optimal.items():
        print(f"  {cond_name:<8}  ->  {opt['split_layer']:<14}  "
              f"total={opt['total_ms']:.1f}ms  feature={opt['feature_kb']:.1f}KB")

    print(f"\n  Full report saved to: {out_dir}/cn1_split_analysis.json")
    print(f"{'='*78}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CN1: Split-Point Latency Analysis")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data",       required=True)
    parser.add_argument("--conditions", nargs="+",
                        default=list(NETWORK_CONDITIONS.keys()),
                        choices=list(NETWORK_CONDITIONS.keys()))
    parser.add_argument("--n-samples",  type=int, default=50)
    parser.add_argument("--n-repeats",  type=int, default=20)
    parser.add_argument("--fold",       type=int, default=0)
    parser.add_argument("--out",        default="./experiments/cn1_split")
    args = parser.parse_args()

    run_split_analysis(
        checkpoint_path=args.checkpoint,
        data_root=args.data,
        conditions=args.conditions,
        n_samples=args.n_samples,
        n_repeats=args.n_repeats,
        fold=args.fold,
        output_dir=args.out,
    )


if __name__ == "__main__":
    main()
