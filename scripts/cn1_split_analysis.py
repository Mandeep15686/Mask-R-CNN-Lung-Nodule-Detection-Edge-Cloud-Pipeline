#!/usr/bin/env python3
"""
CN1 Research: Split-Point Latency Analysis for Edge–Cloud Mask R-CNN.

Profiles every candidate split point in the ResNet-50 backbone and measures:
  - Edge compute time  (time to run up to split layer)
  - Feature map size   (raw float32 KB and serialized INT8 request KB)
  - Cloud compute time (the same checkpoint-checked continuation used by the API)
  - Total latency      (edge + network + cloud)

Network conditions are modeled using configurable bandwidth/latency profiles.
This script measures software components on one host; it does not claim
physical edge/cloud latency or clinical accuracy impact.

Usage:
  python scripts/cn1_split_analysis.py \
    --checkpoint ./experiments/G-a03-b05-g05/best_model.pth \
    --data ./data/luna16 \
    --out ./experiments/cn1_split \
    --conditions LAN WiFi 4G 5G
"""
import argparse
import io
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
    inference_engine=None,
    checkpoint_sha256: str | None = None,
    n_repeats:      int = 20,
) -> dict:
    """
    Measure edge and cloud compute times for a given split point.
    Returns timing and size statistics.
    """
    import torch

    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")
    if not sample_images:
        raise ValueError("At least one sample image is required for profiling")
    if split_layer not in ("input", "cloud_only") and (
        inference_engine is None or not checkpoint_sha256
    ):
        raise ValueError(
            "Backbone split profiling requires the checkpoint-backed "
            "InferenceEngine continuation and checkpoint SHA-256"
        )

    edge_times, cloud_times, feature_sizes, wire_payload_sizes = [], [], [], []

    def synchronize(device):
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)

    def prepare_cloud_features(img):
        features = extractor.extract(img)
        from compressor import FeatureCompressor

        compressed = FeatureCompressor(method="int8", device=img.device).compress(features)
        return features, compressed

    # Run one untimed pass through the exact branch first. This excludes lazy
    # CUDA/context initialization and model warm-up from the reported mean.
    warmup_img = sample_images[0]
    warmup_img = warmup_img.unsqueeze(0) if warmup_img.dim() == 3 else warmup_img
    with torch.no_grad():
        if split_layer in ("input", "cloud_only"):
            synchronize(warmup_img.device)
            model(list(warmup_img))
            synchronize(warmup_img.device)
        else:
            _, warmup_features = prepare_cloud_features(warmup_img)
            synchronize(warmup_img.device)
            inference_engine.predict_from_edge_features(
                warmup_features,
                split_layer,
                "cn1-split-profile-warmup",
                expected_checkpoint_sha256=checkpoint_sha256,
                edge_image_mean=extractor.image_mean,
                edge_image_std=extractor.image_std,
            )
            synchronize(warmup_img.device)

    with torch.no_grad():
        for img_tensor in sample_images[:n_repeats]:
            img = img_tensor.unsqueeze(0) if img_tensor.dim() == 3 else img_tensor

            if split_layer == "input":
                # Input-space split sends the same RGB PNG patch accepted by
                # the backend, rather than charging an uncompressed tensor as
                # though it were the HTTP wire body.
                from PIL import Image

                pixels = np.rint(
                    img[0].detach().cpu().clamp(0.0, 1.0).numpy().transpose(1, 2, 0) * 255.0
                ).astype(np.uint8)
                encoded = io.BytesIO()
                Image.fromarray(pixels, mode="RGB").save(encoded, format="PNG")
                wire_payload_sizes.append(len(encoded.getvalue()) / 1024)
                feature_sizes.append(0.0)
                edge_times.append(0.0)

                synchronize(img.device)
                t0 = time.perf_counter()
                cloud_input = torch.from_numpy(pixels.transpose(2, 0, 1).copy())
                cloud_input = cloud_input.to(device=img.device, dtype=torch.float32) / 255.0
                model([cloud_input])
                synchronize(img.device)
                cloud_times.append((time.perf_counter() - t0) * 1000)

            elif split_layer == "cloud_only":
                # Reference baseline: patch is already at cloud, so no edge
                # compute or network transfer is charged.
                feature_sizes.append(0.0)
                wire_payload_sizes.append(0.0)
                edge_times.append(0.0)
                synchronize(img.device)
                t0 = time.perf_counter()
                model(list(img))
                synchronize(img.device)
                cloud_times.append((time.perf_counter() - t0) * 1000)

            else:
                # Match the live edge path: produce cumulative backbone levels,
                # INT8-quantize and serialize them, then call the cloud's actual
                # checkpoint/hash/normalization-checked continuation method.
                synchronize(img.device)
                t0 = time.perf_counter()
                feats, compressed = prepare_cloud_features(img)
                # Retain the uncompressed feature size for comparison and use
                # the complete INT8 wire body (including JSON header) for the
                # modeled transmission time.
                size_kb = sum(v.numel() * v.element_size() for v in feats.values()) / 1024
                feature_sizes.append(size_kb)
                from edge_node import _serialize_feature_payload

                payload = _serialize_feature_payload(compressed, {
                    "checkpoint_sha256": checkpoint_sha256,
                    "image_mean": extractor.image_mean,
                    "image_std": extractor.image_std,
                    "compression_method": "int8",
                })
                wire_payload_sizes.append(len(payload) / 1024)
                synchronize(img.device)
                edge_ms = (time.perf_counter() - t0) * 1000
                edge_times.append(edge_ms)

                # This method performs the same dequantization, remaining
                # ResNet stages, FPN, RPN, RoI heads, and postprocessing as the
                # live cloud endpoint. A full model forward here would measure
                # a different (and incorrect) cloud path.
                synchronize(img.device)
                t1 = time.perf_counter()
                inference_engine.predict_from_edge_features(
                    compressed,
                    split_layer,
                    "cn1-split-profile",
                    expected_checkpoint_sha256=checkpoint_sha256,
                    edge_image_mean=extractor.image_mean,
                    edge_image_std=extractor.image_std,
                )
                synchronize(img.device)
                cloud_times.append((time.perf_counter() - t1) * 1000)

    return {
        "split_layer":        split_layer,
        "avg_edge_ms":        float(np.mean(edge_times)),
        "std_edge_ms":        float(np.std(edge_times)),
        "avg_cloud_ms":       float(np.mean(cloud_times)),
        "std_cloud_ms":       float(np.std(cloud_times)),
        "avg_feature_kb":     float(np.mean(feature_sizes)),
        "avg_wire_payload_kb": float(np.mean(wire_payload_sizes)),
        "cloud_path": (
            "InferenceEngine.predict_from_edge_features"
            if split_layer not in ("input", "cloud_only")
            else "full_cloud_model_forward"
        ),
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
    payload_kb = profile.get("avg_wire_payload_kb", profile["avg_feature_kb"])
    tx_ms = 0.0 if profile["split_layer"] == "cloud_only" else (
        payload_kb * 8 / (condition["bandwidth_mbps"] * 1000) * 1000
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
    from training.dataset import LUNA16Dataset
    from edge.feature_extractor import EdgeFeatureExtractor
    from backend.app.ml.inference import InferenceEngine
    from torch.utils.data import DataLoader

    if conditions is None:
        conditions = list(NETWORK_CONDITIONS.keys())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    inference_engine = InferenceEngine(
        default_checkpoint_path=checkpoint_path,
        device=device,
    )
    loaded = inference_engine.load_checkpoint(checkpoint_path)
    model = loaded.model
    cfg = loaded.config
    det_model = model.model
    logger.info(f"Model loaded from {checkpoint_path}")

    # Collect sample images
    patch_size = int(cfg.get("patch_size", 64))
    ds = LUNA16Dataset(data_root, subsets=[fold], patch_size=patch_size, augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    samples = []
    with torch.no_grad():
        for batch in loader:
            if len(samples) >= n_samples:
                break
            samples.append(batch["image"][0].to(device))
    logger.info(f"Collected {len(samples)} sample images")
    if not samples:
        raise ValueError(f"No patches found in subset {fold} under {data_root}")

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
                image_mean=tuple(det_model.transform.image_mean),
                image_std=tuple(det_model.transform.image_std),
            )

        profile = profile_split_point(
            split_layer=split_layer,
            model=model,
            extractor=extractor,
            sample_images=samples,
            inference_engine=inference_engine,
            checkpoint_sha256=loaded.sha256,
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
                "network_measurement": "modeled_profile",
                "energy_measurement": "not_measured",
                "clinical_network_measurement": False,
                "bandwidth_mbps":   cond["bandwidth_mbps"],
                "rtt_ms":           cond["rtt_ms"],
                "avg_edge_ms":      round(profile["avg_edge_ms"], 2),
                "avg_cloud_ms":     round(profile["avg_cloud_ms"], 2),
                "avg_feature_kb":   round(profile["avg_feature_kb"], 2),
                "avg_wire_payload_kb": round(profile["avg_wire_payload_kb"], 2),
                "cloud_path": profile["cloud_path"],
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
                "wire_payload_kb": best["avg_wire_payload_kb"],
            }

    # Save results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint":    checkpoint_path,
        "clinical_network_measurement": False,
        "optimization_scope": (
            "Best profiled split among the tested software paths under the selected "
            "modeled network profiles; not a measured hospital or carrier optimum."
        ),
        "energy_measurement": {
            "status": "not_measured",
            "edge_joules": None,
            "cloud_joules": None,
            "reason": "No synchronized power/energy sensors are available in this single-host profiler.",
        },
        "measurement_scope": (
            "Single-host component timings; cloud continuation calls the same "
            "checkpoint-checked InferenceEngine path as the API. Network profiles "
            "are modeled, not measured on a deployed edge/cloud network. No "
            "hospital-to-cloud latency or energy savings are established."
        ),
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

    print("\n  Feature map / request sizes per split layer:")
    for split_layer, p in profiles.items():
        print(f"  {split_layer:<14}  {p['avg_feature_kb']:>8.1f} KB  "
              f"wire={p['avg_wire_payload_kb']:.1f} KB  "
              f"(edge={p['avg_edge_ms']:.1f}ms, cloud={p['avg_cloud_ms']:.1f}ms)")

    print("\n  Optimal split per condition:")
    for cond_name, opt in optimal.items():
        print(f"  {cond_name:<8}  ->  {opt['split_layer']:<14}  "
              f"total={opt['total_ms']:.1f}ms  wire={opt['wire_payload_kb']:.1f}KB")

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
