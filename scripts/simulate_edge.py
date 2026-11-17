#!/usr/bin/env python3
"""
Simulates the edge-cloud pipeline on localhost.
Emulates different network conditions (LAN, 4G, 5G) using configurable delay.

Usage:
  python scripts/simulate_edge.py --image path/to/ct_patch.png --split layer2 --compress int8
"""
import argparse
import asyncio
import sys, os
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../edge"))


async def run_simulation(args):
    import aiohttp
    from preprocessor import CTPreprocessor
    from feature_extractor import EdgeFeatureExtractor
    from compressor import FeatureCompressor
    from edge_node import EDGE_MODEL_PATH, _serialize_feature_payload
    import torch

    device = torch.device("cpu")
    prep   = CTPreprocessor(patch_size=64)
    extractor = EdgeFeatureExtractor(
        split_layer=args.split,
        device=device,
        checkpoint_path=EDGE_MODEL_PATH if args.cloud else None,
    )
    compressor = FeatureCompressor(method=args.compress, device=device)

    with open(args.image, "rb") as f:
        img_bytes = f.read()

    print(f"\n{'='*60}")
    print("Edge–Cloud Pipeline Simulation")
    print(f"  Split layer : {args.split}")
    print(f"  Compression : {args.compress}")
    print(f"  Image       : {args.image} ({len(img_bytes)/1024:.1f} KB)")
    print(f"{'='*60}\n")

    # 1. Preprocess
    t0 = time.perf_counter()
    patch = prep.preprocess_bytes(img_bytes)
    print(f"[Edge] Preprocess:  {(time.perf_counter()-t0)*1000:.2f} ms")

    # 2. Feature extraction
    t1 = time.perf_counter()
    features = extractor.extract(patch.unsqueeze(0))
    t_extract = (time.perf_counter()-t1)*1000
    print(f"[Edge] Feature extraction: {t_extract:.2f} ms")
    for k, v in features.items():
        print(f"[Edge] Features [{k}]: shape={list(v.shape)}, size={v.numel()*4/1024:.1f} KB (fp32)")

    # 3. Compression
    t2 = time.perf_counter()
    compressed = compressor.compress(features)
    t_compress = (time.perf_counter()-t2)*1000
    comp_kb = sum(v.numel()*v.element_size() for v in compressed.values() if hasattr(v,"numel")) / 1024
    raw_kb  = sum(v.numel()*4 for v in features.values()) / 1024
    ratio = raw_kb / max(comp_kb, 0.001)
    print(f"[Edge] Compress:    {t_compress:.2f} ms | {raw_kb:.1f} -> {comp_kb:.1f} KB (ratio={ratio:.2f}x)")
    print(f"[Edge] Total edge:  {(time.perf_counter()-t0)*1000:.2f} ms")

    # 4. Build the same wire payload used by the real edge node.
    payload = _serialize_feature_payload(
        compressed,
        {
            "scan_uid": "sim_001",
            "edge_processing_ms": (time.perf_counter() - t0) * 1000,
            "bandwidth_mbps": 100.0,
            "compression_method": args.compress,
            "compression_ratio": ratio,
            "checkpoint_sha256": extractor.checkpoint_sha256,
            "image_mean": extractor.image_mean,
            "image_std": extractor.image_std,
        },
    )
    print(f"\n[Net]  Payload size:  {len(payload)/1024:.1f} KB")

    # Network conditions
    conditions = {
        "LAN (1Gbps)":  {"delay": 0.001, "bw_mbps": 1000},
        "WiFi (100Mbps)": {"delay": 0.01,  "bw_mbps": 100},
        "4G (20Mbps)":  {"delay": 0.05,  "bw_mbps": 20},
        "5G (200Mbps)": {"delay": 0.005, "bw_mbps": 200},
    }
    print("\n[Net] Estimated transmission times:")
    for cond, params in conditions.items():
        tx_ms = (len(payload) * 8) / (params["bw_mbps"] * 1e6) * 1000 + params["delay"] * 1000
        print(f"  {cond:20s}: {tx_ms:.1f} ms")

    # 5. Try real cloud request (optional)
    if args.cloud:
        print(f"\n[Cloud] Sending to {args.cloud}/api/edge/infer-features ...")
        t_cloud = time.perf_counter()
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
                async with s.post(
                    f"{args.cloud}/api/edge/infer-features",
                    data=payload,
                    params={"node_id": "sim", "scan_uid": "sim_001", "split_layer": args.split},
                    headers={
                        "Content-Type": "application/octet-stream",
                        "X-Edge-Token": os.getenv("EDGE_API_TOKEN", ""),
                    },
                ) as r:
                    result = await r.json()
                    cloud_ms = (time.perf_counter()-t_cloud)*1000
                    print(f"[Cloud] Inference: {cloud_ms:.1f} ms")
                    print(f"[Cloud] Nodules:   {len(result.get('nodules',[]))}")
                    print(f"\nResult: {result}")
        except Exception as e:
            print(f"[Cloud] Request failed: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image",    default=None, help="Path to CT patch image (PNG/JPEG); defaults to the first generated sample")
    p.add_argument("--split",    default="layer2", choices=["layer1","layer2","layer3"])
    p.add_argument("--compress", default="int8",   choices=["none","int8","learned"])
    p.add_argument("--cloud",    default=None,  help="Cloud API URL (optional, e.g. http://localhost:8000)")
    args = p.parse_args()
    if args.image is None:
        project_root = Path(__file__).resolve().parents[1]
        sample_images = sorted((project_root / "data" / "sample_patches").glob("*.png"))
        if not sample_images:
            p.error("No sample patches found. Run 'make data' or pass --image PATH.")
        args.image = str(sample_images[0])
    asyncio.run(run_simulation(args))


if __name__ == "__main__":
    main()
