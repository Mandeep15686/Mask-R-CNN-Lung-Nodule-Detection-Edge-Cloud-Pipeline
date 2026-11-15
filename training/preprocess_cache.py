"""
LUNA16 Patch Pre-extraction Script
====================================
Run this ONCE before training to convert full CT volumes into small pre-extracted
patches saved as .npz files.  After this, training reads tiny files (~12 KB each)
instead of full CT volumes (~150-300 MB each), eliminating the SSD bottleneck.

Usage:
    python -m training.preprocess_cache \
        --data_root ./data/luna16 \
        --out_dir   ./data/patches_cache \
        --patch_size 64 \
        --neg_ratio  1.0 \
        --workers    4

Outputs per patch  (saved as <out_dir>/<uid>_<idx>.npz):
    image   : float32 (3, patch_size, patch_size)   -- 2.5D axial patch, normalised
    mask    : uint8   (patch_size, patch_size)        -- binary nodule mask
    label   : int     0 or 1
    diameter: float   nodule diameter in mm (0 for negatives)
    uid     : str     LUNA16 series UID

A manifest JSON file  <out_dir>/manifest.json  lists every patch path so the
dataset loader can read it instantly without globbing the whole directory.
"""

import os
import csv
import json
import random
import logging
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional
import re
import numpy as np

# ---------------------------------------------------------------------------
# Reuse the same helpers from dataset.py so pre-processing is identical
# ---------------------------------------------------------------------------
from .dataset import (
    load_mhd,
    world_to_voxel,
    clip_and_normalise,
    extract_patch_2_5d,
    create_spherical_mask,
    HU_MIN,
    HU_MAX,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Per-volume worker
# ---------------------------------------------------------------------------

def _process_volume(args: dict) -> list[dict]:
    """
    Load one CT volume, extract all positive patches and neg_count negative
    patches, save each as an .npz file, return list of manifest entries.
    Called in a subprocess so the main process stays free.
    """
    mhd_path   = args["mhd_path"]
    uid        = args["uid"]
    positives  = args["positives"]    # list[{"coord": np.array, "diameter_mm": float}]
    neg_count  = args["neg_count"]
    patch_size = args["patch_size"]
    out_dir    = Path(args["out_dir"])

    try:
        volume, origin, spacing, direction = load_mhd(mhd_path)
        volume = clip_and_normalise(volume)
    except Exception as exc:
        logger.error(f"Failed to load {mhd_path}: {exc}")
        return []

    entries = []

    # ---- Positive patches ------------------------------------------------
    for ann_idx, ann in enumerate(positives):
        center_vox = world_to_voxel(ann["coord"], origin, spacing, direction)
        patch = extract_patch_2_5d(volume, center_vox, patch_size)
        if patch is None:
            logger.warning(f"  OOB positive patch skipped: {uid} ann {ann_idx}")
            continue
        mask = create_spherical_mask(patch_size, center_vox, ann["diameter_mm"], spacing)
        fname = f"{uid}_pos_{ann_idx}.npz"
        np.savez_compressed(
            out_dir / fname,
            image=patch,
            mask=mask,
            label=np.int8(1),
            diameter=np.float32(ann["diameter_mm"]),
        )
        entries.append({"file": fname, "uid": uid, "label": 1,
                        "diameter_mm": float(ann["diameter_mm"])})

    # ---- Negative patches ------------------------------------------------
    Z, Y, X = volume.shape
    half = patch_size // 2
    saved_neg = 0
    attempts  = 0
    max_attempts = neg_count * 20  # avoid infinite loop on unusual volumes

    while saved_neg < neg_count and attempts < max_attempts:
        attempts += 1

        # Sample inside interior of volume (avoid edges)
        cz = random.randint(1, Z - 2)
        cy = random.randint(half, Y - half - 1)
        cx = random.randint(half, X - half - 1)

        # Skip if voxel value looks like air/background (HU < -950 → normalised < 0.036)
        if volume[cz, cy, cx] < 0.036:
            continue

        center_vox = np.array([cx, cy, cz])
        patch = extract_patch_2_5d(volume, center_vox, patch_size)
        if patch is None:
            continue

        mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
        fname = f"{uid}_neg_{saved_neg}.npz"
        np.savez_compressed(
            out_dir / fname,
            image=patch,
            mask=mask,
            label=np.int8(0),
            diameter=np.float32(0.0),
        )
        entries.append({"file": fname, "uid": uid, "label": 0, "diameter_mm": 0.0})
        saved_neg += 1

    logger.info(f"  {uid}: {len(positives)} pos, {saved_neg} neg patches saved")
    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_cache(
    data_root:  str,
    out_dir:    str,
    patch_size: int   = 64,
    neg_ratio:  float = 1.0,
    workers:    int   = 4,
    seed:       int   = 42,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    data_root_path = Path(data_root).expanduser().resolve()
    out_path       = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    manifest_path = out_path / "manifest.json"
    if manifest_path.exists():
        logger.info(f"Cache already exists at {out_path}. Delete manifest.json to rebuild.")
        return

    # ---- Parse annotations.csv ------------------------------------------
    annotations: dict[str, list[dict]] = {}
    ann_csv = data_root_path / "annotations.csv"
    if not ann_csv.exists():
        raise FileNotFoundError(f"annotations.csv not found at {ann_csv}")

    with open(ann_csv) as f:
        for row in csv.DictReader(f):
            uid = row["seriesuid"]
            annotations.setdefault(uid, []).append({
                "coord": np.array([float(row["coordX"]),
                                   float(row["coordY"]),
                                   float(row["coordZ"])]),
                "diameter_mm": float(row["diameter_mm"]),
            })

    # ---- Discover all .mhd files ----------------------------------------
    all_mhd = sorted(
        p for p in data_root_path.rglob("*.mhd")
        if re.fullmatch(r"\d+(?:\.\d+)+", p.stem)
    )
    if not all_mhd:
        raise FileNotFoundError(f"No .mhd files found under {data_root_path}")
    logger.info(f"Found {len(all_mhd)} CT volumes under {data_root_path}")

    # ---- Build job list --------------------------------------------------
    jobs = []
    for mhd_file in all_mhd:
        uid       = mhd_file.stem
        positives = annotations.get(uid, [])
        neg_count = max(1, int(len(positives) * neg_ratio)) if positives else 2
        jobs.append({
            "mhd_path":   str(mhd_file),
            "uid":        uid,
            "positives":  positives,
            "neg_count":  neg_count,
            "patch_size": patch_size,
            "out_dir":    str(out_path),
        })

    logger.info(f"Processing {len(jobs)} volumes with {workers} worker(s)…")

    # ---- Run in parallel -------------------------------------------------
    all_entries: list[dict] = []
    done = 0

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_volume, job): job["uid"] for job in jobs}
            for fut in as_completed(futures):
                try:
                    all_entries.extend(fut.result())
                except Exception as exc:
                    logger.error(f"Worker failed for {futures[fut]}: {exc}")
                done += 1
                if done % 10 == 0:
                    logger.info(f"  Progress: {done}/{len(jobs)} volumes done")
    else:
        for job in jobs:
            all_entries.extend(_process_volume(job))
            done += 1
            if done % 10 == 0:
                logger.info(f"  Progress: {done}/{len(jobs)} volumes done")

    # ---- Write manifest --------------------------------------------------
    random.shuffle(all_entries)
    with open(manifest_path, "w") as f:
        json.dump(all_entries, f, indent=2)

    pos_count = sum(1 for e in all_entries if e["label"] == 1)
    neg_count = sum(1 for e in all_entries if e["label"] == 0)
    logger.info(
        f"\nCache built successfully at {out_path}\n"
        f"  Total patches : {len(all_entries)}\n"
        f"  Positives     : {pos_count}\n"
        f"  Negatives     : {neg_count}\n"
        f"  Manifest      : {manifest_path}\n"
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-extract LUNA16 patches to eliminate SSD bottleneck during training."
    )
    parser.add_argument("--data_root",  default="./data/luna16",       help="Path to LUNA16 root")
    parser.add_argument("--out_dir",    default="./data/patches_cache", help="Output cache directory")
    parser.add_argument("--patch_size", type=int,   default=64,         help="Patch size in voxels")
    parser.add_argument("--neg_ratio",  type=float, default=1.0,        help="Negatives per positive")
    parser.add_argument("--workers",    type=int,   default=4,          help="Parallel workers")
    parser.add_argument("--seed",       type=int,   default=42,         help="Random seed")
    args = parser.parse_args()

    build_cache(
        data_root  = args.data_root,
        out_dir    = args.out_dir,
        patch_size = args.patch_size,
        neg_ratio  = args.neg_ratio,
        workers    = args.workers,
        seed       = args.seed,
    )
