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
import tempfile
from datetime import datetime, timezone
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
    patch_overlaps_nodule,
    TARGET_SPACING_MM,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def validate_cache_manifest(
    cache_dir: str | Path,
    expected_uids: Optional[set[str]] = None,
    expected_patch_size: Optional[int] = None,
    expected_subsets: Optional[set[int]] = None,
    expected_uid_subsets: Optional[dict[int, set[str]]] = None,
    require_both_labels: bool = False,
) -> list[dict]:
    """Validate manifest metadata and the contents of every referenced NPZ."""
    cache_path = Path(cache_dir)
    manifest_path = cache_path / "manifest.json"
    try:
        with manifest_path.open(encoding="utf-8") as f:
            entries = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid cache manifest {manifest_path}: {exc}") from exc
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Cache manifest must be a non-empty JSON list: {manifest_path}")

    seen_files: set[str] = set()
    manifest_uids: set[str] = set()
    manifest_uid_subsets: dict[int, set[str]] = {}
    patch_sizes: set[int] = set()
    subset_labels: dict[int, set[int]] = {}
    spacing_values: set[tuple[float, float, float]] = set()
    missing_files: list[str] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest entry {index} must be an object")
        filename = entry.get("file")
        uid = entry.get("uid")
        label = entry.get("label")
        subset = entry.get("subset")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.endswith(".npz")
            or not isinstance(uid, str)
            or not uid
            or type(label) is not int and not isinstance(label, np.integer)
            or int(label) not in (0, 1)
            or type(subset) is not int and not isinstance(subset, np.integer)
            or int(subset) < 0
        ):
            raise ValueError(f"Manifest entry {index} has invalid or missing fields")
        if filename in seen_files:
            raise ValueError(f"Duplicate patch file in manifest: {filename}")
        seen_files.add(filename)
        manifest_uids.add(uid)
        subset_id = int(subset)
        subset_labels.setdefault(subset_id, set()).add(int(label))
        manifest_uid_subsets.setdefault(subset_id, set()).add(uid)
        patch_size = entry.get("patch_size")
        if patch_size is not None:
            if type(patch_size) is not int or patch_size <= 0:
                raise ValueError(f"Manifest entry {index} has invalid patch_size")
            patch_sizes.add(patch_size)
        spacing = entry.get("spacing_mm")
        if spacing is not None:
            try:
                spacing_tuple = tuple(float(value) for value in spacing)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Manifest entry {index} has invalid spacing_mm") from exc
            if len(spacing_tuple) != 3 or not all(np.isfinite(value) and value > 0 for value in spacing_tuple):
                raise ValueError(f"Manifest entry {index} has invalid spacing_mm")
            spacing_values.add(spacing_tuple)
        if not (cache_path / filename).is_file():
            missing_files.append(filename)

    if missing_files:
        preview = ", ".join(missing_files[:5])
        raise FileNotFoundError(
            f"Cache manifest references {len(missing_files)} missing NPZ file(s): {preview}"
        )
    if len(patch_sizes) > 1:
        raise ValueError(f"Cache manifest mixes patch sizes: {sorted(patch_sizes)}")
    if expected_patch_size is not None and patch_sizes and patch_sizes != {expected_patch_size}:
        raise ValueError(
            f"Cache manifest patch size {sorted(patch_sizes)} does not match "
            f"requested patch size {expected_patch_size}"
        )
    if len(spacing_values) > 1:
        raise ValueError(f"Cache manifest mixes physical spacing values: {sorted(spacing_values)}")

    discovered_patch_size = next(iter(patch_sizes), expected_patch_size)
    for index, entry in enumerate(entries):
        patch_path = cache_path / entry["file"]
        try:
            with np.load(patch_path, allow_pickle=False) as patch:
                if not {"image", "mask", "label", "diameter"}.issubset(patch.files):
                    raise ValueError("NPZ must contain image, mask, label, and diameter arrays")
                image = patch["image"]
                mask = patch["mask"]
                patch_label = np.asarray(patch["label"])
                diameter = np.asarray(patch["diameter"])
                if image.ndim != 3 or image.shape[0] != 3 or image.shape[1] != image.shape[2]:
                    raise ValueError(f"image must have shape (3, H, H), got {image.shape}")
                if discovered_patch_size is None:
                    discovered_patch_size = int(image.shape[1])
                if image.shape != (3, discovered_patch_size, discovered_patch_size):
                    raise ValueError(f"image shape {image.shape} does not match patch size {discovered_patch_size}")
                if expected_patch_size is not None and image.shape[1:] != (expected_patch_size, expected_patch_size):
                    raise ValueError(
                        f"Patch {entry['file']} has dimensions {image.shape[1:]}, "
                        f"expected {(expected_patch_size, expected_patch_size)}"
                    )
                if mask.shape != image.shape[1:]:
                    raise ValueError(f"mask shape {mask.shape} is not aligned with image {image.shape}")
                if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
                    raise ValueError("image values must be finite numbers")
                if image.size and (float(image.min()) < -1e-6 or float(image.max()) > 1.0 + 1e-6):
                    raise ValueError("image values must be normalized to [0, 1]")
                if not np.issubdtype(mask.dtype, np.number) or not np.isfinite(mask).all():
                    raise ValueError("mask values must be finite numbers")
                if not np.isin(mask, (0, 1)).all():
                    raise ValueError("mask values must be binary 0/1")
                if patch_label.size != 1 or int(patch_label.item()) not in (0, 1):
                    raise ValueError("NPZ label must be a scalar 0 or 1")
                if int(patch_label.item()) != int(entry["label"]):
                    raise ValueError("NPZ label does not match manifest label")
                if (
                    diameter.size != 1
                    or not np.issubdtype(diameter.dtype, np.number)
                    or not np.isfinite(diameter).all()
                    or float(diameter.item()) < 0
                ):
                    raise ValueError("NPZ diameter must be a finite non-negative scalar")
                if int(entry["label"]) == 1 and not bool(mask.any()):
                    raise ValueError("positive patch has an empty mask")
                if int(entry["label"]) == 0 and bool(mask.any()):
                    raise ValueError("negative patch has a non-empty mask")
                entry_patch_size = entry.get("patch_size")
                if entry_patch_size is not None and int(entry_patch_size) != int(image.shape[1]):
                    raise ValueError("NPZ dimensions do not match manifest patch_size")
        except (OSError, KeyError, ValueError, EOFError) as exc:
            raise ValueError(f"Invalid cached patch at manifest entry {index} ({patch_path}): {exc}") from exc

    if expected_uids is not None:
        missing_uids = sorted(set(expected_uids) - manifest_uids)
        if missing_uids:
            raise ValueError(
                f"Cache manifest omits {len(missing_uids)} expected scan(s): "
                f"{', '.join(missing_uids[:5])}"
            )
    if expected_subsets is not None:
        missing_subsets = sorted(set(expected_subsets) - set(manifest_uid_subsets))
        if missing_subsets:
            raise ValueError(f"Cache has no patches for requested subsets {missing_subsets}")
        label_check_subsets = expected_subsets if require_both_labels else ()
        for subset_id in label_check_subsets:
            labels = subset_labels.get(subset_id, set())
            if labels != {0, 1}:
                missing = sorted({0, 1} - labels)
                raise ValueError(
                    f"Cache subset{subset_id} is missing "
                    f"{'positive' if missing == [1] else 'negative' if missing == [0] else 'positive and negative'} patches"
                )
    if expected_uid_subsets is not None:
        for subset_id, expected_subset_uids in expected_uid_subsets.items():
            actual_subset_uids = manifest_uid_subsets.get(int(subset_id), set())
            if actual_subset_uids != set(expected_subset_uids):
                missing = sorted(set(expected_subset_uids) - actual_subset_uids)
                unexpected = sorted(actual_subset_uids - set(expected_subset_uids))
                raise ValueError(
                    f"Cache scan UIDs do not match raw subset{subset_id}: "
                    f"missing={missing[:5]}, unexpected={unexpected[:5]}"
                )
    if not any(int(entry["label"]) == 1 for entry in entries):
        raise ValueError("Cache manifest has no positive patches")
    if not any(int(entry["label"]) == 0 for entry in entries):
        raise ValueError("Cache manifest has no negative patches")
    return entries


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
    negative_candidates = args.get("negative_candidates", [])
    neg_count  = args["neg_count"]
    patch_size = args["patch_size"]
    subset     = args["subset"]
    target_spacing_mm = args.get("target_spacing_mm", TARGET_SPACING_MM)
    out_dir    = Path(args["out_dir"])

    try:
        volume, origin, spacing, direction = load_mhd(mhd_path, target_spacing_mm)
        volume = clip_and_normalise(volume)
    except Exception as exc:
        raise RuntimeError(f"Failed to load {mhd_path}: {exc}") from exc

    entries = []
    positive_centers = [
        (world_to_voxel(ann["coord"], origin, spacing, direction), ann["diameter_mm"])
        for ann in positives
    ]

    # ---- Positive patches ------------------------------------------------
    for ann_idx, ann in enumerate(positives):
        center_vox = positive_centers[ann_idx][0]
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
        entries.append({
            "file": fname,
            "uid": uid,
            "subset": subset,
            "label": 1,
            "diameter_mm": float(ann["diameter_mm"]),
            "patch_size": patch_size,
            "spacing_mm": list(target_spacing_mm),
        })

    # ---- Negative patches ------------------------------------------------
    Z, Y, X = volume.shape
    half = patch_size // 2
    saved_neg = 0
    attempts  = 0
    max_attempts = max(neg_count * 50, 100)  # bounded fallback for unusual volumes
    seen_centers: set[tuple[int, int, int]] = set()

    def save_negative(center_vox: np.ndarray) -> None:
        nonlocal saved_neg
        patch = extract_patch_2_5d(volume, center_vox, patch_size)
        if patch is None:
            return
        fname = f"{uid}_neg_{saved_neg}.npz"
        np.savez_compressed(
            out_dir / fname,
            image=patch,
            mask=np.zeros((patch_size, patch_size), dtype=np.uint8),
            label=np.int8(0),
            diameter=np.float32(0.0),
        )
        entries.append({
            "file": fname,
            "uid": uid,
            "subset": subset,
            "label": 0,
            "diameter_mm": 0.0,
            "patch_size": patch_size,
            "spacing_mm": list(target_spacing_mm),
        })
        saved_neg += 1

    # Prefer LUNA16's labeled class-0 candidates, while still excluding any
    # patch whose physical footprint intersects an annotated nodule region.
    candidates = list(negative_candidates)
    random.shuffle(candidates)
    for coord_world in candidates:
        if saved_neg >= neg_count:
            break
        center_vox = world_to_voxel(coord_world, origin, spacing, direction)
        cx, cy, cz = (int(v) for v in center_vox)
        if not (
            1 <= cz < Z - 1
            and half <= cy < Y - half
            and half <= cx < X - half
        ):
            continue
        center_key = tuple(int(v) for v in center_vox)
        if center_key in seen_centers:
            continue
        if volume[cz, cy, cx] < 0.036:
            continue
        if any(
            patch_overlaps_nodule(
                center_vox, nodule_vox, diameter_mm, spacing, patch_size
            )
            for nodule_vox, diameter_mm in positive_centers
        ):
            continue
        seen_centers.add(center_key)
        save_negative(center_vox)

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
        center_key = tuple(int(v) for v in center_vox)
        if center_key in seen_centers:
            continue
        if any(
            patch_overlaps_nodule(
                center_vox, nodule_vox, diameter_mm, spacing, patch_size
            )
            for nodule_vox, diameter_mm in positive_centers
        ):
            continue
        seen_centers.add(center_key)
        save_negative(center_vox)

    logger.info(f"  {uid}: {len(positives)} pos, {saved_neg} neg patches saved")
    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _process_jobs(jobs: list[dict], workers: int) -> tuple[list[dict], list[str]]:
    entries: list[dict] = []
    failures: list[str] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_volume, job): job["uid"] for job in jobs}
            for future in as_completed(futures):
                uid = futures[future]
                try:
                    entries.extend(future.result())
                except Exception as exc:
                    failures.append(f"{uid}: {exc}")
                if len(entries) and len(failures) == 0 and len(entries) % 500 == 0:
                    logger.info("Generated %d patches so far", len(entries))
    else:
        for job in jobs:
            try:
                entries.extend(_process_volume(job))
            except Exception as exc:
                failures.append(f"{job['uid']}: {exc}")
                break
    return entries, failures

def build_cache(
    data_root:  str,
    out_dir:    str,
    patch_size: int   = 64,
    neg_ratio:  float = 1.0,
    workers:    int   = 4,
    seed:       int   = 42,
    rebuild:    bool  = False,
    target_spacing_mm: tuple[float, float, float] = TARGET_SPACING_MM,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    data_root_path = Path(data_root).expanduser().resolve()
    out_path       = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    manifest_path = out_path / "manifest.json"
    if workers < 1:
        raise ValueError("workers must be at least 1")
    target_spacing_mm = tuple(float(v) for v in target_spacing_mm)
    if len(target_spacing_mm) != 3 or any(v <= 0 for v in target_spacing_mm):
        raise ValueError("target_spacing_mm must contain three positive values")

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

    # The candidate file provides independently labeled class-0 locations.
    # They are preferred over arbitrary voxel samples for negative patches.
    candidates_by_uid: dict[str, list[np.ndarray]] = {}
    candidates_csv = data_root_path / "candidates.csv"
    if candidates_csv.exists():
        with open(candidates_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("class") != "0":
                    continue
                candidates_by_uid.setdefault(row["seriesuid"], []).append(
                    np.array([
                        float(row["coordX"]),
                        float(row["coordY"]),
                        float(row["coordZ"]),
                    ], dtype=np.float64)
                )

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
        subset_match = next(
            (
                match
                for parent in mhd_file.parents
                if (match := re.fullmatch(r"subset(\d+)", parent.name))
            ),
            None,
        )
        if subset_match is None:
            raise ValueError(f"Cannot determine LUNA16 subset for {mhd_file}")
        jobs.append({
            "mhd_path":   str(mhd_file),
            "uid":        uid,
            "positives":  positives,
            "negative_candidates": candidates_by_uid.get(uid, []),
            "neg_count":  neg_count,
            "patch_size": patch_size,
            "subset":     int(subset_match.group(1)),
            "target_spacing_mm": target_spacing_mm,
        })

    logger.info(f"Processing {len(jobs)} volumes with {workers} worker(s)…")

    expected_uids = {job["uid"] for job in jobs}
    expected_positive_count = sum(len(job["positives"]) for job in jobs)
    expected_negative_count = sum(job["neg_count"] for job in jobs)
    if manifest_path.exists() and not rebuild:
        try:
            existing = validate_cache_manifest(
                manifest_path.parent, expected_uids=expected_uids
            )
        except (ValueError, FileNotFoundError) as exc:
            raise RuntimeError(
                f"Existing cache is incomplete or invalid: {exc}. "
                "Run again with --rebuild to replace it."
            ) from exc
        existing_uids = {entry["uid"] for entry in existing}
        extra_uids = sorted(existing_uids - expected_uids)
        if extra_uids:
            raise RuntimeError(
                f"Existing cache contains {len(extra_uids)} unexpected scans "
                f"({', '.join(extra_uids[:5])}); pass --rebuild to replace it."
            )
        existing_positive = sum(int(entry["label"]) == 1 for entry in existing)
        existing_negative = sum(int(entry["label"]) == 0 for entry in existing)
        existing_patch_sizes = {
            int(entry["patch_size"])
            for entry in existing
            if entry.get("patch_size") is not None
        }
        existing_spacings = {
            tuple(float(v) for v in entry["spacing_mm"])
            for entry in existing
            if entry.get("spacing_mm") is not None
        }
        if (
            existing_positive != expected_positive_count
            or existing_negative != expected_negative_count
            or existing_patch_sizes != {patch_size}
            or existing_spacings != {target_spacing_mm}
        ):
            raise RuntimeError(
                "Existing cache does not match requested patch counts/size/spacing; "
                "pass --rebuild to replace it."
            )
        logger.info(
            "Valid cache already exists at %s (%d patches); leaving it unchanged.",
            out_path, len(existing),
        )
        return
    if not manifest_path.exists() and not rebuild and any(out_path.glob("*.npz")):
        raise RuntimeError(
            f"NPZ files exist without a manifest in {out_path}; this looks like a "
            "partial cache. Run again with --rebuild to replace it."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out_path.name}-build-", dir=out_path.parent) as stage:
        for job in jobs:
            job["out_dir"] = stage
        all_entries, failures = _process_jobs(jobs, workers)
        if failures:
            raise RuntimeError(
                "Cache generation failed; no manifest was written: "
                + "; ".join(failures[:5])
            )

        pos_count = sum(int(e["label"]) == 1 for e in all_entries)
        neg_count = sum(int(e["label"]) == 0 for e in all_entries)
        if pos_count != expected_positive_count or neg_count != expected_negative_count:
            raise RuntimeError(
                "Patch generation was incomplete; no manifest was written. "
                f"Expected {expected_positive_count} positive and "
                f"{expected_negative_count} negative patches, got "
                f"{pos_count} positive and {neg_count} negative."
            )

        random.shuffle(all_entries)
        stage_path = Path(stage)
        stage_manifest = stage_path / "manifest.json"
        with stage_manifest.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(all_entries, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        validate_cache_manifest(stage_path, expected_uids=expected_uids)

        # Publish the complete directory atomically. When rebuilding, keep
        # the replaced cache as a timestamped backup rather than deleting it.
        backup_path = None
        if out_path.exists():
            if not rebuild and any(out_path.iterdir()):
                raise RuntimeError(
                    f"Output directory {out_path} is not empty; pass --rebuild "
                    "to replace it."
                )
            if rebuild:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                backup_path = out_path.with_name(f"{out_path.name}.backup-{stamp}")
                suffix = 1
                while backup_path.exists():
                    backup_path = out_path.with_name(
                        f"{out_path.name}.backup-{stamp}-{suffix}"
                    )
                    suffix += 1
                os.replace(out_path, backup_path)
            else:
                out_path.rmdir()
        try:
            os.replace(stage_path, out_path)
        except Exception:
            if backup_path is not None and backup_path.exists() and not out_path.exists():
                os.replace(backup_path, out_path)
            raise

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
    parser.add_argument("--rebuild",    action="store_true",           help="Rebuild an existing or partial cache")
    parser.add_argument(
        "--target-spacing-mm", type=float, nargs=3,
        default=TARGET_SPACING_MM,
        metavar=("X", "Y", "Z"),
        help="Resample CTs to this physical spacing in mm (default: 1 1 1)",
    )
    args = parser.parse_args()

    build_cache(
        data_root  = args.data_root,
        out_dir    = args.out_dir,
        patch_size = args.patch_size,
        neg_ratio  = args.neg_ratio,
        workers    = args.workers,
        seed       = args.seed,
        rebuild    = args.rebuild,
        target_spacing_mm=tuple(args.target_spacing_mm),
    )
