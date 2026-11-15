#!/usr/bin/env python3
"""
Generates synthetic CT-like patches and a fake LUNA16 annotation CSV.
Useful for testing the full pipeline without downloading the real dataset.

Usage:
  python scripts/generate_test_data.py --out ./data/luna16 --n-scans 20 --n-nodules 3

FIXES APPLIED
=============
FIX-1  (IndexError / wrong dimension):
  BEFORE: volume[body_mask] = rng.normal(...)
          body_mask is 2D (H,W), so NumPy tried to index the 3D volume on axis-0,
          raising an IndexError.
  AFTER:  volume[:, body_mask] = rng.normal(-300, 150,
              size=(shape[0], int(body_mask.sum()))).astype(np.float32)
          The slice ':' keeps axis-0 intact; body_mask selects the flat Y/X pixels.

FIX-2  (Identical scans / frozen seed):
  BEFORE: rng = np.random.default_rng(seed=42)   # inside generate_ct_volume()
          Every call with the same arguments produced bit-identical volumes.
  AFTER:  The RNG is passed in from the caller (main loop).
          Each scan uses seed = base_seed + scan_index, guaranteeing
          independent, reproducible variation across scans.
"""
import argparse
import csv
import os
import uuid
from pathlib import Path

import numpy as np
from PIL import Image


def gaussian_sphere(shape, center, radius, amplitude=0.6):
    """Draw a 3D Gaussian blob (simulates a nodule)."""
    Z, Y, X = np.ogrid[:shape[0], :shape[1], :shape[2]]
    cz, cy, cx = center
    dist = np.sqrt((Z - cz)**2 + (Y - cy)**2 + (X - cx)**2)
    return amplitude * np.exp(-dist**2 / (2 * (radius / 2)**2))


def generate_ct_volume(shape=(64, 128, 128), n_nodules=2, rng=None):
    """
    Generate a synthetic lung CT volume in Hounsfield-like range [-1000, 400].
    Returns (volume_HU, nodule_annotations, spacing, origin).

    Parameters
    ----------
    rng : np.random.Generator
        Caller-supplied RNG for reproducibility. If None a default is used.
    """
    # FIX-2: accept an external RNG so callers can control the seed per scan.
    if rng is None:
        rng = np.random.default_rng(seed=42)  # kept only as a safe fallback

    # Background: air (~-800 HU) with lung parenchyma noise
    volume = rng.normal(-700, 80, shape).astype(np.float32)

    # --- FIX-1: body_mask is 2D (H,W); index with volume[:, body_mask] -----------
    # Add a soft-tissue cylinder (body boundary)
    _Y, _X = np.ogrid[:shape[1], :shape[2]]
    dist_from_center = np.sqrt(
        (_Y - shape[1] // 2) ** 2 + (_X - shape[2] // 2) ** 2
    )
    body_mask = dist_from_center < (min(shape[1], shape[2]) // 2 - 5)  # (H, W) bool
    # Correct 3D assignment: broadcast over the Z axis
    volume[:, body_mask] = rng.normal(
        -300, 150, size=(shape[0], int(body_mask.sum()))
    ).astype(np.float32)
    # -------------------------------------------------------------------------

    # Add vessel-like structures
    for _ in range(5):
        vx = rng.integers(20, shape[2] - 20)
        vy = rng.integers(20, shape[1] - 20)
        volume[:, vy - 2:vy + 2, vx - 2:vx + 2] = rng.normal(
            -100, 30, (shape[0], 4, 4)
        )

    # Add nodules
    annotations = []
    spacing = np.array([1.5, 0.7, 0.7])   # mm per voxel (Z, Y, X)
    origin  = np.array([0.0, 0.0, 0.0])

    for _ in range(n_nodules):
        margin = 20
        cz = rng.integers(margin, shape[0] - margin)
        cy = rng.integers(margin, shape[1] - margin)
        cx = rng.integers(margin, shape[2] - margin)
        radius_vox  = rng.integers(3, 15)
        diameter_mm = radius_vox * np.mean(spacing[1:]) * 2

        nodule  = gaussian_sphere(shape, (cz, cy, cx), radius_vox, amplitude=400)
        volume += nodule

        # World coordinates
        coord_world = np.array([cx, cy, cz], dtype=float) * spacing[::-1] + origin
        annotations.append({
            "coordX": float(coord_world[0]),
            "coordY": float(coord_world[1]),
            "coordZ": float(coord_world[2]),
            "diameter_mm": float(diameter_mm),
        })

    volume = np.clip(volume, -1000, 400)
    return volume, annotations, spacing, origin


def save_mhd(volume: np.ndarray, spacing: np.ndarray, origin: np.ndarray, out_path: str):
    """Save volume as SimpleITK .mhd / .raw pair."""
    try:
        import SimpleITK as sitk
        itk_img = sitk.GetImageFromArray(volume)
        itk_img.SetSpacing(tuple(spacing[::-1].tolist()))   # ITK: X,Y,Z
        itk_img.SetOrigin(tuple(origin.tolist()))
        sitk.WriteImage(itk_img, out_path)
        return True
    except ImportError:
        raw_path = out_path.replace(".mhd", ".raw")
        volume.tofile(raw_path)
        with open(out_path, "w") as f:
            f.write("ObjectType = Image\n")
            f.write("NDims = 3\n")
            f.write("BinaryData = True\n")
            f.write("BinaryDataByteOrderMSB = False\n")
            f.write("CompressedData = False\n")
            f.write("TransformMatrix = 1 0 0 0 1 0 0 0 1\n")
            f.write(f"Offset = {' '.join(str(o) for o in origin)}\n")
            f.write("CenterOfRotation = 0 0 0\n")
            f.write(f"ElementSpacing = {' '.join(str(s) for s in spacing[::-1])}\n")
            f.write(f"DimSize = {volume.shape[2]} {volume.shape[1]} {volume.shape[0]}\n")
            f.write("ElementType = MET_FLOAT\n")
            f.write(f"ElementDataFile = {os.path.basename(raw_path)}\n")
        return True


def save_png_patch(volume: np.ndarray, center_vox: tuple, out_dir: str, uid: str, patch_size=64):
    """Extract and save a 2.5D PNG patch centred on a nodule."""
    cz, cy, cx = center_vox
    half = patch_size // 2
    Z, Y, X = volume.shape
    cz = max(1, min(Z - 2, cz))
    cy = max(half, min(Y - half, cy))
    cx = max(half, min(X - half, cx))

    hu_min, hu_max = -1000, 400
    slices = []
    for dz in (-1, 0, 1):
        sl = volume[cz + dz, cy - half: cy + half, cx - half: cx + half]
        sl_norm = ((sl - hu_min) / (hu_max - hu_min) * 255).clip(0, 255).astype(np.uint8)
        slices.append(sl_norm)

    rgb = np.stack(slices, axis=2)   # (H, W, 3)
    img = Image.fromarray(rgb)
    path = os.path.join(out_dir, f"{uid}_patch.png")
    img.save(path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic LUNA16-like test data")
    parser.add_argument("--out",       default="./data/luna16",       help="Output root directory")
    parser.add_argument("--n-scans",   type=int, default=20,          help="Number of synthetic CT scans")
    parser.add_argument("--n-nodules", type=int, default=3,           help="Nodules per scan")
    parser.add_argument("--subsets",   type=int, default=2,           help="Number of subset folders (0..N-1)")
    parser.add_argument("--patch-dir", default="./data/sample_patches", help="Where to save PNG patches")
    parser.add_argument("--seed",      type=int, default=0,           help="Base RNG seed for reproducibility")
    args = parser.parse_args()

    out_root  = Path(args.out)
    patch_dir = Path(args.patch_dir)
    patch_dir.mkdir(parents=True, exist_ok=True)

    all_annotations = []
    all_candidates  = []

    print(f"Generating {args.n_scans} synthetic CT scans into {out_root}...")

    for i in range(args.n_scans):
        subset_id  = i % args.subsets
        subset_dir = out_root / f"subset{subset_id}"
        subset_dir.mkdir(parents=True, exist_ok=True)

        uid = str(uuid.uuid4()).replace("-", "")

        # FIX-2: each scan gets its own independent, reproducible RNG
        scan_rng = np.random.default_rng(seed=args.seed + i)
        volume, anns, spacing, origin = generate_ct_volume(
            shape=(64, 128, 128), n_nodules=args.n_nodules, rng=scan_rng
        )

        # Save .mhd / .raw
        mhd_path = str(subset_dir / f"{uid}.mhd")
        save_mhd(volume, spacing, origin, mhd_path)

        # Save PNG patches + annotation CSV rows
        for ann in anns:
            cz = int(ann["coordZ"] / spacing[0])
            cy = int(ann["coordY"] / spacing[1])
            cx = int(ann["coordX"] / spacing[2])
            save_png_patch(volume, (cz, cy, cx), str(patch_dir), uid)

            all_annotations.append({
                "seriesuid":   uid,
                "coordX":      ann["coordX"],
                "coordY":      ann["coordY"],
                "coordZ":      ann["coordZ"],
                "diameter_mm": ann["diameter_mm"],
            })
            all_candidates.append({
                "seriesuid": uid,
                "coordX": ann["coordX"],
                "coordY": ann["coordY"],
                "coordZ": ann["coordZ"],
                "class": 1,
            })

        # Add negative candidates with per-scan seed (FIX-2 extended)
        neg_rng = np.random.default_rng(seed=args.seed + i + args.n_scans)
        for _ in range(args.n_nodules * 5):
            cx_ = float(neg_rng.integers(20, 108)) * spacing[2]
            cy_ = float(neg_rng.integers(20, 108)) * spacing[1]
            cz_ = float(neg_rng.integers(5, 59))   * spacing[0]
            all_candidates.append({
                "seriesuid": uid,
                "coordX": cx_, "coordY": cy_, "coordZ": cz_, "class": 0,
            })

        print(f"  [{i+1}/{args.n_scans}] subset{subset_id}/{uid[:8]}... "
              f"({args.n_nodules} nodules)")

    # Write annotation CSVs
    ann_path = out_root / "annotations.csv"
    with open(ann_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"])
        writer.writeheader()
        writer.writerows(all_annotations)

    cand_path = out_root / "candidates.csv"
    with open(cand_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seriesuid", "coordX", "coordY", "coordZ", "class"])
        writer.writeheader()
        writer.writerows(all_candidates)

    print(f"\n✓ Generated:")
    print(f"  {args.n_scans} CT volumes in {out_root}/subset0..subset{args.subsets-1}/")
    print(f"  {len(all_annotations)} annotations → {ann_path}")
    print(f"  {len(all_candidates)} candidates  → {cand_path}")
    print(f"  {len(list(patch_dir.glob('*.png')))} PNG patches → {patch_dir}/")
    print(f"\nNow run training with:")
    print(f"  python scripts/run_experiments.py --phase 1 --data {out_root} --epochs 5")


if __name__ == "__main__":
    main()
