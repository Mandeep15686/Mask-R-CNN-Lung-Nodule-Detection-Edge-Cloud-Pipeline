"""
LUNA16 Dataset loader for Mask R-CNN training.
Reads .mhd/.raw CT volumes, extracts 2.5D patches centred on annotated nodules.

Performance tip — eliminate SSD bottleneck
-------------------------------------------
Pre-extract all patches once with:

    python -m training.preprocess_cache \\
        --data_root ./data/luna16 \\
        --out_dir   ./data/patches_cache

Then pass ``cache_dir`` to ``build_dataloaders``.  The dataset will load
pre-extracted ~12 KB .npz files instead of 150-300 MB CT volumes, reducing
SSD I/O by ~16 000× and removing the CPU worker bottleneck entirely.
"""
import os
import csv
import json
import logging
import random
import re
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)

# LUNA16 Hounsfield Unit windows for lung parenchyma
HU_MIN, HU_MAX = -1000, 400
TARGET_SPACING_MM = (1.0, 1.0, 1.0)


def load_mhd(path: str, target_spacing_mm=TARGET_SPACING_MM):
    """
    Load MetaImage CT volume resampled to the target physical voxel spacing.
    The default is the documented 1 mm isotropic grid.
    """
    itk_img = sitk.ReadImage(str(path))

    target_spacing_mm = tuple(float(v) for v in target_spacing_mm)
    if len(target_spacing_mm) != 3 or any(v <= 0 for v in target_spacing_mm):
        raise ValueError("target_spacing_mm must contain three positive values")

    original_spacing = np.asarray(itk_img.GetSpacing(), dtype=np.float64)
    if not np.allclose(original_spacing, target_spacing_mm):
        original_size = np.asarray(itk_img.GetSize(), dtype=np.float64)
        target_size = np.maximum(
            np.rint(original_size * original_spacing / target_spacing_mm).astype(int),
            1,
        )
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize([int(v) for v in target_size])
        resampler.SetOutputSpacing(target_spacing_mm)
        resampler.SetOutputOrigin(itk_img.GetOrigin())
        resampler.SetOutputDirection(itk_img.GetDirection())
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(float(HU_MIN))
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        itk_img = resampler.Execute(itk_img)

    volume = sitk.GetArrayFromImage(itk_img).astype(np.float32)

    origin = np.asarray(itk_img.GetOrigin(), dtype=np.float64)
    spacing = np.asarray(itk_img.GetSpacing(), dtype=np.float64)

    direction = np.asarray(
        itk_img.GetDirection(), dtype=np.float64
    ).reshape(3, 3)

    return volume, origin, spacing, direction


def world_to_voxel(
    coord_world,
    origin,
    spacing,
    direction=None
):
    """
    Convert physical coordinates (X,Y,Z) into voxel indices (X,Y,Z).
    Correctly accounts for image direction and spacing.
    """
    coord_world = np.asarray(coord_world, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)

    if direction is None:
        direction = np.eye(3, dtype=np.float64)
    else:
        direction = np.asarray(direction, dtype=np.float64).reshape(3, 3)

    affine = direction @ np.diag(spacing)

    voxel = np.linalg.solve(affine, coord_world - origin)

    return np.rint(voxel).astype(np.int64)


def voxel_to_world(voxel, origin, spacing, direction=None):
    """Convert voxel indices (X,Y,Z) to physical coordinates (X,Y,Z)."""
    voxel = np.asarray(voxel, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)
    if direction is None:
        direction = np.eye(3, dtype=np.float64)
    else:
        direction = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    return origin + direction @ (voxel * spacing)


def patch_overlaps_nodule(
    patch_center_vox,
    nodule_center_vox,
    diameter_mm: float,
    spacing,
    patch_size: int,
    safety_margin_mm: float = 2.0,
) -> bool:
    """Return whether a 2.5D patch intersects an annotated nodule buffer."""
    patch_center_vox = np.asarray(patch_center_vox, dtype=np.float64)
    nodule_center_vox = np.asarray(nodule_center_vox, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)

    half = patch_size / 2.0
    xy_extent = max(half - 0.5, 0.0)
    voxel_gaps = np.array([
        max(abs(nodule_center_vox[0] - patch_center_vox[0]) - xy_extent, 0.0),
        max(abs(nodule_center_vox[1] - patch_center_vox[1]) - xy_extent, 0.0),
        max(abs(nodule_center_vox[2] - patch_center_vox[2]) - 1.0, 0.0),
    ])
    distance_mm = float(np.linalg.norm(voxel_gaps * spacing))
    return distance_mm <= (float(diameter_mm) / 2.0 + safety_margin_mm)


def clip_and_normalise(volume: np.ndarray) -> np.ndarray:
    """Clip HU to lung window and normalise to [0, 1]."""
    vol = np.clip(volume, HU_MIN, HU_MAX)
    vol = (vol - HU_MIN) / (HU_MAX - HU_MIN)
    return vol.astype(np.float32)


def extract_patch_2_5d(
    volume: np.ndarray,
    center_vox: np.ndarray,
    patch_size: int = 64,
) -> Optional[np.ndarray]:
    """
    Extract three axial slices centered on the annotation with zero padding.

    The annotated voxel is always at (patch_size // 2, patch_size // 2) in
    the central channel. Neighboring Z channels are zero-padded at the volume
    ends. Return ``None`` only when the annotation center itself is outside
    the volume.
    """
    cz, cy, cx = int(center_vox[2]), int(center_vox[1]), int(center_vox[0])
    half = patch_size // 2
    Z, Y, X = volume.shape

    if patch_size < 1:
        raise ValueError("patch_size must be at least 1")
    if not (0 <= cz < Z and 0 <= cy < Y and 0 <= cx < X):
        return None

    # Use a fixed-size destination and copy only the in-volume intersection.
    # This preserves the requested center index and fills every out-of-volume
    # pixel (including adjacent Z slices) with normalized zero (-1000 HU).
    patch = np.zeros((3, patch_size, patch_size), dtype=volume.dtype)
    y0, x0 = cy - half, cx - half
    y1, x1 = y0 + patch_size, x0 + patch_size
    src_y0, src_x0 = max(y0, 0), max(x0, 0)
    src_y1, src_x1 = min(y1, Y), min(x1, X)
    dst_y0, dst_x0 = src_y0 - y0, src_x0 - x0
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    for dz in (-1, 0, 1):
        z = cz + dz
        if 0 <= z < Z:
            patch[dz + 1, dst_y0:dst_y1, dst_x0:dst_x1] = volume[
                z, src_y0:src_y1, src_x0:src_x1
            ]

    return patch


def create_spherical_mask(
    patch_size: int,
    center_vox: np.ndarray,
    diameter_mm: float,
    spacing: np.ndarray,
) -> np.ndarray:
    """
    Create a 2D binary mask of an elliptical nodule approximation on the
    central slice, given its diameter in mm and voxel spacing.
    """
    radius_x = (diameter_mm / 2.0) / spacing[0]
    radius_y = (diameter_mm / 2.0) / spacing[1]
    half = patch_size // 2
    yy, xx = np.ogrid[:patch_size, :patch_size]
    mask = (
        ((xx - half) ** 2 / (radius_x ** 2 + 1e-6))
        + ((yy - half) ** 2 / (radius_y ** 2 + 1e-6))
    ) <= 1.0
    return mask.astype(np.uint8)


class CachedLUNA16Dataset(Dataset):
    """
    Fast dataset that loads pre-extracted .npz patch files.

    Use this after running ``preprocess_cache.py`` once.  Each item read is a
    ~12 KB .npz file instead of a ~200 MB CT volume, eliminating the SSD
    bottleneck completely.

    Args:
        cache_dir : directory produced by ``preprocess_cache.py``
        subsets   : list of LUNA16 subset indices to include (filters by UID prefix
                    if subset mapping is recorded in manifest; otherwise uses all)
        augment   : apply random flip/rotate augmentation to positive patches
    """

    def __init__(
        self,
        cache_dir: str,
        subsets: Optional[list[int]] = None,
        augment: bool = True,
        expected_patch_size: Optional[int] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.augment   = augment

        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Cache manifest not found at {manifest_path}. "
                "Run:  python -m training.preprocess_cache --data_root <luna16> --out_dir <cache>"
            )

        with open(manifest_path) as f:
            all_entries: list[dict] = json.load(f)
        if not isinstance(all_entries, list) or not all_entries:
            raise ValueError(f"Cache manifest must be a non-empty list: {manifest_path}")

        # Optionally filter by subset.  The manifest stores UIDs; LUNA16 UIDs
        # don't encode subset numbers, so we keep all entries when subsets=None.
        # If a 'subset' key was added during pre-processing, filter by it.
        if subsets is not None:
            subset_set = {int(s) for s in subsets}

            missing_metadata = [
                e for e in all_entries if "subset" not in e
            ]

            if missing_metadata:
                raise ValueError(
                    "Manifest contains entries without subset metadata."
                )

            all_entries = [
                e for e in all_entries
                if int(e["subset"]) in subset_set
            ]

            if not all_entries:
                raise ValueError(
                    f"No patches found for subsets {sorted(subset_set)}"
                )

        self.entries = all_entries
        missing_files = [
            entry.get("file", "<missing file field>")
            for entry in self.entries
            if not isinstance(entry, dict)
            or not isinstance(entry.get("file"), str)
            or not (self.cache_dir / entry["file"]).is_file()
        ]
        if missing_files:
            raise FileNotFoundError(
                f"Cached subset references missing NPZ file(s): {missing_files[:5]}"
            )
        manifest_patch_sizes = {
            int(entry["patch_size"])
            for entry in self.entries
            if entry.get("patch_size") is not None
        }
        if len(manifest_patch_sizes) > 1:
            raise ValueError("Cache manifest mixes patch sizes")
        self.patch_size = next(iter(manifest_patch_sizes), None)
        if self.patch_size is None:
            with np.load(self.cache_dir / self.entries[0]["file"]) as data:
                self.patch_size = int(data["image"].shape[-1])
        spacing_values = {
            tuple(float(v) for v in entry["spacing_mm"])
            for entry in self.entries
            if entry.get("spacing_mm") is not None
        }
        if len(spacing_values) > 1:
            raise ValueError("Cache manifest mixes physical spacing values")
        self.preprocessing_spacing_mm = (
            next(iter(spacing_values)) if spacing_values else None
        )
        if expected_patch_size is not None and self.patch_size != expected_patch_size:
            raise ValueError(
                f"Cache patch size is {self.patch_size}, requested {expected_patch_size}"
            )
        self.scan_uids = {entry["uid"] for entry in self.entries}
        pos = sum(1 for e in self.entries if e["label"] == 1)
        neg = len(self.entries) - pos
        self.cache_summary = {
            "total_patches": len(self.entries),
            "positive_patches": pos,
            "negative_patches": neg,
            "patch_dimensions": [3, self.patch_size, self.patch_size],
            "channel_count": 3,
            "available_subsets": sorted({int(e["subset"]) for e in self.entries if "subset" in e}),
            "scan_uid_count": len(self.scan_uids),
            "spacing_mm": (
                list(self.preprocessing_spacing_mm)
                if self.preprocessing_spacing_mm is not None
                else None
            ),
        }
        logger.info(
            f"CachedLUNA16Dataset: {pos} positives + {neg} negatives "
            f"= {len(self.entries)} total  (cache: {self.cache_dir})"
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        data  = np.load(self.cache_dir / entry["file"])

        patch  = data["image"].astype(np.float32)   # (3, H, W)
        mask   = data["mask"].astype(np.int64)       # (H, W)
        label  = int(data["label"])
        diam   = float(data["diameter"])

        patch_t = torch.from_numpy(patch)
        mask_t  = torch.from_numpy(mask)

        if self.augment and label == 1:
            patch_t, mask_t = LUNA16Dataset._augment(patch_t, mask_t)

        return {
            "image":       patch_t,
            "mask":        mask_t,
            "label":       torch.tensor(label),
            "diameter_mm": torch.tensor(diam),
            "uid":         entry["uid"],
        }


class LUNA16Dataset(Dataset):
    """
    PyTorch Dataset for LUNA16 lung nodule detection.

    **Prefer CachedLUNA16Dataset** for training — it is 100-1000× faster.
    Use this class only when the pre-extracted cache is not available yet, or
    for a quick one-off inference on a single scan.

    Directory structure expected:
        data_root/
          subset0/ ... subset9/   <- .mhd + .raw files
          annotations.csv         <- seriesuid, coordX, coordY, coordZ, diameter_mm
          candidates.csv          <- seriesuid, coordX, coordY, coordZ, class (0/1)
    """

    def __init__(
        self,
        data_root: str,
        subsets: list[int],
        patch_size: int = 64,
        negative_ratio: float = 1.0,
        augment: bool = True,
        use_candidates: bool = False,
        target_spacing_mm: tuple[float, float, float] = TARGET_SPACING_MM,
    ):
        self.data_root = Path(data_root)
        self.patch_size = patch_size
        self.augment = augment
        self.target_spacing_mm = tuple(float(v) for v in target_spacing_mm)

        # Parse annotations
        self.annotations: dict[str, list[dict]] = {}
        ann_path = self.data_root / "annotations.csv"
        if ann_path.exists():
            with open(ann_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    uid = row["seriesuid"]
                    if uid not in self.annotations:
                        self.annotations[uid] = []
                    self.annotations[uid].append({
                        "coord": np.array([
                            float(row["coordX"]),
                            float(row["coordY"]),
                            float(row["coordZ"]),
                        ]),
                        "diameter_mm": float(row["diameter_mm"]),
                    })

        # Find .mhd files for the specified subsets
        self.samples: list[dict] = []
        self.negatives: list[dict] = []

        for subset in subsets:
            subset_dir = self.data_root / f"subset{subset}"
            if not subset_dir.exists():
                logger.warning(f"Subset directory not found: {subset_dir}")
                continue
            for mhd_file in sorted(subset_dir.glob("*.mhd")):
                if not re.fullmatch(r"\d+(?:\.\d+)+", mhd_file.stem):
                    logger.warning("Skipping non-standard LUNA16 scan filename: %s", mhd_file)
                    continue
                uid = mhd_file.stem
                positives_for_uid = self.annotations.get(uid, [])
                for ann in positives_for_uid:
                    self.samples.append({
                        "mhd_path": str(mhd_file),
                        "uid": uid,
                        "coord_world": ann["coord"],
                        "diameter_mm": ann["diameter_mm"],
                        "label": 1,
                    })

                # Sample negative patches from the same scan
                n_neg = max(1, int(len(positives_for_uid) * negative_ratio))
                for _ in range(n_neg):
                    self.negatives.append({
                        "mhd_path": str(mhd_file),
                        "uid": uid,
                        "coord_world": None,
                        "diameter_mm": 0.0,
                        "label": 0,
                    })

        # Cap negatives to desired ratio
        random.shuffle(self.negatives)
        cap = int(len(self.samples) * negative_ratio)
        self.all_samples = self.samples + self.negatives[:cap]
        random.shuffle(self.all_samples)
        logger.info(
            f"LUNA16Dataset (slow/no-cache): {len(self.samples)} positives + "
            f"{min(cap, len(self.negatives))} negatives = {len(self.all_samples)} total"
        )

        # Per-worker LRU volume cache (max 20 volumes).
        # NOTE: this cache is NOT shared across DataLoader workers.
        # Use CachedLUNA16Dataset to avoid this limitation entirely.
        self._volume_cache: dict[str, tuple] = {}
        self._sampled_negative_centers: dict[str, set[tuple[int, int, int]]] = {}

    def _load_volume(
        self, mhd_path: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if mhd_path not in self._volume_cache:
            # Import here so SimpleITK is only needed when this slow path is used
            vol, origin, spacing, direction = load_mhd(
                mhd_path, self.target_spacing_mm
            )
            vol = clip_and_normalise(vol)
            if len(self._volume_cache) >= 20:
                oldest_key = next(iter(self._volume_cache))
                del self._volume_cache[oldest_key]
            self._volume_cache[mhd_path] = (vol, origin, spacing, direction)
        return self._volume_cache[mhd_path]

    def _random_lung_coord(
        self,
        volume: np.ndarray,
        origin: np.ndarray,
        spacing: np.ndarray,
        direction: np.ndarray,
        positive_annotations: list[dict],
        scan_uid: str,
    ) -> np.ndarray:
        """
        Rejection-sample a crop-safe non-air (z,y,x) voxel outside nodules.
        The HU threshold is not equivalent to a true lung segmentation mask.
        """
        # Draw Z/Y/X together from crop-safe non-air voxels; random X/Y values
        # are never grafted onto another Z slice.
        z_size, y_size, x_size = volume.shape
        half = self.patch_size // 2
        if z_size < 3 or y_size < self.patch_size or x_size < self.patch_size:
            raise RuntimeError(
                f"Volume shape {volume.shape} is too small for a {self.patch_size}px patch"
            )
        seen = self._sampled_negative_centers.setdefault(scan_uid, set())
        positive_regions = [
            (
                world_to_voxel(ann["coord"], origin, spacing, direction),
                ann["diameter_mm"],
            )
            for ann in positive_annotations
        ]
        # Inspect one 2D slice at a time so sampling a full CT does not create
        # a volume-sized coordinate array. Every candidate is a real crop-safe
        # non-air (z,y,x) voxel, sampled in randomized order.
        z_order = np.random.permutation(np.arange(1, z_size - 1))
        for cz_value in z_order:
            cz = int(cz_value)
            valid_xy = np.argwhere(
                volume[cz, half:y_size - half, half:x_size - half] > 0.036
            )
            for candidate_index in np.random.permutation(len(valid_xy)):
                cy0, cx0 = valid_xy[candidate_index]
                cy = int(cy0) + half
                cx = int(cx0) + half
                center_key = (cx, cy, cz)
                if center_key in seen:
                    continue
                vox = np.array([cx, cy, cz], dtype=np.int64)
                if any(
                    patch_overlaps_nodule(
                        vox, center, diameter, spacing, self.patch_size
                    )
                    for center, diameter in positive_regions
                ):
                    continue
                seen.add(center_key)
                return voxel_to_world(vox, origin, spacing, direction)

        raise RuntimeError(
            f"Unable to find a unique crop-safe, non-air, annotation-safe negative "
            f"center for scan {scan_uid}; exhausted all valid crop-safe voxels"
        )

    def __len__(self) -> int:
        return len(self.all_samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.all_samples[idx]
        volume, origin, spacing, direction = self._load_volume(sample["mhd_path"])

        if sample["label"] == 1:
            coord_world = sample["coord_world"]
        else:
            coord_world = self._random_lung_coord(
                volume,
                origin,
                spacing,
                direction,
                self.annotations.get(sample["uid"], []),
                sample["uid"],
            )

        center_vox = world_to_voxel(coord_world, origin, spacing, direction)
        patch = extract_patch_2_5d(volume, center_vox, self.patch_size)

        if patch is None:
            patch = np.zeros((3, self.patch_size, self.patch_size), dtype=np.float32)
            mask  = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
            return {
                "image":       torch.tensor(patch),
                "mask":        torch.tensor(mask).long(),
                "label":       torch.tensor(0),
                "diameter_mm": torch.tensor(0.0),
                "uid":         sample["uid"],
            }

        if sample["label"] == 1:
            mask = create_spherical_mask(
                self.patch_size, center_vox, sample["diameter_mm"], spacing
            )
        else:
            mask = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)

        patch_t = torch.tensor(patch)
        mask_t  = torch.tensor(mask).long()

        if self.augment and sample["label"] == 1:
            patch_t, mask_t = self._augment(patch_t, mask_t)

        return {
            "image":       patch_t,
            "mask":        mask_t,
            "label":       torch.tensor(sample["label"]),
            "diameter_mm": torch.tensor(sample["diameter_mm"]),
            "uid":         sample["uid"],
        }

    @staticmethod
    def _augment(
        patch: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Random flip, rotation, brightness jitter."""
        if random.random() > 0.5:
            patch = TF.hflip(patch)
            mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)
        if random.random() > 0.5:
            patch = TF.vflip(patch)
            mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)
        angle = random.choice([0, 90, 180, 270])
        if angle:
            patch = TF.rotate(patch, angle)
            mask = TF.rotate(mask.unsqueeze(0), angle).squeeze(0)
        # Brightness / contrast jitter on the middle slice only
        brt = random.uniform(0.8, 1.2)
        patch[1] = torch.clamp(patch[1] * brt, 0, 1)
        return patch, mask


def build_dataloaders(
    data_root: str | Path,
    fold: int = 0,
    batch_size: int = 8,
    num_workers: int = 2,
    patch_size: int = 64,
    cache_dir: Optional[str | Path] = None,
    train_subsets: Optional[list[int]] = None,
    validation_subset: Optional[int] = None,
    test_subset: Optional[int] = None,
    target_spacing_mm: tuple[float, float, float] = TARGET_SPACING_MM,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train/validation DataLoaders. Explicit scan-disjoint subset groups
    may be provided; otherwise ``fold`` selects validation and all other
    available subsets are used for training (legacy behavior).

    Args:
        data_root  : path to raw LUNA16 data (used when cache_dir is None)
        fold       : subset index used for validation
        batch_size : samples per batch
        num_workers: DataLoader worker processes (set to 0 on Windows)
        patch_size : voxel size of each patch (must match cache if cache_dir given)
        cache_dir  : path to pre-extracted patch cache from ``preprocess_cache.py``.
                     **Strongly recommended** — eliminates SSD I/O bottleneck.
                     When supplied, raw LUNA16 volumes are never read during training.
        train_subsets, validation_subset, test_subset: explicit disjoint subsets.
        target_spacing_mm: physical voxel spacing for raw-volume resampling.
    """
    root_arg = Path(data_root).expanduser()
    if root_arg.is_absolute():
        root_candidates = [root_arg]
    else:
        project_root = Path(__file__).resolve().parents[1]
        root_candidates = [Path.cwd() / root_arg, project_root / root_arg]

    # Prefer the candidate containing real subset directories. This lets the
    # same relative config work when launched from either the repo root or
    # backend/.
    data_root_path = None
    subset_uids: dict[int, set[str]] = {}
    for candidate in root_candidates:
        candidate = candidate.resolve()
        found: dict[int, set[str]] = {
            int(path.name.removeprefix("subset")): {
                mhd.stem for mhd in path.glob("*.mhd")
                if re.fullmatch(r"\d+(?:\.\d+)+", mhd.stem)
            }
            for path in candidate.glob("subset*")
            if path.is_dir() and path.name.removeprefix("subset").isdigit()
        }
        if sum(map(len, found.values())) > sum(map(len, subset_uids.values())):
            data_root_path = candidate
            subset_uids = found
        if len(found) >= 2:
            break

    if data_root_path is None:
        checked = ", ".join(str(path.resolve()) for path in root_candidates)
        raise FileNotFoundError(
            f"No LUNA16 subset directories found for data_root={data_root!r}. "
            f"Checked: {checked}"
        )

    all_subsets = sorted(subset_uids)
    if len(all_subsets) < 2:
        raise ValueError(
            f"At least 2 LUNA16 subsets are required under {data_root_path}. "
            f"Found: {all_subsets}"
        )

    explicit_split = any(
        value is not None for value in (train_subsets, validation_subset, test_subset)
    )
    if explicit_split:
        if train_subsets is None or validation_subset is None or not train_subsets:
            raise ValueError(
                "Explicit splits require non-empty train_subsets and validation_subset"
            )
        train_subsets = [int(s) for s in train_subsets]
        val_subset = [int(validation_subset)]
        test_subsets = [int(test_subset)] if test_subset is not None else []
    else:
        if fold not in all_subsets:
            raise ValueError(
                f"Validation fold {fold} is unavailable under {data_root_path}. "
                f"Available subsets: {all_subsets}"
            )
        val_subset = [int(fold)]
        train_subsets = [subset for subset in all_subsets if subset != fold]
        test_subsets = []

    groups = {
        "train": train_subsets,
        "validation": val_subset,
        "test": test_subsets,
    }
    for group_name, group_subsets in groups.items():
        if len(group_subsets) != len(set(group_subsets)):
            raise ValueError(f"Duplicate subset in {group_name}_subsets: {group_subsets}")
        missing = sorted(set(group_subsets) - set(all_subsets))
        if missing:
            raise ValueError(
                f"Requested {group_name} subset(s) {missing} are unavailable under "
                f"{data_root_path}; available subsets: {all_subsets}"
            )
    if not train_subsets:
        raise ValueError("At least one training subset is required")

    subset_groups = {
        name: set().union(*(subset_uids[s] for s in values)) if values else set()
        for name, values in groups.items()
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = subset_groups[left] & subset_groups[right]
        if overlap:
            raise ValueError(
                f"Scan UID leakage between {left} and {right}: {sorted(overlap)[:5]}"
            )

    split_info = {
        "train_subsets": list(train_subsets),
        "validation_subset": val_subset[0],
        "test_subset": test_subsets[0] if test_subsets else None,
        "train_scan_uids": sorted(subset_groups["train"]),
        "validation_scan_uids": sorted(subset_groups["validation"]),
        "test_scan_uids": sorted(subset_groups["test"]),
        "independent_test_available": bool(test_subsets),
    }

    logger.info(
        f"Using LUNA16 data at {data_root_path} | Available subsets: "
        f"{all_subsets} | Training: {train_subsets} | Validation: {val_subset}"
    )

    # ------------------------------------------------------------------
    # Choose dataset implementation
    # ------------------------------------------------------------------
    if cache_dir is not None:
        cache_path = Path(cache_dir).expanduser().resolve()
        logger.info(f"Using pre-extracted patch cache at {cache_path} (fast path)")
        from .preprocess_cache import validate_cache_manifest

        cache_entries = validate_cache_manifest(
            cache_path,
            expected_patch_size=patch_size,
            expected_subsets=set(train_subsets) | set(val_subset) | set(test_subsets),
            expected_uid_subsets={
                subset: subset_uids[subset]
                for subset in set(train_subsets) | set(val_subset) | set(test_subsets)
            },
        )
        train_ds = CachedLUNA16Dataset(
            cache_path, subsets=train_subsets, augment=True,
            expected_patch_size=patch_size,
        )
        val_ds = CachedLUNA16Dataset(
            cache_path, subsets=val_subset, augment=False,
            expected_patch_size=patch_size,
        )
        if test_subsets:
            test_ds = CachedLUNA16Dataset(
                cache_path, subsets=test_subsets, augment=False,
                expected_patch_size=patch_size,
            )
            if test_ds.scan_uids != subset_groups["test"]:
                raise ValueError("Test cache scan UIDs do not match the requested raw subset")
        if train_ds.scan_uids != subset_groups["train"]:
            raise ValueError("Training cache scan UIDs do not match requested raw subsets")
        if val_ds.scan_uids != subset_groups["validation"]:
            raise ValueError("Validation cache scan UIDs do not match the requested raw subset")
        cache_summary = {
            "total_patches": len(cache_entries),
            "positive_patches": sum(int(entry["label"]) == 1 for entry in cache_entries),
            "negative_patches": sum(int(entry["label"]) == 0 for entry in cache_entries),
            "patch_dimensions": [3, patch_size, patch_size],
            "channel_count": 3,
            "available_subsets": sorted({int(entry["subset"]) for entry in cache_entries}),
            "scan_uid_count": len({entry["uid"] for entry in cache_entries}),
            "spacing_mm": train_ds.cache_summary["spacing_mm"],
        }
        logger.info("Patch cache validation succeeded: %s", cache_summary)
        logger.info(
            "Cached split loaded: train subsets=%s patches=%d (+%d/- %d) scans=%d; "
            "validation subset=%s patches=%d (+%d/- %d) scans=%d; test subset=%s",
            train_subsets,
            train_ds.cache_summary["total_patches"],
            train_ds.cache_summary["positive_patches"],
            train_ds.cache_summary["negative_patches"],
            train_ds.cache_summary["scan_uid_count"],
            val_subset[0],
            val_ds.cache_summary["total_patches"],
            val_ds.cache_summary["positive_patches"],
            val_ds.cache_summary["negative_patches"],
            val_ds.cache_summary["scan_uid_count"],
            test_subsets,
        )
    else:
        logger.warning(
            "No cache_dir supplied — loading raw CT volumes at runtime. "
            "SSD will run at 100%%. Run preprocess_cache.py once to fix this."
        )
        train_ds = LUNA16Dataset(
            data_root_path, train_subsets, patch_size=patch_size, augment=True,
            target_spacing_mm=target_spacing_mm,
        )
        val_ds = LUNA16Dataset(
            data_root_path, val_subset, patch_size=patch_size, augment=False,
            target_spacing_mm=target_spacing_mm,
        )
        if test_subsets:
            test_ds = LUNA16Dataset(
                data_root_path, test_subsets, patch_size=patch_size, augment=False,
                target_spacing_mm=target_spacing_mm,
            )

    train_ds.split_info = split_info
    val_ds.split_info = split_info
    if not len(train_ds) or not len(val_ds):
        raise ValueError(
            f"Dataset is empty: training={len(train_ds)} from {train_subsets}, "
            f"validation={len(val_ds)} from {val_subset}. "
            "Check annotations.csv and that each subset contains matching .mhd files."
        )

    import platform
    # Raw-volume sampling uses an in-memory per-scan set to prevent repeated
    # negative centers. Keep it in one process; cached loading can be parallel.
    safe_workers = 0 if platform.system() == "Windows" or cache_dir is None else num_workers

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=safe_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(safe_workers > 0),   # avoids worker respawn overhead
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=safe_workers,
        pin_memory=True,
        persistent_workers=(safe_workers > 0),
    )
    return train_loader, val_loader
