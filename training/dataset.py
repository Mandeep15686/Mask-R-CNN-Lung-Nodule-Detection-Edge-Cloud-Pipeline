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


def load_mhd(path: str):
    """
    Load MetaImage CT volume.
    Returns volume, origin, spacing, direction.
    """
    itk_img = sitk.ReadImage(str(path))

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
    Extract a 2.5D patch: 3 consecutive axial slices (z-1, z, z+1).
    Returns: (3, patch_size, patch_size) float32 or None if OOB.
    """
    cz, cy, cx = int(center_vox[2]), int(center_vox[1]), int(center_vox[0])
    half = patch_size // 2
    Z, Y, X = volume.shape

    if (
        cz < 1 or cz >= Z - 1
        or cy < half or cy >= Y - half
        or cx < half or cx >= X - half
    ):
        return None

    slices = []
    for dz in (-1, 0, 1):
        s = volume[cz + dz, cy - half: cy + half, cx - half: cx + half]
        if s.shape != (patch_size, patch_size):
            return None
        slices.append(s)

    return np.stack(slices, axis=0)  # (3, H, W)


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
        pos = sum(1 for e in self.entries if e["label"] == 1)
        neg = len(self.entries) - pos
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
    ):
        self.data_root = Path(data_root)
        self.patch_size = patch_size
        self.augment = augment

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

    def _load_volume(
        self, mhd_path: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if mhd_path not in self._volume_cache:
            # Import here so SimpleITK is only needed when this slow path is used
            vol, origin, spacing, direction = load_mhd(mhd_path)
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
    ) -> np.ndarray:
        """
        Sample a random voxel from the volume interior.
        NOTE: this is NOT a true lung mask — it avoids background air voxels
        (normalised HU < 0.036 ≈ below -950 HU) and volume edges only.
        For a proper lung mask, run a lung segmentation algorithm first.
        """
        # Avoid pure air (background) voxels and patches near annotated nodules.
        interior = np.where(volume[1:-1] > 0.036)   # skip first/last slice
        positive_regions = [
            (
                world_to_voxel(ann["coord"], origin, spacing, direction),
                ann["diameter_mm"],
            )
            for ann in positive_annotations
        ]
        half = self.patch_size // 2
        for _ in range(128):
            if interior[0].size == 0:
                cz = volume.shape[0] // 2
                cy = random.randint(half, volume.shape[1] - half - 1)
                cx = random.randint(half, volume.shape[2] - half - 1)
            else:
                # Pick a non-air voxel, then jitter in XY.
                pick = random.randint(0, interior[0].size - 1)
                cz = int(interior[0][pick]) + 1
                cy = random.randint(half, volume.shape[1] - half - 1)
                cx = random.randint(half, volume.shape[2] - half - 1)

            vox = np.array([cx, cy, cz], dtype=np.float64)
            if any(
                patch_overlaps_nodule(vox, center, diameter, spacing, self.patch_size)
                for center, diameter in positive_regions
            ):
                continue
            return voxel_to_world(vox, origin, spacing, direction)

        raise RuntimeError(
            "Unable to sample an annotation-safe negative patch after 128 attempts"
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
) -> tuple[DataLoader, DataLoader]:
    """
    Build train/val DataLoaders using the LUNA16 subsets available on disk.
    The requested fold is used for validation; all other subsets are training.

    Args:
        data_root  : path to raw LUNA16 data (used when cache_dir is None)
        fold       : subset index used for validation
        batch_size : samples per batch
        num_workers: DataLoader worker processes (set to 0 on Windows)
        patch_size : voxel size of each patch (must match cache if cache_dir given)
        cache_dir  : path to pre-extracted patch cache from ``preprocess_cache.py``.
                     **Strongly recommended** — eliminates SSD I/O bottleneck.
                     When supplied, raw LUNA16 volumes are never read during training.
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
    all_subsets = []
    for candidate in root_candidates:
        candidate = candidate.resolve()
        subsets = sorted(
            int(path.name.removeprefix("subset"))
            for path in candidate.glob("subset*")
            if path.is_dir() and path.name.removeprefix("subset").isdigit()
        )
        if len(subsets) > len(all_subsets):
            data_root_path = candidate
            all_subsets = subsets
        if len(subsets) >= 2:
            break

    if data_root_path is None:
        checked = ", ".join(str(path.resolve()) for path in root_candidates)
        raise FileNotFoundError(
            f"No LUNA16 subset directories found for data_root={data_root!r}. "
            f"Checked: {checked}"
        )

    if len(all_subsets) < 2:
        raise ValueError(
            f"At least 2 LUNA16 subsets are required under {data_root_path}. "
            f"Found: {all_subsets}"
        )
    if fold not in all_subsets:
        raise ValueError(
            f"Validation fold {fold} is unavailable under {data_root_path}. "
            f"Available subsets: {all_subsets}"
        )

    val_subset = [fold]
    train_subsets = [subset for subset in all_subsets if subset != fold]

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
        train_ds = CachedLUNA16Dataset(cache_path, subsets=train_subsets, augment=True)
        val_ds   = CachedLUNA16Dataset(cache_path, subsets=val_subset,    augment=False)
    else:
        logger.warning(
            "No cache_dir supplied — loading raw CT volumes at runtime. "
            "SSD will run at 100%%. Run preprocess_cache.py once to fix this."
        )
        train_ds = LUNA16Dataset(
            data_root_path, train_subsets, patch_size=patch_size, augment=True
        )
        val_ds = LUNA16Dataset(
            data_root_path, val_subset, patch_size=patch_size, augment=False
        )

    if not len(train_ds) or not len(val_ds):
        raise ValueError(
            f"Dataset is empty: training={len(train_ds)} from {train_subsets}, "
            f"validation={len(val_ds)} from {val_subset}. "
            "Check annotations.csv and that each subset contains matching .mhd files."
        )

    import platform
    safe_workers = 0 if platform.system() == "Windows" else num_workers

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
