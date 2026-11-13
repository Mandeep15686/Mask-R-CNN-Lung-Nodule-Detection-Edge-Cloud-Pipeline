"""
LUNA16 Dataset loader for Mask R-CNN training.
Reads .mhd/.raw CT volumes, extracts 2.5D patches centred on annotated nodules.
"""
import os
import csv
import logging
import random
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


def load_mhd(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a .mhd CT volume using SimpleITK.
    Returns: (volume_HU, origin, spacing)  all in world coordinates.
    volume_HU shape: (Z, Y, X)
    """
    itk_img = sitk.ReadImage(str(path))
    volume = sitk.GetArrayFromImage(itk_img).astype(np.float32)   # (Z, Y, X)
    origin = np.array(itk_img.GetOrigin())    # (X, Y, Z)
    spacing = np.array(itk_img.GetSpacing())  # (X, Y, Z)
    return volume, origin, spacing


def world_to_voxel(coord_world: np.ndarray, origin: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """Convert world (mm) coordinate to voxel index."""
    return np.round((coord_world - origin) / spacing).astype(int)


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


class LUNA16Dataset(Dataset):
    """
    PyTorch Dataset for LUNA16 lung nodule detection.

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
            f"Dataset: {len(self.samples)} positives + "
            f"{min(cap, len(self.negatives))} negatives = {len(self.all_samples)} total"
        )

        # Cache loaded volumes to avoid repeated I/O
        self._volume_cache: dict[str, tuple] = {}

    def _load_volume(self, mhd_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if mhd_path not in self._volume_cache:
            vol, origin, spacing = load_mhd(mhd_path)
            vol = clip_and_normalise(vol)
            # Keep cache small
            if len(self._volume_cache) > 20:
                oldest_key = next(iter(self._volume_cache))
                del self._volume_cache[oldest_key]
            self._volume_cache[mhd_path] = (vol, origin, spacing)
        return self._volume_cache[mhd_path]

    def _random_lung_coord(
        self, volume: np.ndarray, origin: np.ndarray, spacing: np.ndarray
    ) -> np.ndarray:
        """Sample a random voxel inside the lung mask (thresholded)."""
        lung_mask = (volume > 0.1) & (volume < 0.9)
        z_range = np.where(lung_mask.any(axis=(1, 2)))[0]
        if len(z_range) == 0:
            cz = volume.shape[0] // 2
        else:
            cz = int(random.choice(z_range))
        half = self.patch_size // 2
        cy = random.randint(half, volume.shape[1] - half - 1)
        cx = random.randint(half, volume.shape[2] - half - 1)
        # Convert voxel back to world
        vox = np.array([cx, cy, cz], dtype=float)
        return vox * spacing + origin

    def __len__(self) -> int:
        return len(self.all_samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.all_samples[idx]
        volume, origin, spacing = self._load_volume(sample["mhd_path"])

        if sample["label"] == 1:
            coord_world = sample["coord_world"]
        else:
            coord_world = self._random_lung_coord(volume, origin, spacing)

        center_vox = world_to_voxel(coord_world, origin, spacing)
        patch = extract_patch_2_5d(volume, center_vox, self.patch_size)

        if patch is None:
            # Fallback: return empty patch
            patch = np.zeros((3, self.patch_size, self.patch_size), dtype=np.float32)
            mask = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
            return {
                "image": torch.tensor(patch),
                "mask": torch.tensor(mask).long(),
                "label": torch.tensor(0),
                "diameter_mm": torch.tensor(0.0),
                "uid": sample["uid"],
            }

        if sample["label"] == 1:
            mask = create_spherical_mask(
                self.patch_size, center_vox, sample["diameter_mm"], spacing
            )
        else:
            mask = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)

        patch_t = torch.tensor(patch)   # (3, H, W)
        mask_t = torch.tensor(mask).long()

        # Augmentation
        if self.augment and sample["label"] == 1:
            patch_t, mask_t = self._augment(patch_t, mask_t)

        return {
            "image": patch_t,
            "mask": mask_t,
            "label": torch.tensor(sample["label"]),
            "diameter_mm": torch.tensor(sample["diameter_mm"]),
            "uid": sample["uid"],
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
) -> tuple[DataLoader, DataLoader]:
    """
    Build train/val DataLoaders using the subsets available under data_root.
    The requested fold is used for validation; all other discovered subsets
    are used for training.
    """
    data_root = Path(data_root)

    # Automatically discover available LUNA16 subsets
    all_subsets = sorted(
        int(p.name.replace("subset", ""))
        for p in data_root.glob("subset*")
        if p.is_dir() and p.name.replace("subset", "").isdigit()
    )

    if len(all_subsets) < 2:
        raise ValueError(
            f"At least 2 subsets are required. Found: {all_subsets}"
        )

    if fold not in all_subsets:
        raise ValueError(
            f"Validation fold {fold} is unavailable. "
            f"Available subsets: {all_subsets}"
        )

    val_subset = [fold]
    train_subsets = [s for s in all_subsets if s != fold]

    logger.info(
        f"Available subsets: {all_subsets} | "
        f"Training: {train_subsets} | Validation: {val_subset}"
    )

    train_ds = LUNA16Dataset(
        data_root, train_subsets, patch_size=patch_size, augment=True
    )
    val_ds = LUNA16Dataset(
        data_root, val_subset, patch_size=patch_size, augment=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
