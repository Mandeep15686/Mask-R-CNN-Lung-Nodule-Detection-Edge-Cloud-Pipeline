"""
Server-side CT preprocessing utilities.
Used by the inference engine to prepare uploaded images for Mask R-CNN.
"""
import io
import logging
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)

HU_MIN  = -1000.0
HU_MAX  =  400.0
PATCH_SIZE_DEFAULT = 64


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def normalise_hu(image_np: np.ndarray) -> np.ndarray:
    """
    Clip and normalise a greyscale CT image to [0, 1].
    Assumes pixel values are in Hounsfield Units or a similar [0, 255] range.
    """
    if image_np.max() > 1.0:
        # Treat as HU-like — remap from pixel range to [0,1]
        v_min = float(image_np.min())
        v_max = float(image_np.max())
        if v_max - v_min < 1e-6:
            return np.zeros_like(image_np, dtype=np.float32)
        return ((image_np.astype(np.float32) - v_min) / (v_max - v_min))
    return image_np.astype(np.float32)


def apply_lung_window(image_np: np.ndarray) -> np.ndarray:
    """
    Apply standard lung window [-1000, 400 HU] and normalise to [0, 1].
    Works even if input is already normalised (passthrough).
    """
    if image_np.max() <= 1.0:
        return image_np.astype(np.float32)
    clipped = np.clip(image_np, HU_MIN, HU_MAX)
    return ((clipped - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def to_three_channel(image_np: np.ndarray) -> np.ndarray:
    """
    Ensure a 2D or 3-channel array → (3, H, W) float32.
    - Greyscale (H, W)         → replicated 3-channel
    - RGB (H, W, 3)            → transposed to (3, H, W)
    - 3-channel (3, H, W)      → returned as-is
    """
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=0)
    elif image_np.ndim == 3 and image_np.shape[-1] == 3:
        image_np = image_np.transpose(2, 0, 1)
    elif image_np.ndim == 3 and image_np.shape[0] != 3:
        image_np = np.stack([image_np[0]] * 3, axis=0)
    return image_np.astype(np.float32)


def resize_patch(image_np: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resize (3, H, W) → (3, target_size, target_size).
    Uses Lanczos resampling via PIL.
    """
    c, h, w = image_np.shape
    if h == target_size and w == target_size:
        return image_np
    slices = []
    for i in range(c):
        pil_slice = Image.fromarray(
            (image_np[i] * 255).clip(0, 255).astype(np.uint8)
        )
        resized = pil_slice.resize((target_size, target_size), Image.LANCZOS)
        slices.append(np.array(resized, dtype=np.float32) / 255.0)
    return np.stack(slices, axis=0)


def preprocess_uploaded_image(
    image_bytes: bytes,
    patch_size: int = PATCH_SIZE_DEFAULT,
    enhance_contrast: bool = True,
) -> np.ndarray:
    """
    Full preprocessing pipeline for an uploaded CT patch.

    1. Decode PNG/JPEG bytes
    2. Convert to greyscale or 3-channel float
    3. Normalise pixel values
    4. Optionally apply CLAHE-like contrast enhancement
    5. Resize to patch_size × patch_size
    6. Return (3, H, W) float32 numpy array

    Args:
        image_bytes: raw PNG or JPEG bytes
        patch_size:  spatial dimension for the model (default 64)
        enhance_contrast: apply contrast-limited enhancement

    Returns:
        (3, patch_size, patch_size) float32 ndarray in [0, 1]
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"Cannot decode image: {e}")

    # Convert to RGB always (handles RGBA, P, L modes)
    img_rgb = img.convert("RGB")

    if enhance_contrast:
        enhancer = ImageEnhance.Contrast(img_rgb)
        img_rgb  = enhancer.enhance(1.5)

    arr = np.array(img_rgb, dtype=np.float32) / 255.0   # (H, W, 3) in [0, 1]
    arr = to_three_channel(arr)                           # (3, H, W)
    arr = resize_patch(arr, patch_size)                   # (3, ps, ps)

    return arr


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def box_from_mask(mask: np.ndarray) -> Optional[list[float]]:
    """
    Compute axis-aligned bounding box from a binary 2D mask.
    Returns [x1, y1, x2, y2] or None if mask is empty.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [float(cmin), float(rmin), float(cmax + 1), float(rmax + 1)]


def estimate_diameter_mm(
    box: list[float],
    spacing_mm: float = 0.7,
) -> float:
    """
    Estimate nodule diameter from a bounding box and voxel spacing.
    Uses the geometric mean of width and height.
    """
    w = box[2] - box[0]
    h = box[3] - box[1]
    return float(((w * h) ** 0.5) * spacing_mm)


def compute_iou_2d(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
