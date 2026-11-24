"""Preprocess exported 2.5D RGB patches on the edge device.

PNG/JPEG data is already windowed and quantized for model input. It is not
raw DICOM data and cannot be reconstructed into Hounsfield units here.
"""
import io
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

class CTPreprocessor:
    def __init__(self, patch_size: int = 64):
        self.patch_size = patch_size

    def preprocess_bytes(self, image_bytes: bytes, fmt: str = "png") -> torch.Tensor:
        """
        Decode a model-ready RGB 2.5D patch as (3, H, W) in [0,1].

        Grayscale PNGs and resized images are rejected: repeating or resizing
        channels would silently create a different model input distribution.
        PNG/JPEG pixel values are not interpreted as Hounsfield units.
        """
        img = Image.open(io.BytesIO(image_bytes))
        if img.format not in {"PNG", "JPEG"}:
            raise ValueError(f"Unsupported image format {img.format!r}; use PNG or JPEG")
        if img.mode != "RGB":
            raise ValueError(
                f"Expected an RGB 2.5D patch (three adjacent slices), got mode {img.mode!r}"
            )
        if img.size != (self.patch_size, self.patch_size):
            raise ValueError(
                f"Expected {self.patch_size}x{self.patch_size} RGB patch, got "
                f"{img.width}x{img.height}; input is not resized"
            )
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr3 = np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32)
        return torch.from_numpy(arr3)

    def preprocess_numpy(self, patch_np: np.ndarray) -> torch.Tensor:
        """
        Convert (3, H, W) or (H, W) numpy float32 array → tensor.
        """
        if patch_np.ndim == 2:
            patch_np = np.stack([patch_np] * 3, axis=0)
        t = torch.tensor(patch_np, dtype=torch.float32)
        if t.shape[-2:] != (self.patch_size, self.patch_size):
            t = TF.resize(t, [self.patch_size, self.patch_size])
        return t
