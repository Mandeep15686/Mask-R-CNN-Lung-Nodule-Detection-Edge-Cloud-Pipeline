"""CT patch preprocessing on the edge device."""
import io
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

HU_MIN, HU_MAX = -1000, 400


class CTPreprocessor:
    def __init__(self, patch_size: int = 64):
        self.patch_size = patch_size

    def preprocess_bytes(self, image_bytes: bytes, fmt: str = "png") -> torch.Tensor:
        """
        Convert raw CT patch bytes → (3, H, W) float32 tensor in [0,1].
        Input can be PNG/JPEG grey CT or 3-channel 2.5D patch.
        """
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode == "L":
            # Single-channel grey → replicate to 3 channels (2.5D approximation)
            arr = np.array(img, dtype=np.float32)
            arr = np.clip(arr, 0, 255) / 255.0
            arr3 = np.stack([arr, arr, arr], axis=0)   # (3, H, W)
        else:
            img = img.convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            arr3 = arr.transpose(2, 0, 1)             # (3, H, W)

        t = torch.tensor(arr3)
        if t.shape[-2:] != (self.patch_size, self.patch_size):
            t = TF.resize(t, [self.patch_size, self.patch_size])
        return t

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
