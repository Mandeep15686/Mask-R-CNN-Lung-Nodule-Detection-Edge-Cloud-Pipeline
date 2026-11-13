"""
Feature map compression for edge→cloud transmission.
Methods: none | int8 | learned (auto-encoder bottleneck)
"""
import torch
import torch.nn as nn
from typing import Literal

CompressMethod = Literal["none", "int8", "learned"]


class FeatureCompressor:
    """
    Compress feature tensors before network transmission.

    CN Research: Measuring compression_ratio vs Dice degradation
    provides the Rate-Distortion curve for the edge-cloud pipeline.
    """

    def __init__(
        self,
        method: CompressMethod = "int8",
        device: torch.device = torch.device("cpu"),
    ):
        self.method = method
        self.device = device
        self._autoencoder: nn.Module | None = None
        if method == "learned":
            self._autoencoder = _LearnedBottleneck().to(device)

    def compress(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compress a dict of feature tensors. Returns compressed tensors."""
        if self.method == "none":
            return {k: v.cpu() for k, v in features.items()}

        elif self.method == "int8":
            compressed = {}
            for k, v in features.items():
                # Quantise to INT8 (scale + zero-point per tensor)
                vmin, vmax = v.min().item(), v.max().item()
                scale = (vmax - vmin) / 255.0 if vmax != vmin else 1.0
                zero_point = -round(vmin / scale) if scale != 0 else 0
                v_int8 = torch.quantize_per_tensor(
                    v.cpu().float(), scale=scale, zero_point=zero_point,
                    dtype=torch.qint8
                )
                # Store as dict with metadata for de-quantisation
                compressed[k] = v_int8
            return compressed

        elif self.method == "learned":
            if self._autoencoder is None:
                return {k: v.cpu() for k, v in features.items()}
            with torch.no_grad():
                return {
                    k: self._autoencoder.encode(v.to(self.device)).cpu()
                    for k, v in features.items()
                }

        return {k: v.cpu() for k, v in features.items()}

    def decompress(self, compressed: dict) -> dict[str, torch.Tensor]:
        """Decompress received tensors back to float32."""
        result = {}
        for k, v in compressed.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.qint8:
                result[k] = v.dequantize().float()
            elif self.method == "learned" and self._autoencoder is not None:
                result[k] = self._autoencoder.decode(v.to(self.device)).cpu()
            else:
                result[k] = v.float() if isinstance(v, torch.Tensor) else v
        return result

    def compression_ratio(self, original: dict, compressed: dict) -> float:
        """Compute bytes_original / bytes_compressed."""
        def _bytes(d: dict) -> int:
            total = 0
            for v in d.values():
                if isinstance(v, torch.Tensor):
                    total += v.numel() * v.element_size()
            return max(total, 1)
        return _bytes(original) / _bytes(compressed)


class _LearnedBottleneck(nn.Module):
    """
    Lightweight convolutional auto-encoder bottleneck.
    Encoder (edge): 512ch → 64ch (8× compression).
    Decoder (cloud): 64ch → 512ch.
    """

    def __init__(self, in_channels: int = 512, bottleneck: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, bottleneck, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(bottleneck, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, in_channels, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
