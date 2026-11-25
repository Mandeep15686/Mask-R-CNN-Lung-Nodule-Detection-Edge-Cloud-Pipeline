"""
Feature map compression for edge→cloud transmission.
Methods: none | int8 | learned (auto-encoder bottleneck)
"""
import hashlib
import torch
import torch.nn as nn
from typing import Literal
import os
import sys
from pathlib import Path

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
        expected_backbone_sha256: str | None = None,
        checkpoint_path: str | None = None,
    ):
        self.method = method
        self.device = device
        self._autoencoder: _LearnedCodecBank | None = None
        self.codec_sha256: str | None = None
        if method == "learned":
            checkpoint = checkpoint_path or os.getenv("LEARNED_COMPRESSOR_CHECKPOINT")
            if not checkpoint or not os.path.isfile(checkpoint):
                raise ValueError(
                    "Learned compression requires a trained codec checkpoint set "
                    "with LEARNED_COMPRESSOR_CHECKPOINT"
                )
            project_root = Path(__file__).resolve().parents[1]
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            from checkpoint_io import load_checkpoint_weights_only

            state = load_checkpoint_weights_only(checkpoint, map_location=device)
            metadata = state.get("metadata", {}) if isinstance(state, dict) else {}
            model_state = state.get("codec_state") if isinstance(state, dict) else None
            if not isinstance(model_state, dict) or metadata.get("format") != "learned_feature_codec_v1":
                raise ValueError("Codec checkpoint is missing learned_feature_codec_v1 metadata/state")
            trained_backbone = metadata.get("backbone_checkpoint_sha256")
            if expected_backbone_sha256 and trained_backbone != expected_backbone_sha256:
                raise ValueError("Learned codec was trained for a different backbone checkpoint")
            self._autoencoder = _LearnedCodecBank().to(device)
            self._autoencoder.load_state_dict(model_state, strict=True)
            self._autoencoder.eval()
            self.codec_sha256 = _sha256_file(checkpoint)

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
                zero_point = (
                    max(-128, min(127, round(-128 - vmin / scale)))
                    if scale != 0 else 0
                )
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
                    k: self._autoencoder.encode(k, v.to(self.device)).cpu()
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
                result[k] = self._autoencoder.decode(k, v.to(self.device)).cpu()
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


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _LearnedBottleneck(nn.Module):
    """
    Lightweight convolutional auto-encoder bottleneck.
    Encoder (edge): 512ch → 64ch (8× compression).
    Decoder (cloud): 64ch → 512ch.
    """

    def __init__(self, in_channels: int, bottleneck: int):
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


class _LearnedCodecBank(nn.Module):
    """Independent learned 1x1 codecs for the three ResNet split levels."""

    CHANNELS = {"layer1": 256, "layer2": 512, "layer3": 1024}
    LATENT_CHANNELS = {"layer1": 32, "layer2": 64, "layer3": 128}

    def __init__(self):
        super().__init__()
        self.codecs = nn.ModuleDict({
            name: _LearnedBottleneck(channels, self.LATENT_CHANNELS[name])
            for name, channels in self.CHANNELS.items()
        })

    def encode(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if name not in self.codecs or value.shape[1] != self.CHANNELS[name]:
            raise ValueError(f"Unexpected feature level or channel count for {name}")
        return self.codecs[name].encode(value)

    def decode(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if name not in self.codecs or value.shape[1] != self.LATENT_CHANNELS[name]:
            raise ValueError(f"Unexpected learned latent level or channel count for {name}")
        return self.codecs[name].decode(value)
