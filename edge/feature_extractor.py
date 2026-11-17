"""
Early-layer feature extractor for split computing.
Runs ResNet-50 up to the configured split layer on the edge device.
"""
import torch
import torch.nn as nn
import hashlib
from pathlib import Path
from typing import Literal

SplitLayer = Literal["layer1", "layer2", "layer3"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EdgeFeatureExtractor(nn.Module):
    """
    Extracts intermediate feature maps from a ResNet-50 backbone.
    Only the layers up to `split_layer` run on the edge.

    Split-layer options (output channels × stride vs input):
      layer1 → 256ch,  stride ×4   (C2)
      layer2 → 512ch,  stride ×8   (C3)  ← default, best bandwidth/accuracy tradeoff
      layer3 → 1024ch, stride ×16  (C4)
    """

    def __init__(
        self,
        split_layer: SplitLayer = "layer2",
        pretrained: bool = False,
        device: torch.device = torch.device("cpu"),
        checkpoint_path: str | Path | None = None,
        image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.split_layer = split_layer
        self.device = device
        self.image_mean = [float(v) for v in image_mean]
        self.image_std = [float(v) for v in image_std]
        if len(self.image_mean) != 3 or len(self.image_std) != 3 or any(v <= 0 for v in self.image_std):
            raise ValueError("Image normalization must contain three means and positive standard deviations")

        import torchvision.models as tv
        resnet = tv.resnet50(weights="DEFAULT" if pretrained else None)

        self.checkpoint_sha256: str | None = None
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path).expanduser().resolve()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Edge model checkpoint not found: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict):
                raise ValueError("Checkpoint root must be a dictionary")
            state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
            prefix = "model.backbone.body."
            backbone_state = {
                key[len(prefix):]: value
                for key, value in state.items()
                if isinstance(key, str) and key.startswith(prefix)
            }
            if not backbone_state:
                raise ValueError("Checkpoint contains no model.backbone.body ResNet weights")
            required_prefixes = ["conv1.", "bn1.", "layer1."]
            if split_layer in ("layer2", "layer3"):
                required_prefixes.append("layer2.")
            if split_layer == "layer3":
                required_prefixes.append("layer3.")
            missing = [
                key for key in resnet.state_dict()
                if any(key.startswith(prefix) for prefix in required_prefixes)
                and not key.endswith("num_batches_tracked")
                and key not in backbone_state
            ]
            if missing:
                raise ValueError(f"Checkpoint is missing edge backbone weights: {missing[:5]}")
            resnet.load_state_dict(backbone_state, strict=False)
            self.checkpoint_sha256 = _sha256_file(checkpoint_path)

        # Build sequential up to the split
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1

        if split_layer in ("layer2", "layer3"):
            self.layer2 = resnet.layer2
        if split_layer == "layer3":
            self.layer3 = resnet.layer3

        self.register_buffer("_image_mean", torch.tensor(self.image_mean).view(1, 3, 1, 1))
        self.register_buffer("_image_std", torch.tensor(self.image_std).view(1, 3, 1, 1))

        self.to(device)
        self.eval()

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward x through layers up to the split point.
        Returns a dict of named feature tensors for transmission.
        """
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected input shape (N,3,H,W), got {tuple(x.shape)}")
        # Match torchvision GeneralizedRCNNTransform used by cloud Mask R-CNN.
        x = (x - self._image_mean.to(dtype=x.dtype)) / self._image_std.to(dtype=x.dtype)
        x = self.stem(x)
        x = self.layer1(x)
        features = {"layer1": x}

        if self.split_layer in ("layer2", "layer3"):
            x = self.layer2(x)
            features["layer2"] = x

        if self.split_layer == "layer3":
            x = self.layer3(x)
            features["layer3"] = x

        # Return ALL computed feature levels so the cloud can run the FPN.
        # The FPN needs every backbone level (C2…C5); returning only the final
        # split-layer tensor discards the lower-level maps required for FPN
        # reconstruction, which caused HTTP 501 on the cloud endpoint.
        return features

    def feature_map_size(self, input_size: int = 64) -> dict:
        dummy = torch.zeros(1, 3, input_size, input_size, device=self.device)
        feats = self.extract(dummy)
        info = {}
        for name, t in feats.items():
            info[name] = {
                "shape": list(t.shape),
                "numel": t.numel(),
                "size_kb_fp32": t.numel() * 4 / 1024,
            }
        return info
