"""
Early-layer feature extractor for split computing.
Runs ResNet-50 up to the configured split layer on the edge device.
"""
import torch
import torch.nn as nn
from typing import Literal

SplitLayer = Literal["layer1", "layer2", "layer3"]


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
        pretrained: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.split_layer = split_layer
        self.device = device

        import torchvision.models as tv
        resnet = tv.resnet50(weights="DEFAULT" if pretrained else None)

        # Build sequential up to the split
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1

        if split_layer in ("layer2", "layer3"):
            self.layer2 = resnet.layer2
        if split_layer == "layer3":
            self.layer3 = resnet.layer3

        self.to(device)
        self.eval()

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward x through layers up to the split point.
        Returns a dict of named feature tensors for transmission.
        """
        x = self.stem(x)
        x = self.layer1(x)
        features = {"layer1": x}

        if self.split_layer in ("layer2", "layer3"):
            x = self.layer2(x)
            features["layer2"] = x

        if self.split_layer == "layer3":
            x = self.layer3(x)
            features["layer3"] = x

        return {self.split_layer: features[self.split_layer]}

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
