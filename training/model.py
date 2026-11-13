"""
Mask R-CNN wrapper for lung nodule detection + segmentation.
Injects custom composite mask loss into torchvision's MaskRCNN.
"""
import torch
import torch.nn as nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import roi_align
import torch.nn.functional as F
from typing import Optional
from .losses import CompositeMaskLoss


# ---------------------------------------------------------------------------
# Custom Mask Head that uses our composite loss
# ---------------------------------------------------------------------------

class CustomMaskRCNNHeads(MaskRCNNHeads):
    """Drop-in replacement for MaskRCNNHeads; identical forward pass."""
    pass


class LungNoduleMaskRCNN(nn.Module):
    """
    Wrapper around torchvision MaskRCNN with:
      1. Custom composite mask loss (BCE + Dice + Boundary + ...)
      2. Lung-nodule-tuned anchor sizes (3mm–30mm nodules)
      3. 2.5D input support (3-channel axial slices)
    """

    # Typical LUNA16 spacing: ~0.7mm/vox. Nodule diameters: 3–30mm → 4–43 vox.
    ANCHOR_SIZES = ((8,), (16,), (32,), (64,), (128,))
    ANCHOR_RATIOS = ((0.5, 1.0, 2.0),) * 5

    def __init__(
        self,
        num_classes: int = 2,          # background + nodule
        mask_loss_cfg: Optional[dict] = None,
        pretrained_backbone: bool = True,
        trainable_backbone_layers: int = 3,
        min_size: int = 64,
        max_size: int = 256,
        box_score_thresh: float = 0.05,
        box_nms_thresh: float = 0.5,
        box_detections_per_img: int = 100,
        mask_roi_pool_output_size: int = 14,
        hidden_dim: int = 256,
    ):
        super().__init__()

        if mask_loss_cfg is None:
            mask_loss_cfg = {"alpha": 1.0, "beta": 0.0, "gamma": 0.0}

        # Build FPN backbone (ResNet-50)
        backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights="DEFAULT" if pretrained_backbone else None,
            trainable_layers=trainable_backbone_layers,
        )

        # Modify first conv to accept 3-channel 2.5D input (default: RGB → still 3ch)
        # (No change needed; torchvision expects 3ch by default)

        # Anchor generator tuned for nodule sizes
        anchor_generator = AnchorGenerator(
            sizes=self.ANCHOR_SIZES,
            aspect_ratios=self.ANCHOR_RATIOS,
        )

        # Build Mask R-CNN
        self.model = MaskRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            min_size=min_size,
            max_size=max_size,
            box_score_thresh=box_score_thresh,
            box_nms_thresh=box_nms_thresh,
            box_detections_per_img=box_detections_per_img,
        )

        # Replace mask predictor for binary nodule masks
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_channels=in_features_mask,
            dim_reduced=hidden_dim,
            num_classes=num_classes,
        )

        # Our custom composite mask loss
        self.composite_mask_loss = CompositeMaskLoss.from_config(mask_loss_cfg)

        # Monkey-patch: replace the internal mask loss with our custom one
        self._patch_mask_loss()

    def _patch_mask_loss(self):
        """Replace torchvision's default BCE mask loss with our composite loss."""
        composite = self.composite_mask_loss

        def custom_maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
            """
            Custom drop-in for torchvision's maskrcnn_loss.
            Accepts the same signature as the original.
            """
            discretization_size = mask_logits.shape[-1]
            labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
            mask_targets = []
            for m, p, i in zip(gt_masks, proposals, mask_matched_idxs):
                if m.numel() == 0 or p.shape[0] == 0:
                    mask_targets.append(
                        torch.zeros(0, discretization_size, discretization_size, device=mask_logits.device)
                    )
                    continue
                # Resize GT masks to prediction size
                gt_m = m[i].to(dtype=torch.float32, device=mask_logits.device)
                # gt_m shape: (N, H, W) → need to resize via roi_align
                gt_m_resized = F.interpolate(
                    gt_m.unsqueeze(1),
                    size=(discretization_size, discretization_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                mask_targets.append(gt_m_resized)

            if len(mask_targets) == 0 or all(mt.numel() == 0 for mt in mask_targets):
                return mask_logits.sum() * 0

            mask_targets = torch.cat(mask_targets, dim=0)
            labels_all = torch.cat(labels, dim=0)

            # Only compute loss for positive labels (class 1 = nodule)
            pos_idx = torch.where(labels_all > 0)[0]
            if pos_idx.numel() == 0:
                return mask_logits.sum() * 0

            # Select the mask logit for the GT class
            pos_logits = mask_logits[pos_idx, labels_all[pos_idx]]  # (N_pos, H, W)
            pos_targets = mask_targets[pos_idx]                       # (N_pos, H, W)

            loss_dict = composite(pos_logits, pos_targets)
            return loss_dict["total"]

        # Inject into the roi_heads
        import types
        self.model.roi_heads.maskrcnn_loss = types.MethodType(
            lambda self_roi, *args: custom_maskrcnn_loss(*args),
            self.model.roi_heads,
        )

    def forward(self, images: list, targets: Optional[list] = None):
        """
        Training: images list[Tensor(C,H,W)], targets list[dict]
        Inference: images list[Tensor(C,H,W)]
        Returns losses dict (train) or list of prediction dicts (eval).
        """
        return self.model(images, targets)

    def get_loss_description(self) -> str:
        return self.composite_mask_loss.describe()

    @classmethod
    def from_config(cls, cfg: dict) -> "LungNoduleMaskRCNN":
        return cls(
            num_classes=cfg.get("num_classes", 2),
            mask_loss_cfg=cfg.get("mask_loss", {}),
            pretrained_backbone=cfg.get("pretrained_backbone", True),
            trainable_backbone_layers=cfg.get("trainable_backbone_layers", 3),
            min_size=cfg.get("min_size", 64),
            max_size=cfg.get("max_size", 256),
        )


# ---------------------------------------------------------------------------
# Lightweight edge model (early-layer feature extractor)
# ---------------------------------------------------------------------------

class EdgeFeatureExtractor(nn.Module):
    """
    Runs only the first two ResNet stages on the edge device.
    Output: intermediate feature maps to transmit to cloud.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision.models as tv_models
        resnet = tv_models.resnet50(weights="DEFAULT" if pretrained else None)
        # Keep: conv1, bn1, relu, maxpool, layer1, layer2
        self.edge_layers = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # C2: stride 4, 256 ch
            resnet.layer2,  # C3: stride 8, 512 ch
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.edge_layers(x)

    def feature_size(self, input_size: int = 64) -> int:
        dummy = torch.zeros(1, 3, input_size, input_size)
        with torch.no_grad():
            out = self.forward(dummy)
        return out.numel()


def build_model(config: dict) -> LungNoduleMaskRCNN:
    """Factory function used by trainer."""
    return LungNoduleMaskRCNN.from_config(config)
