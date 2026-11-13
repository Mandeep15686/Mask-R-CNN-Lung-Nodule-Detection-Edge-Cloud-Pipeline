"""
Composite loss functions for Mask R-CNN lung nodule segmentation.
Supports: BCE, Soft-Dice, Boundary, Focal, Tversky, Hausdorff-DT losses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np


# ---------------------------------------------------------------------------
# 1. Binary Cross-Entropy Loss
# ---------------------------------------------------------------------------
class BCELoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        pw = torch.tensor([pos_weight])
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, target.float())


# ---------------------------------------------------------------------------
# 2. Soft Dice Loss
# ---------------------------------------------------------------------------
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_p = torch.sigmoid(pred)
        target_f = target.float()
        intersection = (pred_p * target_f).sum(dim=(-2, -1))
        union = pred_p.sum(dim=(-2, -1)) + target_f.sum(dim=(-2, -1))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


# ---------------------------------------------------------------------------
# 3. Boundary Loss (distance-transform based)
# ---------------------------------------------------------------------------
class BoundaryLoss(nn.Module):
    """
    Boundary loss from Kervadec et al. (2019).
    Uses the signed distance map of the ground-truth boundary.
    L_boundary = sum_q [ D_G(q) * sigmoid(pred(q)) ]
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def _compute_distance_map(mask: np.ndarray) -> np.ndarray:
        """Compute signed distance transform for a binary mask."""
        if mask.max() == 0:
            return np.zeros_like(mask, dtype=np.float32)
        if mask.min() == 1:
            return np.zeros_like(mask, dtype=np.float32)
        dist_inside = distance_transform_edt(mask)
        dist_outside = distance_transform_edt(1 - mask)
        # Signed: positive inside, negative outside
        return (dist_inside - dist_outside).astype(np.float32)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: (N, H, W) logits; target: (N, H, W) binary
        pred_p = torch.sigmoid(pred)
        batch_size = target.shape[0]
        loss = torch.tensor(0.0, device=pred.device, requires_grad=False)
        dist_maps = []

        with torch.no_grad():
            for i in range(batch_size):
                mask_np = target[i].cpu().numpy().astype(np.uint8)
                dist = self._compute_distance_map(mask_np)
                dist_maps.append(dist)

        dist_tensor = torch.tensor(
            np.stack(dist_maps), dtype=torch.float32, device=pred.device
        )
        # Normalise distance map to [-1, 1]
        d_max = dist_tensor.abs().max().clamp(min=1e-6)
        dist_norm = dist_tensor / d_max
        # Boundary loss: minimize prediction on positive-distance boundary regions
        loss = (dist_norm * pred_p).mean()
        return loss


# ---------------------------------------------------------------------------
# 4. Focal Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_f = target.float()
        bce = F.binary_cross_entropy_with_logits(pred, target_f, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = self.alpha * target_f + (1 - self.alpha) * (1 - target_f)
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()


# ---------------------------------------------------------------------------
# 5. Tversky Loss
# ---------------------------------------------------------------------------
class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        """
        alpha: weight for FP penalties
        beta: weight for FN penalties (beta > alpha → penalise false negatives more)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_p = torch.sigmoid(pred)
        target_f = target.float()
        tp = (pred_p * target_f).sum(dim=(-2, -1))
        fp = (pred_p * (1 - target_f)).sum(dim=(-2, -1))
        fn = ((1 - pred_p) * target_f).sum(dim=(-2, -1))
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1.0 - tversky.mean()


# ---------------------------------------------------------------------------
# 6. Hausdorff Distance Loss (DT-based differentiable approximation)
# ---------------------------------------------------------------------------
class HausdorffDTLoss(nn.Module):
    """
    Differentiable 95th-percentile Hausdorff distance loss via distance transforms.
    From Karimi & Salcudean (2019).
    """
    def __init__(self, percentile: float = 95.0):
        super().__init__()
        self.percentile = percentile

    @staticmethod
    def _dt(mask_np: np.ndarray) -> np.ndarray:
        if mask_np.max() == 0:
            return np.zeros_like(mask_np, dtype=np.float32)
        return distance_transform_edt(1 - mask_np).astype(np.float32)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_bin = (torch.sigmoid(pred) > 0.5).float()
        batch_size = target.shape[0]
        total_loss = torch.tensor(0.0, device=pred.device)

        for i in range(batch_size):
            pred_np = pred_bin[i].detach().cpu().numpy().astype(np.uint8)
            tgt_np = target[i].cpu().numpy().astype(np.uint8)

            # Distance transforms
            dt_pred = torch.tensor(
                self._dt(pred_np), dtype=torch.float32, device=pred.device
            )
            dt_tgt = torch.tensor(
                self._dt(tgt_np), dtype=torch.float32, device=pred.device
            )

            pred_p = torch.sigmoid(pred[i])
            # Hausdorff: sum of DT * prediction error
            loss_i = (dt_tgt ** 2 * pred_p + dt_pred ** 2 * (1 - pred_p)).mean()
            total_loss = total_loss + loss_i

        return total_loss / max(batch_size, 1)


# ---------------------------------------------------------------------------
# COMPOSITE LOSS (the main research object)
# ---------------------------------------------------------------------------
class CompositeMaskLoss(nn.Module):
    """
    Configurable composite loss:
      L = alpha*BCE + beta*Dice + gamma*Boundary + delta*Focal + epsilon*HD95

    Research Usage:
      loss_fn = CompositeMaskLoss(alpha=0.3, beta=0.5, gamma=0.5, delta=0.0, epsilon=0.0)
    """

    def __init__(
        self,
        alpha: float = 1.0,   # BCE weight
        beta: float = 0.0,    # Dice weight
        gamma: float = 0.0,   # Boundary weight
        delta: float = 0.0,   # Focal weight
        epsilon: float = 0.0, # HD95 weight
        bce_pos_weight: float = 1.0,
        dice_smooth: float = 1e-6,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon

        self.bce = BCELoss(pos_weight=bce_pos_weight)
        self.dice = SoftDiceLoss(smooth=dice_smooth)
        self.boundary = BoundaryLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.hd = HausdorffDTLoss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pred:   (N, H, W) logits
            target: (N, H, W) binary masks

        Returns:
            dict with 'total' and individual component losses
        """
        components: dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=pred.device)

        if self.alpha > 0:
            l_bce = self.bce(pred, target)
            components["bce"] = l_bce
            total = total + self.alpha * l_bce

        if self.beta > 0:
            l_dice = self.dice(pred, target)
            components["dice"] = l_dice
            total = total + self.beta * l_dice

        if self.gamma > 0:
            l_boundary = self.boundary(pred, target)
            components["boundary"] = l_boundary
            total = total + self.gamma * l_boundary

        if self.delta > 0:
            l_focal = self.focal(pred, target)
            components["focal"] = l_focal
            total = total + self.delta * l_focal

        if self.epsilon > 0:
            l_hd = self.hd(pred, target)
            components["hd"] = l_hd
            total = total + self.epsilon * l_hd

        components["total"] = total
        return components

    @classmethod
    def from_config(cls, cfg: dict) -> "CompositeMaskLoss":
        return cls(
            alpha=cfg.get("alpha", 1.0),
            beta=cfg.get("beta", 0.0),
            gamma=cfg.get("gamma", 0.0),
            delta=cfg.get("delta", 0.0),
            epsilon=cfg.get("epsilon", 0.0),
        )

    def describe(self) -> str:
        parts = []
        names = {
            "alpha": "BCE",
            "beta": "Dice",
            "gamma": "Boundary",
            "delta": "Focal",
            "epsilon": "HD95",
        }
        for attr, name in names.items():
            w = getattr(self, attr)
            if w > 0:
                parts.append(f"{w:.2f}·{name}")
        return " + ".join(parts) if parts else "None"
