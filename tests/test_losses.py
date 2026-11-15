"""
Unit tests for all loss function implementations.
Run: pytest tests/test_losses.py -v
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
import numpy as np
from training.losses import (
    BCELoss, SoftDiceLoss, BoundaryLoss, FocalLoss,
    TverskyLoss, HausdorffDTLoss, CompositeMaskLoss,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def perfect_pred():
    """Prediction = ground truth (all correct)."""
    target = torch.zeros(2, 32, 32)
    target[:, 10:20, 10:20] = 1.0
    # Perfect logits: very high where GT=1, very low where GT=0
    logits = target.clone() * 20.0 - 10.0
    return logits, target


@pytest.fixture
def zero_pred():
    """All-zero prediction (no nodule detected)."""
    target = torch.zeros(2, 32, 32)
    target[:, 10:20, 10:20] = 1.0
    logits = torch.full_like(target, -10.0)  # sigmoid → 0
    return logits, target


@pytest.fixture
def random_pred():
    """Random prediction and binary target."""
    torch.manual_seed(42)
    target = (torch.rand(4, 32, 32) > 0.7).float()
    logits = torch.randn(4, 32, 32)
    return logits, target


# ---------------------------------------------------------------------------
# BCELoss tests
# ---------------------------------------------------------------------------
class TestBCELoss:
    def test_perfect_prediction_low_loss(self, perfect_pred):
        logits, target = perfect_pred
        loss = BCELoss()(logits, target)
        assert loss.item() < 0.01, f"Perfect pred should have ~0 loss, got {loss.item()}"

    def test_zero_prediction_nonzero_loss(self, zero_pred):
        logits, target = zero_pred
        loss = BCELoss()(logits, target)
        assert loss.item() > 0.5, f"All-zero pred should have high loss, got {loss.item()}"

    def test_output_is_scalar(self, random_pred):
        logits, target = random_pred
        loss = BCELoss()(logits, target)
        assert loss.shape == torch.Size([]), "Loss must be scalar"

    def test_gradient_flows(self, random_pred):
        logits, target = random_pred
        logits = logits.requires_grad_(True)
        loss = BCELoss()(logits, target)
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_pos_weight_increases_loss_on_fn(self):
        # Pred = 0, GT = 1 (false negative)
        logits = torch.tensor([[[-10.0, -10.0]]])
        target = torch.tensor([[[1.0, 1.0]]])
        loss_w1 = BCELoss(pos_weight=1.0)(logits, target)
        loss_w5 = BCELoss(pos_weight=5.0)(logits, target)
        assert loss_w5.item() > loss_w1.item()


# ---------------------------------------------------------------------------
# SoftDiceLoss tests
# ---------------------------------------------------------------------------
class TestSoftDiceLoss:
    def test_perfect_prediction_zero_loss(self, perfect_pred):
        logits, target = perfect_pred
        loss = SoftDiceLoss()(logits, target)
        assert loss.item() < 0.001, f"Perfect pred: Dice loss should be ~0, got {loss.item()}"

    def test_range_zero_to_one(self, random_pred):
        logits, target = random_pred
        loss = SoftDiceLoss()(logits, target)
        assert 0.0 <= loss.item() <= 1.0, f"Dice loss must be in [0,1], got {loss.item()}"

    def test_all_zeros_pred_loss_one(self):
        """All-zero prediction on a non-empty target → Dice = 0 → loss = 1."""
        logits = torch.full((1, 32, 32), -10.0)
        target = torch.zeros(1, 32, 32)
        target[:, 10:20, 10:20] = 1.0
        loss = SoftDiceLoss()(logits, target)
        assert loss.item() > 0.9, f"All-zero pred: loss should be ~1, got {loss.item()}"

    def test_gradient_flows(self, random_pred):
        logits, target = random_pred
        logits = logits.requires_grad_(True)
        loss = SoftDiceLoss()(logits, target)
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_empty_target(self):
        """Empty GT mask (no nodule): dice loss should be low if pred is also zero."""
        logits = torch.full((1, 16, 16), -10.0)
        target = torch.zeros(1, 16, 16)
        loss = SoftDiceLoss()(logits, target)
        assert loss.item() < 0.01


# ---------------------------------------------------------------------------
# BoundaryLoss tests
# ---------------------------------------------------------------------------
class TestBoundaryLoss:
    def test_output_is_scalar(self, random_pred):
        logits, target = random_pred
        loss = BoundaryLoss()(logits, target)
        assert loss.shape == torch.Size([])

    def test_finite_output(self, random_pred):
        logits, target = random_pred
        loss = BoundaryLoss()(logits, target)
        assert torch.isfinite(loss), f"Boundary loss must be finite, got {loss.item()}"

    def test_all_zero_target(self):
        """All-zero target (no nodule): boundary loss should not error."""
        logits = torch.randn(2, 16, 16)
        target = torch.zeros(2, 16, 16)
        loss = BoundaryLoss()(logits, target)
        assert torch.isfinite(loss)

    def test_gradient_flows(self, random_pred):
        logits, target = random_pred
        logits = logits.clone().requires_grad_(True)
        loss = BoundaryLoss()(logits, target)
        loss.backward()
        assert logits.grad is not None


# ---------------------------------------------------------------------------
# FocalLoss tests
# ---------------------------------------------------------------------------
class TestFocalLoss:
    def test_focal_less_than_bce_on_easy_negatives(self):
        """Focal loss should down-weight easy (confident negative) predictions."""
        logits = torch.full((1, 16, 16), -5.0)   # All confidently background
        target = torch.zeros(1, 16, 16)
        bce_loss   = BCELoss()(logits, target).item()
        focal_loss = FocalLoss(alpha=0.5, gamma=2.0)(logits, target).item()
        assert focal_loss <= bce_loss + 1e-5, (
            f"Focal ({focal_loss:.5f}) should be <= BCE ({bce_loss:.5f}) on easy negatives"
        )

    def test_range(self, random_pred):
        logits, target = random_pred
        loss = FocalLoss()(logits, target)
        assert loss.item() >= 0.0
        assert torch.isfinite(loss)

    def test_gradient_flows(self, random_pred):
        logits, target = random_pred
        logits = logits.requires_grad_(True)
        FocalLoss()(logits, target).backward()
        assert logits.grad is not None


# ---------------------------------------------------------------------------
# TverskyLoss tests
# ---------------------------------------------------------------------------
class TestTverskyLoss:
    def test_symmetric_at_half(self, random_pred):
        """At alpha=beta=0.5, Tversky == Dice."""
        logits, target = random_pred
        dice   = SoftDiceLoss()(logits, target).item()
        tversky = TverskyLoss(alpha=0.5, beta=0.5)(logits, target).item()
        assert abs(dice - tversky) < 0.01, (
            f"Tversky(0.5,0.5) should ≈ Dice. Got dice={dice:.4f} tversky={tversky:.4f}"
        )

    def test_fn_penalisation(self, zero_pred):
        """Higher beta (penalise FN more) → higher loss on all-zero pred."""
        logits, target = zero_pred
        t_low  = TverskyLoss(alpha=0.3, beta=0.7)(logits, target).item()
        t_high = TverskyLoss(alpha=0.1, beta=0.9)(logits, target).item()
        # Both high, but high beta should produce ≥ low beta loss
        assert t_high >= t_low - 1e-4

    def test_gradient_flows(self, random_pred):
        logits, target = random_pred
        logits = logits.requires_grad_(True)
        TverskyLoss()(logits, target).backward()
        assert logits.grad is not None


# ---------------------------------------------------------------------------
# HausdorffDTLoss tests
# ---------------------------------------------------------------------------
class TestHausdorffDTLoss:
    def test_output_finite(self, random_pred):
        logits, target = random_pred
        loss = HausdorffDTLoss()(logits, target)
        assert torch.isfinite(loss), f"HD loss must be finite, got {loss.item()}"

    def test_all_zero_target_no_error(self):
        logits = torch.randn(2, 16, 16)
        target = torch.zeros(2, 16, 16)
        loss = HausdorffDTLoss()(logits, target)
        assert torch.isfinite(loss)

    def test_nonnegative(self, random_pred):
        logits, target = random_pred
        loss = HausdorffDTLoss()(logits, target)
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# CompositeMaskLoss tests
# ---------------------------------------------------------------------------
class TestCompositeMaskLoss:
    def test_bce_only(self, random_pred):
        logits, target = random_pred
        comp = CompositeMaskLoss(alpha=1.0, beta=0.0, gamma=0.0)
        result = comp(logits, target)
        standalone = BCELoss()(logits, target)
        assert abs(result["total"].item() - standalone.item()) < 1e-5

    def test_dice_only(self, random_pred):
        logits, target = random_pred
        comp = CompositeMaskLoss(alpha=0.0, beta=1.0, gamma=0.0)
        result = comp(logits, target)
        standalone = SoftDiceLoss()(logits, target)
        assert abs(result["total"].item() - standalone.item()) < 1e-5

    def test_composite_sum(self, random_pred):
        logits, target = random_pred
        alpha, beta, gamma = 0.3, 0.5, 0.5
        comp = CompositeMaskLoss(alpha=alpha, beta=beta, gamma=gamma)
        result = comp(logits, target)
        expected = (
            alpha * result["bce"].item()
            + beta  * result["dice"].item()
            + gamma * result["boundary"].item()
        )
        assert abs(result["total"].item() - expected) < 1e-4

    def test_all_five_losses(self, random_pred):
        logits, target = random_pred
        comp = CompositeMaskLoss(alpha=0.2, beta=0.3, gamma=0.3, delta=0.1, epsilon=0.1)
        result = comp(logits, target)
        assert "bce" in result
        assert "dice" in result
        assert "boundary" in result
        assert "focal" in result
        assert "hd" in result
        assert torch.isfinite(result["total"])

    def test_from_config(self, random_pred):
        logits, target = random_pred
        cfg = {"alpha": 0.3, "beta": 0.5, "gamma": 0.5, "delta": 0.0, "epsilon": 0.0}
        comp = CompositeMaskLoss.from_config(cfg)
        result = comp(logits, target)
        assert torch.isfinite(result["total"])

    def test_describe(self):
        comp = CompositeMaskLoss(alpha=0.3, beta=0.5, gamma=0.5)
        desc = comp.describe()
        assert "BCE" in desc
        assert "Dice" in desc
        assert "Boundary" in desc

    def test_gradient_flows_through_total(self, random_pred):
        logits, target = random_pred
        logits = logits.requires_grad_(True)
        comp = CompositeMaskLoss(alpha=0.3, beta=0.5, gamma=0.5)
        result = comp(logits, target)
        result["total"].backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()
        assert not torch.isinf(logits.grad).any()

    def test_zero_weights_raises_no_key(self, random_pred):
        """Components with weight=0 should not appear in result dict."""
        logits, target = random_pred
        comp = CompositeMaskLoss(alpha=1.0, beta=0.0, gamma=0.0)
        result = comp(logits, target)
        assert "dice" not in result
        assert "boundary" not in result
        assert "bce" in result

    @pytest.mark.parametrize("alpha,beta,gamma", [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.5, 0.5, 0.0),
        (0.0, 0.5, 0.5),
        (0.3, 0.5, 0.5),
        (1.0, 1.0, 1.0),
    ])
    def test_all_weight_combos(self, random_pred, alpha, beta, gamma):
        logits, target = random_pred
        comp = CompositeMaskLoss(alpha=alpha, beta=beta, gamma=gamma)
        result = comp(logits, target)
        assert torch.isfinite(result["total"]), (
            f"Non-finite loss for α={alpha} β={beta} γ={gamma}"
        )
        assert result["total"].item() >= 0.0
