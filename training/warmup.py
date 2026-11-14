"""
Boundary Loss Warm-up Curriculum (Hypothesis H1).

Strategy: start with stable (BCE + Dice) until overlap learning is established,
then gradually introduce boundary loss to sharpen edges without divergence.

Three schedules:
  - step:    boundary weight is 0 for N epochs, then jumps to target value
  - linear:  boundary weight ramps linearly from 0 → target over N epochs
  - cosine:  boundary weight follows a cosine ramp (smooth start)

Usage in trainer:
    warmup = BoundaryWarmupScheduler(
        total_epochs=30,
        warmup_epochs=10,
        target_gamma=0.5,
        schedule='linear',
    )
    for epoch in range(1, 31):
        gamma_this_epoch = warmup.get_gamma(epoch)
        loss_fn.gamma = gamma_this_epoch   # update live
"""
import math
import logging
from typing import Literal

logger = logging.getLogger(__name__)

Schedule = Literal["step", "linear", "cosine"]


class BoundaryWarmupScheduler:
    """
    Gradually introduces boundary loss weight γ during training.

    Research motivation (H1): applying boundary loss from epoch 0 before
    the network has learned stable overlap causes gradient conflicts and
    training divergence. Delaying it until a baseline Dice > 0.3 is
    established yields better final HD95 without sacrificing Dice.
    """

    def __init__(
        self,
        total_epochs:   int,
        warmup_epochs:  int   = 10,
        target_gamma:   float = 0.5,
        start_gamma:    float = 0.0,
        schedule:       Schedule = "linear",
    ):
        """
        Args:
            total_epochs:   total training epochs
            warmup_epochs:  number of epochs before γ reaches target
            target_gamma:   final boundary loss weight (γ_max)
            start_gamma:    initial boundary loss weight (usually 0)
            schedule:       'step' | 'linear' | 'cosine'
        """
        assert warmup_epochs <= total_epochs, "warmup_epochs must be ≤ total_epochs"
        assert 0.0 <= start_gamma <= target_gamma

        self.total_epochs  = total_epochs
        self.warmup_epochs = warmup_epochs
        self.target_gamma  = target_gamma
        self.start_gamma   = start_gamma
        self.schedule      = schedule
        self._history: list[tuple[int, float]] = []

    def get_gamma(self, epoch: int) -> float:
        """
        Return the γ value to use at the given epoch (1-indexed).

        Args:
            epoch: current epoch number (starts at 1)

        Returns:
            gamma value in [start_gamma, target_gamma]
        """
        if epoch <= 0:
            return self.start_gamma
        if epoch > self.warmup_epochs:
            return self.target_gamma

        progress = (epoch - 1) / max(self.warmup_epochs - 1, 1)  # 0 → 1

        if self.schedule == "step":
            # Binary: 0 until last warmup epoch, then target
            gamma = self.target_gamma if epoch == self.warmup_epochs else self.start_gamma

        elif self.schedule == "linear":
            gamma = self.start_gamma + progress * (self.target_gamma - self.start_gamma)

        elif self.schedule == "cosine":
            # Cosine ramp: slow start, fast middle, slow end
            cos_progress = (1.0 - math.cos(math.pi * progress)) / 2.0
            gamma = self.start_gamma + cos_progress * (self.target_gamma - self.start_gamma)

        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        gamma = round(float(gamma), 6)
        self._history.append((epoch, gamma))
        return gamma

    def schedule_summary(self) -> list[dict]:
        """Return the full γ schedule as a list for logging/plotting."""
        return [
            {"epoch": ep, "gamma": self.get_gamma(ep)}
            for ep in range(1, self.total_epochs + 1)
        ]

    def log_schedule(self):
        """Print the warm-up schedule to logger."""
        logger.info(
            f"Boundary warm-up: {self.schedule} schedule, "
            f"γ: {self.start_gamma} → {self.target_gamma} over {self.warmup_epochs} epochs"
        )
        summary = self.schedule_summary()
        for s in summary[:5]:
            logger.info(f"  Epoch {s['epoch']:>3}: γ = {s['gamma']:.4f}")
        if len(summary) > 5:
            logger.info(f"  ... ")
            logger.info(f"  Epoch {summary[-1]['epoch']:>3}: γ = {summary[-1]['gamma']:.4f}")


class AdaptiveLossWeightScheduler:
    """
    Adaptive loss weight scheduler that adjusts α, β, γ based on
    training metrics — specifically implements the warm-up for γ
    while optionally decaying α (BCE) as training stabilises.

    Rule:
      - Phase A (epochs 1..warmup):   use (alpha_start, beta, 0)
      - Phase B (epochs warmup..end): use (alpha_end,   beta, gamma_schedule)
    """

    def __init__(
        self,
        total_epochs:    int,
        warmup_epochs:   int   = 10,
        alpha_start:     float = 0.5,
        alpha_end:       float = 0.3,
        beta:            float = 0.5,
        gamma_target:    float = 0.5,
        gamma_schedule:  Schedule = "linear",
        delta:           float = 0.0,
        epsilon:         float = 0.0,
    ):
        self.alpha_start    = alpha_start
        self.alpha_end      = alpha_end
        self.beta           = beta
        self.delta          = delta
        self.epsilon        = epsilon
        self._gamma_sched   = BoundaryWarmupScheduler(
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            target_gamma=gamma_target,
            schedule=gamma_schedule,
        )
        self.warmup_epochs  = warmup_epochs
        self.total_epochs   = total_epochs

    def get_weights(self, epoch: int) -> dict[str, float]:
        """
        Return full loss weight dict for the given epoch.

        Returns:
            {"alpha": ..., "beta": ..., "gamma": ..., "delta": ..., "epsilon": ...}
        """
        gamma = self._gamma_sched.get_gamma(epoch)

        # Linearly decay alpha from alpha_start to alpha_end over warmup period
        if epoch <= self.warmup_epochs:
            t     = (epoch - 1) / max(self.warmup_epochs - 1, 1)
            alpha = self.alpha_start + t * (self.alpha_end - self.alpha_start)
        else:
            alpha = self.alpha_end

        return {
            "alpha":   round(alpha, 6),
            "beta":    self.beta,
            "gamma":   gamma,
            "delta":   self.delta,
            "epsilon": self.epsilon,
        }

    def update_loss_fn(self, loss_fn, epoch: int):
        """
        Convenience: directly update a CompositeMaskLoss instance's weights.

        Args:
            loss_fn: CompositeMaskLoss instance
            epoch:   current epoch number
        """
        weights = self.get_weights(epoch)
        loss_fn.alpha   = weights["alpha"]
        loss_fn.beta    = weights["beta"]
        loss_fn.gamma   = weights["gamma"]
        loss_fn.delta   = weights["delta"]
        loss_fn.epsilon = weights["epsilon"]
        return weights
