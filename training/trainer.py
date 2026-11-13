"""
Training loop for Mask R-CNN lung nodule experiments.
Tracks loss components, evaluation metrics, and saves checkpoints + JSON logs.
"""
import os
import json
import time
import logging
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from .model import build_model
from .dataset import build_dataloaders
from .metrics import MetricTracker, evaluate_batch
from .losses import CompositeMaskLoss

logger = logging.getLogger(__name__)


class Trainer:
    """
    Manages a single training run for a given loss weight configuration.
    """

    def __init__(self, config: dict, run_dir: str):
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Device: {self.device}")

        # Save config
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Build model
        self.model = build_model(config).to(self.device)
        logger.info(f"Loss: {self.model.get_loss_description()}")

        # Optimizer
        lr = config.get("lr", 0.005)
        momentum = config.get("momentum", 0.9)
        weight_decay = config.get("weight_decay", 1e-4)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )

        self.epochs = config.get("epochs", 30)
        warmup_epochs = config.get("warmup_epochs", 5)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs - warmup_epochs,
            eta_min=lr * 0.01,
        )
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = lr / 10.0

        # Data
        self.train_loader, self.val_loader = build_dataloaders(
            data_root=config["data_root"],
            fold=config.get("fold", 0),
            batch_size=config.get("batch_size", 4),
            num_workers=config.get("num_workers", 2),
            patch_size=config.get("patch_size", 64),
        )

        self.history: list[dict] = []
        self.best_dice = 0.0
        self.best_epoch = 0

    # ------------------------------------------------------------------
    # Warmup LR
    # ------------------------------------------------------------------
    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    # ------------------------------------------------------------------
    # Single training epoch
    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        tracker = MetricTracker()
        t0 = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)   # (B, 3, H, W)
            masks = batch["mask"].to(self.device)      # (B, H, W)
            labels = batch["label"].to(self.device)    # (B,)

            # Build Mask R-CNN targets
            targets = []
            for i in range(images.shape[0]):
                h, w = masks[i].shape
                target: dict = {}
                if labels[i].item() == 1 and masks[i].sum() > 0:
                    # Create bounding box from mask
                    msk = masks[i].cpu().numpy().astype(bool)
                    rows = np.any(msk, axis=1)
                    cols = np.any(msk, axis=0)
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    box = torch.tensor(
                        [[cmin, rmin, cmax + 1, rmax + 1]],
                        dtype=torch.float32, device=self.device,
                    )
                    target["boxes"] = box
                    target["labels"] = torch.tensor([1], dtype=torch.int64, device=self.device)
                    target["masks"] = masks[i].unsqueeze(0).to(torch.uint8)
                else:
                    target["boxes"] = torch.zeros((0, 4), dtype=torch.float32, device=self.device)
                    target["labels"] = torch.zeros(0, dtype=torch.int64, device=self.device)
                    target["masks"] = torch.zeros(
                        (0, masks[i].shape[0], masks[i].shape[1]),
                        dtype=torch.uint8, device=self.device,
                    )
                targets.append(target)

            self.optimizer.zero_grad()
            loss_dict = self.model(list(images), targets)

            if isinstance(loss_dict, dict):
                total_loss = sum(loss_dict.values())
                tracker.update({k: v.item() for k, v in loss_dict.items()})
                tracker.update({"total_loss": total_loss.item()})
            else:
                total_loss = torch.tensor(0.0, device=self.device)
                logger.warning("Model returned non-dict in training mode")

            if total_loss.requires_grad:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

            if (batch_idx + 1) % 20 == 0:
                avgs = tracker.averages()
                logger.info(
                    f"Epoch {epoch} [{batch_idx+1}/{len(self.train_loader)}] "
                    f"loss={avgs.get('total_loss', 0):.4f}"
                )

        elapsed = time.time() - t0
        result = tracker.averages()
        result["epoch_time_s"] = elapsed
        return result

    # ------------------------------------------------------------------
    # Validation epoch
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        loss_tracker = MetricTracker()

        # Use our standalone mask loss for validation loss computation
        standalone_loss = self.model.composite_mask_loss

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"]
            labels = batch["label"]

            # Run inference
            outputs = self.model(list(images))

            for i, out in enumerate(outputs):
                h, w = masks[i].shape
                pred_mask = np.zeros((h, w), dtype=np.float32)
                if len(out.get("masks", [])) > 0:
                    # Take the highest-scoring mask
                    scores = out["scores"].cpu().numpy()
                    if len(scores) > 0:
                        best = int(np.argmax(scores))
                        pred_mask = (
                            out["masks"][best, 0].cpu().numpy()
                        )  # (H, W) probabilities

                all_preds.append(pred_mask)
                all_targets.append(masks[i].numpy())

                # Compute standalone validation loss
                if labels[i].item() == 1 and masks[i].sum() > 0:
                    pred_t = torch.tensor(pred_mask, device=self.device).unsqueeze(0)
                    tgt_t = masks[i].to(self.device).unsqueeze(0).float()
                    # logit approximation (clamp to avoid inf)
                    logit = torch.log(
                        pred_t.clamp(1e-7, 1 - 1e-7) /
                        (1 - pred_t.clamp(1e-7, 1 - 1e-7))
                    )
                    loss_dict = standalone_loss(logit, tgt_t)
                    loss_tracker.update({f"val_{k}": v.item() for k, v in loss_dict.items()})

        preds_arr = np.stack(all_preds)
        targets_arr = np.stack(all_targets)
        seg_metrics = evaluate_batch(preds_arr, targets_arr)

        result = loss_tracker.averages()
        result.update({f"val_{k}": v for k, v in seg_metrics.items()})
        return result

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(self) -> dict:
        logger.info(f"Starting training: {self.epochs} epochs | Loss: {self.model.get_loss_description()}")
        best_ckpt = self.run_dir / "best_model.pth"

        for epoch in range(1, self.epochs + 1):
            # Warmup LR
            if epoch <= self.warmup_epochs:
                warmup_lr = self.warmup_lr + (
                    self.config.get("lr", 0.005) - self.warmup_lr
                ) * (epoch / self.warmup_epochs)
                self._set_lr(warmup_lr)
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            train_metrics = self._train_epoch(epoch)
            val_metrics = self._val_epoch(epoch)

            epoch_log = {
                "epoch": epoch,
                "lr": current_lr,
                **train_metrics,
                **val_metrics,
            }
            self.history.append(epoch_log)

            # Checkpoint
            val_dice = val_metrics.get("val_dice", 0.0)
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                self.best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "metrics": epoch_log,
                    "config": self.config,
                }, best_ckpt)
                logger.info(f"  ✓ New best Dice: {val_dice:.4f} → saved checkpoint")

            # Save latest
            torch.save({
                "epoch": epoch,
                "model_state": self.model.state_dict(),
            }, self.run_dir / "latest_model.pth")

            # Save history
            with open(self.run_dir / "history.json", "w") as f:
                json.dump(self.history, f, indent=2)

            logger.info(
                f"Epoch {epoch}/{self.epochs} | "
                f"train_loss={train_metrics.get('total_loss', 0):.4f} | "
                f"val_dice={val_dice:.4f} | "
                f"val_hd95={val_metrics.get('val_hd95', 999):.2f} | "
                f"val_boundary_f1={val_metrics.get('val_boundary_f1', 0):.4f}"
            )

        # Final summary
        summary = {
            "config_id": self.config.get("config_id", "unknown"),
            "loss_description": self.model.get_loss_description(),
            "best_epoch": self.best_epoch,
            "best_dice": self.best_dice,
            "loss_weights": {
                "alpha": self.config.get("mask_loss", {}).get("alpha", 0),
                "beta": self.config.get("mask_loss", {}).get("beta", 0),
                "gamma": self.config.get("mask_loss", {}).get("gamma", 0),
                "delta": self.config.get("mask_loss", {}).get("delta", 0),
                "epsilon": self.config.get("mask_loss", {}).get("epsilon", 0),
            },
            "final_metrics": self.history[-1] if self.history else {},
        }

        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            f"Training complete. Best Dice: {self.best_dice:.4f} at epoch {self.best_epoch}"
        )
        return summary
