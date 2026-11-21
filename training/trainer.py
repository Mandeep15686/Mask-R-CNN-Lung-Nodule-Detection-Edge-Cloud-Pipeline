"""
Training loop for Mask R-CNN lung nodule experiments.
Tracks loss components, evaluation metrics, and saves checkpoints + JSON logs.
"""
import os
import json
import time
import logging
import shutil
import hashlib
import csv
import random
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
from .mgda import MGDASolver, LambdaHistory, mgda_capture_context, solve_mgda

logger = logging.getLogger(__name__)


def _fmt_metric(value, fmt: str = ".4f") -> str:
    """Format a metric safely, showing unavailable values as N/A."""
    if value is None or not np.isfinite(value):
        return "N/A"
    return format(float(value), fmt)


def _global_grad_norm(parameters) -> float:
    """L2 norm across all present parameter gradients, without concatenating them."""
    total_sq = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        norm = torch.linalg.vector_norm(parameter.grad.detach().float())
        total_sq = norm.square() if total_sq is None else total_sq + norm.square()
    return float(total_sq.sqrt().item()) if total_sq is not None else 0.0


def _estimate_sgd_step_norm(optimizer: torch.optim.Optimizer) -> float:
    """Estimate the exact current SGD parameter-step norm before ``step()``.

    Includes the configured learning rate, weight decay, momentum, dampening,
    and Nesterov term. This is a scalar diagnostic and does not mutate optimizer
    state or gradients.
    """
    total_sq = None
    with torch.no_grad():
        for group in optimizer.param_groups:
            lr = float(group["lr"])
            momentum = float(group.get("momentum", 0.0))
            dampening = float(group.get("dampening", 0.0))
            weight_decay = float(group.get("weight_decay", 0.0))
            nesterov = bool(group.get("nesterov", False))
            maximize = bool(group.get("maximize", False))
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                direction = parameter.grad.detach()
                if maximize:
                    direction = -direction
                if weight_decay:
                    direction = direction.add(parameter.detach(), alpha=weight_decay)
                if momentum:
                    previous = optimizer.state.get(parameter, {}).get("momentum_buffer")
                    if previous is None:
                        momentum_direction = direction
                    else:
                        momentum_direction = previous.detach().mul(momentum).add(
                            direction, alpha=1.0 - dampening
                        )
                    direction = (
                        direction.add(momentum_direction, alpha=momentum)
                        if nesterov
                        else momentum_direction
                    )
                step_norm = torch.linalg.vector_norm(direction.float()) * lr
                total_sq = step_norm.square() if total_sq is None else total_sq + step_norm.square()
    return float(total_sq.sqrt().item()) if total_sq is not None else 0.0


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch CPU, and all visible CUDA devices."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))


class Trainer:
    """
    Manages a single training run for a given loss weight configuration.
    """

    def __init__(self, config: dict, run_dir: str):
        self.config = config
        self.seed = config.get("seed")
        if self.seed is not None:
            seed_everything(int(self.seed))
        self.training_strategy = config.get("training_strategy", "fixed")
        if self.training_strategy not in {"fixed", "mgda"}:
            raise ValueError("training_strategy must be 'fixed' or 'mgda'")
        self.config["training_strategy"] = self.training_strategy
        self.mgda_gradient_normalization = config.get(
            "mgda_gradient_normalization", "raw"
        )
        self.mgda_parameter_scope = config.get("mgda_parameter_scope", "mask_head")
        if self.mgda_gradient_normalization not in {"raw", "unit"}:
            raise ValueError("mgda_gradient_normalization must be 'raw' or 'unit'")
        if self.mgda_parameter_scope not in {"mask_head", "shared_mask_path"}:
            raise ValueError(
                "mgda_parameter_scope must be 'mask_head' or 'shared_mask_path'"
            )
        if self.training_strategy == "mgda":
            self.config["mgda_gradient_normalization"] = self.mgda_gradient_normalization
            self.config["mgda_parameter_scope"] = self.mgda_parameter_scope
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        if config.get("require_cuda", False) and not torch.cuda.is_available():
            raise RuntimeError("This experiment requires CUDA, but torch.cuda.is_available() is False")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

        # Build model
        model_config = dict(config)
        if self.training_strategy == "mgda" and not config.get("mgda_use_configured_tasks", False):
            # The model's Mask R-CNN mask-loss hook returns a scalar. Unit
            # weights expose all five component graphs; MGDA scalarizes them
            # in _train_epoch after measuring their shared-head gradients.
            model_config["mask_loss"] = {
                "alpha": 1.0,
                "beta": 1.0,
                "gamma": 1.0,
                "delta": 1.0,
                "epsilon": 1.0,
            }
        self.model = build_model(model_config).to(self.device)
        initial_state_path = config.get("initial_state_path")
        if initial_state_path:
            initial_path = Path(initial_state_path).expanduser().resolve()
            if not initial_path.is_file():
                raise FileNotFoundError(f"Shared initial model state not found: {initial_path}")
            digest = hashlib.sha256(initial_path.read_bytes()).hexdigest()
            expected_digest = config.get("initial_state_sha256")
            if expected_digest and digest != expected_digest:
                raise ValueError("Shared initial state SHA256 does not match experiment config")
            initial_state = torch.load(initial_path, map_location="cpu", weights_only=True)
            if isinstance(initial_state, dict) and "model_state" in initial_state:
                initial_state = initial_state["model_state"]
            self.model.load_state_dict(initial_state, strict=True)
            for value in self.model.state_dict().values():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    raise ValueError("Shared initial checkpoint contains non-finite model tensors")
            self.config["initial_state_sha256"] = digest
            self.config["initial_state_path"] = str(initial_path)
        if self.training_strategy == "mgda":
            self.model.composite_mask_loss.capture_components = True
            logger.info(
                "Loss strategy: MGDA adaptive objectives %s | "
                "gradient_normalization=%s | parameter_scope=%s",
                self.model.composite_mask_loss.active_tasks(),
                self.mgda_gradient_normalization,
                self.mgda_parameter_scope,
            )
        else:
            logger.info(f"Loss: {self.model.get_loss_description()}")

        all_mgda_tasks = ("bce", "dice", "boundary", "focal", "hd")
        configured_tasks = tuple(self.model.composite_mask_loss.active_tasks())
        self.mgda_task_names = configured_tasks if config.get("mgda_use_configured_tasks", False) else all_mgda_tasks
        def is_mask_branch(name: str) -> bool:
            return "roi_heads.mask_head." in name or "roi_heads.mask_predictor." in name

        self.mgda_shared_parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
            and (
                is_mask_branch(name)
                or (
                    self.mgda_parameter_scope == "shared_mask_path"
                    and ("backbone.body." in name or "backbone.fpn." in name)
                )
            )
        ]
        self.mgda_weights = {name: 1.0 / len(self.mgda_task_names) if name in self.mgda_task_names else 0.0 for name in all_mgda_tasks}
        self.mgda_coefficients = dict(self.mgda_weights)
        if self.training_strategy == "mgda" and not self.mgda_shared_parameters:
            raise ValueError("MGDA could not find trainable Mask R-CNN mask-head parameters")

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

        # Data — use pre-extracted cache if available (eliminates SSD bottleneck)
        self.train_loader, self.val_loader = build_dataloaders(
            data_root=config["data_root"],
            fold=config.get("fold", 0),
            batch_size=config.get("batch_size", 4),
            num_workers=config.get("num_workers", 2),
            patch_size=config.get("patch_size", 64),
            cache_dir=config.get("cache_dir"),   # None → slow raw-volume path
            train_subsets=config.get("train_subsets"),
            validation_subset=config.get("validation_subset"),
            test_subset=config.get("test_subset"),
            target_spacing_mm=tuple(
                config.get("target_spacing_mm", (1.0, 1.0, 1.0))
            ),
            seed=self.seed,
        )
        self.data_split = self.train_loader.dataset.split_info
        self.config["data_split"] = self.data_split
        if self.config.get("cache_dir"):
            self.config["cache_mode"] = "cached"
            self.config["cache_dir"] = str(
                self.train_loader.dataset.cache_dir.resolve()
            )
            self.config.setdefault(
                "cache_summary",
                self.train_loader.dataset.cache_summary,
            )
            self.config["cache_spacing_mm"] = (
                list(self.train_loader.dataset.preprocessing_spacing_mm)
                if self.train_loader.dataset.preprocessing_spacing_mm is not None
                else None
            )
        with open(self.run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

        self.history: list[dict] = []
        self.training_history: list[dict] = []
        self.validation_history: list[dict] = []
        self.best_dice = float("-inf")
        self.best_epoch = 0
        self.best_metrics: dict = {}

    # ------------------------------------------------------------------
    # Warmup LR
    # ------------------------------------------------------------------
    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _forward_train(self, images, targets):
        loss_dict = self.model(list(images), targets)
        return loss_dict, getattr(self.model.composite_mask_loss, "last_components", None)

    def _solve_mgda(self, objectives):
        result = solve_mgda(
            objectives, self.mgda_shared_parameters,
            gradient_normalization=self.mgda_gradient_normalization,
        )
        return result.weights, result.gradient_norms, result.applied_coefficients, result.diagnostics

    # ------------------------------------------------------------------
    # Single training epoch
    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        tracker = MetricTracker()
        t0 = time.time()
        mgda_active_batches = 0

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
            loss_dict, component_losses = self._forward_train(images, targets)

            if isinstance(loss_dict, dict):
                if self.training_strategy == "mgda":
                    if component_losses and all(
                        name in component_losses for name in self.mgda_task_names
                    ):
                        objectives = {
                            name: component_losses[name]
                            for name in self.mgda_task_names
                        }
                        weights, gradient_norms, applied_coefficients, diagnostics = self._solve_mgda(objectives)
                        weighted_mask_loss = sum(
                            applied_coefficients[name] * objectives[name]
                            for name in self.mgda_task_names
                        )
                        detector_loss = sum(
                            value for name, value in loss_dict.items()
                            if name != "loss_mask"
                        )
                        total_loss = detector_loss + weighted_mask_loss
                        tracker.update({
                            name: value.item()
                            for name, value in loss_dict.items()
                            if name != "loss_mask"
                        })
                        tracker.update({
                            "loss_mask": weighted_mask_loss.item(),
                            **{
                                f"mgda_loss_{name}": objectives[name].item()
                                for name in self.mgda_task_names
                            },
                            **{
                                f"mgda_weight_{name}": weights[name]
                                for name in self.mgda_task_names
                            },
                            **{
                                f"mgda_grad_norm_{name}": gradient_norms[name]
                                for name in self.mgda_task_names
                            },
                            **{
                                f"mgda_coefficient_{name}": applied_coefficients[name]
                                for name in self.mgda_task_names
                            },
                            **diagnostics,
                        })
                        mgda_active_batches += 1
                        self.mgda_weights = weights
                        self.mgda_coefficients = applied_coefficients
                    else:
                        # A negative-only batch can have no positive mask ROIs.
                        # In that case torchvision supplies a graph-connected
                        # zero mask loss and only the detector losses are active.
                        total_loss = sum(loss_dict.values())
                        tracker.update({k: v.item() for k, v in loss_dict.items()})
                else:
                    total_loss = sum(loss_dict.values())
                    tracker.update({k: v.item() for k, v in loss_dict.items()})
                tracker.update({"total_loss": total_loss.item()})
            else:
                total_loss = torch.tensor(0.0, device=self.device)
                logger.warning("Model returned non-dict in training mode")

            if total_loss.requires_grad:
                total_loss.backward()
                pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=5.0
                )
                if self.training_strategy == "mgda":
                    post_clip_norm = _global_grad_norm(self.model.parameters())
                    learning_rate = float(self.optimizer.param_groups[0]["lr"])
                    tracker.update({
                        "mgda_optimizer_grad_norm_pre_clip": float(pre_clip_norm.item()),
                        "mgda_optimizer_grad_norm_post_clip": post_clip_norm,
                        "mgda_learning_rate": learning_rate,
                        # First-order gradient contribution of LR; momentum and
                        # weight decay are accounted for by the step estimate.
                        "mgda_lr_times_postclip_grad_norm": learning_rate * post_clip_norm,
                        "mgda_optimizer_step_norm_estimate": _estimate_sgd_step_norm(
                            self.optimizer
                        ),
                    })
                self.optimizer.step()

            if (batch_idx + 1) % 20 == 0:
                avgs = tracker.averages()
                logger.info(
                    f"Epoch {epoch} [{batch_idx+1}/{len(self.train_loader)}] "
                    f"loss={avgs.get('total_loss', 0):.4f}"
                )

        elapsed = time.time() - t0
        result = tracker.averages()
        sampler = getattr(self.train_loader, "sampler", None)
        sampled_indices = getattr(sampler, "last_indices", None)
        dataset_entries = getattr(getattr(self.train_loader, "dataset", None), "entries", None)
        if sampled_indices is not None and dataset_entries is not None:
            actual_count = len(self.train_loader) * int(self.train_loader.batch_size)
            order_digest = hashlib.sha256()
            for index in sampled_indices[:actual_count]:
                entry = dataset_entries[index]
                order_digest.update(f"{entry['uid']}|{entry['file']}\n".encode("utf-8"))
            result["train_sample_order_sha256"] = order_digest.hexdigest()
        if self.training_strategy == "mgda":
            for name in self.mgda_task_names:
                result.setdefault(f"mgda_loss_{name}", 0.0)
                result.setdefault(f"mgda_weight_{name}", self.mgda_weights[name])
                result.setdefault(f"mgda_grad_norm_{name}", 0.0)
                result.setdefault(f"mgda_coefficient_{name}", self.mgda_coefficients[name])
            result.setdefault("mgda_optimizer_grad_norm_pre_clip", 0.0)
            result.setdefault("mgda_optimizer_grad_norm_post_clip", 0.0)
            result.setdefault("mgda_learning_rate", float(self.optimizer.param_groups[0]["lr"]))
            result.setdefault("mgda_lr_times_postclip_grad_norm", 0.0)
            result.setdefault("mgda_optimizer_step_norm_estimate", 0.0)
            result.setdefault("mgda_shared_parameter_count", float(len(self.mgda_shared_parameters)))
            result.setdefault(
                "mgda_shared_parameter_numel",
                float(sum(parameter.numel() for parameter in self.mgda_shared_parameters)),
            )
            result["mgda_active_batches"] = float(mgda_active_batches)
            self.mgda_weights = {
                name: result.get(f"mgda_weight_{name}", 0.0) for name in self.mgda_weights
            }
            self.mgda_coefficients = {
                name: result.get(f"mgda_coefficient_{name}", 0.0) for name in self.mgda_coefficients
            }
        result["epoch_time_s"] = elapsed
        if self.device.type == "cuda":
            result["gpu_peak_memory_allocated_mb"] = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            result["gpu_peak_memory_reserved_mb"] = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
        return result

    # ------------------------------------------------------------------
    # Validation epoch
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> dict[str, float | None]:
        self.model.eval()
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        all_uids: list[str] = []
        false_positive_detections: list[int] = []
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
                predicted_instances = 0
                if len(out.get("masks", [])) > 0:
                    predicted_instances = int(
                        (out["masks"][:, 0] > 0.5)
                        .flatten(start_dim=1)
                        .any(dim=1)
                        .sum()
                        .item()
                    )
                    # Take the highest-scoring mask
                    scores = out["scores"].cpu().numpy()
                    if len(scores) > 0:
                        best = int(np.argmax(scores))
                        pred_mask = (
                            out["masks"][best, 0].cpu().numpy()
                        )  # (H, W) probabilities

                all_preds.append(pred_mask)
                all_targets.append(masks[i].numpy())
                all_uids.append(str(batch["uid"][i]))
                false_positive_detections.append(predicted_instances)

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
                    if self.training_strategy == "mgda":
                        loss_tracker.update({
                            f"val_{name}": loss_dict[name].item()
                            for name in self.mgda_task_names
                        })
                        val_total = sum(
                            self.mgda_coefficients[name] * loss_dict[name]
                            for name in self.mgda_task_names
                        )
                        loss_tracker.update({"val_total": val_total.item()})
                    else:
                        loss_tracker.update({f"val_{k}": v.item() for k, v in loss_dict.items()})

        preds_arr = np.stack(all_preds)
        targets_arr = np.stack(all_targets)
        # Both CLI and API training use the same finite-safe metric aggregation.
        # None means a metric is not defined for this validation set.
        seg_metrics = evaluate_batch(
            preds_arr,
            targets_arr,
            scan_uids=all_uids,
            false_positive_detections=false_positive_detections,
        )
        # Guard the API training/history path as well, so a non-finite metric
        # from any future metric implementation cannot reach JSON or SQLite.
        seg_metrics = {
            key: (float(value) if value is not None and np.isfinite(value) else None)
            for key, value in seg_metrics.items()
        }

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
            if self.device.type == "cuda":
                train_metrics["gpu_peak_memory_allocated_mb"] = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                train_metrics["gpu_peak_memory_reserved_mb"] = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)

            epoch_log = {
                "epoch": epoch,
                "lr": current_lr,
                **train_metrics,
                **val_metrics,
            }
            self.history.append(epoch_log)
            self.training_history.append({"epoch": epoch, "lr": current_lr, **train_metrics})
            self.validation_history.append({"epoch": epoch, **val_metrics})

            # Checkpoint
            val_dice = val_metrics.get("val_dice")
            if val_dice is not None and np.isfinite(val_dice):
                val_dice = float(val_dice)
                if val_dice > self.best_dice:
                    self.best_dice = val_dice
                    self.best_epoch = epoch
                    self.best_metrics = dict(epoch_log)
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
                json.dump(self.history, f, indent=2, allow_nan=False)
            with open(self.run_dir / "training_history.json", "w", encoding="utf-8") as f:
                json.dump(self.training_history, f, indent=2, allow_nan=False)
            with open(self.run_dir / "validation_history.json", "w", encoding="utf-8") as f:
                json.dump(self.validation_history, f, indent=2, allow_nan=False)
            self._write_history_csv("history.csv", self.history)
            self._write_history_csv("training_history.csv", self.training_history)
            self._write_history_csv("validation_history.csv", self.validation_history)

            logger.info(
                f"Epoch {epoch}/{self.epochs} | "
                f"train_loss={train_metrics.get('total_loss', 0):.4f} | "
                f"val_dice={_fmt_metric(val_metrics.get('val_dice'))} | "
                f"sampled_neg_patch_fp_rate={_fmt_metric(val_metrics.get('val_sampled_negative_patch_false_positive_rate'))} | "
                f"sampled_patch_fp_per_scan_proxy={_fmt_metric(val_metrics.get('val_sampled_patch_false_detections_per_scan_proxy'))} | "
                f"val_hd95={_fmt_metric(val_metrics.get('val_hd95'), '.2f')} | "
                f"val_boundary_f1={_fmt_metric(val_metrics.get('val_boundary_f1'))}"
            )
            if self.training_strategy == "mgda":
                logger.info(
                    "MGDA epoch %d | simplex_weights=%s | applied_coefficients=%s | "
                    "losses=%s | task_grad_norms=%s | gradient_cosines=%s | "
                    "optimizer_grad_norm(pre/post_clip)=%s | lr=%s | lr*grad_norm=%s | "
                    "step_norm_estimate=%s | active_batches=%d",
                    epoch,
                    {
                        name: _fmt_metric(train_metrics.get(f"mgda_weight_{name}"))
                        for name in self.mgda_task_names
                    },
                    {
                        name: _fmt_metric(train_metrics.get(f"mgda_coefficient_{name}"), ".6g")
                        for name in self.mgda_task_names
                    },
                    {
                        name: _fmt_metric(train_metrics.get(f"mgda_loss_{name}"))
                        for name in self.mgda_task_names
                    },
                    {
                        name: _fmt_metric(train_metrics.get(f"mgda_grad_norm_{name}"), ".6g")
                        for name in self.mgda_task_names
                    },
                    {
                        key.removeprefix("mgda_cosine_"): _fmt_metric(value, ".3f")
                        for key, value in train_metrics.items()
                        if key.startswith("mgda_cosine_")
                    },
                    (
                        _fmt_metric(train_metrics.get("mgda_optimizer_grad_norm_pre_clip"), ".6g"),
                        _fmt_metric(train_metrics.get("mgda_optimizer_grad_norm_post_clip"), ".6g"),
                    ),
                    _fmt_metric(train_metrics.get("mgda_learning_rate"), ".6g"),
                    _fmt_metric(train_metrics.get("mgda_lr_times_postclip_grad_norm"), ".6g"),
                    _fmt_metric(train_metrics.get("mgda_optimizer_step_norm_estimate"), ".6g"),
                    int(train_metrics.get("mgda_active_batches", 0.0)),
                )

        # Final summary
        summary = {
            "config_id": self.config.get("config_id", "unknown"),
            "training_strategy": self.training_strategy,
            "use_mgda": bool(self.config.get("use_mgda", self.training_strategy == "mgda")),
            "mgda_param_scope": self.config.get("mgda_param_scope", self.mgda_parameter_scope),
            "mgda_tasks": list(self.mgda_task_names) if self.training_strategy == "mgda" else [],
            "mgda_max_iter": self.config.get("mgda_max_iter"),
            "mgda_tol": self.config.get("mgda_tol"),
            "seed": self.seed,
            "initial_state_sha256": self.config.get("initial_state_sha256"),
            "train_batches_per_epoch": len(self.train_loader),
            "validation_batches_per_epoch": len(self.val_loader),
            "train_samples": len(self.train_loader.dataset),
            "validation_samples": len(self.val_loader.dataset),
            "batch_size": self.config.get("batch_size"),
            "hardware": self.config.get("hardware"),
            "mgda_profile": self.config.get("mgda_profile", "custom"),
            "mgda_gradient_normalization": self.mgda_gradient_normalization,
            "mgda_parameter_scope": self.mgda_parameter_scope,
            "mgda_shared_parameter_count": (
                len(self.mgda_shared_parameters) if self.training_strategy == "mgda" else None
            ),
            "mgda_shared_parameter_numel": (
                sum(parameter.numel() for parameter in self.mgda_shared_parameters)
                if self.training_strategy == "mgda"
                else None
            ),
            "loss_description": (
                "MGDA over BCE, Dice, Boundary, Focal, and HD-distance surrogate"
                if self.training_strategy == "mgda"
                else self.model.get_loss_description()
            ),
            "best_epoch": self.best_epoch,
            "best_metrics": self.best_metrics,
            "best_dice": (
                float(self.best_dice) if np.isfinite(self.best_dice) else None
            ),
            "loss_weights": {
                "alpha": self.mgda_weights["bce"] if self.training_strategy == "mgda" else self.config.get("mask_loss", {}).get("alpha", 0),
                "beta": self.mgda_weights["dice"] if self.training_strategy == "mgda" else self.config.get("mask_loss", {}).get("beta", 0),
                "gamma": self.mgda_weights["boundary"] if self.training_strategy == "mgda" else self.config.get("mask_loss", {}).get("gamma", 0),
                "delta": self.mgda_weights["focal"] if self.training_strategy == "mgda" else self.config.get("mask_loss", {}).get("delta", 0),
                "epsilon": self.mgda_weights["hd"] if self.training_strategy == "mgda" else self.config.get("mask_loss", {}).get("epsilon", 0),
            },
            "effective_loss_coefficients": (
                {
                    "alpha": self.mgda_coefficients["bce"],
                    "beta": self.mgda_coefficients["dice"],
                    "gamma": self.mgda_coefficients["boundary"],
                    "delta": self.mgda_coefficients["focal"],
                    "epsilon": self.mgda_coefficients["hd"],
                }
                if self.training_strategy == "mgda"
                else None
            ),
            "final_metrics": self.history[-1] if self.history else {},
            "data_split": self.data_split,
        }

        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, allow_nan=False)

        logger.info(
            f"Training complete. Best Dice: {_fmt_metric(self.best_dice)} "
            f"at epoch {self.best_epoch}"
        )
        return summary

    def _write_history_csv(self, filename: str, rows: list[dict]) -> None:
        if not rows:
            return
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (self.run_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


class MGDATrainer(Trainer):
    """MGDA trainer with explicit task subsets and the Frank-Wolfe solver.

    The legacy ``Trainer(training_strategy='mgda')`` remains available. This
    class is the separate Phase 4 strategy and captures positive ROI mask
    logits so task losses can be scalarized before the single optimizer step.
    """

    def __init__(self, config: dict, run_dir: str):
        cfg = dict(config)
        cfg["training_strategy"] = "mgda"
        cfg["mgda_use_configured_tasks"] = True
        self.mgda_param_scope = cfg.get("mgda_param_scope", "mask_head")
        if self.mgda_param_scope not in {"mask_head", "full"}:
            raise ValueError("mgda_param_scope must be 'mask_head' or 'full'")
        # Base Trainer has a historical spelling for the larger shared path.
        cfg["mgda_parameter_scope"] = (
            "mask_head" if self.mgda_param_scope == "mask_head" else "shared_mask_path"
        )
        super().__init__(cfg, run_dir)
        self.config["use_mgda"] = True
        self.config["mgda_param_scope"] = self.mgda_param_scope
        self.config["mgda_log_conflict"] = bool(cfg.get("mgda_log_conflict", False))
        self.mgda_solver = MGDASolver(
            self.model,
            param_scope=self.mgda_param_scope,
            max_iter=int(cfg.get("mgda_max_iter", 200)),
            tol=float(cfg.get("mgda_tol", 1e-8)),
            gradient_normalization=cfg.get("mgda_gradient_normalization", "raw"),
        )
        self.mgda_shared_parameters = self.mgda_solver.parameters
        # Rewrite after base initialization so config/checkpoints record the
        # actual selectable strategy and parameter scope.
        with (self.run_dir / "config.json").open("w", encoding="utf-8") as stream:
            json.dump(self.config, stream, indent=2)
        self.lambda_history = LambdaHistory()

    def _forward_train(self, images, targets):
        cache: dict = {}
        with mgda_capture_context(cache):
            loss_dict = self.model.forward_mgda(list(images), targets, cache)
        logits = cache.get("logits")
        mask_targets = cache.get("targets")
        objectives = None
        if logits is not None and mask_targets is not None:
            objectives = self.model.composite_mask_loss.forward_all_tasks(logits, mask_targets)
        return loss_dict, objectives

    def _solve_mgda(self, objectives):
        result = self.mgda_solver.solve(objectives)
        weights = {
            name: float(result.lambdas[index].item())
            for index, name in enumerate(result.task_names)
        }
        self.lambda_history.record(weights)
        diagnostics = dict(result.diagnostics)
        if not self.config.get("mgda_log_conflict", False):
            diagnostics = {
                key: value for key, value in diagnostics.items()
                if not key.startswith("mgda_cosine_")
                and key not in {"conflict_fraction", "mean_pairwise_cosine"}
            }
        return weights, result.gradient_norms, result.applied_coefficients, diagnostics

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.lambda_history.reset()
        metrics = super()._train_epoch(epoch)
        for name, average in self.lambda_history.epoch_averages().items():
            metrics[f"mgda_epoch_lambda_{name}"] = average
        return metrics


def build_trainer(config: dict, run_dir: str) -> Trainer:
    """Construct the configured trainer while preserving legacy defaults."""
    if config.get("use_mgda", False):
        return MGDATrainer(config, run_dir)
    return Trainer(config, run_dir)
