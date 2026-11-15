"""Strict checkpoint-backed inference for exported LUNA16 2.5D patches."""
from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from ..schemas import NoduleDetection

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    path = Path(raw).expanduser() if raw else default
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path.resolve()


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


MODEL_PATH = str(_path_from_env("MODEL_PATH", _PROJECT_ROOT / "experiments" / "best_model.pth"))
EXPERIMENTS_DIR = _path_from_env("EXPERIMENTS_DIR", _PROJECT_ROOT / "experiments")


def _select_device() -> torch.device:
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return torch.device("cuda")
    except Exception:
        logger.exception("CUDA device discovery failed; using CPU inference")
    return torch.device("cpu")


DEVICE = _select_device()
SCORE_THRESHOLD = _env_float("SCORE_THRESHOLD", 0.05, 0.0, 1.0)
NMS_THRESHOLD = _env_float("NMS_THRESHOLD", 0.5, 0.0, 1.0)
PATCH_SIZE = int(os.getenv("PATCH_SIZE", "64"))
DEFAULT_SPACING_MM = _env_float("PATCH_SPACING_MM", 1.0, 1e-6, 100.0)


class InferenceUnavailableError(RuntimeError):
    """Raised when a trained checkpoint cannot provide predictions."""


class CheckpointNotFoundError(FileNotFoundError):
    """Raised when the specifically requested checkpoint does not exist."""


def _checkpoint_state_dict(checkpoint: dict) -> dict:
    """Extract checkpoint weights and migrate the legacy BCE buffer key."""
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint does not contain a model state dictionary")
    state = dict(state)
    old_key = "composite_mask_loss.bce.loss_fn.pos_weight"
    new_key = "composite_mask_loss.bce.pos_weight"
    if old_key in state and new_key not in state:
        state[new_key] = state.pop(old_key)
    return state


@dataclass
class LoadedCheckpoint:
    model: torch.nn.Module
    path: Path
    config_id: str
    config: dict
    epoch: Optional[int]
    metrics: dict
    spacing_mm: tuple[float, float, float]
    inference_lock: threading.RLock = field(default_factory=threading.RLock)


class InferenceEngine:
    """Loads trained models by exact checkpoint path and caches them read-only."""

    def __init__(
        self,
        default_checkpoint_path: str | Path | None = None,
        experiments_dir: str | Path | None = None,
        device: torch.device | str | None = None,
        model_factory: Callable[[dict], torch.nn.Module] | None = None,
    ):
        self.device = torch.device(device) if device is not None else DEVICE
        self.default_checkpoint_path = (
            Path(default_checkpoint_path).expanduser().resolve()
            if default_checkpoint_path is not None
            else Path(MODEL_PATH)
        )
        self.experiments_dir = (
            Path(experiments_dir).expanduser().resolve()
            if experiments_dir is not None
            else Path(EXPERIMENTS_DIR)
        )
        self.model_factory = model_factory
        self._models: dict[str, LoadedCheckpoint] = {}
        self._models_lock = threading.RLock()
        self.model: Optional[torch.nn.Module] = None  # default checkpoint compatibility
        self.model_loading_error: Optional[str] = None
        self.default_config_id: Optional[str] = None

        if os.getenv("ALLOW_UNTRAINED_MODEL", "").strip().lower() in {"1", "true", "yes"}:
            logger.warning("ALLOW_UNTRAINED_MODEL is ignored; predictions require trained checkpoint weights")
        try:
            loaded = self._load_checkpoint(self.default_checkpoint_path)
            self.model = loaded.model
            self.default_config_id = loaded.config_id
        except Exception as exc:
            self.model_loading_error = str(exc)
            logger.error(
                "Default inference checkpoint unavailable at %s: %s",
                self.default_checkpoint_path,
                exc,
            )

    @property
    def status(self) -> dict:
        key = str(self.default_checkpoint_path)
        loaded = self._models.get(key)
        return {
            "model_loaded": loaded is not None,
            "checkpoint_path": str(self.default_checkpoint_path),
            "config_id": loaded.config_id if loaded else None,
            "device": str(self.device),
            "score_threshold": SCORE_THRESHOLD,
            "nms_threshold": NMS_THRESHOLD,
            "model_loading_error": None if loaded else self.model_loading_error,
        }

    def checkpoint_for_experiment(self, config_id: str) -> Path:
        """Resolve only the requested config's checkpoint; never fall back."""
        if not config_id or Path(config_id).name != config_id or config_id in {".", ".."}:
            raise ValueError("Invalid experiment config_id")
        root = self.experiments_dir.resolve()
        checkpoint = (root / config_id / "best_model.pth").resolve()
        if root not in checkpoint.parents:
            raise ValueError("Experiment checkpoint path escapes the experiments directory")
        if not checkpoint.is_file():
            raise CheckpointNotFoundError(
                f"No best_model.pth checkpoint exists for experiment {config_id} at {checkpoint}"
            )
        return checkpoint

    def _make_model(self, config: dict) -> torch.nn.Module:
        model_config = dict(config)
        model_config["pretrained_backbone"] = False
        model_config["box_score_thresh"] = SCORE_THRESHOLD
        model_config["box_nms_thresh"] = NMS_THRESHOLD
        if "mask_loss" not in model_config:
            model_config["mask_loss"] = {"alpha": 1.0, "beta": 0.0, "gamma": 0.0}
        factory = self.model_factory
        if factory is None:
            if str(_PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(_PROJECT_ROOT))
            from training.model import LungNoduleMaskRCNN
            factory = LungNoduleMaskRCNN.from_config
        return factory(model_config)

    def _load_checkpoint(self, checkpoint_path: str | Path) -> LoadedCheckpoint:
        path = Path(checkpoint_path).expanduser().resolve()
        key = str(path)
        with self._models_lock:
            loaded = self._models.get(key)
            if loaded is not None:
                return loaded
            if not path.is_file():
                raise CheckpointNotFoundError(f"Trained checkpoint not found: {path}")

            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            if not isinstance(checkpoint, dict):
                raise ValueError("Checkpoint root must be a dictionary")
            state = _checkpoint_state_dict(checkpoint)
            config = checkpoint.get("config", {})
            if not isinstance(config, dict):
                raise ValueError("Checkpoint config metadata must be a dictionary")
            config = dict(config)
            model = self._make_model(config)
            try:
                model.load_state_dict(state, strict=True)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Checkpoint keys or architecture do not match {path}: {exc}"
                ) from exc
            model.to(self.device)
            model.eval()

            spacing = config.get("target_spacing_mm")
            if spacing is None:
                spacing = (DEFAULT_SPACING_MM,) * 3
                logger.warning(
                    "Checkpoint %s has no target_spacing_mm metadata; using PATCH_SPACING_MM=%.4f",
                    path,
                    DEFAULT_SPACING_MM,
                )
            if not isinstance(spacing, (list, tuple)) or len(spacing) != 3:
                raise ValueError(f"Invalid target_spacing_mm in checkpoint config: {spacing!r}")
            spacing_mm = tuple(float(v) for v in spacing)
            if any(not np.isfinite(v) or v <= 0 for v in spacing_mm):
                raise ValueError(f"Invalid target_spacing_mm in checkpoint config: {spacing!r}")

            metrics = checkpoint.get("metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}
            loaded = LoadedCheckpoint(
                model=model,
                path=path,
                config_id=str(config.get("config_id") or path.parent.name),
                config=config,
                epoch=checkpoint.get("epoch"),
                metrics=metrics,
                spacing_mm=spacing_mm,
            )
            self._models[key] = loaded
            if path == self.default_checkpoint_path:
                self.model_loading_error = None
                self.model = model
                self.default_config_id = loaded.config_id
            logger.info(
                "Loaded trained checkpoint=%s config_id=%s epoch=%s val_dice=%s val_hd95=%s device=%s",
                path,
                loaded.config_id,
                loaded.epoch,
                metrics.get("val_dice"),
                metrics.get("val_hd95"),
                self.device,
            )
            return loaded

    def _get_checkpoint(self, checkpoint_path: str | Path | None) -> LoadedCheckpoint:
        path = Path(checkpoint_path).resolve() if checkpoint_path is not None else self.default_checkpoint_path
        try:
            return self._load_checkpoint(path)
        except CheckpointNotFoundError:
            raise
        except Exception as exc:
            if path == self.default_checkpoint_path:
                self.model_loading_error = str(exc)
            raise InferenceUnavailableError(
                f"Could not load trained checkpoint {path}: {exc}"
            ) from exc

    def checkpoint_info(self, checkpoint_path: str | Path | None = None) -> dict:
        loaded = self._get_checkpoint(checkpoint_path)
        return {
            "checkpoint_path": str(loaded.path),
            "config_id": loaded.config_id,
            "epoch": loaded.epoch,
            "device": str(self.device),
            "score_threshold": SCORE_THRESHOLD,
            "nms_threshold": NMS_THRESHOLD,
        }

    def _preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        from ..utils.preprocessing import preprocess_uploaded_image

        tensor = torch.from_numpy(
            preprocess_uploaded_image(image_bytes, patch_size=PATCH_SIZE, enhance_contrast=False)
        )
        logger.info(
            "Prepared RGB 2.5D patch shape=%s range=[%.5f, %.5f]",
            tuple(tensor.shape),
            float(tensor.min()),
            float(tensor.max()),
        )
        return tensor

    @staticmethod
    def _postprocess(
        outputs: list[dict],
        spacing_mm: tuple[float, float, float],
        score_threshold: float = SCORE_THRESHOLD,
    ) -> list[NoduleDetection]:
        if not outputs:
            logger.info("Raw detections=0; after threshold=0")
            return []
        out = outputs[0]
        boxes = out.get("boxes", torch.zeros((0, 4))).detach().cpu().numpy()
        scores = out.get("scores", torch.zeros(0)).detach().cpu().numpy()
        masks = out.get("masks", torch.zeros((0, 1, PATCH_SIZE, PATCH_SIZE))).detach().cpu().numpy()
        raw_count = min(len(boxes), len(scores))
        detections = []
        for i in range(raw_count):
            score = float(scores[i])
            if not np.isfinite(score) or score < score_threshold:
                continue
            x1, y1, x2, y2 = (float(v) for v in boxes[i])
            width_mm = max(0.0, x2 - x1) * spacing_mm[0]
            height_mm = max(0.0, y2 - y1) * spacing_mm[1]
            diameter_mm = float(np.sqrt(width_mm * height_mm))
            mask_rle = None
            if i < len(masks):
                mask_rle = _rle_encode((masks[i, 0] > 0.5).astype(np.uint8))
            detections.append(NoduleDetection(
                box=[x1, y1, x2, y2],
                score=score,
                mask_rle=mask_rle,
                diameter_estimate_mm=round(diameter_mm, 2),
            ))
        logger.info(
            "Raw detections=%d; after score threshold %.3f=%d",
            raw_count,
            score_threshold,
            len(detections),
        )
        return detections

    @torch.inference_mode()
    def predict_from_bytes(
        self,
        image_bytes: bytes,
        experiment_id: Optional[int] = None,
        checkpoint_path: str | Path | None = None,
    ) -> list[NoduleDetection]:
        # experiment_id is retained for compatible callers; DB resolution is
        # deliberately done in the API route, where a DB session is available.
        del experiment_id
        loaded = self._get_checkpoint(checkpoint_path)
        image = self._preprocess_image(image_bytes).to(self.device)
        with loaded.inference_lock:
            outputs = loaded.model([image])
        detections = self._postprocess(outputs, loaded.spacing_mm)
        logger.info("Final inference detection count=%d", len(detections))
        return detections

    @torch.inference_mode()
    def predict_from_features(self, features: dict, scan_uid: str) -> list[NoduleDetection]:
        """Run the edge continuation using the configured default checkpoint."""
        del scan_uid  # patient/scan identifiers are never logged
        loaded = self._get_checkpoint(None)
        features_device = {
            k: (v.dequantize().to(self.device) if v.is_quantized else v.to(self.device))
            if isinstance(v, torch.Tensor) else v
            for k, v in features.items()
        }
        try:
            with loaded.inference_lock:
                detection_model = loaded.model.model
                dummy_img = torch.zeros(1, 3, PATCH_SIZE, PATCH_SIZE, device=self.device)
                image_list, _ = detection_model.transform([dummy_img], None)
                proposals, _ = detection_model.rpn(image_list, features_device, None)
                detections, _ = detection_model.roi_heads(
                    features_device, proposals, image_list.image_sizes, None
                )
                outputs = detection_model.transform.postprocess(
                    detections,
                    image_list.image_sizes,
                    [(PATCH_SIZE, PATCH_SIZE)],
                )
            result = self._postprocess(outputs, loaded.spacing_mm)
            logger.info("Final edge inference detection count=%d", len(result))
            return result
        except Exception as exc:
            logger.exception("Feature-based inference failed")
            raise InferenceUnavailableError(f"Feature inference failed: {exc}") from exc

    def load_checkpoint(self, checkpoint_path: str):
        """Load/cache a checkpoint without mutating any model already in use."""
        return self._load_checkpoint(checkpoint_path)


def _rle_encode(mask: np.ndarray) -> str:
    """Simple run-length encoding of a 2D binary mask → string."""
    flat = mask.flatten()
    runs = []
    pos = 0
    while pos < len(flat):
        value = flat[pos]
        start = pos
        while pos < len(flat) and flat[pos] == value:
            pos += 1
        runs.append(f"{int(value)}:{pos - start}")
    return ",".join(runs)


def _rle_decode(rle: str, shape: tuple) -> np.ndarray:
    """Decode a binary mask from this module's simple RLE representation."""
    flat = []
    for token in rle.split(","):
        value, count = token.split(":")
        flat.extend([int(value)] * int(count))
    return np.asarray(flat, dtype=np.uint8).reshape(shape)
