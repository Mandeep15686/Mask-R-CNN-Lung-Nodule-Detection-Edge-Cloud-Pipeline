"""
Inference engine: loads a trained Mask R-CNN model and runs prediction.
Handles both full-image cloud inference and feature-continuation from edge.
"""
import io
import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..schemas import NoduleDetection

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = os.getenv("MODEL_PATH", str(_PROJECT_ROOT / "experiments" / "best_model.pth"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.05"))
NMS_THRESHOLD = float(os.getenv("NMS_THRESHOLD", "0.5"))
PATCH_SIZE = int(os.getenv("PATCH_SIZE", "64"))
APP_ENV = os.getenv("APP_ENV", "development").lower()
_allow_untrained_env = os.getenv("ALLOW_UNTRAINED_MODEL", "").strip().lower()
ALLOW_UNTRAINED_MODEL = (
    _allow_untrained_env == "true"
    if _allow_untrained_env
    else APP_ENV != "production"
)


def _checkpoint_state_dict(checkpoint):
    """Extract weights and migrate the pre-buffer BCE key name if present."""
    state = dict(checkpoint.get("model_state", checkpoint))
    old_key = "composite_mask_loss.bce.loss_fn.pos_weight"
    new_key = "composite_mask_loss.bce.pos_weight"
    if old_key in state and new_key not in state:
        state[new_key] = state.pop(old_key)
    return state

# Average CT spacing for diameter estimation
DEFAULT_SPACING_MM = 0.7


class InferenceEngine:
    """
    Lazy-loads the best checkpoint and provides prediction methods.
    Falls back to a randomly-initialised model if no checkpoint exists
    (useful for development without training data).
    """

    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
            from training.model import LungNoduleMaskRCNN

            self.model = LungNoduleMaskRCNN(
                num_classes=2,
                mask_loss_cfg={"alpha": 0.3, "beta": 0.5, "gamma": 0.5},
                # Checkpoints contain the complete backbone; inference must not
                # download unrelated ImageNet weights during API startup.
                pretrained_backbone=False,
            )
            self.model.to(DEVICE)

            checkpoint_path = Path(MODEL_PATH)
            if not checkpoint_path.exists():
                candidates = sorted(
                    _PROJECT_ROOT.joinpath("experiments").glob("*/best_model.pth"),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
                if candidates:
                    checkpoint_path = candidates[0]
            if checkpoint_path.exists():
                ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
                state_dict = _checkpoint_state_dict(ckpt)
                self.model.load_state_dict(state_dict, strict=True)
                logger.info(f"Loaded model weights from {checkpoint_path}")
            else:
                message = f"No trained checkpoint found at {MODEL_PATH}"
                if not ALLOW_UNTRAINED_MODEL:
                    raise FileNotFoundError(message)
                logger.warning("%s. Using uninitialised model (development only).", message)

            self.model.eval()

        except Exception as e:
            logger.exception("Failed to load model; inference will return empty results.")
            self.model = None
            if not ALLOW_UNTRAINED_MODEL:
                raise

    def _preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """Convert PNG/JPEG bytes → (3, H, W) float tensor in [0,1]."""
        from ..utils.preprocessing import preprocess_uploaded_image
        array = preprocess_uploaded_image(image_bytes, patch_size=PATCH_SIZE)
        return torch.from_numpy(array)

    def _postprocess(
        self, outputs: list[dict], original_size: Optional[tuple] = None
    ) -> list[NoduleDetection]:
        """Convert raw model output to NoduleDetection list."""
        detections = []
        if not outputs:
            return detections
        out = outputs[0]

        boxes = out.get("boxes", torch.zeros(0, 4)).cpu().numpy()
        scores = out.get("scores", torch.zeros(0)).cpu().numpy()
        masks = out.get("masks", torch.zeros(0, 1, 1, 1)).cpu().numpy()  # (N, 1, H, W)

        for i, (box, score) in enumerate(zip(boxes, scores)):
            if score < SCORE_THRESHOLD:
                continue
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            # Estimate diameter from bounding box (in voxels → mm)
            width_px = x2 - x1
            height_px = y2 - y1
            diameter_px = (width_px + height_px) / 2.0
            diameter_mm = diameter_px * DEFAULT_SPACING_MM

            # RLE encode mask
            if i < len(masks):
                mask_arr = (masks[i, 0] > 0.5).astype(np.uint8)
                rle = _rle_encode(mask_arr)
            else:
                rle = None

            detections.append(NoduleDetection(
                box=[x1, y1, x2, y2],
                score=float(score),
                mask_rle=rle,
                diameter_estimate_mm=round(diameter_mm, 2),
            ))

        return detections

    @torch.no_grad()
    def predict_from_bytes(
        self, image_bytes: bytes, experiment_id: Optional[int] = None
    ) -> list[NoduleDetection]:
        """Full cloud inference from raw image bytes."""
        if self.model is None:
            return []
        img_tensor = self._preprocess_image(image_bytes).to(DEVICE)
        outputs = self.model([img_tensor])
        return self._postprocess(outputs)

    @torch.no_grad()
    def predict_from_features(
        self, features: dict, scan_uid: str
    ) -> list[NoduleDetection]:
        """
        Continue inference from intermediate feature maps (edge-cloud split).
        Features are the output of edge's early layers.
        """
        if self.model is None:
            return []
        # Move features to device
        features_device = {
            k: (v.dequantize().to(DEVICE) if v.is_quantized else v.to(DEVICE))
            if isinstance(v, torch.Tensor) else v
            for k, v in features.items()
        }
        try:
            # Pass through remaining model layers (FPN + RPN + RoI + Mask head)
            # This requires the model to support partial forward — handled via backbone hook
            backbone = self.model.model.backbone
            # Inject pre-computed features into FPN
            dummy_img = torch.zeros(1, 3, PATCH_SIZE, PATCH_SIZE, device=DEVICE)
            image_list, _ = self.model.model.transform([dummy_img], None)
            # Directly pass features dict to RPN
            proposals, _ = self.model.model.rpn(image_list, features_device, None)
            detections, _ = self.model.model.roi_heads(
                features_device, proposals, image_list.image_sizes, None
            )
            outputs = self.model.model.transform.postprocess(detections, image_list.image_sizes, [(PATCH_SIZE, PATCH_SIZE)])
            return self._postprocess(outputs)
        except Exception as e:
            logger.exception("Feature-based inference failed")
            return []

    def load_checkpoint(self, checkpoint_path: str):
        """Hot-reload a new checkpoint."""
        if self.model is None:
            return
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        state_dict = _checkpoint_state_dict(ckpt)
        self.model.load_state_dict(state_dict, strict=True)
        logger.info(f"Reloaded checkpoint from {checkpoint_path}")


def _rle_encode(mask: np.ndarray) -> str:
    """Simple run-length encoding of a 2D binary mask → string."""
    flat = mask.flatten()
    runs = []
    pos = 0
    while pos < len(flat):
        val = flat[pos]
        run_start = pos
        while pos < len(flat) and flat[pos] == val:
            pos += 1
        runs.append(f"{int(val)}:{pos - run_start}")
    return ",".join(runs)


def _rle_decode(rle: str, shape: tuple) -> np.ndarray:
    """Decode RLE string back to 2D binary mask."""
    flat = []
    for token in rle.split(","):
        val, count = token.split(":")
        flat.extend([int(val)] * int(count))
    return np.array(flat, dtype=np.uint8).reshape(shape)
