"""
Evaluation metrics for lung nodule segmentation.
  - Dice Score
  - IoU / Jaccard
  - 95th-percentile Hausdorff Distance
  - Boundary F1
  - Average Surface Distance
  - Volume Error %
  - FROC (Free-Response ROC)
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import torch


EPS = 1e-7


# ---------------------------------------------------------------------------
# Core binary metrics (batch-aware)
# ---------------------------------------------------------------------------

def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """
    pred, target: binary 2D/3D numpy arrays.
    Returns Dice in [0, 1].
    """
    pred_f = pred.astype(bool).flatten()
    tgt_f = target.astype(bool).flatten()
    intersection = (pred_f & tgt_f).sum()
    return (2.0 * intersection + EPS) / (pred_f.sum() + tgt_f.sum() + EPS)


def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred_f = pred.astype(bool).flatten()
    tgt_f = target.astype(bool).flatten()
    intersection = (pred_f & tgt_f).sum()
    union = (pred_f | tgt_f).sum()
    return (intersection + EPS) / (union + EPS)


def sensitivity(pred: np.ndarray, target: np.ndarray) -> float:
    tp = (pred.astype(bool) & target.astype(bool)).sum()
    fn = (~pred.astype(bool) & target.astype(bool)).sum()
    return float(tp) / float(tp + fn + EPS)


def specificity(pred: np.ndarray, target: np.ndarray) -> float:
    tn = (~pred.astype(bool) & ~target.astype(bool)).sum()
    fp = (pred.astype(bool) & ~target.astype(bool)).sum()
    return float(tn) / float(tn + fp + EPS)


# ---------------------------------------------------------------------------
# Surface / boundary metrics
# ---------------------------------------------------------------------------

def _surface_distances(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute surface distances between binary masks.
    Returns (distances from pred surface to target, distances from target surface to pred).
    """
    pred_b = pred.astype(bool)
    tgt_b = target.astype(bool)

    if not pred_b.any() and not tgt_b.any():
        zeros = np.array([0.0])
        return zeros, zeros
    if not pred_b.any() or not tgt_b.any():
        nan = np.array([np.nan])
        return nan, nan

    dt_pred = distance_transform_edt(~pred_b)
    dt_tgt = distance_transform_edt(~tgt_b)

    # Surface = boundary pixels
    pred_surface = pred_b ^ _erode(pred_b)
    tgt_surface = tgt_b ^ _erode(tgt_b)

    dist_pred_to_tgt = dt_tgt[pred_surface]
    dist_tgt_to_pred = dt_pred[tgt_surface]
    return dist_pred_to_tgt, dist_tgt_to_pred


def _empty_mask_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """Use 0 for two empty masks and the image diagonal for one empty mask."""
    pred_empty = not np.any(pred)
    target_empty = not np.any(target)
    if pred_empty and target_empty:
        return 0.0
    extents = np.maximum(np.asarray(pred.shape, dtype=float) - 1.0, 0.0)
    return float(np.linalg.norm(extents))


def _erode(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion
    return binary_erosion(mask)


def hausdorff_distance_95(pred: np.ndarray, target: np.ndarray) -> float:
    """95th-percentile Hausdorff distance in voxels.

    Empty-empty masks score 0; a single empty mask receives the image-diagonal
    penalty so aggregate metrics remain finite and JSON-safe.
    """
    if not np.any(pred) or not np.any(target):
        return _empty_mask_distance(pred, target)
    d1, d2 = _surface_distances(pred, target)
    if np.isnan(d1).all() or np.isnan(d2).all():
        return _empty_mask_distance(pred, target)
    combined = np.concatenate([d1, d2])
    combined = combined[~np.isnan(combined)]
    return float(np.percentile(combined, 95)) if len(combined) > 0 else _empty_mask_distance(pred, target)


def average_surface_distance(pred: np.ndarray, target: np.ndarray) -> float:
    if not np.any(pred) or not np.any(target):
        return _empty_mask_distance(pred, target)
    d1, d2 = _surface_distances(pred, target)
    if np.isnan(d1).all() or np.isnan(d2).all():
        return _empty_mask_distance(pred, target)
    combined = np.concatenate([d1, d2])
    combined = combined[~np.isnan(combined)]
    return float(combined.mean()) if len(combined) > 0 else _empty_mask_distance(pred, target)


def boundary_f1(pred: np.ndarray, target: np.ndarray, tolerance: int = 2) -> float:
    """
    Boundary F1 score: precision/recall on boundary pixels within tolerance px.
    """
    pred_b = pred.astype(bool)
    tgt_b = target.astype(bool)

    if pred_b.sum() == 0 and tgt_b.sum() == 0:
        return 1.0
    if pred_b.sum() == 0 or tgt_b.sum() == 0:
        return 0.0

    pred_surf = pred_b ^ _erode(pred_b)
    tgt_surf = tgt_b ^ _erode(tgt_b)

    dt_tgt = distance_transform_edt(~tgt_surf)
    dt_pred = distance_transform_edt(~pred_surf)

    prec = float((dt_tgt[pred_surf] <= tolerance).mean()) if pred_surf.sum() > 0 else 0.0
    rec = float((dt_pred[tgt_surf] <= tolerance).mean()) if tgt_surf.sum() > 0 else 0.0

    if prec + rec < EPS:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def volume_error_percent(pred: np.ndarray, target: np.ndarray) -> float:
    """Absolute relative volume error, defined for positive target masks."""
    pred_vol = pred.astype(bool).sum()
    tgt_vol = target.astype(bool).sum()
    if tgt_vol == 0:
        return 0.0 if pred_vol == 0 else 100.0
    return float(abs(pred_vol - tgt_vol) / tgt_vol) * 100.0


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def _safe_mean(values: list[float]) -> float | None:
    """Average finite metric values; return None when none are applicable."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else None

def evaluate_batch(
    preds: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
    scan_uids: list[str] | None = None,
    false_positive_detections: list[int] | None = None,
) -> dict[str, float | None]:
    """
    Evaluate a batch of predictions.
    preds:   (N, H, W) sigmoid probabilities [0,1]
    targets: (N, H, W) binary masks
    scan_uids: optional scan UID per patch, for scan-normalized false positives
    false_positive_detections: optional count of predicted instances per patch
    """
    if scan_uids is not None and len(scan_uids) != len(preds):
        raise ValueError("scan_uids must contain one UID per prediction")
    if (
        false_positive_detections is not None
        and len(false_positive_detections) != len(preds)
    ):
        raise ValueError("false_positive_detections must contain one count per prediction")

    pred_bin = (preds > threshold).astype(np.uint8)
    metrics: dict[str, list] = {
        "dice": [], "positive_dice": [], "iou": [],
        "sensitivity": [], "specificity": [],
        "hd95": [], "boundary_f1": [], "asd": [], "vol_err": [],
        "negative_patch_false_positive_rate": [],
    }
    false_positives_by_scan: dict[str, int] = {}

    for i in range(len(preds)):
        p, t = pred_bin[i], targets[i]
        has_gt = bool(np.any(t))
        # Keep foreground overlap metrics on positive patches and separately
        # report false-positive behavior on negative patches.
        metrics["specificity"].append(specificity(p, t))
        if has_gt:
            positive_dice = dice_score(p, t)
            metrics["dice"].append(positive_dice)
            metrics["positive_dice"].append(positive_dice)
            metrics["iou"].append(iou_score(p, t))
            metrics["sensitivity"].append(sensitivity(p, t))
            metrics["hd95"].append(hausdorff_distance_95(p, t))
            metrics["boundary_f1"].append(boundary_f1(p, t))
            metrics["asd"].append(average_surface_distance(p, t))
            metrics["vol_err"].append(volume_error_percent(p, t))
        else:
            uid = scan_uids[i] if scan_uids is not None else None
            detection_count = (
                int(false_positive_detections[i])
                if false_positive_detections is not None
                else int(np.any(p))
            )
            metrics["negative_patch_false_positive_rate"].append(
                float(detection_count > 0)
            )
            if uid is not None:
                false_positives_by_scan[uid] = (
                    false_positives_by_scan.get(uid, 0) + detection_count
                )

    result = {key: _safe_mean(values) for key, values in metrics.items()}
    result["false_positives_per_scan"] = (
        float(sum(false_positives_by_scan.values()) / len(set(scan_uids)))
        if scan_uids is not None
        and scan_uids
        and metrics["negative_patch_false_positive_rate"]
        else None
    )
    return result


# ---------------------------------------------------------------------------
# FROC curve
# ---------------------------------------------------------------------------

def compute_froc(
    all_probs: list[float],
    all_labels: list[int],
    n_scans: int,
    thresholds: np.ndarray = None,
) -> dict:
    """
    Compute FROC (Free-Response ROC) curve.
    all_probs:  list of detection probability scores
    all_labels: list of 0/1 (1 = true nodule)
    n_scans:    total number of scans (for FP/scan normalisation)

    Returns dict with 'fps_per_scan', 'sensitivities', 'froc_score'
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)

    probs = np.array(all_probs)
    labels = np.array(all_labels)
    n_positives = labels.sum()

    fps_per_scan_list = []
    sensitivities = []

    for thr in thresholds:
        pred_pos = probs >= thr
        tp = (pred_pos & (labels == 1)).sum()
        fp = (pred_pos & (labels == 0)).sum()
        fps_per_scan_list.append(fp / max(n_scans, 1))
        sensitivities.append(tp / max(n_positives, 1))

    # FROC score = average sensitivity at [0.125, 0.25, 0.5, 1, 2, 4, 8] FPs/scan
    froc_points = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    fps_arr = np.asarray(fps_per_scan_list, dtype=float)
    sens_arr = np.asarray(sensitivities, dtype=float)
    sort_idx = np.argsort(fps_arr, kind="stable")
    fps_arr = fps_arr[sort_idx]
    sens_arr = sens_arr[sort_idx]
    # Collapse duplicate FP rates to their best sensitivity, then interpolate
    # on the increasing FP axis as required by the FROC operating points.
    unique_fps = np.unique(fps_arr)
    unique_sens = np.asarray([
        sens_arr[fps_arr == fp].max() for fp in unique_fps
    ])
    froc_sens = np.interp(
        np.asarray(froc_points), unique_fps, unique_sens,
        left=float(unique_sens[0]), right=float(unique_sens[-1]),
    ).tolist()

    return {
        "fps_per_scan": fps_per_scan_list,
        "sensitivities": sensitivities,
        "froc_score": float(np.mean(froc_sens)),
        "froc_per_point": dict(zip(froc_points, froc_sens)),
        "thresholds": thresholds.tolist(),
    }


# ---------------------------------------------------------------------------
# Metric tracker for training
# ---------------------------------------------------------------------------

class MetricTracker:
    """Running averages for training metrics."""

    def __init__(self):
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def update(self, metrics: dict[str, Union[float, "torch.Tensor"]]):
        for k, v in metrics.items():
            val = float(v) if hasattr(v, "item") else float(v)
            if not np.isfinite(val):
                continue
            self._totals[k] = self._totals.get(k, 0.0) + val
            self._counts[k] = self._counts.get(k, 0) + 1

    def averages(self) -> dict[str, float]:
        return {
            k: self._totals[k] / self._counts[k]
            for k in self._totals if self._counts[k] > 0
        }

    def reset(self):
        self._totals.clear()
        self._counts.clear()
