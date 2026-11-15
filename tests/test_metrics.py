"""
Tests for all segmentation metrics.
Run: pytest tests/test_metrics.py -v
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
from training.metrics import (
    dice_score, iou_score, sensitivity, specificity,
    hausdorff_distance_95, average_surface_distance,
    boundary_f1, volume_error_percent, evaluate_batch,
    compute_froc, MetricTracker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def square_mask(size=32, inner=10, offset=5):
    """Create a binary mask with a square blob."""
    m = np.zeros((size, size), dtype=np.uint8)
    m[offset:offset+inner, offset:offset+inner] = 1
    return m


# ---------------------------------------------------------------------------
# Dice Score
# ---------------------------------------------------------------------------
class TestDiceScore:
    def test_perfect_match(self):
        m = square_mask()
        assert dice_score(m, m) == pytest.approx(1.0, abs=1e-5)

    def test_no_overlap(self):
        pred = square_mask(offset=0, inner=5)
        gt   = square_mask(offset=20, inner=5)
        assert dice_score(pred, gt) == pytest.approx(0.0, abs=1e-5)

    def test_range(self):
        pred = square_mask(offset=5, inner=10)
        gt   = square_mask(offset=7, inner=10)
        d = dice_score(pred, gt)
        assert 0.0 <= d <= 1.0

    def test_symmetry(self):
        pred = square_mask(offset=3, inner=8)
        gt   = square_mask(offset=7, inner=8)
        assert dice_score(pred, gt) == pytest.approx(dice_score(gt, pred), abs=1e-6)

    def test_empty_both(self):
        empty = np.zeros((32, 32), dtype=np.uint8)
        d = dice_score(empty, empty)
        assert 0.0 <= d <= 1.0   # Smooth avoids division by zero

    def test_full_mask(self):
        full = np.ones((16, 16), dtype=np.uint8)
        assert dice_score(full, full) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------
class TestIoU:
    def test_perfect(self):
        m = square_mask()
        assert iou_score(m, m) == pytest.approx(1.0, abs=1e-5)

    def test_no_overlap(self):
        p = square_mask(offset=0, inner=4)
        g = square_mask(offset=20, inner=4)
        assert iou_score(p, g) == pytest.approx(0.0, abs=1e-5)

    def test_iou_less_equal_dice(self):
        p = square_mask(offset=5, inner=8)
        g = square_mask(offset=7, inner=8)
        assert iou_score(p, g) <= dice_score(p, g) + 1e-6


# ---------------------------------------------------------------------------
# Sensitivity / Specificity
# ---------------------------------------------------------------------------
class TestSensSpec:
    def test_all_correct_sensitivity_1(self):
        m = square_mask()
        assert sensitivity(m, m) == pytest.approx(1.0, abs=1e-5)
        assert specificity(m, m) == pytest.approx(1.0, abs=1e-5)

    def test_all_zero_pred_sensitivity_0(self):
        pred = np.zeros((32, 32), dtype=np.uint8)
        gt   = square_mask()
        assert sensitivity(pred, gt) == pytest.approx(0.0, abs=1e-5)

    def test_all_one_pred_specificity_0(self):
        pred = np.ones((32, 32), dtype=np.uint8)
        gt   = square_mask()
        assert specificity(pred, gt) == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Hausdorff Distance
# ---------------------------------------------------------------------------
class TestHausdorff:
    def test_perfect_match_zero(self):
        m = square_mask(inner=10)
        hd = hausdorff_distance_95(m, m)
        assert hd == pytest.approx(0.0, abs=1e-5)

    def test_shifted_mask_positive(self):
        p = square_mask(offset=5,  inner=10)
        g = square_mask(offset=10, inner=10)
        hd = hausdorff_distance_95(p, g)
        assert hd > 0.0

    def test_empty_pred_inf(self):
        pred = np.zeros((32, 32), dtype=np.uint8)
        gt   = square_mask()
        hd = hausdorff_distance_95(pred, gt)
        expected = float(np.linalg.norm(np.array([31.0, 31.0])))
        assert hd == pytest.approx(expected, abs=1e-4)

    def test_larger_shift_larger_hd(self):
        base = square_mask(offset=5, inner=8)
        near = square_mask(offset=7, inner=8)
        far  = square_mask(offset=15, inner=8)
        hd_near = hausdorff_distance_95(base, near)
        hd_far  = hausdorff_distance_95(base, far)
        assert hd_far >= hd_near


# ---------------------------------------------------------------------------
# Boundary F1
# ---------------------------------------------------------------------------
class TestBoundaryF1:
    def test_perfect_match_one(self):
        m = square_mask()
        assert boundary_f1(m, m, tolerance=1) == pytest.approx(1.0, abs=0.05)

    def test_both_empty(self):
        empty = np.zeros((16, 16), dtype=np.uint8)
        assert boundary_f1(empty, empty) == 1.0

    def test_one_empty(self):
        empty = np.zeros((16, 16), dtype=np.uint8)
        m = square_mask(size=16)
        assert boundary_f1(empty, m) == 0.0

    def test_range(self):
        p = square_mask(offset=5, inner=8)
        g = square_mask(offset=6, inner=8)
        bf1 = boundary_f1(p, g)
        assert 0.0 <= bf1 <= 1.0


# ---------------------------------------------------------------------------
# Volume Error
# ---------------------------------------------------------------------------
class TestVolumeError:
    def test_perfect_zero(self):
        m = square_mask()
        assert volume_error_percent(m, m) == pytest.approx(0.0, abs=1e-5)

    def test_empty_pred_on_nonempty_gt(self):
        pred = np.zeros((32, 32), dtype=np.uint8)
        gt   = square_mask()
        err = volume_error_percent(pred, gt)
        assert err == pytest.approx(100.0, abs=1e-5)

    def test_double_volume(self):
        gt = np.zeros((32, 32), dtype=np.uint8)
        gt[5:15, 5:15] = 1
        pred = np.zeros((32, 32), dtype=np.uint8)
        pred[5:15, 5:20] = 1   # 1.5× the volume
        err = volume_error_percent(pred, gt)
        assert err == pytest.approx(50.0, abs=1.0)


# ---------------------------------------------------------------------------
# Batch Evaluation
# ---------------------------------------------------------------------------
class TestEvaluateBatch:
    def test_shape_and_keys(self):
        preds   = np.random.rand(4, 32, 32).astype(np.float32)
        targets = (np.random.rand(4, 32, 32) > 0.7).astype(np.uint8)
        result  = evaluate_batch(preds, targets)
        for key in ["dice","iou","sensitivity","specificity","hd95","boundary_f1","asd","vol_err"]:
            assert key in result, f"Missing metric: {key}"
            assert isinstance(result[key], float)

    def test_perfect_preds(self):
        targets = (np.random.rand(2, 16, 16) > 0.5).astype(np.float32)
        preds   = targets.copy()
        result  = evaluate_batch(preds, targets, threshold=0.5)
        assert result["dice"] == pytest.approx(1.0, abs=0.05)
        assert result["iou"]  == pytest.approx(1.0, abs=0.05)

    def test_negative_patch_false_positive_metrics_are_labeled_as_proxies(self):
        targets = np.zeros((2, 8, 8), dtype=np.uint8)
        preds = np.zeros((2, 8, 8), dtype=np.float32)
        preds[0, 2:4, 2:4] = 1.0
        result = evaluate_batch(
            preds,
            targets,
            scan_uids=["scan-a", "scan-a"],
            false_positive_detections=[2, 0],
        )
        assert result["sampled_negative_patch_false_positive_rate"] == pytest.approx(0.5)
        assert result["sampled_patch_false_detections_per_scan_proxy"] == pytest.approx(2.0)
        assert "false_positives_per_scan" not in result


# ---------------------------------------------------------------------------
# FROC
# ---------------------------------------------------------------------------
class TestFROC:
    def test_basic_structure(self):
        probs  = [0.9, 0.8, 0.3, 0.1, 0.95]
        labels = [1,   0,   0,   0,   1  ]
        result = compute_froc(probs, labels, n_scans=5)
        assert "froc_score" in result
        assert 0.0 <= result["froc_score"] <= 1.0
        assert len(result["fps_per_scan"]) == len(result["sensitivities"])

    def test_perfect_detector(self):
        probs  = [1.0] * 5 + [0.0] * 20
        labels = [1]   * 5 + [0]   * 20
        result = compute_froc(probs, labels, n_scans=5)
        assert result["froc_score"] > 0.8   # Should be high

    def test_all_negative_preds(self):
        probs  = [0.0] * 10
        labels = [1, 0, 1, 0, 1, 0, 1, 0, 0, 0]
        result = compute_froc(probs, labels, n_scans=3)
        assert isinstance(result["froc_score"], float)


# ---------------------------------------------------------------------------
# MetricTracker
# ---------------------------------------------------------------------------
class TestMetricTracker:
    def test_running_average(self):
        t = MetricTracker()
        t.update({"loss": 1.0})
        t.update({"loss": 3.0})
        avg = t.averages()
        assert avg["loss"] == pytest.approx(2.0)

    def test_ignores_nan(self):
        t = MetricTracker()
        t.update({"loss": 1.0})
        t.update({"loss": float("nan")})
        avg = t.averages()
        assert avg["loss"] == pytest.approx(1.0)

    def test_reset_clears(self):
        t = MetricTracker()
        t.update({"loss": 5.0})
        t.reset()
        assert t.averages() == {}

    def test_multiple_metrics(self):
        t = MetricTracker()
        t.update({"loss": 2.0, "dice": 0.8})
        t.update({"loss": 4.0, "dice": 0.6})
        avg = t.averages()
        assert avg["loss"] == pytest.approx(3.0)
        assert avg["dice"] == pytest.approx(0.7)
