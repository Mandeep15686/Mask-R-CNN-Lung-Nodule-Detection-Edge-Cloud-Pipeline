"""
Tests for edge node pipeline: preprocessor, feature extractor, compressor.
Run: pytest tests/test_edge_pipeline.py -v
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../edge")))

import io
import pytest
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_png_bytes():
    """64×64 synthetic CT patch as PNG bytes."""
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[20:44, 20:44] = 160    # simulated nodule blob
    arr += np.random.randint(0, 30, arr.shape, dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_grey_bytes():
    """64×64 single-channel CT patch as PNG."""
    arr = (np.random.rand(64, 64) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CTPreprocessor tests
# ---------------------------------------------------------------------------
class TestCTPreprocessor:
    def test_rgb_input_shape(self, sample_png_bytes):
        from preprocessor import CTPreprocessor
        p = CTPreprocessor(patch_size=64)
        t = p.preprocess_bytes(sample_png_bytes)
        assert t.shape == (3, 64, 64), f"Expected (3,64,64), got {t.shape}"

    def test_grayscale_input_is_rejected(self, sample_grey_bytes):
        from preprocessor import CTPreprocessor
        p = CTPreprocessor(patch_size=64)
        with pytest.raises(ValueError, match="Expected an RGB 2.5D patch"):
            p.preprocess_bytes(sample_grey_bytes)

    def test_output_range(self, sample_png_bytes):
        from preprocessor import CTPreprocessor
        import torch
        p = CTPreprocessor(patch_size=64)
        t = p.preprocess_bytes(sample_png_bytes)
        assert float(t.min()) >= 0.0
        assert float(t.max()) <= 1.0 + 1e-5

    def test_wrong_dimensions_are_rejected_without_resizing(self, sample_png_bytes):
        from preprocessor import CTPreprocessor
        p = CTPreprocessor(patch_size=32)
        with pytest.raises(ValueError, match="input is not resized"):
            p.preprocess_bytes(sample_png_bytes)

    def test_edge_preprocessing_matches_cloud_pixel_mapping(self, sample_png_bytes):
        import torch
        from preprocessor import CTPreprocessor
        from backend.app.utils.preprocessing import preprocess_uploaded_image

        edge = CTPreprocessor(patch_size=64).preprocess_bytes(sample_png_bytes)
        cloud = torch.from_numpy(preprocess_uploaded_image(sample_png_bytes))
        torch.testing.assert_close(edge, cloud)

    def test_numpy_input(self):
        from preprocessor import CTPreprocessor
        p = CTPreprocessor(patch_size=64)
        arr = np.random.rand(3, 64, 64).astype(np.float32)
        t = p.preprocess_numpy(arr)
        assert t.shape == (3, 64, 64)

    def test_numpy_2d_input(self):
        from preprocessor import CTPreprocessor
        p = CTPreprocessor(patch_size=64)
        arr = np.random.rand(64, 64).astype(np.float32)
        t = p.preprocess_numpy(arr)
        assert t.shape == (3, 64, 64)


# ---------------------------------------------------------------------------
# Backend preprocessing utils tests
# ---------------------------------------------------------------------------
class TestBackendPreprocessing:
    def test_normalise_hu_range(self):
        from backend.app.utils.preprocessing import normalise_hu
        arr = np.array([-1000.0, 0.0, 400.0])
        out = normalise_hu(arr)
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-5

    def test_normalise_already_01(self):
        from backend.app.utils.preprocessing import normalise_hu
        arr = np.array([0.0, 0.5, 1.0])
        out = normalise_hu(arr)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_to_three_channel_hw(self):
        from backend.app.utils.preprocessing import to_three_channel
        arr = np.random.rand(32, 32).astype(np.float32)
        out = to_three_channel(arr)
        assert out.shape == (3, 32, 32)
        # All three channels identical
        np.testing.assert_array_almost_equal(out[0], out[1])
        np.testing.assert_array_almost_equal(out[1], out[2])

    def test_to_three_channel_hwc(self):
        from backend.app.utils.preprocessing import to_three_channel
        arr = np.random.rand(32, 32, 3).astype(np.float32)
        out = to_three_channel(arr)
        assert out.shape == (3, 32, 32)

    def test_to_three_channel_chw(self):
        from backend.app.utils.preprocessing import to_three_channel
        arr = np.random.rand(3, 32, 32).astype(np.float32)
        out = to_three_channel(arr)
        assert out.shape == (3, 32, 32)

    def test_resize_patch(self):
        from backend.app.utils.preprocessing import resize_patch
        arr = np.random.rand(3, 128, 128).astype(np.float32)
        out = resize_patch(arr, target_size=64)
        assert out.shape == (3, 64, 64)

    def test_resize_already_correct(self):
        from backend.app.utils.preprocessing import resize_patch
        arr = np.random.rand(3, 64, 64).astype(np.float32)
        out = resize_patch(arr, target_size=64)
        assert out.shape == (3, 64, 64)

    def test_preprocess_uploaded_image(self, sample_png_bytes):
        from backend.app.utils.preprocessing import preprocess_uploaded_image
        out = preprocess_uploaded_image(sample_png_bytes, patch_size=64)
        assert out.shape == (3, 64, 64)
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-5

    def test_preprocess_bad_bytes_raises(self):
        from backend.app.utils.preprocessing import preprocess_uploaded_image
        with pytest.raises(ValueError):
            preprocess_uploaded_image(b"not_an_image_at_all")

    def test_box_from_mask_square(self):
        from backend.app.utils.preprocessing import box_from_mask
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:20, 8:18] = 1
        box = box_from_mask(mask)
        assert box is not None
        assert box == [8.0, 10.0, 18.0, 20.0]

    def test_box_from_mask_empty_returns_none(self):
        from backend.app.utils.preprocessing import box_from_mask
        mask = np.zeros((32, 32), dtype=np.uint8)
        assert box_from_mask(mask) is None

    def test_estimate_diameter_mm(self):
        from backend.app.utils.preprocessing import estimate_diameter_mm
        # 10px × 10px box at 0.7mm/px → ~7mm diameter
        box = [5.0, 5.0, 15.0, 15.0]
        d = estimate_diameter_mm(box, spacing_mm=0.7)
        assert abs(d - 7.0) < 0.5

    def test_compute_iou_2d_perfect(self):
        from backend.app.utils.preprocessing import compute_iou_2d
        b = [0.0, 0.0, 10.0, 10.0]
        assert compute_iou_2d(b, b) == pytest.approx(1.0)

    def test_compute_iou_2d_no_overlap(self):
        from backend.app.utils.preprocessing import compute_iou_2d
        b1 = [0.0, 0.0, 5.0, 5.0]
        b2 = [10.0, 10.0, 15.0, 15.0]
        assert compute_iou_2d(b1, b2) == pytest.approx(0.0)

    def test_compute_iou_2d_partial(self):
        from backend.app.utils.preprocessing import compute_iou_2d
        b1 = [0.0, 0.0, 10.0, 10.0]
        b2 = [5.0, 5.0, 15.0, 15.0]
        iou = compute_iou_2d(b1, b2)
        assert 0.0 < iou < 1.0
        # Overlap is 5×5=25, union is 100+100-25=175
        assert abs(iou - 25/175) < 1e-5


# ---------------------------------------------------------------------------
# DP noise tests
# ---------------------------------------------------------------------------
class TestDifferentialPrivacy:
    def _dummy_features(self):
        try:
            import torch
            return {"layer2": torch.zeros(1, 512, 8, 8)}
        except ImportError:
            return {"layer2": np.zeros((1, 512, 8, 8))}

    def test_noise_changes_values(self):
        from scripts.cn5_differential_privacy import add_gaussian_dp_noise
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        features = {"layer2": torch.zeros(1, 64, 8, 8)}
        noised = add_gaussian_dp_noise(features, epsilon=1.0)
        # With ε=1.0, noise should be non-zero
        assert not (noised["layer2"] == 0).all()

    def test_no_noise_at_inf_epsilon(self):
        """σ(ε=∞) = 0 → features unchanged."""
        from scripts.cn5_differential_privacy import add_gaussian_dp_noise
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        features = {"layer2": torch.ones(1, 64, 8, 8) * 5.0}
        # Mimic ε=∞ by passing extremely large value
        noised = add_gaussian_dp_noise(features, epsilon=1e9)
        # std should be ~1e-9, effectively zero
        diff = (noised["layer2"] - features["layer2"]).abs().max().item()
        assert diff < 1e-3

    def test_larger_epsilon_less_noise(self):
        """Higher ε (weaker privacy) → lower σ → less noise."""
        from scripts.cn5_differential_privacy import add_gaussian_dp_noise
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        torch.manual_seed(0)
        base = torch.zeros(1, 256, 8, 8)
        features1 = {"layer2": base.clone()}
        features2 = {"layer2": base.clone()}
        noised_low  = add_gaussian_dp_noise(features1, epsilon=0.1)   # strong privacy
        noised_high = add_gaussian_dp_noise(features2, epsilon=10.0)  # weak privacy
        std_low  = noised_low["layer2"].std().item()
        std_high = noised_high["layer2"].std().item()
        assert std_low > std_high


# ---------------------------------------------------------------------------
# Network monitor tests (pure-python, no network calls)
# ---------------------------------------------------------------------------
def test_v2_feature_serializer_preserves_int8_bytes():
    import json
    import torch
    from edge_node import _serialize_feature_payload

    source = torch.linspace(-1.0, 1.0, steps=128).reshape(1, 2, 8, 8)
    quantized = torch.quantize_per_tensor(source, scale=1 / 64, zero_point=0, dtype=torch.qint8)
    payload = _serialize_feature_payload({"layer1": quantized}, {"split_layer": "layer1"})
    header_bytes, raw = payload.split(b"\x00", 1)
    header = json.loads(header_bytes)

    assert header["version"] == 2
    assert header["dtypes"] == ["int8"]
    assert header["scales"] == [pytest.approx(1 / 64)]
    assert header["zero_points"] == [0]
    assert len(raw) == quantized.numel()
    restored_int8 = torch.frombuffer(bytearray(raw), dtype=torch.int8).reshape(quantized.shape)
    restored = torch._make_per_tensor_quantized_tensor(
        restored_int8, header["scales"][0], header["zero_points"][0]
    )
    assert torch.equal(restored.int_repr(), quantized.int_repr())


def test_wire_compression_ratio_uses_complete_serialized_body():
    import torch
    from edge_node import _serialize_feature_payload, _wire_compression_ratio

    source = torch.ones(1, 8, 4, 4)
    quantized = torch.quantize_per_tensor(source, scale=0.1, zero_point=0, dtype=torch.qint8)
    payload = _serialize_feature_payload({"layer1": quantized}, {"scan_uid": "test"})

    assert _wire_compression_ratio({"layer1": source}, payload) == pytest.approx(
        source.numel() * 4 / len(payload)
    )
    assert len(payload) > quantized.numel()  # header bytes are included


def test_live_edge_node_disables_unvalidated_dp_and_unwired_learned_compression(monkeypatch):
    import edge_node

    monkeypatch.setattr(edge_node, "DP_EPSILON", 0.5)
    with pytest.raises(ValueError, match="Live DP feature release is disabled"):
        edge_node.EdgeNode()

    monkeypatch.setattr(edge_node, "DP_EPSILON", 0.0)
    monkeypatch.setattr(edge_node, "COMPRESSION", "learned")
    with pytest.raises(ValueError, match="cloud decoder.*not implemented"):
        edge_node.EdgeNode()


@pytest.mark.parametrize(
    ("split_layer", "expected_keys"),
    [
        ("layer1", {"layer1"}),
        ("layer2", {"layer1", "layer2"}),
        ("layer3", {"layer1", "layer2", "layer3"}),
    ],
)
def test_feature_extractor_returns_all_levels_through_split(split_layer, expected_keys):
    import torch
    from feature_extractor import EdgeFeatureExtractor

    extractor = EdgeFeatureExtractor(split_layer=split_layer, pretrained=False)
    features = extractor.extract(torch.rand(1, 3, 64, 64))
    assert set(features) == expected_keys


class TestNetworkMonitor:
    def test_should_use_cloud_default(self):
        from network_monitor import NetworkMonitor
        mon = NetworkMonitor(cloud_url="http://localhost:8000")
        # Default state: last_bw=100, last_latency=10 → should use cloud
        assert mon.should_use_cloud() is True

    def test_should_not_use_cloud_high_latency(self):
        from network_monitor import NetworkMonitor
        mon = NetworkMonitor(cloud_url="http://localhost:8000")
        mon._last_latency = 9999.0   # Simulate network down
        assert mon.should_use_cloud() is False

    def test_should_not_use_cloud_low_bandwidth(self):
        from network_monitor import NetworkMonitor
        mon = NetworkMonitor(cloud_url="http://localhost:8000")
        mon._last_bw = 0.1   # 0.1 Mbps — below threshold
        assert mon.should_use_cloud() is False

    def test_stats_empty(self):
        from network_monitor import NetworkMonitor
        mon = NetworkMonitor(cloud_url="http://x")
        stats = mon.stats()
        assert stats == {}

    def test_stats_after_manual_history(self):
        from network_monitor import NetworkMonitor
        mon = NetworkMonitor(cloud_url="http://x")
        mon._history = [
            {"timestamp": 0, "bandwidth_mbps": 100.0, "latency_ms": 10.0},
            {"timestamp": 1, "bandwidth_mbps": 200.0, "latency_ms": 20.0},
        ]
        s = mon.stats()
        assert s["avg_bandwidth_mbps"] == pytest.approx(150.0)
        assert s["avg_latency_ms"]     == pytest.approx(15.0)
        assert s["n_probes"] == 2
