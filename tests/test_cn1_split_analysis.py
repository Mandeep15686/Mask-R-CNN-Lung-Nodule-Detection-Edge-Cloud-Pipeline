import pytest
import torch

from scripts.cn1_split_analysis import compute_total_latency, profile_split_point


class _FullModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, images):
        self.calls += 1
        return []


class _Extractor:
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

    def extract(self, image):
        return {"layer1": torch.ones((1, 2, 2, 2), device=image.device)}


class _Continuation:
    def __init__(self):
        self.calls = []

    def predict_from_edge_features(self, features, split_layer, scan_uid, **metadata):
        self.calls.append((features, split_layer, scan_uid, metadata))
        return []


def test_split_profile_uses_cloud_continuation_not_full_model_forward():
    model = _FullModel()
    continuation = _Continuation()
    profile = profile_split_point(
        "layer1",
        model,
        _Extractor(),
        [torch.zeros((3, 8, 8))],
        inference_engine=continuation,
        checkpoint_sha256="checkpoint-hash",
        n_repeats=1,
    )

    assert model.calls == 0
    # One warm-up call plus the measured call; both use the real continuation.
    assert len(continuation.calls) == 2
    features, split_layer, scan_uid, metadata = continuation.calls[-1]
    assert features["layer1"].is_quantized
    assert split_layer == "layer1"
    assert scan_uid == "cn1-split-profile"
    assert metadata["expected_checkpoint_sha256"] == "checkpoint-hash"
    assert metadata["edge_image_mean"] == _Extractor.image_mean
    assert profile["cloud_path"] == "InferenceEngine.predict_from_edge_features"
    assert profile["avg_wire_payload_kb"] > 0


def test_split_profile_requires_the_real_cloud_continuation():
    with pytest.raises(ValueError, match="checkpoint-backed InferenceEngine"):
        profile_split_point(
            "layer2",
            _FullModel(),
            _Extractor(),
            [torch.zeros((3, 8, 8))],
            n_repeats=1,
        )


def test_total_latency_uses_wire_payload_and_cloud_only_has_no_transfer():
    condition = {"bandwidth_mbps": 10.0, "rtt_ms": 20.0}
    split = {
        "split_layer": "layer2",
        "avg_edge_ms": 5.0,
        "avg_cloud_ms": 7.0,
        "avg_feature_kb": 100.0,
        "avg_wire_payload_kb": 10.0,
    }
    result = compute_total_latency(split, condition)
    assert result["transmission_ms"] == pytest.approx(28.0)
    assert result["total_ms"] == pytest.approx(40.0)

    cloud_only = dict(split, split_layer="cloud_only")
    cloud_result = compute_total_latency(cloud_only, condition)
    assert cloud_result["transmission_ms"] == 0.0
    assert cloud_result["total_ms"] == pytest.approx(12.0)
