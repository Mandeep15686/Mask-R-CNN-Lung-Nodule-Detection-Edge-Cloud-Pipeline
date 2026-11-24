import pytest

from scripts.evaluate_model import evaluation_data_role_for_checkpoint


def test_two_subset_style_split_is_never_classified_as_independent_test():
    config = {
        "train_subsets": [1],
        "validation_subset": 0,
        "test_subset": None,
        "data_split": {
            "train_subsets": [1],
            "validation_subset": 0,
            "test_subset": None,
        },
    }

    assert evaluation_data_role_for_checkpoint(config, 0) == "no_independent_test_subset_configured"
    assert evaluation_data_role_for_checkpoint(config, 1) == "no_independent_test_subset_configured"


@pytest.mark.parametrize(
    "config",
    [
        {"train_subsets": [1], "validation_subset": 0, "test_subset": 1},
        {"data_split": {"train_subsets": [1], "validation_subset": 0, "test_subset": 0}},
    ],
)
def test_evaluator_rejects_reserved_test_subset_that_overlaps_train_or_validation(config):
    with pytest.raises(ValueError, match="overlaps the reserved test subset"):
        evaluation_data_role_for_checkpoint(config, config["test_subset"] if "test_subset" in config else 0)


def test_evaluator_accepts_only_the_disjoint_reserved_test_subset():
    config = {
        "data_split": {
            "train_subsets": [1],
            "validation_subset": 0,
            "test_subset": 2,
        }
    }

    assert evaluation_data_role_for_checkpoint(config, 2) == "reserved_test_subset"
    with pytest.raises(ValueError, match="reserves subset 2"):
        evaluation_data_role_for_checkpoint(config, 1)


def test_edge_cloud_failure_records_fallback_decision_and_probe(monkeypatch):
    import asyncio
    import sys
    from types import SimpleNamespace
    from pathlib import Path

    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "edge"))
    import edge_node

    class Monitor:
        async def measure(self):
            return 12.0, 45.0

        def should_use_cloud(self):
            return True

    node = object.__new__(edge_node.EdgeNode)
    node.preprocessor = SimpleNamespace(
        preprocess_bytes=lambda image, source: torch.zeros(3, 64, 64)
    )
    node.net_monitor = Monitor()
    node.extractor = SimpleNamespace(
        extract=lambda image: {"layer1": torch.ones(1, 1, 1, 1)},
        checkpoint_sha256="same-checkpoint",
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )
    node.compressor = SimpleNamespace(compress=lambda features: features)
    node.result_cache = {}
    events = []

    async def send_failure(payload, scan_uid):
        raise RuntimeError("cloud unavailable")

    async def local_result(image, scan_uid):
        return {"source": "edge_fallback"}

    async def log_event(event):
        events.append(event)

    node._send_features = send_failure
    node.predict_local = local_result
    node._log_network_event = log_event
    monkeypatch.setattr(edge_node, "DP_EPSILON", 0.0)
    monkeypatch.setattr(edge_node, "COMPRESSION", "none")
    monkeypatch.setattr(edge_node, "_serialize_feature_payload", lambda features, meta: b"payload")
    monkeypatch.setattr(edge_node, "_wire_compression_ratio", lambda features, payload: 1.0)

    result = asyncio.run(node.process_scan(b"image", "scan-1"))

    assert result["adaptive_decision"] == "local_fallback_cloud_error"
    assert result["probe_bandwidth_mbps"] == 12.0
    assert result["probe_latency_ms"] == 45.0
    assert result["fallback_reason"] == "cloud unavailable"
    assert events[0]["compression_method"] == "local_fallback_cloud_error"
    assert events[0]["payload_size_kb"] > 0
