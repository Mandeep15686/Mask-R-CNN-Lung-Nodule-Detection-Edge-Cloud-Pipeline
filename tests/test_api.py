"""
Integration tests for the FastAPI backend.
Run: pytest tests/test_api.py -v
"""
import sys, os, tempfile, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
import base64
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from unittest.mock import patch


class TestInferenceEngine:
    last_edge_feature_call = None
    status = {
        "model_loaded": True,
        "checkpoint_path": "test-checkpoint.pth",
        "config_id": "test-model",
        "device": "cpu",
        "score_threshold": 0.05,
        "nms_threshold": 0.5,
        "model_loading_error": None,
    }

    def predict_from_bytes(self, image_bytes, experiment_id=None, checkpoint_path=None):
        return []

    def checkpoint_info(self, checkpoint_path=None):
        return {
            "checkpoint_path": checkpoint_path or self.status["checkpoint_path"],
            "config_id": self.status["config_id"],
            "score_threshold": self.status["score_threshold"],
        }

    def predict_from_features(self, features, scan_uid):
        return []

    def predict_from_edge_features(self, features, split_layer, scan_uid, **metadata):
        TestInferenceEngine.last_edge_feature_call = (features, split_layer, scan_uid)
        return []

    def load_checkpoint(self, checkpoint_path):
        return None
# ---------------------------------------------------------------------------
# App fixture — uses an in-memory SQLite DB for isolation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def client():
    with tempfile.TemporaryDirectory(prefix="lung-nodule-api-tests-") as db_dir:
        db_path = os.path.join(db_dir, "test_lung_nodule.db")
        os.environ["DATABASE_URL"] = f"sqlite:///{db_path.replace(os.sep, '/')}"
        os.environ["MODEL_PATH"] = "/nonexistent/model.pth"
        os.environ["APP_ENV"] = "test"
        raw_root = os.path.join(db_dir, "raw")
        cache_dir = os.path.join(db_dir, "patch-cache")
        os.makedirs(cache_dir)
        cache_entries = []
        for subset in (0, 1):
            uid = f"1.2.840.{subset + 1}.7"
            subset_dir = os.path.join(raw_root, f"subset{subset}")
            os.makedirs(subset_dir)
            with open(os.path.join(subset_dir, f"{uid}.mhd"), "w", encoding="utf-8") as f:
                f.write("synthetic header")
            for label in (0, 1):
                filename = f"subset{subset}-{label}.npz"
                mask = np.zeros((64, 64), dtype=np.uint8)
                if label:
                    mask[32, 32] = 1
                np.savez_compressed(
                    os.path.join(cache_dir, filename),
                    image=np.zeros((3, 64, 64), dtype=np.float32),
                    mask=mask,
                    label=np.int8(label),
                    diameter=np.float32(4 if label else 0),
                )
                cache_entries.append({
                    "file": filename,
                    "uid": uid,
                    "subset": subset,
                    "label": label,
                    "patch_size": 64,
                    "spacing_mm": [1.0, 1.0, 1.0],
                })
        with open(os.path.join(raw_root, "annotations.csv"), "w", encoding="utf-8") as f:
            f.write("seriesuid,coordX,coordY,coordZ,diameter_mm\n")
        with open(os.path.join(cache_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(cache_entries, f)
        os.environ["DATA_ROOT"] = raw_root
        os.environ["LUNA16_CACHE_DIR"] = cache_dir

        from backend.app.main import app
        from backend.app.database import engine

        try:
            from backend.app.ml import inference as inference_module

            with patch.object(
                inference_module,
                "InferenceEngine",
                TestInferenceEngine,
            ):
                with TestClient(app) as c:
                    yield c
        finally:
            # Close pooled SQLite connections before the temporary directory
            # removes the test database.
            engine.dispose()

            import gc
            gc.collect()



@pytest.fixture
def sample_image_bytes():
    """Generate a small synthetic CT patch as PNG bytes."""
    arr = np.random.randint(0, 200, (64, 64, 3), dtype=np.uint8)
    # Add a bright circle to simulate a nodule
    cy, cx, r = 32, 32, 8
    yy, xx = np.ogrid[:64, :64]
    arr[(yy-cy)**2 + (xx-cx)**2 <= r**2] = 220
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_image_b64(sample_image_bytes):
    return base64.b64encode(sample_image_bytes).decode()


# ---------------------------------------------------------------------------
# Health / info
# ---------------------------------------------------------------------------
class TestHealth:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_info(self, client):
        r = client.get("/api/info")
        assert r.status_code == 200
        data = r.json()
        assert "model" in data
        assert "loss_functions" in data


# ---------------------------------------------------------------------------
# Experiments CRUD
# ---------------------------------------------------------------------------
class TestExperiments:
    EXP_PAYLOAD = {
        "config_id":    "test-bce-dice",
        "phase":        1,
        "loss_weights": {"alpha": 0.3, "beta": 0.5, "gamma": 0.5, "delta": 0.0, "epsilon": 0.0},
        "epochs":       5,
        "batch_size":   2,
        "fold":         0,
        "lr":           0.005,
    }

    def test_create_experiment(self, client):
        r = client.post("/api/experiments/", json=self.EXP_PAYLOAD)
        assert r.status_code == 201
        data = r.json()
        assert data["config_id"] == "test-bce-dice"
        assert data["alpha"] == pytest.approx(0.3)
        assert data["beta"]  == pytest.approx(0.5)
        assert data["gamma"] == pytest.approx(0.5)
        assert data["status"] == "pending"

    def test_create_duplicate_fails(self, client):
        r = client.post("/api/experiments/", json=self.EXP_PAYLOAD)
        assert r.status_code == 409

    def test_list_experiments(self, client):
        r = client.get("/api/experiments/")
        assert r.status_code == 200
        assert isinstance(r.json(), list)
        assert len(r.json()) >= 1

    def test_get_experiment(self, client):
        # First create a fresh one
        payload = {**self.EXP_PAYLOAD, "config_id": "test-get-exp"}
        r_create = client.post("/api/experiments/", json=payload)
        assert r_create.status_code == 201
        exp_id = r_create.json()["id"]

        r = client.get(f"/api/experiments/{exp_id}")
        assert r.status_code == 200
        assert r.json()["id"] == exp_id

    def test_get_nonexistent(self, client):
        r = client.get("/api/experiments/999999")
        assert r.status_code == 404

    def test_zero_weights_rejected(self, client):
        bad = {**self.EXP_PAYLOAD, "config_id": "bad-zero",
               "loss_weights": {"alpha": 0.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0, "epsilon": 0.0}}
        r = client.post("/api/experiments/", json=bad)
        assert r.status_code == 422

    def test_delete_experiment(self, client):
        payload = {**self.EXP_PAYLOAD, "config_id": "test-delete-exp"}
        r_create = client.post("/api/experiments/", json=payload)
        exp_id = r_create.json()["id"]

        r_del = client.delete(f"/api/experiments/{exp_id}")
        assert r_del.status_code == 204

        r_get = client.get(f"/api/experiments/{exp_id}")
        assert r_get.status_code == 404

    def test_summary_no_runs(self, client):
        payload = {**self.EXP_PAYLOAD, "config_id": "test-summary-exp"}
        r_create = client.post("/api/experiments/", json=payload)
        exp_id = r_create.json()["id"]

        r = client.get(f"/api/experiments/{exp_id}/summary")
        assert r.status_code == 200
        data = r.json()
        assert data["best_dice"] == 0.0
        assert data["status"] == "pending"


# ---------------------------------------------------------------------------
# Results endpoints
# ---------------------------------------------------------------------------
class TestResults:
    def test_stats(self, client):
        r = client.get("/api/results/stats")
        assert r.status_code == 200
        data = r.json()
        assert "total_experiments" in data
        assert isinstance(data["total_experiments"], int)

    def test_leaderboard(self, client):
        r = client.get("/api/results/leaderboard")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_pareto(self, client):
        r = client.get("/api/results/pareto")
        assert r.status_code == 200
        assert isinstance(r.json(), list)


# ---------------------------------------------------------------------------
# Inference endpoints
# ---------------------------------------------------------------------------
class TestInference:
    def test_predict_base64(self, client, sample_image_b64):
        payload = {
            "scan_uid":   "test_scan_001",
            "image_b64":  sample_image_b64,
            "source":     "cloud",
        }
        r = client.post("/api/inference/predict", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["scan_uid"] == "test_scan_001"
        assert "nodules" in data
        assert isinstance(data["nodules"], list)
        assert "inference_time_ms" in data
        assert data["inference_time_ms"] > 0

    def test_predict_upload(self, client, sample_image_bytes):
        r = client.post(
            "/api/inference/upload",
            data={"scan_uid": "test_upload_001", "source": "cloud"},
            files={"file": ("patch.png", sample_image_bytes, "image/png")},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["scan_uid"] == "test_upload_001"
        assert "result_id" in data

    def test_predict_invalid_b64(self, client):
        r = client.post("/api/inference/predict", json={
            "scan_uid": "bad", "image_b64": "NOT_VALID_BASE64!!!", "source": "cloud"
        })
        assert r.status_code == 400

    def test_list_inference_results(self, client):
        r = client.get("/api/inference/results")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_get_specific_result(self, client, sample_image_b64):
        # Create a result first
        r_create = client.post("/api/inference/predict", json={
            "scan_uid": "test_get_result",
            "image_b64": sample_image_b64,
            "source": "cloud",
        })
        result_id = r_create.json()["result_id"]

        r = client.get(f"/api/inference/results/{result_id}")
        assert r.status_code == 200
        assert r.json()["id"] == result_id

    def test_get_nonexistent_result(self, client):
        r = client.get("/api/inference/results/999999")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Edge endpoints
# ---------------------------------------------------------------------------
class TestEdge:
    NODE_PAYLOAD = {
        "node_id":    "test-edge-01",
        "hostname":   "test-host",
        "ip_address": "10.0.0.1",
        "split_layer": "layer2",
        "compression_method": "int8",
    }

    def test_register_node(self, client):
        r = client.post("/api/edge/nodes/register", json=self.NODE_PAYLOAD)
        assert r.status_code in (200, 201)
        data = r.json()
        assert data["node_id"] == "test-edge-01"
        assert data["is_online"] is True

    def test_list_nodes(self, client):
        r = client.get("/api/edge/nodes")
        assert r.status_code == 200
        nodes = r.json()
        assert any(n["node_id"] == "test-edge-01" for n in nodes)

    def test_heartbeat(self, client):
        r = client.post("/api/edge/nodes/test-edge-01/heartbeat")
        assert r.status_code == 200

    def test_heartbeat_unknown_node(self, client):
        r = client.post("/api/edge/nodes/nonexistent-node/heartbeat")
        assert r.status_code == 404

    def test_network_probe_reports_received_bytes(self, client):
        payload = b"p" * 65536
        response = client.post(
            "/api/edge/network-probe",
            content=payload,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert response.status_code == 200
        assert response.json()["bytes_received"] == len(payload)

    def test_feature_endpoint_routes_cumulative_backbone_maps(self, client, monkeypatch):
        monkeypatch.delenv("EDGE_API_TOKEN", raising=False)
        TestInferenceEngine.last_edge_feature_call = None
        header = {
            "version": 2,
            "keys": ["layer1", "layer2"],
            "shapes": [[1, 1, 1, 1], [1, 1, 1, 1]],
            "dtypes": ["int8", "int8"],
            "scales": [0.25, 0.5],
            "zero_points": [0, -1],
            "checkpoint_sha256": "test-checkpoint",
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
        }
        body = json.dumps(header).encode("utf-8") + b"\x00" + b"\x01\x02"
        response = client.post(
            "/api/edge/infer-features",
            params={"node_id": "test-edge-01", "scan_uid": "scan-feature", "split_layer": "layer2"},
            content=body,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert response.status_code == 200, response.text
        features, split_layer, scan_uid = TestInferenceEngine.last_edge_feature_call
        assert set(features) == {"layer1", "layer2"}
        assert all(value.is_quantized for value in features.values())
        assert (split_layer, scan_uid) == ("layer2", "scan-feature")

    def test_log_network_event(self, client):
        r = client.post("/api/edge/network-log", json={
            "node_id": "test-edge-01",
            "latency_ms": 12.5,
            "bandwidth_mbps": 95.0,
            "payload_size_kb": 32.0,
            "inference_time_ms": 150.0,
            "split_layer": "layer2",
            "compression_method": "int8",
            "compression_ratio": 3.8,
        })
        assert r.status_code in (200, 201)
        data = r.json()
        assert data["node_id"] == "test-edge-01"

    def test_network_stats(self, client):
        r = client.get("/api/edge/network-stats")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_latency_analysis(self, client):
        r = client.get("/api/edge/split-latency-analysis")
        assert r.status_code == 200
        assert isinstance(r.json(), list)
