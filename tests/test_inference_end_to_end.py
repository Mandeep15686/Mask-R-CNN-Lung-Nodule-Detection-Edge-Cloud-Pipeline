import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from backend.app.ml.inference import (
    CheckpointNotFoundError,
    InferenceEngine,
    InferenceUnavailableError,
)
from backend.app.routes.inference import _checkpoint_for_request, _execute_inference
from backend.app.schemas import InferenceResponse, NoduleDetection
from backend.app.utils.preprocessing import preprocess_uploaded_image


class _EmptyDetectionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, images):
        return [
            {
                "boxes": torch.zeros((0, 4)),
                "scores": torch.zeros(0),
                "masks": torch.zeros((0, 1, 64, 64)),
            }
            for _ in images
        ]


def _fake_checkpoint(path: Path, config_id="test-config", model_state=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state = model_state or _EmptyDetectionModel().state_dict()
    torch.save({
        "epoch": 3,
        "model_state": model_state,
        "metrics": {"val_dice": 0.4, "val_hd95": 4.0},
        "config": {
            "config_id": config_id,
            "num_classes": 2,
            "mask_loss": {"alpha": 1.0},
            "pretrained_backbone": True,
            "target_spacing_mm": [1.0, 1.0, 1.0],
        },
    }, path)


def _model_factory(_config):
    return _EmptyDetectionModel()


def _rgb_png(rgb):
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def test_existing_valid_checkpoint_loads_strictly_and_logs_metadata(tmp_path):
    checkpoint = tmp_path / "best_model.pth"
    _fake_checkpoint(checkpoint, config_id="trained-config")

    engine = InferenceEngine(
        default_checkpoint_path=checkpoint,
        model_factory=_model_factory,
        device="cpu",
    )

    assert engine.status["model_loaded"] is True
    assert engine.status["checkpoint_path"] == str(checkpoint.resolve())
    assert engine.status["config_id"] == "trained-config"
    assert engine.checkpoint_info()["epoch"] == 3
    assert engine._models[str(checkpoint.resolve())].model.training is False


def test_missing_checkpoint_disables_inference_instead_of_returning_empty(tmp_path):
    engine = InferenceEngine(
        default_checkpoint_path=tmp_path / "missing.pth",
        model_factory=_model_factory,
        device="cpu",
    )

    assert engine.status["model_loaded"] is False
    assert "not found" in engine.status["model_loading_error"].lower()
    with pytest.raises(CheckpointNotFoundError):
        engine.predict_from_bytes(_rgb_png(np.zeros((64, 64, 3), dtype=np.uint8)))


def test_incompatible_checkpoint_is_reported_and_not_loaded(tmp_path):
    checkpoint = tmp_path / "incompatible.pth"
    _fake_checkpoint(checkpoint, model_state={"not_a_model_weight": torch.ones(1)})

    engine = InferenceEngine(
        default_checkpoint_path=checkpoint,
        model_factory=_model_factory,
        device="cpu",
    )

    assert engine.status["model_loaded"] is False
    assert "do not match" in engine.status["model_loading_error"].lower()


def test_explicit_model_path_is_used_without_fallback(tmp_path, monkeypatch):
    checkpoint = tmp_path / "selected.pth"
    _fake_checkpoint(checkpoint, config_id="selected-config")
    monkeypatch.setattr("backend.app.ml.inference.MODEL_PATH", str(checkpoint))

    engine = InferenceEngine(model_factory=_model_factory, device="cpu")

    assert engine.status["model_loaded"] is True
    assert engine.status["checkpoint_path"] == str(checkpoint.resolve())


def test_experiment_checkpoint_selection_is_exact_and_cached(tmp_path):
    root = tmp_path / "experiments"
    default = root / "default" / "best_model.pth"
    selected = root / "requested-exp" / "best_model.pth"
    _fake_checkpoint(default, config_id="default")
    _fake_checkpoint(selected, config_id="requested-exp")
    engine = InferenceEngine(
        default_checkpoint_path=default,
        experiments_dir=root,
        model_factory=_model_factory,
        device="cpu",
    )

    path = engine.checkpoint_for_experiment("requested-exp")
    first = engine._load_checkpoint(path)
    second = engine._load_checkpoint(path)

    assert path == selected.resolve()
    assert first is second
    assert first.config_id == "requested-exp"
    with pytest.raises(CheckpointNotFoundError):
        engine.checkpoint_for_experiment("missing-exp")


def test_checkpoint_overwrite_at_same_path_invalidates_loaded_model(tmp_path):
    checkpoint = tmp_path / "best_model.pth"
    _fake_checkpoint(checkpoint, config_id="model-v1")
    engine = InferenceEngine(
        default_checkpoint_path=checkpoint,
        model_factory=_model_factory,
        device="cpu",
    )
    previous = engine._load_checkpoint(checkpoint)
    previous_signature = previous.file_signature
    previous_hash = previous.sha256
    previous_size = checkpoint.stat().st_size

    replacement = _EmptyDetectionModel()
    with torch.no_grad():
        replacement.weight.fill_(2.0)
    _fake_checkpoint(
        checkpoint,
        config_id="model-v2",
        model_state=replacement.state_dict(),
    )
    assert checkpoint.stat().st_size == previous_size

    refreshed = engine._load_checkpoint(checkpoint)

    assert refreshed is not previous
    assert refreshed.file_signature != previous_signature
    assert refreshed.sha256 != previous_hash
    assert refreshed.config_id == "model-v2"
    assert refreshed.model.weight.item() == pytest.approx(2.0)
    assert engine.model is refreshed.model
    assert engine.status["config_id"] == "model-v2"


def test_preprocessing_preserves_three_rgb_channels_without_enhancement():
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[:, :, 0] = 32
    rgb[:, :, 1] = 128
    rgb[:, :, 2] = 240

    tensor = preprocess_uploaded_image(_rgb_png(rgb))

    assert tensor.shape == (3, 64, 64)
    np.testing.assert_allclose(tensor[:, 0, 0], np.array([32, 128, 240]) / 255.0)
    assert np.isclose(tensor.min(), 32 / 255.0)
    assert np.isclose(tensor.max(), 240 / 255.0)


def test_grayscale_patch_is_rejected_as_not_equivalent_to_2_5d():
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((64, 64), dtype=np.uint8), mode="L").save(buffer, format="PNG")

    with pytest.raises(ValueError, match="true three-channel RGB"):
        preprocess_uploaded_image(buffer.getvalue())


def test_full_ct_image_is_not_silently_resized():
    with pytest.raises(ValueError, match="whole CT images are not supported"):
        preprocess_uploaded_image(_rgb_png(np.zeros((128, 128, 3), dtype=np.uint8)))


def test_score_threshold_filters_detections_and_diameter_uses_patch_spacing():
    outputs = [{
        "boxes": torch.tensor([[0, 0, 10, 10], [1, 1, 5, 5]], dtype=torch.float32),
        "scores": torch.tensor([0.049, 0.05]),
        "masks": torch.ones((2, 1, 64, 64)),
    }]

    detections = InferenceEngine._postprocess(outputs, (1.0, 1.0, 1.0), 0.05)

    assert len(detections) == 1
    assert detections[0].score == pytest.approx(0.05)
    assert detections[0].diameter_estimate_mm == pytest.approx(4.0)
    assert detections[0].mask_rle is not None


def test_empty_detection_is_distinct_from_model_unavailable(tmp_path):
    checkpoint = tmp_path / "trained.pth"
    _fake_checkpoint(checkpoint)
    engine = InferenceEngine(
        default_checkpoint_path=checkpoint,
        model_factory=_model_factory,
        device="cpu",
    )

    assert engine.predict_from_bytes(_rgb_png(np.zeros((64, 64, 3), dtype=np.uint8))) == []

    unavailable = InferenceEngine(
        default_checkpoint_path=tmp_path / "absent.pth",
        model_factory=_model_factory,
        device="cpu",
    )
    with pytest.raises(CheckpointNotFoundError):
        unavailable.predict_from_bytes(_rgb_png(np.zeros((64, 64, 3), dtype=np.uint8)))


class _FakeDB:
    def __init__(self, experiment=None):
        self.experiment = experiment

    def query(self, _model):
        return self

    def filter(self, _query):
        return self

    def first(self):
        return self.experiment

    def add(self, result):
        self.result = result

    def commit(self):
        pass

    def refresh(self, result):
        result.id = 17


class _FakeAPIEngine:
    status = {"model_loaded": True}

    def __init__(self, fail=False):
        self.fail = fail
        self.selected = None

    def checkpoint_for_experiment(self, config_id):
        return Path("/models") / config_id / "best_model.pth"

    def predict_from_bytes(self, _image_bytes, checkpoint_path=None):
        self.selected = checkpoint_path
        if self.fail:
            raise InferenceUnavailableError("no trained weights")
        return [NoduleDetection(
            box=[1, 2, 10, 12], score=0.8,
            mask_rle="0:4096", diameter_estimate_mm=9.0,
        )]

    def checkpoint_info(self, checkpoint_path=None):
        return {
            "checkpoint_path": checkpoint_path or "/models/default/best_model.pth",
            "config_id": "chosen-config",
            "score_threshold": 0.05,
        }


def test_api_result_contains_checkpoint_and_detection_details():
    db = _FakeDB()
    engine = _FakeAPIEngine()
    detections, metadata, _ = _execute_inference(b"synthetic-rgb-patch", None, db, engine)
    response = InferenceResponse(
        scan_uid="synthetic-scan",
        nodules=detections,
        inference_time_ms=2.0,
        source="cloud",
        result_id=17,
        checkpoint_path=metadata["checkpoint_path"],
        config_id=metadata["config_id"],
        score_threshold=metadata["score_threshold"],
    )

    assert response.result_id == 17
    assert response.config_id == "chosen-config"
    assert response.score_threshold == 0.05
    assert len(response.nodules) == 1
    assert response.nodules[0].mask_rle == "0:4096"


def test_api_explicit_experiment_maps_to_its_own_checkpoint(tmp_path):
    experiment = type("ExperimentStub", (), {"config_id": "selected-config"})()
    engine = _FakeAPIEngine()

    path, config_id = _checkpoint_for_request(9, _FakeDB(experiment), engine)

    assert path == str(Path("/models/selected-config/best_model.pth"))
    assert config_id == "selected-config"


def test_api_reports_model_unavailable_as_503():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        _execute_inference(b"synthetic-rgb-patch", None, _FakeDB(), _FakeAPIEngine(fail=True))
    assert exc.value.status_code == 503


def _cache_png(path: Path) -> bytes:
    with np.load(path) as patch:
        image = patch["image"]
    return _rgb_png(np.rint(image.transpose(1, 2, 0) * 255).astype(np.uint8))


@pytest.mark.slow
def test_real_v3_validation_positive_and_negative_patches():
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "experiments" / "best_model.pth"
    cache_dir = root / "data" / "patches_cache_v3_smoke"
    if not checkpoint.is_file() or not (cache_dir / "manifest.json").is_file():
        pytest.skip("Local trained checkpoint or V3 smoke cache is unavailable")
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    positives = [e for e in manifest if int(e["subset"]) == 0 and int(e["label"]) == 1]
    negatives = [e for e in manifest if int(e["subset"]) == 0 and int(e["label"]) == 0]
    if not positives or not negatives:
        pytest.skip("V3 smoke cache does not include both validation labels")

    engine = InferenceEngine(default_checkpoint_path=checkpoint)
    assert engine.status["model_loaded"], engine.status["model_loading_error"]
    positive_path = cache_dir / positives[0]["file"]
    negative_path = cache_dir / negatives[0]["file"]
    positive_detections = engine.predict_from_bytes(_cache_png(positive_path))
    negative_detections = engine.predict_from_bytes(_cache_png(negative_path))

    assert positive_detections, f"No detections for cached positive {positives[0]['file']}"
    assert not negative_detections, f"Unexpected detections for cached negative {negatives[0]['file']}"
