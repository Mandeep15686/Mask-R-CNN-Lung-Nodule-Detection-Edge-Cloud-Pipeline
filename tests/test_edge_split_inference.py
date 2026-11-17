"""Focused checks for the cloud continuation of edge backbone features."""
from collections import OrderedDict
from threading import RLock
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from backend.app.ml.inference import InferenceEngine, InferenceUnavailableError


class _Stage(nn.Module):
    def __init__(self, name, calls):
        super().__init__()
        self.name = name
        self.calls = calls

    def forward(self, value):
        self.calls.append(self.name)
        return value


class _FPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.keys = None

    def forward(self, features):
        self.keys = list(features)
        return OrderedDict(features)


class _Transform:
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

    def __call__(self, images, targets):
        return SimpleNamespace(image_sizes=[(64, 64)]), targets

    @staticmethod
    def postprocess(detections, image_sizes, original_image_sizes):
        return []


@pytest.mark.parametrize(
    ("split_layer", "edge_layers", "cloud_layers"),
    [
        ("layer1", ("layer1",), ("layer2", "layer3", "layer4")),
        ("layer2", ("layer1", "layer2"), ("layer3", "layer4")),
        ("layer3", ("layer1", "layer2", "layer3"), ("layer4",)),
    ],
)
def test_edge_split_runs_remaining_stages_in_order_and_builds_all_fpn_levels(
    split_layer, edge_layers, cloud_layers
):
    calls = []
    fpn = _FPN()
    body = nn.ModuleDict({
        name: _Stage(name, calls)
        for name in ("layer2", "layer3", "layer4")
    })

    class _RPN:
        def __call__(self, image_list, features, targets):
            assert list(features) == ["0", "1", "2", "3"]
            return [], None

    class _ROIHeads:
        def __call__(self, features, proposals, image_sizes, targets):
            return [{}], None

    detection_model = SimpleNamespace(
        backbone=SimpleNamespace(body=body, fpn=fpn),
        transform=_Transform(),
        rpn=_RPN(),
        roi_heads=_ROIHeads(),
    )
    loaded = SimpleNamespace(
        model=SimpleNamespace(model=detection_model),
        inference_lock=RLock(),
        spacing_mm=(1.0, 1.0, 1.0),
        sha256="same-checkpoint",
    )
    engine = object.__new__(InferenceEngine)
    engine.device = torch.device("cpu")
    engine._get_checkpoint = lambda checkpoint_path=None: loaded

    features = {
        name: torch.zeros(1, 1, 2, 2)
        for name in edge_layers
    }
    assert engine.predict_from_edge_features(
        features,
        split_layer,
        "scan-test",
        expected_checkpoint_sha256="same-checkpoint",
        edge_image_mean=_Transform.image_mean,
        edge_image_std=_Transform.image_std,
    ) == []
    assert calls == list(cloud_layers)
    assert fpn.keys == ["0", "1", "2", "3"]


def test_edge_split_rejects_mismatched_checkpoint_and_normalization():
    loaded = SimpleNamespace(
        sha256="cloud-checkpoint",
        model=SimpleNamespace(model=SimpleNamespace(transform=_Transform())),
        inference_lock=RLock(),
    )
    engine = object.__new__(InferenceEngine)
    engine.device = torch.device("cpu")
    engine._get_checkpoint = lambda checkpoint_path=None: loaded

    with pytest.raises(InferenceUnavailableError, match="different model checkpoints"):
        engine.predict_from_edge_features(
            {"layer1": torch.zeros(1, 1, 2, 2)}, "layer1", "scan",
            expected_checkpoint_sha256="other-checkpoint",
        )

    with pytest.raises(InferenceUnavailableError, match="normalization does not match"):
        engine.predict_from_edge_features(
            {"layer1": torch.zeros(1, 1, 2, 2)}, "layer1", "scan",
            expected_checkpoint_sha256="cloud-checkpoint",
            edge_image_mean=[0.0, 0.0, 0.0],
            edge_image_std=[1.0, 1.0, 1.0],
        )


def test_edge_auth_bypass_requires_explicit_dev_or_test_mode(monkeypatch):
    from fastapi import HTTPException
    from backend.app.routes.edge import _validate_edge_token

    monkeypatch.delenv("EDGE_API_TOKEN", raising=False)
    monkeypatch.delenv("APP_ENV", raising=False)
    with pytest.raises(HTTPException) as exc:
        _validate_edge_token("")
    assert exc.value.status_code == 503

    _validate_edge_token("", app_env="development")
    _validate_edge_token("", app_env="test")


def test_edge_auth_requires_configured_token_outside_dev(monkeypatch):
    from fastapi import HTTPException
    from backend.app.routes.edge import _validate_edge_token

    monkeypatch.setenv("EDGE_API_TOKEN", "secret")
    with pytest.raises(HTTPException) as exc:
        _validate_edge_token("wrong", app_env="production")
    assert exc.value.status_code == 401
    _validate_edge_token("secret", app_env="production")
