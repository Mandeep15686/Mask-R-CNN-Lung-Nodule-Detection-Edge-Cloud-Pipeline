import json
import os
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import BackgroundTasks, HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app import data_config
from backend.app.database import Base, Experiment
from backend.app.routes import experiments
from backend.app.schemas import ExperimentCreate


def _run_sync_route(coroutine):
    """Run route coroutines that contain no await points without Windows asyncio."""
    try:
        coroutine.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError("Expected this API route to complete without awaiting")


def _raw_root(parent: Path) -> Path:
    root = parent / "raw"
    for subset in (0, 1):
        directory = root / f"subset{subset}"
        directory.mkdir(parents=True)
        (directory / f"1.2.840.{subset + 1}.7.mhd").write_text("fake header", encoding="utf-8")
    (root / "annotations.csv").write_text("seriesuid,coordX,coordY,coordZ,diameter_mm\n", encoding="utf-8")
    return root


def _cache(path: Path, raw_root: Path, *, patch_size: int = 8) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    entries = []
    for subset in (0, 1):
        uid = f"1.2.840.{subset + 1}.7"
        for label in (0, 1):
            filename = f"subset{subset}-{label}.npz"
            image = np.full((3, patch_size, patch_size), 0.5, dtype=np.float32)
            mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
            if label:
                mask[patch_size // 2, patch_size // 2] = 1
            np.savez_compressed(
                path / filename,
                image=image,
                mask=mask,
                label=np.int8(label),
                diameter=np.float32(4 if label else 0),
            )
            entries.append({
                "file": filename,
                "uid": uid,
                "subset": subset,
                "label": label,
                "patch_size": patch_size,
                "spacing_mm": [1.0, 1.0, 1.0],
            })
    (path / "manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    return path


def _db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return engine, Session(engine)


def _payload(config_id="cache-test", **overrides):
    values = {
        "config_id": config_id,
        "loss_weights": {"alpha": 0.3, "beta": 0.5, "gamma": 0.5, "delta": 0, "epsilon": 0},
        "epochs": 1,
        "batch_size": 2,
        "lr": 0.005,
        "patch_size": 8,
        "train_subsets": [1],
        "validation_subset": 0,
    }
    values.update(overrides)
    return ExperimentCreate(**values)


def test_path_resolution_priority_and_project_relative_paths(tmp_path, monkeypatch):
    env_path = tmp_path / "environment-cache"
    env_path.mkdir()
    monkeypatch.setenv("LUNA16_CACHE_DIR", str(env_path))

    explicit = data_config.resolve_cache_dir("custom/cache")
    assert explicit == (data_config.PROJECT_ROOT / "custom/cache").resolve()
    assert data_config.resolve_cache_dir() == env_path.resolve()

    monkeypatch.delenv("LUNA16_CACHE_DIR")
    assert data_config.resolve_cache_dir() == (
        data_config.PROJECT_ROOT / "data/patches_cache_v3_smoke"
    ).resolve()


def test_windows_and_docker_absolute_paths_are_not_made_project_relative():
    windows_path = r"D:\lung-nodule\data\patches_cache_v3_smoke"
    if os.name == "nt":
        assert data_config.resolve_cache_dir(windows_path) == Path(windows_path).resolve()
    else:
        assert PureWindowsPath(windows_path).is_absolute()
        with pytest.raises(ValueError, match="Windows path"):
            data_config.resolve_cache_dir(windows_path)

    if os.name != "nt":
        assert data_config.resolve_cache_dir("/data/patches_cache_v3_smoke") == Path(
            "/data/patches_cache_v3_smoke"
        )


def test_validate_cached_split_reports_summary_and_rejects_missing_cache(tmp_path):
    raw = _raw_root(tmp_path)
    cache = _cache(tmp_path / "cache", raw)
    split = data_config.resolve_subset_split(
        data_config.available_subset_uids(raw), train_subsets=[1], validation_subset=0
    )
    summary = data_config.validate_cached_split(raw, cache, 8, split)
    assert summary["total_patches"] == 4
    assert summary["positive_patches"] == summary["negative_patches"] == 2
    assert summary["patch_dimensions"] == [3, 8, 8]
    assert summary["available_subsets"] == [0, 1]
    assert summary["scan_uid_count"] == 2
    assert summary["spacing_mm"] == [1.0, 1.0, 1.0]

    with pytest.raises(FileNotFoundError, match="Patch cache directory does not exist"):
        data_config.validate_cached_split(raw, tmp_path / "missing", 8, split)


def test_api_create_persists_resolved_paths_split_and_cache_override(tmp_path, monkeypatch):
    raw = _raw_root(tmp_path)
    default_cache = _cache(tmp_path / "default-cache", raw)
    custom_cache = _cache(tmp_path / "custom-cache", raw)
    monkeypatch.setenv("DATA_ROOT", str(raw))
    monkeypatch.setenv("LUNA16_CACHE_DIR", str(default_cache))
    engine, db = _db_session()
    try:
        created = _run_sync_route(experiments.create_experiment(
            _payload(cache_dir=str(custom_cache)), db
        ))
        assert created.config_json["cache_dir"] == str(custom_cache.resolve())
        assert created.config_json["data_root"] == str(raw.resolve())
        assert created.config_json["cache_mode"] == "cached"
        assert created.config_json["train_subsets"] == [1]
        assert created.config_json["validation_subset"] == 0
        assert created.config_json["test_subset"] is None
        assert created.config_json["training_strategy"] == "fixed"
        assert created.config_json["cache_summary"]["total_patches"] == 4
        assert set(created.config_json["data_split"]["train_scan_uids"]).isdisjoint(
            created.config_json["data_split"]["validation_scan_uids"]
        )
    finally:
        db.close()
        engine.dispose()


def test_api_create_persists_mgda_training_strategy(tmp_path, monkeypatch):
    raw = _raw_root(tmp_path)
    cache = _cache(tmp_path / "mgda-cache", raw)
    monkeypatch.setenv("DATA_ROOT", str(raw))
    engine, db = _db_session()
    try:
        created = _run_sync_route(experiments.create_experiment(
            _payload(
                "mgda-exp",
                cache_dir=str(cache),
                training_strategy="mgda",
                mgda_gradient_normalization="unit",
                mgda_parameter_scope="shared_mask_path",
            )
        , db))
        assert created.config_json["training_strategy"] == "mgda"
        assert created.config_json["mgda_gradient_normalization"] == "unit"
        assert created.config_json["mgda_parameter_scope"] == "shared_mask_path"
        assert created.loss_description.startswith("MGDA")
    finally:
        db.close()
        engine.dispose()


def test_api_create_uses_environment_default_and_no_row_on_invalid_cache(tmp_path, monkeypatch):
    raw = _raw_root(tmp_path)
    cache = _cache(tmp_path / "default-cache", raw)
    monkeypatch.setenv("DATA_ROOT", str(raw))
    monkeypatch.setenv("LUNA16_CACHE_DIR", str(cache))
    engine, db = _db_session()
    try:
        created = _run_sync_route(experiments.create_experiment(_payload("env-cache"), db))
        assert created.config_json["cache_dir"] == str(cache.resolve())

        with pytest.raises(HTTPException) as error:
            _run_sync_route(experiments.create_experiment(
                _payload("bad-cache", cache_dir=str(tmp_path / "not-a-cache")), db
            ))
        assert error.value.status_code == 422
        assert db.query(Experiment).filter(Experiment.config_id == "bad-cache").count() == 0
    finally:
        db.close()
        engine.dispose()


def test_training_configuration_keeps_persisted_cache_and_uses_cached_loaders(tmp_path, monkeypatch):
    raw = _raw_root(tmp_path)
    cache = _cache(tmp_path / "selected-cache", raw)
    other_cache = _cache(tmp_path / "environment-cache", raw)
    monkeypatch.setenv("LUNA16_CACHE_DIR", str(other_cache))
    config = experiments._prepare_training_configuration({
        "data_root": str(raw),
        "cache_mode": "cached",
        "cache_dir": str(cache),
        "patch_size": 8,
        "train_subsets": [1],
        "validation_subset": 0,
        "test_subset": None,
    }, require_resolved_cache=True)
    assert config["cache_dir"] == str(cache.resolve())
    train, validation = __import__("training.dataset", fromlist=["build_dataloaders"]).build_dataloaders(
        config["data_root"],
        cache_dir=config["cache_dir"],
        train_subsets=config["train_subsets"],
        validation_subset=config["validation_subset"],
        test_subset=config["test_subset"],
        patch_size=config["patch_size"],
        batch_size=1,
        num_workers=0,
    )
    from training.dataset import CachedLUNA16Dataset
    assert isinstance(train.dataset, CachedLUNA16Dataset)
    assert isinstance(validation.dataset, CachedLUNA16Dataset)
    assert train.dataset.cache_dir == cache.resolve()
    assert train.dataset.scan_uids == {"1.2.840.2.7"}
    assert validation.dataset.scan_uids == {"1.2.840.1.7"}


def test_start_revalidates_missing_cache_marks_failed_and_logs(tmp_path, monkeypatch):
    raw = _raw_root(tmp_path)
    cache = _cache(tmp_path / "cache", raw)
    monkeypatch.setattr(experiments, "_experiment_run_dir", lambda config_id: tmp_path / "runs" / config_id)
    engine, db = _db_session()
    exp = Experiment(
        config_id="cache-gone",
        status="pending",
        config_json={
            "data_root": str(raw.resolve()),
            "cache_mode": "cached",
            "cache_dir": str(cache.resolve()),
            "patch_size": 8,
            "train_subsets": [1],
            "validation_subset": 0,
            "test_subset": None,
        },
    )
    db.add(exp)
    db.commit()
    (cache / "manifest.json").unlink()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(inference_engine=None)))
    try:
        with pytest.raises(HTTPException) as error:
            _run_sync_route(experiments.start_experiment(exp.id, BackgroundTasks(), request, db))
        assert error.value.status_code == 422
        db.refresh(exp)
        assert exp.status == "failed"
        assert "manifest.json" in (tmp_path / "runs" / "cache-gone" / "terminal.log").read_text(encoding="utf-8")
    finally:
        db.close()
        engine.dispose()


def test_runtime_info_reports_default_cache_and_survives_missing_cache(tmp_path, monkeypatch):
    raw = _raw_root(tmp_path)
    cache = _cache(tmp_path / "cache", raw)
    monkeypatch.setenv("DATA_ROOT", str(raw))
    monkeypatch.setenv("LUNA16_CACHE_DIR", str(cache))
    monkeypatch.setenv("PATCH_SIZE", "8")
    info = data_config.runtime_data_info()
    assert info["cache_valid"] is True
    assert info["default_split"] == {"train_subsets": [1], "validation_subset": 0, "test_subset": None}
    assert info["cache_summary"]["total_patches"] == 4

    monkeypatch.setenv("LUNA16_CACHE_DIR", str(tmp_path / "missing"))
    unavailable = data_config.runtime_data_info()
    assert unavailable["cache_valid"] is False
    assert "does not exist" in unavailable["cache_error"]


def test_raw_mode_requires_explicit_selection_and_never_uses_cache_env(tmp_path, monkeypatch):
    raw = _raw_root(tmp_path)
    cache = _cache(tmp_path / "cache", raw)
    monkeypatch.setenv("LUNA16_CACHE_DIR", str(cache))
    config = experiments._prepare_training_configuration({
        "data_root": str(raw),
        "cache_mode": "raw",
        "train_subsets": [1],
        "validation_subset": 0,
    })
    assert config["cache_dir"] is None
    assert config["cache_mode"] == "raw"
