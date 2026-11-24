import json

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.database import Base, Experiment, TrainingRun
from scripts import import_experiment_folders as importer


def _write_run(root, folder_name, config_id, *, epochs=2, history=None):
    folder = root / folder_name
    folder.mkdir(parents=True)
    (folder / "best_model.pth").write_bytes(b"checkpoint")
    (folder / "config.json").write_text(json.dumps({
        "config_id": config_id,
        "phase": 2,
        "epochs": epochs,
        "fold": 0,
        "mask_loss": {"alpha": 0.3, "beta": 0.5, "gamma": 0.2},
    }), encoding="utf-8")
    if history is None:
        history = [
            {"epoch": 1, "total_loss": 1.2, "val_total": 0.9, "val_dice": 0.4, "lr": 0.005},
            {"epoch": 2, "total_loss": 0.8, "val_total": 0.7, "val_dice": 0.6, "lr": 0.004},
        ][:epochs]
    (folder / "history.json").write_text(json.dumps(history), encoding="utf-8")
    (folder / "summary.json").write_text(json.dumps({
        "config_id": config_id,
        "best_dice": 0.6,
        "loss_weights": {"alpha": 0.3, "beta": 0.5, "gamma": 0.2},
    }), encoding="utf-8")
    return folder


def _session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return engine, Session(engine)


def test_collect_skips_absent_optional_roots_and_imports_existing_runs(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "experiments").mkdir(parents=True)
    _write_run(root / "experiments", "run-a", "run-a")
    monkeypatch.setattr(importer, "PROJECT_ROOT", root)

    collected = importer._collect_runs()

    assert [item["config_id"] for item in collected] == ["run-a"]


def test_duplicate_config_uses_canonical_run_folder(tmp_path, monkeypatch):
    root = tmp_path / "project"
    runs = root / "experiments"
    runs.mkdir(parents=True)
    canonical = _write_run(runs, "run-a", "run-a")
    _write_run(runs, "all_results_copy", "run-a")
    monkeypatch.setattr(importer, "PROJECT_ROOT", root)

    collected = importer._collect_runs()

    assert len(collected) == 1
    assert collected[0]["config_json"]["artifact_dir"] == canonical.relative_to(root).as_posix()


def test_sync_imports_epoch_details_and_is_idempotent(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "experiments").mkdir(parents=True)
    _write_run(root / "experiments", "run-a", "run-a")
    monkeypatch.setattr(importer, "PROJECT_ROOT", root)
    imports = importer._collect_runs()
    engine, db = _session()
    try:
        created = importer.sync_experiment_folders(
            db, imports, experiment_model=Experiment, training_run_model=TrainingRun
        )
        assert created["created"] == 1
        assert created["complete"] == 1
        assert created["history_rows"] == 2

        experiment = db.query(Experiment).filter_by(config_id="run-a").one()
        epochs = db.query(TrainingRun).filter_by(experiment_id=experiment.id).order_by(TrainingRun.epoch).all()
        assert [run.epoch for run in epochs] == [1, 2]
        assert epochs[1].val_dice == 0.6
        assert epochs[1].val_total_loss == 0.7
        assert experiment.config_json["artifact_dir"] == "experiments/run-a"
        assert experiment.config_json["artifact_summary"]["best_dice"] == 0.6

        refreshed = importer.sync_experiment_folders(
            db, imports, experiment_model=Experiment, training_run_model=TrainingRun
        )
        assert refreshed["updated"] == 1
        assert db.query(TrainingRun).filter_by(experiment_id=experiment.id).count() == 2
    finally:
        db.close()
        engine.dispose()


def test_sync_preserves_active_api_experiment(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "experiments").mkdir(parents=True)
    _write_run(root / "experiments", "run-a", "run-a")
    monkeypatch.setattr(importer, "PROJECT_ROOT", root)
    imports = importer._collect_runs()
    engine, db = _session()
    try:
        db.add(Experiment(config_id="run-a", status="running"))
        db.commit()

        stats = importer.sync_experiment_folders(
            db, imports, experiment_model=Experiment, training_run_model=TrainingRun
        )

        assert stats["skipped_running"] == 1
        assert db.query(Experiment).filter_by(config_id="run-a").one().status == "running"
        assert db.query(TrainingRun).count() == 0
    finally:
        db.close()
        engine.dispose()
