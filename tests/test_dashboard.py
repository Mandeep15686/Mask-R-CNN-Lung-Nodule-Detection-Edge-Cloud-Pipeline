from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.database import Base, EdgeNode, Experiment, InferenceResult, NetworkLog, TrainingRun
from backend.app.routes import dashboard, results


def _session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return engine, Session(engine)


def _run(coroutine):
    try:
        coroutine.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError("Expected dashboard handler to complete synchronously")


def test_dashboard_summary_empty_database_returns_zero_counts_and_null_metrics():
    engine, db = _session()
    try:
        result = _run(dashboard.summary(db))
        assert result["experiment_counts"] == {
            "total": 0, "running": 0, "pending": 0, "done": 0, "failed": 0,
        }
        assert result["inference_requests"] == 0
        assert result["active_edge_nodes"] == 0
        assert result["recorded_network_requests"] == 0
        assert result["best_experiment"] is None
        assert result["recent_experiments"] == []
    finally:
        db.close()
        engine.dispose()


def test_dashboard_summary_uses_metrics_from_same_best_epoch_and_real_counts():
    engine, db = _session()
    try:
        exp = Experiment(
            config_id="best-run",
            phase=2,
            alpha=0.3,
            beta=0.5,
            gamma=0.2,
            delta=0.0,
            epsilon=0.0,
            status="done",
            created_at=datetime.now(timezone.utc),
            started_at=datetime.now(timezone.utc) - timedelta(seconds=90),
            finished_at=datetime.now(timezone.utc),
            config_json={"cache_mode": "cached", "data_split": {"train_subsets": [1], "validation_subset": 0}},
        )
        db.add(exp)
        db.flush()
        db.add_all([
            TrainingRun(experiment_id=exp.id, epoch=1, val_dice=0.4, val_iou=0.2, val_hd95=20.0, val_boundary_f1=0.1),
            TrainingRun(experiment_id=exp.id, epoch=2, val_dice=0.7, val_iou=0.5, val_hd95=8.0, val_boundary_f1=0.6),
            TrainingRun(experiment_id=exp.id, epoch=3, val_dice=0.6, val_iou=0.4, val_hd95=2.0, val_boundary_f1=0.8),
        ])
        db.add(EdgeNode(node_id="online", hostname="edge-a", ip_address="127.0.0.1", is_online=True, last_seen=datetime.now(timezone.utc)))
        db.add(EdgeNode(node_id="stale", hostname="edge-b", ip_address="127.0.0.2", is_online=True, last_seen=datetime.now(timezone.utc) - timedelta(minutes=10)))
        db.add(InferenceResult(scan_uid="scan-a", nodules=[], source="cloud"))
        db.add(NetworkLog(node_id="online", latency_ms=5, bandwidth_mbps=100, payload_size_kb=12, inference_time_ms=8, split_layer="layer2", compression_method="int8", compression_ratio=2.0))
        db.commit()

        result = _run(dashboard.summary(db))
        best = result["best_experiment"]
        assert result["experiment_counts"]["done"] == 1
        assert result["inference_requests"] == 1
        assert result["active_edge_nodes"] == 1
        assert result["recorded_network_requests"] == 1
        assert best["best_epoch"] == 2
        assert best["best_dice"] == 0.7
        assert best["best_hd95"] == 8.0
        assert best["best_boundary_f1"] == 0.6
        assert best["best_specificity"] is None
        assert best["training_duration_s"] is not None
    finally:
        db.close()
        engine.dispose()


def test_dashboard_health_reports_checkpoint_state_and_cache_validation(monkeypatch):
    engine, db = _session()
    try:
        monkeypatch.setattr(dashboard, "runtime_data_info", lambda: {
            "cache_exists": True,
            "cache_valid": True,
            "cache_error": None,
            "cache_summary": {"total_patches": 12},
        })
        app = SimpleNamespace(state=SimpleNamespace(inference_engine=SimpleNamespace(status={
            "model_loaded": True,
            "checkpoint_path": "experiments/test/best_model.pth",
            "device": "cpu",
            "model_loading_error": None,
        })))
        request = SimpleNamespace(app=app)
        result = _run(dashboard.health(request, db))
        assert result["statuses"]["database"]["state"] == "healthy"
        assert result["statuses"]["inference"]["state"] == "healthy"
        assert result["statuses"]["checkpoint"]["detail"] == "experiments/test/best_model.pth"
        assert result["statuses"]["dataset_cache"]["state"] == "healthy"
        assert result["statuses"]["edge_nodes"]["state"] == "not_configured"
    finally:
        db.close()
        engine.dispose()


def test_dashboard_activity_contains_only_persisted_event_rows():
    engine, db = _session()
    try:
        exp = Experiment(config_id="run-a", status="failed", created_at=datetime.now(timezone.utc), finished_at=datetime.now(timezone.utc), config_json={})
        db.add(exp)
        db.flush()
        db.add(InferenceResult(scan_uid="scan-a", nodules=[{"score": 0.9}], source="edge"))
        db.commit()
        result = _run(dashboard.activity(50, db))
        assert {event["event"] for event in result} == {"Experiment created", "Experiment failed", "Inference performed"}
        assert next(event for event in result if event["event"] == "Inference performed")["source"] == "edge"
    finally:
        db.close()
        engine.dispose()


def test_leaderboard_filters_and_keeps_same_epoch_metrics():
    engine, db = _session()
    try:
        first = Experiment(config_id="phase1", phase=1, status="done", alpha=1, beta=0, gamma=0, delta=0, epsilon=0)
        second = Experiment(config_id="phase2", phase=2, status="failed", alpha=0.5, beta=0.5, gamma=0, delta=0, epsilon=0)
        db.add_all([first, second])
        db.flush()
        db.add_all([
            TrainingRun(experiment_id=first.id, epoch=1, val_dice=0.4, val_iou=0.2, val_hd95=2.0, val_boundary_f1=0.3),
            TrainingRun(experiment_id=first.id, epoch=2, val_dice=0.8, val_iou=0.6, val_hd95=9.0, val_boundary_f1=0.7),
            TrainingRun(experiment_id=first.id, epoch=3, val_dice=0.7, val_iou=0.5, val_hd95=1.0, val_boundary_f1=0.9),
        ])
        db.commit()

        leaderboard = _run(results.leaderboard(phase=1, status="done", db=db))
        assert len(leaderboard) == 1
        assert leaderboard[0]["config_id"] == "phase1"
        assert leaderboard[0]["best_dice"] == 0.8
        assert leaderboard[0]["best_hd95"] == 9.0
        assert leaderboard[0]["best_boundary_f1"] == 0.7
        assert leaderboard[0]["best_iou"] == 0.6
        assert _run(results.pareto_frontier(phase=1, status="done", db=db))[0]["config_id"] == "phase1"
    finally:
        db.close()
        engine.dispose()
