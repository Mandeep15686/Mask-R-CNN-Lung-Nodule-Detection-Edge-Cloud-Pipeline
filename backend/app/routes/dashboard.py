"""Research dashboard summaries built only from persisted runtime data."""
from __future__ import annotations

import math
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from ..data_config import runtime_data_info
from ..database import EdgeNode, Experiment, InferenceResult, NetworkLog, TrainingRun, get_db
from ..schemas import DashboardActivityEvent, DashboardHealthResponse, DashboardSummaryResponse, SystemMetricsResponse

router = APIRouter()
ARCHITECTURE = "Mask R-CNN ResNet-50-FPN"
HEARTBEAT_TIMEOUT_SECONDS = int(os.getenv("EDGE_HEARTBEAT_TIMEOUT_SECONDS", "60"))


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _node_online(node: EdgeNode, now: datetime) -> bool:
    last_seen = _utc(node.last_seen)
    return bool(last_seen and now - last_seen <= timedelta(seconds=HEARTBEAT_TIMEOUT_SECONDS))


@router.get("/health", response_model=DashboardHealthResponse)
async def health(request: Request, db: Session = Depends(get_db)):
    now = datetime.now(timezone.utc)
    statuses: dict[str, dict[str, Any]] = {
        "api": {"state": "healthy", "detail": "Dashboard API is responding"},
    }
    try:
        db.execute(text("SELECT 1"))
        statuses["database"] = {"state": "healthy", "detail": "Database query succeeded"}
    except Exception as exc:
        statuses["database"] = {"state": "unavailable", "detail": f"Database query failed: {exc}"}

    engine = getattr(request.app.state, "inference_engine", None)
    engine_status = getattr(engine, "status", {}) if engine is not None else {}
    loaded = bool(engine_status.get("model_loaded"))
    statuses["inference"] = {
        "state": "healthy" if loaded else "unavailable",
        "detail": "A trained checkpoint is loaded" if loaded else (
            engine_status.get("model_loading_error") or "No trained checkpoint is loaded"
        ),
    }
    checkpoint_path = engine_status.get("checkpoint_path")
    statuses["checkpoint"] = {
        "state": "healthy" if loaded else "unavailable" if checkpoint_path else "not_configured",
        "detail": checkpoint_path if loaded else engine_status.get("model_loading_error") or ("Configured checkpoint did not load" if checkpoint_path else "No checkpoint configured"),
    }
    device = engine_status.get("device")
    gpu_name = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            statuses["gpu"] = {"state": "healthy", "detail": gpu_name}
        else:
            statuses["gpu"] = {"state": "not_configured", "detail": "CUDA unavailable; inference can use CPU"}
    except Exception as exc:
        statuses["gpu"] = {"state": "unknown", "detail": f"GPU discovery unavailable: {exc}"}

    data = runtime_data_info()
    if data.get("cache_valid"):
        statuses["dataset_cache"] = {
            "state": "healthy",
            "detail": f"Validated cache: {data['cache_summary']['total_patches']} patches",
        }
    elif data.get("cache_exists"):
        statuses["dataset_cache"] = {"state": "degraded", "detail": data.get("cache_error") or "Cache is invalid"}
    else:
        statuses["dataset_cache"] = {"state": "unavailable", "detail": data.get("cache_error") or "Cache is missing"}

    nodes = db.query(EdgeNode).all() if statuses["database"]["state"] == "healthy" else []
    online_count = sum(_node_online(node, now) for node in nodes)
    statuses["edge_nodes"] = {
        "state": "healthy" if online_count else "not_configured" if not nodes else "unavailable",
        "detail": f"{online_count} of {len(nodes)} node(s) have a heartbeat within {HEARTBEAT_TIMEOUT_SECONDS}s",
    }
    required = ("api", "database", "dataset_cache")
    if all(statuses[key]["state"] == "healthy" for key in required) and loaded:
        overall = "healthy"
    elif statuses["api"]["state"] == "healthy" and statuses["database"]["state"] == "healthy":
        overall = "degraded"
    else:
        overall = "unavailable"
    return {
        "overall": overall,
        "updated_at": now,
        "architecture": ARCHITECTURE,
        "device": device,
        "gpu_name": gpu_name,
        "statuses": statuses,
    }


@router.get("/summary", response_model=DashboardSummaryResponse)
async def summary(db: Session = Depends(get_db)):
    now = datetime.now(timezone.utc)
    all_experiments = db.query(Experiment).order_by(Experiment.created_at.desc()).all()
    experiments = all_experiments[:100]
    counts = {"total": len(all_experiments), "running": 0, "pending": 0, "done": 0, "failed": 0}
    for exp in all_experiments:
        counts[exp.status if exp.status in counts and exp.status != "total" else "pending"] += 1

    ids = [exp.id for exp in all_experiments]
    runs = (
        db.query(TrainingRun)
        .filter(TrainingRun.experiment_id.in_(ids))
        .order_by(TrainingRun.epoch)
        .all()
    ) if ids else []
    best_by_exp: dict[int, TrainingRun] = {}
    for run in runs:
        score = run.val_dice
        if score is None or not math.isfinite(float(score)):
            continue
        previous = best_by_exp.get(run.experiment_id)
        if previous is None or score > (previous.val_dice if previous.val_dice is not None else float("-inf")):
            best_by_exp[run.experiment_id] = run

    def experiment_row(exp: Experiment) -> dict[str, Any]:
        cfg = exp.config_json or {}
        best = best_by_exp.get(exp.id)
        start, finish = _utc(exp.started_at), _utc(exp.finished_at)
        checkpoint = Path(__file__).resolve().parents[3] / "experiments" / exp.config_id / "best_model.pth"
        return {
            "id": exp.id,
            "config_id": exp.config_id,
            "phase": exp.phase,
            "status": exp.status,
            "created_at": _utc(exp.created_at),
            "started_at": start,
            "finished_at": finish,
            "epochs_completed": sum(run.experiment_id == exp.id for run in runs),
            "alpha": exp.alpha,
            "beta": exp.beta,
            "gamma": exp.gamma,
            "delta": exp.delta,
            "epsilon": exp.epsilon,
            "loss_description": exp.loss_description,
            "best_epoch": best.epoch if best else None,
            "best_dice": best.val_dice if best else None,
            "best_iou": best.val_iou if best else None,
            "best_hd95": best.val_hd95 if best else None,
            "best_boundary_f1": best.val_boundary_f1 if best else None,
            "best_sensitivity": best.val_sensitivity if best else None,
            "best_specificity": best.val_specificity if best else None,
            "best_asd": best.val_asd if best else None,
            "best_volume_error": best.val_vol_err if best else None,
            "training_duration_s": (finish - start).total_seconds() if start and finish else None,
            "loss_description": exp.loss_description,
            "data_split": cfg.get("data_split"),
            "cache_dir": cfg.get("cache_dir"),
            "cache_mode": cfg.get("cache_mode"),
            "checkpoint_path": str(checkpoint),
            "checkpoint_available": checkpoint.is_file(),
        }

    recent = [experiment_row(exp) for exp in experiments]
    best_exp = max(
        (experiment_row(exp) for exp in all_experiments if best_by_exp.get(exp.id) is not None),
        key=lambda row: row["best_dice"],
        default=None,
    )
    online_nodes = sum(_node_online(node, now) for node in db.query(EdgeNode).all())
    return {
        "experiment_counts": counts,
        "inference_requests": db.query(func.count(InferenceResult.id)).scalar() or 0,
        "active_edge_nodes": online_nodes,
        "recorded_network_requests": db.query(func.count(NetworkLog.id)).scalar() or 0,
        "best_experiment": best_exp,
        "recent_experiments": recent,
    }


@router.get("/activity", response_model=list[DashboardActivityEvent])
async def activity(limit: int = 50, db: Session = Depends(get_db)):
    limit = min(max(limit, 1), 200)
    events: list[dict[str, Any]] = []
    for exp in db.query(Experiment).all():
        if exp.created_at:
            events.append({"timestamp": _utc(exp.created_at), "event": "Experiment created", "detail": exp.config_id, "source": "DIP", "entity_id": str(exp.id)})
        if exp.started_at:
            events.append({"timestamp": _utc(exp.started_at), "event": "Training started", "detail": exp.config_id, "source": "DIP", "entity_id": str(exp.id)})
        if exp.finished_at:
            label = "Experiment completed" if exp.status == "done" else "Experiment failed" if exp.status == "failed" else "Experiment stopped"
            events.append({"timestamp": _utc(exp.finished_at), "event": label, "detail": exp.config_id, "source": "DIP", "entity_id": str(exp.id)})
    for run in db.query(TrainingRun).order_by(TrainingRun.created_at.desc()).limit(limit).all():
        events.append({"timestamp": _utc(run.created_at), "event": "Epoch completed", "detail": f"Epoch {run.epoch} · experiment #{run.experiment_id}", "source": "DIP", "entity_id": str(run.experiment_id)})
    for inference in db.query(InferenceResult).order_by(InferenceResult.created_at.desc()).limit(limit).all():
        events.append({"timestamp": _utc(inference.created_at), "event": "Inference performed", "detail": f"{inference.scan_uid} · {len(inference.nodules or [])} detection(s)", "source": inference.source or "cloud", "entity_id": str(inference.id)})
    for node in db.query(EdgeNode).all():
        if node.created_at:
            events.append({"timestamp": _utc(node.created_at), "event": "Edge node registered", "detail": node.node_id, "source": "CN", "entity_id": node.node_id})
    for log in db.query(NetworkLog).order_by(NetworkLog.timestamp.desc()).limit(limit).all():
        events.append({"timestamp": _utc(log.timestamp), "event": "Network measurement recorded", "detail": f"{log.split_layer} / {log.compression_method} · {log.payload_size_kb:g} KB", "source": "CN", "entity_id": str(log.id)})
    events.sort(key=lambda event: event["timestamp"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return events[:limit]


@router.get("/dataset")
async def dataset_status():
    return runtime_data_info()


@router.get("/system-metrics", response_model=SystemMetricsResponse)
async def system_metrics():
    metrics: dict[str, Any] = {
        "cpu_percent": None,
        "ram_used_bytes": None,
        "ram_total_bytes": None,
        "gpu_name": None,
        "gpu_memory_used_bytes": None,
        "gpu_memory_total_bytes": None,
        "gpu_utilization_percent": None,
        "gpu_temperature_c": None,
        "updated_at": datetime.now(timezone.utc),
    }
    try:
        import psutil
        metrics["cpu_percent"] = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        metrics["ram_used_bytes"] = memory.used
        metrics["ram_total_bytes"] = memory.total
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            metrics["gpu_name"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info(0)
            metrics["gpu_memory_used_bytes"] = total - free
            metrics["gpu_memory_total_bytes"] = total
    except Exception:
        pass
    return metrics
