"""Experiment management routes (create, list, start, track progress)."""
import asyncio
import json
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from ..database import get_db, Experiment, TrainingRun
from ..json_utils import json_safe
from ..schemas import (
    ExperimentCreate, ExperimentResponse,
    TrainingRunResponse, ExperimentSummaryResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# WebSocket connections for live progress streaming
_ws_connections: dict[int, list[WebSocket]] = {}
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _experiment_run_dir(config_id: str) -> Path:
    # Keep generated artifacts under the repository's ignored experiments dir,
    # independent of whether Uvicorn was launched from backend/ or repo root.
    return PROJECT_ROOT / "experiments" / Path(config_id).name


# ---------------------------------------------------------------------------
# POST /api/experiments
# ---------------------------------------------------------------------------
@router.post("/", response_model=ExperimentResponse, status_code=201)
async def create_experiment(
    payload: ExperimentCreate,
    db: Session = Depends(get_db),
):
    if not payload.loss_weights.is_valid():
        raise HTTPException(status_code=400, detail="At least one loss weight must be > 0")

    # Check duplicate
    existing = db.query(Experiment).filter(Experiment.config_id == payload.config_id).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"config_id '{payload.config_id}' already exists")

    lw = payload.loss_weights
    exp = Experiment(
        config_id=payload.config_id,
        phase=payload.phase,
        alpha=lw.alpha,
        beta=lw.beta,
        gamma=lw.gamma,
        delta=lw.delta,
        epsilon=lw.epsilon,
        loss_description=_describe_loss(lw),
        status="pending",
        config_json={
            "epochs": payload.epochs,
            "batch_size": payload.batch_size,
            "fold": payload.fold,
            "lr": payload.lr,
            "data_root": payload.data_root,
        },
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)
    return exp


# ---------------------------------------------------------------------------
# GET /api/experiments
# ---------------------------------------------------------------------------
@router.get("/", response_model=list[ExperimentResponse])
async def list_experiments(
    phase: Optional[int] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    q = db.query(Experiment)
    if phase:
        q = q.filter(Experiment.phase == phase)
    if status:
        q = q.filter(Experiment.status == status)
    return q.order_by(Experiment.created_at.desc()).offset(offset).limit(limit).all()


@router.get("/{exp_id}", response_model=ExperimentResponse)
async def get_experiment(exp_id: int, db: Session = Depends(get_db)):
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp


@router.get("/{exp_id}/logs")
async def get_experiment_logs(exp_id: int, db: Session = Depends(get_db)):
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    config_id, status = exp.config_id, exp.status
    db.close()
    log_path = _experiment_run_dir(config_id) / "terminal.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    return {"experiment_id": exp_id, "status": status, "log": log_text}


@router.get("/{exp_id}/details")
async def get_experiment_details(exp_id: int, db: Session = Depends(get_db)):
    """Fetch experiment configuration, epoch metrics, and log in one request."""
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    runs = (
        db.query(TrainingRun)
        .filter(TrainingRun.experiment_id == exp_id)
        .order_by(TrainingRun.epoch)
        .all()
    )
    experiment_data = ExperimentResponse.model_validate(exp).model_dump()
    run_data = [
        {column.name: getattr(run, column.name) for column in TrainingRun.__table__.columns}
        for run in runs
    ]
    config_id = exp.config_id
    db.close()
    log_path = _experiment_run_dir(config_id) / "terminal.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    return json_safe({
        "experiment": experiment_data,
        "runs": run_data,
        "log": log_text,
    })


# ---------------------------------------------------------------------------
# POST /api/experiments/{exp_id}/start
# ---------------------------------------------------------------------------
@router.post("/{exp_id}/start")
async def start_experiment(
    exp_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if exp.status == "running":
        raise HTTPException(status_code=409, detail="Already running")

    exp.status = "running"
    exp.started_at = datetime.utcnow()
    db.commit()

    background_tasks.add_task(_run_experiment_bg, exp_id, exp.config_json or {})
    return {"message": f"Experiment {exp.config_id} started", "experiment_id": exp_id}


async def _run_experiment_bg(exp_id: int, config_json: dict):
    """Background task: run full training and log each epoch."""
    from ..database import SessionLocal
    event_loop = asyncio.get_running_loop()
    db = SessionLocal()
    log_handler = None
    try:
        exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
        if not exp:
            return

        experiment_id = exp.config_id
        experiment_config = exp.config_json or config_json or {}
        fold = experiment_config.get("fold", 0)
        # Build full config
        full_cfg = {
            **experiment_config,
            "config_id": experiment_id,
            "mask_loss": {
                "alpha": exp.alpha,
                "beta": exp.beta,
                "gamma": exp.gamma,
                "delta": exp.delta,
                "epsilon": exp.epsilon,
            },
        }

        # Do not hold a pooled DB connection during model initialization or
        # training. Epoch/status writes below each use short-lived sessions.
        db.close()
        db = None

        run_dir = _experiment_run_dir(experiment_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_handler = logging.FileHandler(run_dir / "terminal.log", mode="a", encoding="utf-8")
        log_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)

        try:
            import sys
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
            from training.trainer import Trainer

            logger.info("Starting experiment %s (id=%s)", experiment_id, exp_id)
            logger.info("Run directory: %s", run_dir)
            logger.info("Configuration: %s", json.dumps(full_cfg, sort_keys=True))

            class DBCallbackTrainer(Trainer):
                def _train_epoch(self, epoch):
                    self._latest_train_metrics = super()._train_epoch(epoch)
                    return self._latest_train_metrics

                def _val_epoch(self, epoch):
                    result = super()._val_epoch(epoch)
                    # Log to DB
                    train_metrics = getattr(self, "_latest_train_metrics", {})
                    train_metrics = {
                        **train_metrics,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                    _log_epoch_to_db(
                        exp_id, fold, epoch, train_metrics, result,
                    )
                    _broadcast_progress(exp_id, epoch, {
                        **train_metrics,
                        **result,
                    }, event_loop)
                    return result

            def train_run():
                trainer = DBCallbackTrainer(full_cfg, str(run_dir))
                return trainer.train()

            summary = await run_in_threadpool(train_run)

            _set_experiment_status(exp_id, "done")
            logger.info("Experiment %s completed successfully", experiment_id)

        except Exception as e:
            logger.exception(f"Training failed for experiment {exp_id}")
            _set_experiment_status(exp_id, "failed")
        finally:
            root_logger.removeHandler(log_handler)
            log_handler.close()

    finally:
        if db is not None:
            db.close()


def _set_experiment_status(exp_id: int, status: str):
    from ..database import SessionLocal
    db = SessionLocal()
    try:
        exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
        if exp:
            exp.status = status
            exp.finished_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()


def _log_epoch_to_db(exp_id: int, fold: int, epoch: int, train_m: dict, val_m: dict):
    from ..database import SessionLocal
    db = SessionLocal()
    try:
        run = TrainingRun(
            experiment_id=exp_id,
            fold=fold,
            epoch=epoch,
            train_total_loss=train_m.get("total_loss"),
            train_bce_loss=train_m.get("bce"),
            train_dice_loss=train_m.get("dice"),
            train_boundary_loss=train_m.get("boundary"),
            train_focal_loss=train_m.get("focal"),
            train_hd_loss=train_m.get("hd"),
            lr=train_m.get("lr"),
            val_dice=val_m.get("val_dice"),
            val_iou=val_m.get("val_iou"),
            val_hd95=val_m.get("val_hd95"),
            val_boundary_f1=val_m.get("val_boundary_f1"),
            val_sensitivity=val_m.get("val_sensitivity"),
            val_specificity=val_m.get("val_specificity"),
            val_asd=val_m.get("val_asd"),
            val_vol_err=val_m.get("val_vol_err"),
        )
        db.add(run)
        db.commit()
    finally:
        db.close()


async def _send_progress_to_clients(exp_id: int, message: str):
    """Send one progress message on the application event loop."""
    if exp_id not in _ws_connections:
        return
    dead = []
    for ws in list(_ws_connections[exp_id]):
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _ws_connections.get(exp_id, []):
            _ws_connections[exp_id].remove(ws)


def _broadcast_progress(exp_id: int, epoch: int, metrics: dict, event_loop):
    """Bridge worker-thread progress events to active WebSocket clients."""
    if not _ws_connections.get(exp_id) or not event_loop.is_running():
        return
    message = json.dumps(json_safe({"epoch": epoch, "metrics": metrics}), allow_nan=False)
    future = asyncio.run_coroutine_threadsafe(
        _send_progress_to_clients(exp_id, message), event_loop
    )
    try:
        future.result(timeout=5)
    except Exception:
        logger.debug("Could not broadcast progress for experiment %s", exp_id, exc_info=True)


# ---------------------------------------------------------------------------
# GET /api/experiments/{exp_id}/runs
# ---------------------------------------------------------------------------
@router.get("/{exp_id}/runs", response_model=list[TrainingRunResponse])
async def get_runs(exp_id: int, db: Session = Depends(get_db)):
    runs = (
        db.query(TrainingRun)
        .filter(TrainingRun.experiment_id == exp_id)
        .order_by(TrainingRun.epoch)
        .all()
    )
    return json_safe([
        {column.name: getattr(run, column.name) for column in TrainingRun.__table__.columns}
        for run in runs
    ])


# ---------------------------------------------------------------------------
# GET /api/experiments/{exp_id}/summary
# ---------------------------------------------------------------------------
@router.get("/{exp_id}/summary", response_model=ExperimentSummaryResponse)
async def get_summary(exp_id: int, db: Session = Depends(get_db)):
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    runs = (
        db.query(TrainingRun)
        .filter(TrainingRun.experiment_id == exp_id)
        .order_by(TrainingRun.epoch)
        .all()
    )
    if not runs:
        best_dice, best_epoch, final_hd95, final_bdf1 = 0.0, 0, None, None
    else:
        best_run = max(runs, key=lambda r: r.val_dice or 0.0)
        last_run = runs[-1]
        best_dice = best_run.val_dice or 0.0
        best_epoch = best_run.epoch
        final_hd95 = last_run.val_hd95
        final_bdf1 = last_run.val_boundary_f1

    return json_safe(ExperimentSummaryResponse(
        experiment_id=exp.id,
        config_id=exp.config_id,
        loss_description=exp.loss_description or "",
        best_dice=best_dice,
        best_epoch=best_epoch,
        final_hd95=final_hd95,
        final_boundary_f1=final_bdf1,
        total_epochs=len(runs),
        status=exp.status,
    ).model_dump())


# ---------------------------------------------------------------------------
# WebSocket /api/experiments/{exp_id}/ws
# ---------------------------------------------------------------------------
@router.websocket("/{exp_id}/ws")
async def experiment_ws(exp_id: int, websocket: WebSocket):
    await websocket.accept()
    if exp_id not in _ws_connections:
        _ws_connections[exp_id] = []
    _ws_connections[exp_id].append(websocket)
    try:
        while True:
            await asyncio.sleep(1)
            await websocket.send_text(json.dumps({"ping": True}))
    except WebSocketDisconnect:
        if exp_id in _ws_connections:
            _ws_connections[exp_id].remove(websocket)


# ---------------------------------------------------------------------------
# DELETE /api/experiments/{exp_id}
# ---------------------------------------------------------------------------
@router.delete("/{exp_id}", status_code=204)
async def delete_experiment(exp_id: int, db: Session = Depends(get_db)):
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if exp.status == "running":
        raise HTTPException(status_code=409, detail="Cannot delete a running experiment")
    db.delete(exp)
    db.commit()


def _describe_loss(lw) -> str:
    names = [("alpha", "BCE"), ("beta", "Dice"), ("gamma", "Boundary"),
             ("delta", "Focal"), ("epsilon", "HD95")]
    parts = [f"{getattr(lw, a):.2f}·{n}" for a, n in names if getattr(lw, a) > 0]
    return " + ".join(parts)
