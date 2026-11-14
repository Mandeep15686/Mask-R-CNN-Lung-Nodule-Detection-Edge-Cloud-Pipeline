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
    log_path = _experiment_run_dir(exp.config_id) / "terminal.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    return {"experiment_id": exp_id, "status": exp.status, "log": log_text}


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
    db = SessionLocal()
    try:
        exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
        if not exp:
            return

        # Build full config
        full_cfg = {
            **config_json,
            "config_id": exp.config_id,
            "mask_loss": {
                "alpha": exp.alpha,
                "beta": exp.beta,
                "gamma": exp.gamma,
                "delta": exp.delta,
                "epsilon": exp.epsilon,
            },
        }

        run_dir = _experiment_run_dir(exp.config_id)
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

            logger.info("Starting experiment %s (id=%s)", exp.config_id, exp_id)
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
                        db, exp_id, exp.config_json.get("fold", 0),
                        epoch, train_metrics, result,
                    )
                    _broadcast_progress(exp_id, epoch, {
                        **train_metrics,
                        **result,
                    })
                    return result

            trainer = DBCallbackTrainer(full_cfg, str(run_dir))
            summary = trainer.train()

            exp.status = "done"
            exp.finished_at = datetime.utcnow()
            db.commit()
            logger.info("Experiment %s completed successfully", exp.config_id)

        except Exception as e:
            logger.exception(f"Training failed for experiment {exp_id}")
            exp.status = "failed"
            exp.finished_at = datetime.utcnow()
            db.commit()
        finally:
            root_logger.removeHandler(log_handler)
            log_handler.close()

    finally:
        db.close()


def _log_epoch_to_db(db, exp_id: int, fold: int, epoch: int, train_m: dict, val_m: dict):
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


def _broadcast_progress(exp_id: int, epoch: int, metrics: dict):
    """Send epoch update to all WebSocket listeners."""
    if exp_id not in _ws_connections:
        return
    msg = json.dumps(json_safe({"epoch": epoch, "metrics": metrics}), allow_nan=False)
    dead = []
    for ws in _ws_connections[exp_id]:
        try:
            asyncio.get_event_loop().run_until_complete(ws.send_text(msg))
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_connections[exp_id].remove(ws)


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
