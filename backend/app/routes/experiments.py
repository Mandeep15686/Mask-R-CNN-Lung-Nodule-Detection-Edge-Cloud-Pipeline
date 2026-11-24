"""Experiment management routes (create, list, start, track progress)."""
import asyncio
import json
import logging
import os
import shutil
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from ..database import get_db, Experiment, TrainingRun
from ..data_config import (
    available_subset_uids,
    resolve_cache_dir,
    resolve_data_root,
    resolve_subset_split,
    validate_cached_split,
)
from ..json_utils import json_safe
from ..schemas import (
    ExperimentCreate, ExperimentResponse,
    TrainingRunResponse, ExperimentSummaryResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# WebSocket connections for live progress streaming
_ws_connections: dict[int, list[WebSocket]] = {}
_experiment_log_context = threading.local()


class _ExperimentLogFilter(logging.Filter):
    def __init__(self, experiment_id: int):
        super().__init__()
        self.experiment_id = experiment_id

    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(_experiment_log_context, "experiment_id", None) == self.experiment_id
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _experiment_run_dir(config_id: str) -> Path:
    # Keep generated artifacts under the repository's ignored experiments dir,
    # independent of whether Uvicorn was launched from backend/ or repo root.
    return PROJECT_ROOT / "experiments" / Path(config_id).name


def _experiment_artifact_dir(exp: Experiment) -> Path:
    """Resolve imported runs to their source folder, with traversal protection."""
    config = exp.config_json or {}
    relative_dir = config.get("artifact_dir")
    if not relative_dir:
        return _experiment_run_dir(exp.config_id)

    project_root = PROJECT_ROOT.resolve()
    artifact_dir = (project_root / str(relative_dir)).resolve()
    allowed_roots = [
        (project_root / name).resolve()
        for name in (
            "experiments",
            "experiments_phase2_5ep_20cfg",
            "experiments_phase2_screen",
            "experiments_phase2_v3_5ep_20cfg",
        )
    ]
    if not any(root == artifact_dir or root in artifact_dir.parents for root in allowed_roots):
        raise ValueError(f"Experiment artifact path is outside the configured run folders: {relative_dir}")
    return artifact_dir


def _prepare_training_configuration(
    config: dict,
    *,
    require_resolved_cache: bool = False,
) -> dict:
    """Resolve and validate the exact data paths and split used by Trainer."""
    prepared = dict(config)
    mode = prepared.get("cache_mode")
    if mode is None:
        # Existing database rows created before cache_mode was introduced keep
        # their explicit cache setting or their prior raw-volume behavior.
        mode = "cached" if prepared.get("cache_dir") else "raw"
    if mode not in {"cached", "raw"}:
        raise ValueError(f"Unsupported cache_mode: {mode!r}")

    data_root = resolve_data_root(prepared.get("data_root"))
    subset_uids = available_subset_uids(data_root)
    split = resolve_subset_split(
        subset_uids,
        train_subsets=prepared.get("train_subsets"),
        validation_subset=prepared.get("validation_subset"),
        test_subset=prepared.get("test_subset"),
        fold=int(prepared.get("fold", 0)),
    )
    previous_split = prepared.get("data_split")
    if isinstance(previous_split, dict):
        for key in ("train_scan_uids", "validation_scan_uids", "test_scan_uids"):
            if key in previous_split and set(previous_split[key]) != set(split[key]):
                raise ValueError(
                    f"Raw dataset scan membership changed since experiment creation ({key})"
                )

    cache_dir = None
    cache_summary = None
    if mode == "cached":
        if require_resolved_cache and not prepared.get("cache_dir"):
            raise ValueError("This experiment requires cached training but has no persisted cache_dir")
        cache_dir = resolve_cache_dir(prepared.get("cache_dir"))
        cache_summary = validate_cached_split(
            data_root,
            cache_dir,
            int(prepared.get("patch_size", 64)),
            split,
        )
        logger.info(
            "Validated patch cache at %s: patches=%d (+%d/- %d), shape=%s, "
            "subsets=%s, scan_uids=%d, spacing_mm=%s",
            cache_dir,
            cache_summary["total_patches"],
            cache_summary["positive_patches"],
            cache_summary["negative_patches"],
            cache_summary["patch_dimensions"],
            cache_summary["available_subsets"],
            cache_summary["scan_uid_count"],
            cache_summary["spacing_mm"],
        )
    else:
        if prepared.get("cache_dir"):
            raise ValueError("cache_dir cannot be set when cache_mode is 'raw'")
        if not (data_root / "annotations.csv").is_file():
            raise FileNotFoundError(f"Raw-volume mode requires {data_root / 'annotations.csv'}")

    prepared.update({
        "data_root": str(data_root),
        "cache_mode": mode,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "patch_size": int(prepared.get("patch_size", 64)),
        "train_subsets": split["train_subsets"],
        "validation_subset": split["validation_subset"],
        "test_subset": split["test_subset"],
        "fold": split["validation_subset"],
        "data_split": split,
        "cache_summary": cache_summary,
    })
    return prepared


def _write_cache_validation_failure(config_id: str, message: str) -> None:
    run_dir = _experiment_run_dir(config_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "terminal.log").open("a", encoding="utf-8") as log:
        log.write(f"Cache/data validation failed before training: {message}\n")


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

    try:
        experiment_config = _prepare_training_configuration({
            "data_root": payload.data_root,
            "cache_dir": payload.cache_dir,
            "cache_mode": payload.cache_mode,
            "patch_size": payload.patch_size,
            "fold": payload.fold,
            "train_subsets": payload.train_subsets,
            "validation_subset": payload.validation_subset,
            "test_subset": payload.test_subset,
            "epochs": payload.epochs,
            "batch_size": payload.batch_size,
            "lr": payload.lr,
            "training_strategy": payload.training_strategy,
            "mgda_gradient_normalization": payload.mgda_gradient_normalization,
            "mgda_parameter_scope": payload.mgda_parameter_scope,
        })
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

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
        loss_description=(
            "MGDA (adaptive BCE, Dice, Boundary, Focal, and HD-distance surrogate)"
            if payload.training_strategy == "mgda"
            else _describe_loss(lw)
        ),
        status="pending",
        config_json=experiment_config,
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
    limit: int = Query(500, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    q = db.query(Experiment)
    if phase:
        q = q.filter(Experiment.phase == phase)
    if status:
        q = q.filter(Experiment.status == status)
    return q.order_by(Experiment.created_at.desc()).offset(offset).limit(limit).all()


@router.post("/sync-artifacts")
async def sync_experiment_artifacts(db: Session = Depends(get_db)):
    """Refresh website records from saved experiment folders and their histories."""
    from ..artifact_sync import sync_saved_experiment_artifacts

    try:
        stats = sync_saved_experiment_artifacts(db)
    except Exception as exc:
        db.rollback()
        logger.exception("Experiment artifact sync failed")
        raise HTTPException(status_code=500, detail=f"Experiment artifact sync failed: {exc}") from exc
    return {"status": "ok", **stats}


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
    log_path = _experiment_artifact_dir(exp) / "terminal.log"
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
    log_path = _experiment_artifact_dir(exp) / "terminal.log"
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
    request: Request,
    db: Session = Depends(get_db),
):
    exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if exp.status == "running":
        raise HTTPException(status_code=409, detail="Already running")

    try:
        prepared_config = _prepare_training_configuration(
            exp.config_json or {},
            require_resolved_cache=True,
        )
    except (FileNotFoundError, ValueError) as exc:
        exp.status = "failed"
        exp.finished_at = datetime.now(timezone.utc)
        db.commit()
        _write_cache_validation_failure(exp.config_id, str(exc))
        raise HTTPException(status_code=422, detail=f"Experiment data validation failed: {exc}") from exc

    exp.config_json = prepared_config
    exp.status = "running"
    exp.started_at = datetime.now(timezone.utc)
    db.commit()

    background_tasks.add_task(
        _run_experiment_bg, exp_id, prepared_config,
        request.app.state.inference_engine,
    )
    return {"message": f"Experiment {exp.config_id} started", "experiment_id": exp_id}


async def _run_experiment_bg(exp_id: int, config_json: dict, inference_engine=None):
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
        fold = experiment_config.get("validation_subset", experiment_config.get("fold", 0))

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
        log_handler.addFilter(_ExperimentLogFilter(exp_id))
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)

        try:
            # Revalidate at execution time as the cache may have changed after
            # creation or even after the start endpoint checked it.
            full_cfg = _prepare_training_configuration(
                experiment_config,
                require_resolved_cache=True,
            )
            full_cfg["config_id"] = experiment_id
            full_cfg["mask_loss"] = {
                "alpha": exp.alpha,
                "beta": exp.beta,
                "gamma": exp.gamma,
                "delta": exp.delta,
                "epsilon": exp.epsilon,
            }
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
                previous_id = getattr(_experiment_log_context, "experiment_id", None)
                _experiment_log_context.experiment_id = exp_id
                try:
                    trainer = DBCallbackTrainer(full_cfg, str(run_dir))
                    # Persist the resolved UID lists (not just subset numbers)
                    # before training and checkpoint creation.
                    from ..database import SessionLocal as SplitSessionLocal
                    split_db = SplitSessionLocal()
                    try:
                        split_exp = split_db.query(Experiment).filter(
                            Experiment.id == exp_id
                        ).first()
                        if split_exp:
                            split_config = dict(split_exp.config_json or {})
                            split_config["data_split"] = trainer.data_split
                            split_exp.config_json = split_config
                            split_db.commit()
                    finally:
                        split_db.close()
                    return trainer.train()
                finally:
                    _experiment_log_context.experiment_id = previous_id

            summary = await run_in_threadpool(train_run)

            best_checkpoint = run_dir / "best_model.pth"
            if inference_engine is not None and best_checkpoint.exists():
                canonical_checkpoint = PROJECT_ROOT / "experiments" / "best_model.pth"
                temporary_checkpoint = canonical_checkpoint.with_name(
                    f"best_model.{exp_id}.tmp"
                )
                shutil.copy2(best_checkpoint, temporary_checkpoint)
                temporary_checkpoint.replace(canonical_checkpoint)
                inference_engine.load_checkpoint(str(canonical_checkpoint))

            _set_experiment_status(exp_id, "done", summary=summary)
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


def _set_experiment_status(exp_id: int, status: str, summary: Optional[dict] = None):
    from ..database import SessionLocal
    db = SessionLocal()
    try:
        exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
        if exp:
            exp.status = status
            exp.finished_at = datetime.now(timezone.utc)
            if summary and summary.get("training_strategy") == "mgda":
                weights = summary.get("loss_weights", {})
                exp.alpha = weights.get("alpha", exp.alpha)
                exp.beta = weights.get("beta", exp.beta)
                exp.gamma = weights.get("gamma", exp.gamma)
                exp.delta = weights.get("delta", exp.delta)
                exp.epsilon = weights.get("epsilon", exp.epsilon)
                exp.loss_description = (
                    "MGDA final adaptive weights: "
                    f"BCE={exp.alpha:.3f}, Dice={exp.beta:.3f}, "
                    f"Boundary={exp.gamma:.3f}, Focal={exp.delta:.3f}, "
                    f"HD-distance={exp.epsilon:.3f}"
                )
                config = dict(exp.config_json or {})
                config["adaptive_loss_weights"] = weights
                exp.config_json = config
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
            train_bce_loss=train_m.get("bce", train_m.get("mgda_loss_bce")),
            train_dice_loss=train_m.get("dice", train_m.get("mgda_loss_dice")),
            train_boundary_loss=train_m.get("boundary", train_m.get("mgda_loss_boundary")),
            train_focal_loss=train_m.get("focal", train_m.get("mgda_loss_focal")),
            train_hd_loss=train_m.get("hd", train_m.get("mgda_loss_hd")),
            lr=train_m.get("lr"),
            val_dice=val_m.get("val_dice"),
            val_iou=val_m.get("val_iou"),
            val_hd95=val_m.get("val_hd95"),
            val_boundary_f1=val_m.get("val_boundary_f1"),
            val_sensitivity=val_m.get("val_sensitivity"),
            val_specificity=val_m.get("val_specificity"),
            val_asd=val_m.get("val_asd"),
            val_vol_err=val_m.get("val_vol_err"),
            val_total_loss=val_m.get("val_total"),
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
        pass
    finally:
        connections = _ws_connections.get(exp_id, [])
        if websocket in connections:
            connections.remove(websocket)
        if not connections:
            _ws_connections.pop(exp_id, None)


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
