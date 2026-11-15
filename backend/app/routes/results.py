"""Results aggregation and comparison analytics.

FIXES APPLIED
=============
FIX-8  Cross-epoch metric mixing in leaderboard / Pareto:
  BEFORE: SELECT max(val_dice), min(val_hd95), max(val_boundary_f1) in one GROUP BY.
          These three aggregates can come from three *different* epochs, producing
          a row that was never achieved in any single checkpoint.
  AFTER:  For each experiment we find the epoch with the best primary objective
          (val_dice), then report all metrics for *that one epoch*.
          The Pareto endpoint uses the same per-epoch row approach.

FIX-9  Falsy check converts valid zero to None:
  BEFORE: round(r.best_dice, 4) if r.best_dice else None
          A Dice of 0.0 is a valid (bad) result; the old check mapped it to None.
  AFTER:  Explicit `is not None` check throughout.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..database import get_db, Experiment, TrainingRun, InferenceResult
from ..json_utils import json_safe

router = APIRouter()


def _leaderboard_rows(db: Session, phase: int | None = None, status: str | None = None):
    """Build one best-Dice-epoch row per experiment with a bulk run query."""
    query = db.query(Experiment)
    if phase is not None:
        query = query.filter(Experiment.phase == phase)
    if status:
        query = query.filter(Experiment.status == status)
    experiments = query.all()
    ids = [exp.id for exp in experiments]
    runs = (
        db.query(TrainingRun)
        .filter(TrainingRun.experiment_id.in_(ids))
        .order_by(TrainingRun.epoch)
        .all()
    ) if ids else []
    best_by_exp = {}
    for run in runs:
        if run.val_dice is None:
            continue
        old = best_by_exp.get(run.experiment_id)
        if old is None or run.val_dice > old.val_dice:
            best_by_exp[run.experiment_id] = run

    rows = []
    for exp in experiments:
        best = best_by_exp.get(exp.id)
        duration = None
        if exp.started_at and exp.finished_at:
            start = exp.started_at
            finish = exp.finished_at
            if start.tzinfo is None:
                from datetime import timezone
                start = start.replace(tzinfo=timezone.utc)
            if finish.tzinfo is None:
                from datetime import timezone
                finish = finish.replace(tzinfo=timezone.utc)
            duration = max(0.0, (finish - start).total_seconds())
        rows.append({
            "experiment_id": exp.id,
            "config_id": exp.config_id,
            "phase": exp.phase,
            "alpha": exp.alpha,
            "beta": exp.beta,
            "gamma": exp.gamma,
            "delta": exp.delta,
            "epsilon": exp.epsilon,
            "loss_description": exp.loss_description,
            "status": exp.status,
            "best_dice": best.val_dice if best is not None else None,
            "best_iou": best.val_iou if best is not None else None,
            "best_hd95": best.val_hd95 if best is not None else None,
            "best_boundary_f1": best.val_boundary_f1 if best is not None else None,
            "best_sensitivity": best.val_sensitivity if best is not None else None,
            "best_specificity": best.val_specificity if best is not None else None,
            "best_asd": best.val_asd if best is not None else None,
            "best_volume_error": best.val_vol_err if best is not None else None,
            "best_epoch": best.epoch if best is not None else None,
            "training_duration_s": duration,
        })
    rows.sort(key=lambda row: row["best_dice"] if row["best_dice"] is not None else -1.0, reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


@router.get("/leaderboard")
async def leaderboard(phase: int | None = None, status: str | None = None, db: Session = Depends(get_db)):
    """
    Return all experiments ranked by best-epoch validation Dice score.
    All three metrics (Dice, HD95, BF1) are taken from the *same* epoch.
    """
    rows = _leaderboard_rows(db, phase=phase, status=status)
    for row in rows:
        for key, digits in (("best_dice", 4), ("best_iou", 4), ("best_boundary_f1", 4), ("best_sensitivity", 4), ("best_specificity", 4), ("best_hd95", 2), ("best_asd", 2), ("best_volume_error", 2)):
            if row[key] is not None:
                row[key] = round(row[key], digits)
    return json_safe(rows)


@router.get("/pareto")
async def pareto_frontier(phase: int | None = None, status: str | None = None, db: Session = Depends(get_db)):
    """
    Compute Pareto frontier: experiments where no other experiment
    dominates on both Dice (↑) and HD95 (↓).

    FIX-8: metrics are taken from the same best-Dice epoch for each experiment.
    """
    rows = []
    for row in _leaderboard_rows(db, phase=phase, status=status):
        if row["best_dice"] is None or row["best_hd95"] is None:
            continue
        rows.append(row)

    pareto = []
    for r in rows:
        dominated = any(
            other["best_dice"] >= r["best_dice"]
            and other["best_hd95"] <= r["best_hd95"]
            and (
                other["best_dice"] > r["best_dice"]
                or other["best_hd95"] < r["best_hd95"]
            )
            and other["experiment_id"] != r["experiment_id"]
            for other in rows
        )
        if not dominated:
            pareto.append({
                "experiment_id":    r["experiment_id"],
                "config_id":        r["config_id"],
                "loss_description": r["loss_description"],
                "phase": r["phase"],
                "status": r["status"],
                "best_dice":        round(r["best_dice"], 4),
                "best_hd95":        round(r["best_hd95"], 2),
                "best_boundary_f1": round(r["best_boundary_f1"], 4) if r["best_boundary_f1"] is not None else None,  # FIX-9
            })

    return json_safe(pareto)


@router.get("/stats")
async def stats(db: Session = Depends(get_db)):
    """High-level statistics for the dashboard."""
    n_experiments = db.query(func.count(Experiment.id)).scalar()
    n_running     = db.query(func.count(Experiment.id)).filter(Experiment.status == "running").scalar()
    n_done        = db.query(func.count(Experiment.id)).filter(Experiment.status == "done").scalar()
    n_inferences  = db.query(func.count(InferenceResult.id)).scalar()
    best_dice_row = db.query(func.max(TrainingRun.val_dice)).scalar()
    return json_safe({
        "total_experiments": n_experiments,
        "running":           n_running,
        "done":              n_done,
        "total_inferences":  n_inferences,
        # FIX-9: explicit is not None
        "best_dice_overall": round(best_dice_row, 4) if best_dice_row is not None else None,
    })


@router.get("/experiment/{exp_id}/learning-curve")
async def learning_curve(exp_id: int, db: Session = Depends(get_db)):
    """Return epoch-by-epoch metrics for plotting."""
    runs = (
        db.query(TrainingRun)
        .filter(TrainingRun.experiment_id == exp_id)
        .order_by(TrainingRun.epoch)
        .all()
    )
    if not runs:
        raise HTTPException(status_code=404, detail="No runs for this experiment")
    return json_safe({
        "epochs":          [r.epoch             for r in runs],
        "train_loss":      [r.train_total_loss   for r in runs],
        "val_dice":        [r.val_dice           for r in runs],
        "val_hd95":        [r.val_hd95           for r in runs],
        "val_boundary_f1": [r.val_boundary_f1    for r in runs],
        "val_sensitivity": [r.val_sensitivity     for r in runs],
        "lr":              [r.lr                 for r in runs],
    })
