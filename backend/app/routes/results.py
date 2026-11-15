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


def _best_epoch_row(db: Session, exp_id: int):
    """
    FIX-8: Return the single TrainingRun row that achieved the highest val_dice
    for the given experiment.  All metrics on the returned row correspond to
    the same checkpoint.
    """
    return (
        db.query(TrainingRun)
        .filter(TrainingRun.experiment_id == exp_id)
        .order_by(TrainingRun.val_dice.desc().nullslast())
        .first()
    )


@router.get("/leaderboard")
async def leaderboard(db: Session = Depends(get_db)):
    """
    Return all experiments ranked by best-epoch validation Dice score.
    All three metrics (Dice, HD95, BF1) are taken from the *same* epoch.
    """
    experiments = db.query(Experiment).all()

    result_rows = []
    for i, exp in enumerate(experiments):
        best = _best_epoch_row(db, exp.id)
        result_rows.append({
            "rank":             i + 1,       # re-ranked below after sort
            "experiment_id":    exp.id,
            "config_id":        exp.config_id,
            "alpha":            exp.alpha,
            "beta":             exp.beta,
            "gamma":            exp.gamma,
            "delta":            exp.delta,
            "epsilon":          exp.epsilon,
            "loss_description": exp.loss_description,
            "status":           exp.status,
            # FIX-8: all from the same checkpoint
            "best_dice":        round(best.val_dice, 4)        if best is not None and best.val_dice        is not None else None,  # FIX-9
            "best_hd95":        round(best.val_hd95, 2)        if best is not None and best.val_hd95        is not None else None,  # FIX-9
            "best_boundary_f1": round(best.val_boundary_f1, 4) if best is not None and best.val_boundary_f1 is not None else None,  # FIX-9
            "best_epoch":       best.epoch if best is not None else None,
        })

    # Sort by best_dice descending (None last)
    result_rows.sort(
        key=lambda r: r["best_dice"] if r["best_dice"] is not None else -1.0,
        reverse=True,
    )
    for i, row in enumerate(result_rows):
        row["rank"] = i + 1

    return json_safe(result_rows)


@router.get("/pareto")
async def pareto_frontier(db: Session = Depends(get_db)):
    """
    Compute Pareto frontier: experiments where no other experiment
    dominates on both Dice (↑) and HD95 (↓).

    FIX-8: metrics are taken from the same best-Dice epoch for each experiment.
    """
    experiments = db.query(Experiment).all()
    rows = []
    for exp in experiments:
        best = _best_epoch_row(db, exp.id)
        if best is None or best.val_dice is None or best.val_hd95 is None:  # FIX-9
            continue
        rows.append({
            "experiment_id":    exp.id,
            "config_id":        exp.config_id,
            "loss_description": exp.loss_description,
            "best_dice":        best.val_dice,
            "best_hd95":        best.val_hd95,
            "best_boundary_f1": best.val_boundary_f1,
        })

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
