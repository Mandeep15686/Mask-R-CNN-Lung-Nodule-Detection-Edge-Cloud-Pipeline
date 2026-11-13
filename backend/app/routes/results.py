"""Results aggregation and comparison analytics."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..database import get_db, Experiment, TrainingRun, InferenceResult

router = APIRouter()


@router.get("/leaderboard")
async def leaderboard(db: Session = Depends(get_db)):
    """
    Return all experiments ranked by best validation Dice score.
    Useful for the Phase 2 grid search comparison table.
    """
    rows = (
        db.query(
            Experiment.id,
            Experiment.config_id,
            Experiment.alpha,
            Experiment.beta,
            Experiment.gamma,
            Experiment.delta,
            Experiment.epsilon,
            Experiment.loss_description,
            Experiment.status,
            func.max(TrainingRun.val_dice).label("best_dice"),
            func.min(TrainingRun.val_hd95).label("best_hd95"),
            func.max(TrainingRun.val_boundary_f1).label("best_boundary_f1"),
        )
        .outerjoin(TrainingRun, TrainingRun.experiment_id == Experiment.id)
        .group_by(Experiment.id)
        .order_by(func.max(TrainingRun.val_dice).desc().nullslast())
        .all()
    )

    return [
        {
            "rank": i + 1,
            "experiment_id": r.id,
            "config_id": r.config_id,
            "alpha": r.alpha,
            "beta": r.beta,
            "gamma": r.gamma,
            "delta": r.delta,
            "epsilon": r.epsilon,
            "loss_description": r.loss_description,
            "status": r.status,
            "best_dice": round(r.best_dice, 4) if r.best_dice else None,
            "best_hd95": round(r.best_hd95, 2) if r.best_hd95 else None,
            "best_boundary_f1": round(r.best_boundary_f1, 4) if r.best_boundary_f1 else None,
        }
        for i, r in enumerate(rows)
    ]


@router.get("/pareto")
async def pareto_frontier(db: Session = Depends(get_db)):
    """
    Compute Pareto frontier: experiments where no other experiment
    dominates on both Dice (↑) and HD95 (↓).
    """
    rows = (
        db.query(
            Experiment.id,
            Experiment.config_id,
            Experiment.loss_description,
            func.max(TrainingRun.val_dice).label("best_dice"),
            func.min(TrainingRun.val_hd95).label("best_hd95"),
            func.max(TrainingRun.val_boundary_f1).label("best_bf1"),
        )
        .join(TrainingRun, TrainingRun.experiment_id == Experiment.id)
        .group_by(Experiment.id)
        .all()
    )

    # Filter: keep row i if no other row j has dice_j >= dice_i AND hd95_j <= hd95_i
    pareto = []
    for i, r in enumerate(rows):
        if r.best_dice is None or r.best_hd95 is None:
            continue
        dominated = any(
            (other.best_dice or 0) >= r.best_dice
            and (other.best_hd95 or 9999) <= r.best_hd95
            and other is not r
            for other in rows
            if other.best_dice is not None and other.best_hd95 is not None
        )
        if not dominated:
            pareto.append({
                "experiment_id": r.id,
                "config_id": r.config_id,
                "loss_description": r.loss_description,
                "best_dice": round(r.best_dice, 4),
                "best_hd95": round(r.best_hd95, 2),
                "best_boundary_f1": round(r.best_bf1, 4) if r.best_bf1 else None,
            })

    return pareto


@router.get("/stats")
async def stats(db: Session = Depends(get_db)):
    """High-level statistics for the dashboard."""
    n_experiments = db.query(func.count(Experiment.id)).scalar()
    n_running = db.query(func.count(Experiment.id)).filter(Experiment.status == "running").scalar()
    n_done = db.query(func.count(Experiment.id)).filter(Experiment.status == "done").scalar()
    n_inferences = db.query(func.count(InferenceResult.id)).scalar()
    best_dice_row = (
        db.query(func.max(TrainingRun.val_dice)).scalar()
    )
    return {
        "total_experiments": n_experiments,
        "running": n_running,
        "done": n_done,
        "total_inferences": n_inferences,
        "best_dice_overall": round(best_dice_row, 4) if best_dice_row else None,
    }


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
    return {
        "epochs":               [r.epoch for r in runs],
        "train_loss":           [r.train_total_loss for r in runs],
        "val_dice":             [r.val_dice for r in runs],
        "val_hd95":             [r.val_hd95 for r in runs],
        "val_boundary_f1":      [r.val_boundary_f1 for r in runs],
        "val_sensitivity":      [r.val_sensitivity for r in runs],
        "lr":                   [r.lr for r in runs],
    }
