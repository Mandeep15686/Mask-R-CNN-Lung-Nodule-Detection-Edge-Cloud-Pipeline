"""Bridge saved experiment folders into the API's configured database."""
from __future__ import annotations

import sys
from pathlib import Path


def sync_saved_experiment_artifacts(db) -> dict[str, int]:
    project_root = Path(__file__).resolve().parents[2]
    root_text = str(project_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    from scripts.import_experiment_folders import sync_experiment_folders

    from .database import Experiment, TrainingRun

    return sync_experiment_folders(
        db,
        experiment_model=Experiment,
        training_run_model=TrainingRun,
    )


def sync_on_startup() -> dict[str, int]:
    """Refresh saved runs once during normal app startup."""
    from .database import SessionLocal

    with SessionLocal() as db:
        try:
            return sync_saved_experiment_artifacts(db)
        except Exception:
            db.rollback()
            raise
