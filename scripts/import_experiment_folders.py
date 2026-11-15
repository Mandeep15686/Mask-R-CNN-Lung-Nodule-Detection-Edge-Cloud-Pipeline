"""Import saved experiment histories into the website's SQLite database.

This indexes checkpoint folders only. It does not move, rewrite, or delete
model checkpoints, histories, summaries, or logs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FOLDERS = {
    "experiments": "",
    "experiments_phase2_5ep_20cfg": "phase2-5ep20",
    "experiments_phase2_screen": "phase2-screen",
    "experiments_phase2_v3_5ep_20cfg": "phase2-v3-5ep20",
}
WEIGHT_KEYS = ("alpha", "beta", "gamma", "delta", "epsilon")
WEIGHT_NAMES = {
    "alpha": "BCE",
    "beta": "Dice",
    "gamma": "Boundary",
    "delta": "Focal",
    "epsilon": "HD95",
}


def _read_json(path: Path, *, required: bool) -> Any:
    if not path.is_file():
        if required:
            raise ValueError(f"Required experiment file is missing: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read valid JSON from {path}: {exc}") from exc


def _number(record: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in record or record[key] is None:
            continue
        try:
            value = float(record[key])
        except (TypeError, ValueError):
            continue
        return value if math.isfinite(value) else None
    return None


def _weight_config(config: dict[str, Any], summary: dict[str, Any]) -> dict[str, float]:
    configured = summary.get("loss_weights") or config.get("mask_loss") or {}
    if not isinstance(configured, dict):
        configured = {}
    return {key: _number(configured, key) or 0.0 for key in WEIGHT_KEYS}


def _loss_description(weights: dict[str, float]) -> str:
    items = [f"{weights[key]:.2f} {WEIGHT_NAMES[key]}" for key in WEIGHT_KEYS if weights[key] > 0]
    return " + ".join(items) if items else "No configured mask loss"


def _phase(config_id: str, config: dict[str, Any]) -> int:
    if config.get("phase") is not None:
        try:
            return int(config["phase"])
        except (TypeError, ValueError):
            pass
    if config_id.startswith("P3-"):
        return 3
    if config_id.startswith("C-"):
        return 1
    return 2


def _config_for_database(
    config: dict[str, Any],
    summary: dict[str, Any],
    folder: Path,
    original_config_id: str,
) -> dict[str, Any]:
    source = folder.relative_to(PROJECT_ROOT).as_posix()
    # Avoid duplicating full scan UID lists for every imported run.
    details = dict(config)
    details.pop("data_split", None)
    details.pop("train_scan_uids", None)
    details.pop("validation_scan_uids", None)
    details.pop("test_scan_uids", None)
    split = summary.get("data_split") or config.get("data_split") or {}
    if isinstance(split, dict):
        split_summary: dict[str, Any] = {}
        for key in ("train_subsets", "validation_subset", "test_subset"):
            if key in split:
                split_summary[key] = split[key]
        for key in ("train_scan_uids", "validation_scan_uids", "test_scan_uids"):
            if isinstance(split.get(key), list):
                split_summary[f"{key}_count"] = len(split[key])
        if split_summary:
            details["data_split_summary"] = split_summary
    details.update({
        "source_folder": folder.parent.name,
        "artifact_dir": source,
        "source_config_id": original_config_id,
        "imported_from_artifacts": True,
        "completed_epochs": 0,
    })
    return details


def _database_config_id(source_root: Path, original_config_id: str) -> str:
    prefix = SOURCE_FOLDERS[source_root.name]
    result = f"{prefix}--{original_config_id}" if prefix else original_config_id
    if len(result) > 64:
        raise ValueError(f"Generated config_id exceeds database limit: {result}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", result):
        raise ValueError(f"Generated config_id contains unsupported characters: {result}")
    return result


def _collect_runs() -> list[dict[str, Any]]:
    imported: list[dict[str, Any]] = []
    for root_name in SOURCE_FOLDERS:
        source_root = PROJECT_ROOT / root_name
        if not source_root.is_dir():
            raise ValueError(f"Experiment folder does not exist: {source_root}")
        for folder in sorted(path for path in source_root.iterdir() if path.is_dir()):
            checkpoint = folder / "best_model.pth"
            if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                continue
            config = _read_json(folder / "config.json", required=True)
            history = _read_json(folder / "history.json", required=True)
            summary = _read_json(folder / "summary.json", required=False) or {}
            if not isinstance(config, dict) or not isinstance(history, list) or not isinstance(summary, dict):
                raise ValueError(f"Invalid config/history/summary structure in {folder}")
            if not history or not all(isinstance(item, dict) for item in history):
                raise ValueError(f"No valid epoch history found in {folder}")

            original_id = str(config.get("config_id") or folder.name)
            db_id = _database_config_id(source_root, original_id)
            weights = _weight_config(config, summary)
            requested_epochs = config.get("epochs") or len(history)
            try:
                requested_epochs = int(requested_epochs)
            except (TypeError, ValueError):
                requested_epochs = len(history)
            completed_epochs = len(history)
            config_json = _config_for_database(config, summary, folder, original_id)
            config_json["completed_epochs"] = completed_epochs
            config_json["epochs"] = requested_epochs

            # Partial runs are included with a failed status; they are never
            # presented as completed experiments.
            status = "done" if completed_epochs >= requested_epochs else "failed"
            history_stat = (folder / "history.json").stat()
            imported.append({
                "config_id": db_id,
                "original_config_id": original_id,
                "phase": _phase(original_id, config),
                "weights": weights,
                "loss_description": _loss_description(weights),
                "status": status,
                "requested_epochs": requested_epochs,
                "config_json": config_json,
                "history": history,
                "artifact_mtime": datetime.fromtimestamp(history_stat.st_mtime, timezone.utc),
            })
    return imported


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="List imports without writing the database")
    parser.add_argument(
        "--database-url",
        default=f"sqlite:///{(PROJECT_ROOT / 'backend' / 'lung_nodule.db').as_posix()}",
        help="Database used by the website backend",
    )
    args = parser.parse_args()
    os.environ["DATABASE_URL"] = args.database_url
    sys.path.insert(0, str(PROJECT_ROOT))

    try:
        imports = _collect_runs()
        if not imports:
            raise ValueError("No trained experiment checkpoints were found")

        counts: dict[str, int] = {}
        for item in imports:
            source = item["config_json"]["source_folder"]
            counts[source] = counts.get(source, 0) + 1
        print(f"Found {len(imports)} checkpoint histories across {len(counts)} folders:")
        for source, count in sorted(counts.items()):
            print(f"  {source}: {count}")
        incomplete = [item for item in imports if item["status"] != "done"]
        print(f"Complete: {len(imports) - len(incomplete)} | incomplete: {len(incomplete)}")
        for item in incomplete:
            print(
                f"  Incomplete {item['config_json']['artifact_dir']}: "
                f"{len(item['history'])}/{item['requested_epochs']} epochs"
            )
        if args.dry_run:
            return 0

        from backend.app.database import Experiment, SessionLocal, TrainingRun

        with SessionLocal() as db:
            for item in imports:
                experiment = (
                    db.query(Experiment)
                    .filter(Experiment.config_id == item["config_id"])
                    .one_or_none()
                )
                if experiment is None:
                    experiment = Experiment(config_id=item["config_id"])
                    db.add(experiment)
                    db.flush()

                experiment.phase = item["phase"]
                for key, value in item["weights"].items():
                    setattr(experiment, key, value)
                experiment.loss_description = item["loss_description"]
                experiment.status = item["status"]
                experiment.config_json = item["config_json"]
                if experiment.created_at is None:
                    experiment.created_at = item["artifact_mtime"]
                experiment.finished_at = item["artifact_mtime"]

                db.query(TrainingRun).filter(
                    TrainingRun.experiment_id == experiment.id
                ).delete(synchronize_session=False)
                fold = int(item["config_json"].get("fold") or 0)
                for ordinal, epoch_data in enumerate(item["history"], start=1):
                    epoch = int(epoch_data.get("epoch") or ordinal)
                    db.add(TrainingRun(
                        experiment_id=experiment.id,
                        fold=fold,
                        epoch=epoch,
                        train_total_loss=_number(epoch_data, "total_loss", "train_total_loss"),
                        train_bce_loss=_number(epoch_data, "bce", "train_bce_loss"),
                        train_dice_loss=_number(epoch_data, "dice", "train_dice_loss"),
                        train_boundary_loss=_number(epoch_data, "boundary", "train_boundary_loss"),
                        train_focal_loss=_number(epoch_data, "focal", "train_focal_loss"),
                        train_hd_loss=_number(epoch_data, "hd", "train_hd_loss"),
                        lr=_number(epoch_data, "lr"),
                        val_dice=_number(epoch_data, "val_dice"),
                        val_iou=_number(epoch_data, "val_iou"),
                        val_hd95=_number(epoch_data, "val_hd95"),
                        val_boundary_f1=_number(epoch_data, "val_boundary_f1"),
                        val_sensitivity=_number(epoch_data, "val_sensitivity"),
                        val_specificity=_number(epoch_data, "val_specificity"),
                        val_asd=_number(epoch_data, "val_asd"),
                        val_vol_err=_number(epoch_data, "val_vol_err"),
                    ))
            db.commit()
        print(f"Imported or refreshed {len(imports)} experiment entries and epoch histories.")
        return 0
    except Exception as exc:
        print(f"Import failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
