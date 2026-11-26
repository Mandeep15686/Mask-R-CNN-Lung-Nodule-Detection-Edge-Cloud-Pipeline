import asyncio
import csv
import json
from backend.app.routes.experiments import list_phase3_studies


def test_phase3_studies_endpoint_reads_trial_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENTS_DIR", str(tmp_path))
    study_dir = tmp_path / "phase3" / "studies" / "unit_study"
    trial_dir = study_dir / "trial_0002"
    trial_dir.mkdir(parents=True)

    (study_dir / "phase3_best.json").write_text(json.dumps({
        "study_name": "unit_study",
        "best_trial": 2,
        "best_score": 0.75,
        "best_norm_weights": {"alpha": 0.2, "beta": 0.2, "gamma": 0.2, "delta": 0.2, "epsilon": 0.2},
    }), encoding="utf-8")
    with (study_dir / "trials.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "trial_number", "state", "elapsed_s", "score", "dice", "hd95",
            "boundary_f1", "best_epoch", "alpha_raw", "beta_raw", "gamma_raw",
            "delta_raw", "epsilon_raw", "alpha_norm", "beta_norm", "gamma_norm",
            "delta_norm", "epsilon_norm",
        ])
        writer.writeheader()
        writer.writerow({
            "trial_number": 2, "state": "COMPLETE", "elapsed_s": 42,
            "score": 0.75, "dice": 0.8, "hd95": 2.0, "boundary_f1": 0.9,
            "best_epoch": 4, "alpha_raw": 1, "beta_raw": 1, "gamma_raw": 1,
            "delta_raw": 1, "epsilon_raw": 1, "alpha_norm": .2, "beta_norm": .2,
            "gamma_norm": .2, "delta_norm": .2, "epsilon_norm": .2,
        })
    (trial_dir / "config.json").write_text(json.dumps({
        "epochs": 5, "batch_size": 4, "lr": .005, "warmup_epochs": 1,
        "cache_dir": "cache", "patch_size": 64,
        "mask_loss": {"bce_pos_weight": 1},
        "data_split": {"train_subsets": [1], "validation_subset": 0,
                        "test_subset": None, "train_scan_uids": ["train"],
                        "validation_scan_uids": ["validation"],
                        "independent_test_available": False},
        "phase3_raw_weights": {"alpha": 1, "beta": 1, "gamma": 1, "delta": 1, "epsilon": 1},
        "phase3_norm_weights": {"alpha": .2, "beta": .2, "gamma": .2, "delta": .2, "epsilon": .2},
    }), encoding="utf-8")
    (trial_dir / "summary.json").write_text(json.dumps({
        "best_epoch": 4,
        "best_metrics": {"val_dice": .8, "val_hd95": 2, "val_boundary_f1": .9},
        "final_metrics": {"val_dice": .7},
    }), encoding="utf-8")
    (trial_dir / "history.json").write_text(json.dumps([{"epoch": 1, "val_dice": .5}]), encoding="utf-8")
    (trial_dir / "best_model.pth").write_bytes(b"test checkpoint marker")

    response = asyncio.run(list_phase3_studies())

    assert len(response["studies"]) == 1
    study = response["studies"][0]
    assert study["study_name"] == "unit_study"
    assert study["completed_trials"] == 1
    trial = study["trials"][0]
    assert trial["trial_number"] == 2
    assert trial["metrics"]["val_dice"] == .8
    assert trial["training_config"]["train_subsets"] == [1]
    assert trial["training_config"]["independent_test_available"] is False
    assert trial["history"] == [{"epoch": 1, "val_dice": .5}]
    assert trial["checkpoint_available"] is True


def test_phase3_studies_endpoint_returns_empty_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENTS_DIR", str(tmp_path))
    assert asyncio.run(list_phase3_studies())["studies"] == []
