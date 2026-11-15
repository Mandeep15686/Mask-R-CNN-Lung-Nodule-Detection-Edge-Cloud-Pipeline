import csv
import json

import pytest

from training.experiment import ExperimentRunner, select_phase2_configs


def _selected_config():
    return select_phase2_configs(1)[0]


def _write_complete_artifacts(output_dir, config_id, epochs=3, summary=None):
    run_dir = output_dir / config_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = summary or {
        "config_id": config_id,
        "final_metrics": {"val_dice": 0.25},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "history.json").write_text(
        json.dumps([{"epoch": epoch} for epoch in range(1, epochs + 1)]),
        encoding="utf-8",
    )
    (run_dir / "latest_model.pth").write_bytes(b"synthetic-checkpoint")


class _FakeTrainer:
    calls = []

    def __init__(self, config, run_dir):
        self.config = config
        self.run_dir = run_dir
        self.calls.append(config["config_id"])

    def train(self):
        from pathlib import Path

        run_dir = Path(self.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        epochs = self.config["epochs"]
        summary = {
            "config_id": self.config["config_id"],
            "epochs": epochs,
            "final_metrics": {"val_dice": 0.5},
        }
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (run_dir / "history.json").write_text(
            json.dumps([{"epoch": epoch} for epoch in range(1, epochs + 1)]),
            encoding="utf-8",
        )
        (run_dir / "latest_model.pth").write_bytes(b"new-checkpoint")
        return summary


@pytest.fixture(autouse=True)
def reset_fake_trainer_calls():
    _FakeTrainer.calls = []


def _runner(tmp_path, monkeypatch, epochs=3):
    import training.trainer as trainer_module

    monkeypatch.setattr(trainer_module, "Trainer", _FakeTrainer)
    return ExperimentRunner({"epochs": epochs}, str(tmp_path))


def test_resume_skips_configuration_with_all_completed_artifacts(tmp_path, monkeypatch):
    config_id = _selected_config()["config_id"]
    _write_complete_artifacts(tmp_path, config_id, epochs=3)
    runner = _runner(tmp_path, monkeypatch)

    runner.run_phase2(max_configs=1, resume=True)

    assert _FakeTrainer.calls == []
    assert [result["config_id"] for result in runner.results] == [config_id]
    saved = json.loads((tmp_path / "all_results.json").read_text(encoding="utf-8"))
    assert saved[0]["config_id"] == config_id


def test_resume_reruns_incomplete_configuration_from_epoch_one(tmp_path, monkeypatch):
    config_id = _selected_config()["config_id"]
    run_dir = tmp_path / config_id
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps({"config_id": config_id}), encoding="utf-8"
    )
    (run_dir / "history.json").write_text(
        json.dumps([{"epoch": 1}]), encoding="utf-8"
    )
    (run_dir / "latest_model.pth").write_bytes(b"partial-checkpoint")
    runner = _runner(tmp_path, monkeypatch)

    runner.run_phase2(max_configs=1, resume=True)

    assert _FakeTrainer.calls == [config_id]
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    assert [entry["epoch"] for entry in history] == [1, 2, 3]
    assert (run_dir / "latest_model.pth").read_bytes() == b"new-checkpoint"


def test_resume_preserves_previous_results_in_json_and_csv(tmp_path, monkeypatch):
    config_id = _selected_config()["config_id"]
    _write_complete_artifacts(tmp_path, config_id, epochs=3)
    previous = {"config_id": "previous-phase-result", "score": 0.75}
    (tmp_path / "all_results.json").write_text(
        json.dumps([previous]), encoding="utf-8"
    )
    runner = _runner(tmp_path, monkeypatch)

    runner.run_phase2(max_configs=1, resume=True)

    saved = json.loads((tmp_path / "all_results.json").read_text(encoding="utf-8"))
    assert [item["config_id"] for item in saved] == [
        "previous-phase-result", config_id
    ]
    with (tmp_path / "all_results.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {row["config_id"] for row in rows} == {
        "previous-phase-result", config_id
    }
    assert rows[0]["score"] == "0.75"


@pytest.mark.parametrize("summary_text", [
    "{invalid json",
    "NaN",
    json.dumps({"config_id": "different-config"}),
])
def test_resume_reruns_invalid_or_mismatched_summary(
    tmp_path, monkeypatch, summary_text
):
    config_id = _selected_config()["config_id"]
    run_dir = tmp_path / config_id
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(summary_text, encoding="utf-8")
    (run_dir / "history.json").write_text(
        json.dumps([{"epoch": epoch} for epoch in range(1, 4)]),
        encoding="utf-8",
    )
    (run_dir / "latest_model.pth").write_bytes(b"checkpoint")
    runner = _runner(tmp_path, monkeypatch)

    runner.run_phase2(max_configs=1, resume=True)

    assert _FakeTrainer.calls == [config_id]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["config_id"] == config_id
    assert summary["epochs"] == 3


def test_phase2_completes_exactly_twenty_unique_deterministic_configs(
    tmp_path, monkeypatch
):
    expected_ids = [config["config_id"] for config in select_phase2_configs(20)]
    runner = _runner(tmp_path, monkeypatch)

    runner.run_phase2(max_configs=20)

    saved = json.loads((tmp_path / "all_results.json").read_text(encoding="utf-8"))
    assert len(_FakeTrainer.calls) == 20
    assert len(set(_FakeTrainer.calls)) == 20
    assert _FakeTrainer.calls == expected_ids
    assert [item["config_id"] for item in saved] == expected_ids
    with (tmp_path / "all_results.csv").open(newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert [row["config_id"] for row in csv_rows] == expected_ids
