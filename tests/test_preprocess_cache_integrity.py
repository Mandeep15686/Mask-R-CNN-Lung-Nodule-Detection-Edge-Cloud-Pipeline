import csv
import json

import numpy as np
import pytest

from training import preprocess_cache


def _fake_dataset(tmp_path):
    root = tmp_path / "raw"
    subset = root / "subset0"
    subset.mkdir(parents=True)
    uid = "1.2.3.4"
    (subset / f"{uid}.mhd").write_text("synthetic", encoding="utf-8")
    with (root / "annotations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"]
        )
        writer.writeheader()
        writer.writerow({
            "seriesuid": uid, "coordX": 4, "coordY": 4,
            "coordZ": 4, "diameter_mm": 4,
        })
    return root, uid


def _fake_process(job):
    uid = job["uid"]
    out = __import__("pathlib").Path(job["out_dir"])
    entries = []
    for label, suffix in ((1, "pos"), (0, "neg")):
        filename = f"{uid}_{suffix}_0.npz"
        np.savez_compressed(
            out / filename,
            image=np.zeros((3, job["patch_size"], job["patch_size"]), dtype=np.float32),
            mask=np.zeros((job["patch_size"], job["patch_size"]), dtype=np.uint8),
            label=np.int8(label), diameter=np.float32(4 if label else 0),
        )
        entries.append({
            "file": filename,
            "uid": uid,
            "subset": job["subset"],
            "label": label,
            "diameter_mm": 4.0 if label else 0.0,
            "patch_size": job["patch_size"],
            "spacing_mm": list(job["target_spacing_mm"]),
        })
    return entries


def test_manifest_validation_catches_missing_npz_and_empty_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        preprocess_cache.validate_cache_manifest(tmp_path)

    (tmp_path / "manifest.json").write_text(
        json.dumps([{"file": "missing.npz", "uid": "u", "subset": 0, "label": 1}]),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="missing NPZ"):
        preprocess_cache.validate_cache_manifest(tmp_path)


def test_cache_generation_writes_subset_and_counts_atomically(tmp_path, monkeypatch):
    raw, uid = _fake_dataset(tmp_path)
    cache = tmp_path / "cache"
    monkeypatch.setattr(preprocess_cache, "_process_volume", _fake_process)

    preprocess_cache.build_cache(str(raw), str(cache), patch_size=8, neg_ratio=1, workers=1)

    entries = preprocess_cache.validate_cache_manifest(cache, expected_uids={uid})
    assert sum(entry["label"] == 1 for entry in entries) == 1
    assert sum(entry["label"] == 0 for entry in entries) == 1
    assert {entry["subset"] for entry in entries} == {0}
    assert all(entry["spacing_mm"] == [1.0, 1.0, 1.0] for entry in entries)


def test_valid_cache_is_not_overwritten_without_rebuild(tmp_path, monkeypatch):
    raw, _ = _fake_dataset(tmp_path)
    cache = tmp_path / "cache"
    monkeypatch.setattr(preprocess_cache, "_process_volume", _fake_process)
    preprocess_cache.build_cache(str(raw), str(cache), patch_size=8, neg_ratio=1, workers=1)
    before = (cache / "manifest.json").read_bytes()

    preprocess_cache.build_cache(str(raw), str(cache), patch_size=8, neg_ratio=1, workers=1)

    assert (cache / "manifest.json").read_bytes() == before
    assert not list(tmp_path.glob("cache.backup-*"))


def test_rebuild_preserves_previous_cache_as_backup(tmp_path, monkeypatch):
    raw, _ = _fake_dataset(tmp_path)
    cache = tmp_path / "cache"
    monkeypatch.setattr(preprocess_cache, "_process_volume", _fake_process)
    preprocess_cache.build_cache(str(raw), str(cache), patch_size=8, neg_ratio=1, workers=1)
    old_manifest = (cache / "manifest.json").read_bytes()
    (cache / "user-marker").write_text("keep backup", encoding="utf-8")

    preprocess_cache.build_cache(
        str(raw), str(cache), patch_size=8, neg_ratio=1, workers=1, rebuild=True
    )

    backups = list(tmp_path.glob("cache.backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "manifest.json").read_bytes() == old_manifest
    assert (backups[0] / "user-marker").read_text(encoding="utf-8") == "keep backup"
    preprocess_cache.validate_cache_manifest(cache)


def test_failed_rebuild_does_not_replace_existing_cache(tmp_path, monkeypatch):
    raw, _ = _fake_dataset(tmp_path)
    cache = tmp_path / "cache"
    monkeypatch.setattr(preprocess_cache, "_process_volume", _fake_process)
    preprocess_cache.build_cache(str(raw), str(cache), patch_size=8, neg_ratio=1, workers=1)
    old_manifest = (cache / "manifest.json").read_bytes()
    monkeypatch.setattr(preprocess_cache, "_process_volume", lambda job: [])

    with pytest.raises(RuntimeError, match="incomplete"):
        preprocess_cache.build_cache(
            str(raw), str(cache), patch_size=8, neg_ratio=1, workers=1, rebuild=True
        )

    assert (cache / "manifest.json").read_bytes() == old_manifest
    assert not list(tmp_path.glob("cache.backup-*"))


def test_incomplete_existing_manifest_is_not_silently_accepted(tmp_path, monkeypatch):
    raw, _ = _fake_dataset(tmp_path)
    cache = tmp_path / "cache"
    monkeypatch.setattr(preprocess_cache, "_process_volume", _fake_process)
    preprocess_cache.build_cache(str(raw), str(cache), patch_size=8, neg_ratio=1, workers=1)
    manifest_path = cache / "manifest.json"
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(json.dumps(entries[:1]), encoding="utf-8")

    with pytest.raises(RuntimeError, match="incomplete or invalid"):
        preprocess_cache.build_cache(
            str(raw), str(cache), patch_size=8, neg_ratio=1, workers=1
        )
