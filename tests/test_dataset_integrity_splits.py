import json

import numpy as np
import pytest
import torch

from training.dataset import (
    CachedLUNA16Dataset,
    LUNA16Dataset,
    build_dataloaders,
    load_mhd,
    patch_overlaps_nodule,
    voxel_to_world,
)


def _raw_root(tmp_path, subset_ids=(0, 1, 2)):
    root = tmp_path / "raw"
    for subset in subset_ids:
        folder = root / f"subset{subset}"
        folder.mkdir(parents=True)
        (folder / f"1.2.3.{subset + 1}.mhd").touch()
    return root


def _cache(root, entries):
    root.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        np.savez_compressed(
            root / entry["file"],
            image=np.zeros((3, 8, 8), dtype=np.float32),
            mask=np.zeros((8, 8), dtype=np.uint8),
            label=np.int8(entry["label"]),
            diameter=np.float32(entry.get("diameter_mm", 0)),
        )
    (root / "manifest.json").write_text(json.dumps(entries), encoding="utf-8")


def test_raw_negative_samples_actual_crop_safe_non_air_voxel():
    dataset = LUNA16Dataset.__new__(LUNA16Dataset)
    dataset.patch_size = 4
    dataset._sampled_negative_centers = {}
    volume = np.zeros((7, 10, 10), dtype=np.float32)
    volume[3, 5, 6] = 1.0
    origin = np.array([10.0, 20.0, 30.0])
    spacing = np.array([1.0, 1.0, 1.0])
    direction = np.eye(3)

    world = dataset._random_lung_coord(
        volume, origin, spacing, direction, [], "scan-1"
    )

    voxel = np.rint((world - origin) / spacing).astype(int)
    assert tuple(voxel) == (6, 5, 3)
    assert volume[voxel[2], voxel[1], voxel[0]] > 0.036
    assert patch_overlaps_nodule(voxel, voxel, 2.0, spacing, 4) is True


def test_raw_negative_excludes_nodule_and_duplicate_centers():
    dataset = LUNA16Dataset.__new__(LUNA16Dataset)
    dataset.patch_size = 4
    dataset._sampled_negative_centers = {}
    volume = np.zeros((7, 10, 10), dtype=np.float32)
    volume[3, 5, 5] = 1.0
    origin, spacing, direction = np.zeros(3), np.ones(3), np.eye(3)
    annotation = {
        "coord": voxel_to_world((5, 5, 3), origin, spacing, direction),
        "diameter_mm": 4.0,
    }

    with pytest.raises(RuntimeError, match="Unable to find a unique crop-safe"):
        dataset._random_lung_coord(
            volume, origin, spacing, direction, [annotation], "scan-nodule"
        )

    safe = np.zeros_like(volume)
    safe[3, 5, 5] = 1.0
    dataset._sampled_negative_centers.clear()
    first = dataset._random_lung_coord(safe, origin, spacing, direction, [], "scan-safe")
    assert np.allclose(first, voxel_to_world((5, 5, 3), origin, spacing, direction))
    with pytest.raises(RuntimeError, match="unique crop-safe"):
        dataset._random_lung_coord(safe, origin, spacing, direction, [], "scan-safe")


def test_load_mhd_resamples_to_documented_isotropic_spacing(tmp_path):
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(np.zeros((4, 8, 8), dtype=np.int16))
    image.SetSpacing((2.0, 2.0, 2.0))
    path = tmp_path / "volume.mhd"
    sitk.WriteImage(image, str(path))

    volume, _origin, spacing, _direction = load_mhd(path)
    assert np.allclose(spacing, (1.0, 1.0, 1.0))
    assert volume.shape == (8, 16, 16)


def test_cached_dataset_subset_filter_and_missing_metadata(tmp_path):
    cache = tmp_path / "cache"
    entries = [
        {"file": "a.npz", "uid": "scan-a", "subset": 0, "label": 1},
        {"file": "b.npz", "uid": "scan-b", "subset": 1, "label": 0},
    ]
    _cache(cache, entries)
    subset = CachedLUNA16Dataset(cache, subsets=[1], augment=False, expected_patch_size=8)
    assert len(subset) == 1
    assert subset.scan_uids == {"scan-b"}

    (cache / "manifest.json").write_text(
        json.dumps([{"file": "a.npz", "uid": "scan-a", "label": 1}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="subset metadata"):
        CachedLUNA16Dataset(cache, subsets=[0])


def test_cached_dataset_detects_missing_patch(tmp_path):
    cache = tmp_path / "cache"
    (cache / "manifest.json").parent.mkdir(parents=True)
    (cache / "manifest.json").write_text(
        json.dumps([{"file": "missing.npz", "uid": "s", "subset": 0, "label": 1}]),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="missing NPZ"):
        CachedLUNA16Dataset(cache, subsets=[0])


def test_explicit_train_validation_test_uid_separation(tmp_path):
    raw = _raw_root(tmp_path)
    cache = tmp_path / "cache"
    entries = [
        {"file": f"{subset}.npz", "uid": f"1.2.3.{subset + 1}",
         "subset": subset, "label": 1}
        for subset in range(3)
    ]
    _cache(cache, entries)

    train, validation = build_dataloaders(
        raw, train_subsets=[0], validation_subset=1, test_subset=2,
        cache_dir=cache, patch_size=8, batch_size=1, num_workers=0,
    )
    split = train.dataset.split_info
    assert train.dataset.scan_uids.isdisjoint(validation.dataset.scan_uids)
    assert train.dataset.scan_uids.isdisjoint(set(split["test_scan_uids"]))
    assert validation.dataset.scan_uids.isdisjoint(set(split["test_scan_uids"]))
    assert split["independent_test_available"] is True
    assert split["test_subset"] == 2


def test_explicit_splits_reject_unavailable_or_overlapping_subsets(tmp_path):
    raw = _raw_root(tmp_path, subset_ids=(0, 1))
    cache = tmp_path / "cache"
    _cache(cache, [
        {"file": "a.npz", "uid": "1.2.3.1", "subset": 0, "label": 1},
        {"file": "b.npz", "uid": "1.2.3.2", "subset": 1, "label": 0},
    ])
    with pytest.raises(ValueError, match="unavailable"):
        build_dataloaders(
            raw, train_subsets=[0], validation_subset=1, test_subset=2,
            cache_dir=cache, patch_size=8,
        )
    with pytest.raises(ValueError, match="Scan UID leakage"):
        build_dataloaders(
            raw, train_subsets=[0], validation_subset=0,
            cache_dir=cache, patch_size=8,
        )


def test_api_experiment_schema_validates_explicit_split():
    from backend.app.schemas import ExperimentCreate, LossWeights

    valid = ExperimentCreate(
        config_id="explicit-split",
        loss_weights=LossWeights(alpha=1.0),
        train_subsets=[0],
        validation_subset=1,
        test_subset=2,
    )
    assert valid.train_subsets == [0]
    assert valid.test_subset == 2

    with pytest.raises(ValueError, match="disjoint"):
        ExperimentCreate(
            config_id="overlap",
            loss_weights=LossWeights(alpha=1.0),
            train_subsets=[0],
            validation_subset=1,
            test_subset=0,
        )
