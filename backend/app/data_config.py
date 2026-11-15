"""Backend-owned LUNA16 path resolution and cache validation."""
from __future__ import annotations

import os
import re
import sys
from collections import Counter
from pathlib import Path, PureWindowsPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path("data/luna16")
DEFAULT_CACHE_DIR = Path("data/patches_cache_v3_smoke")
DEFAULT_PATCH_SIZE = 64
_LUNA_UID = re.compile(r"\d+(?:\.\d+)+")


def resolve_project_path(
    supplied: str | os.PathLike[str] | None,
    *,
    environment_variable: str,
    default: str | os.PathLike[str],
) -> Path:
    """Resolve an explicit, environment, or default path from the repo root."""
    value = str(supplied).strip() if supplied is not None else ""
    if not value:
        value = os.getenv(environment_variable, "").strip()
    if not value:
        value = str(default)

    # Do not misinterpret a Windows absolute path as a relative Linux path
    # when the API happens to run in a container.
    if PureWindowsPath(value).is_absolute() and os.name != "nt":
        raise ValueError(
            f"{environment_variable} path {value!r} is a Windows path, but this backend "
            "is not running on Windows. Configure a backend-visible path such as /data/..."
        )

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def resolve_data_root(supplied: str | os.PathLike[str] | None = None) -> Path:
    return resolve_project_path(
        supplied,
        environment_variable="DATA_ROOT",
        default=DEFAULT_DATA_ROOT,
    )


def resolve_cache_dir(supplied: str | os.PathLike[str] | None = None) -> Path:
    return resolve_project_path(
        supplied,
        environment_variable="LUNA16_CACHE_DIR",
        default=DEFAULT_CACHE_DIR,
    )


def available_subset_uids(data_root: str | Path) -> dict[int, set[str]]:
    """Collect standard LUNA16 scan UIDs without opening CT volume contents."""
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"LUNA16 data root does not exist: {root}")
    subsets: dict[int, set[str]] = {}
    for directory in root.glob("subset*"):
        suffix = directory.name.removeprefix("subset")
        if not directory.is_dir() or not suffix.isdigit():
            continue
        subset = int(suffix)
        uids = {
            path.stem
            for path in directory.glob("*.mhd")
            if _LUNA_UID.fullmatch(path.stem)
        }
        if uids:
            subsets[subset] = uids
    if not subsets:
        raise ValueError(f"No LUNA16 subset directories with standard .mhd UIDs found under {root}")
    return subsets


def resolve_subset_split(
    subset_uids: dict[int, set[str]],
    *,
    train_subsets: list[int] | None = None,
    validation_subset: int | None = None,
    test_subset: int | None = None,
    fold: int = 0,
) -> dict[str, Any]:
    available = sorted(subset_uids)
    validation = int(validation_subset if validation_subset is not None else fold)
    if train_subsets is None:
        training = [subset for subset in available if subset != validation and subset != test_subset]
    else:
        training = [int(subset) for subset in train_subsets]
    testing = [int(test_subset)] if test_subset is not None else []

    if not training:
        raise ValueError("At least one training subset is required")
    groups = {"training": training, "validation": [validation], "test": testing}
    for name, ids in groups.items():
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate subset ID in {name} split: {ids}")
        missing = sorted(set(ids) - set(available))
        if missing:
            raise ValueError(f"Requested {name} subset(s) {missing} are unavailable; available subsets: {available}")
    if set(training) & {validation} or set(training) & set(testing) or {validation} & set(testing):
        raise ValueError("Training, validation, and test subsets must be disjoint")

    return {
        "train_subsets": training,
        "validation_subset": validation,
        "test_subset": testing[0] if testing else None,
        "train_scan_uids": sorted(set().union(*(subset_uids[s] for s in training))),
        "validation_scan_uids": sorted(subset_uids[validation]),
        "test_scan_uids": sorted(subset_uids[testing[0]]) if testing else [],
        "independent_test_available": bool(testing),
    }


def summarize_cache(entries: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(int(entry["label"]) for entry in entries)
    subset_counts: dict[int, Counter[int]] = {}
    subset_uids: dict[int, set[str]] = {}
    patch_sizes = {int(entry["patch_size"]) for entry in entries if entry.get("patch_size") is not None}
    spacing = {
        tuple(float(value) for value in entry["spacing_mm"])
        for entry in entries
        if entry.get("spacing_mm") is not None
    }
    for entry in entries:
        subset = int(entry["subset"])
        subset_counts.setdefault(subset, Counter())[int(entry["label"])] += 1
        subset_uids.setdefault(subset, set()).add(str(entry["uid"]))
    patch_size = next(iter(patch_sizes), None)
    spacing_value = list(next(iter(spacing))) if len(spacing) == 1 else None
    return {
        "total_patches": len(entries),
        "positive_patches": labels[1],
        "negative_patches": labels[0],
        "patch_dimensions": [3, patch_size, patch_size] if patch_size is not None else None,
        "channel_count": 3,
        "available_subsets": sorted(subset_counts),
        "scan_uid_count": len({str(entry["uid"]) for entry in entries}),
        "spacing_mm": spacing_value,
        "subsets": {
            str(subset): {
                "patch_count": sum(counts.values()),
                "positive_patches": counts[1],
                "negative_patches": counts[0],
                "scan_uid_count": len(subset_uids[subset]),
            }
            for subset, counts in sorted(subset_counts.items())
        },
    }


def validate_cached_split(
    data_root: str | Path,
    cache_dir: str | Path,
    patch_size: int,
    split: dict[str, Any],
) -> dict[str, Any]:
    """Validate NPZ contents and exact raw-to-cache UID membership for a split."""
    root = Path(data_root).resolve()
    cache = Path(cache_dir).resolve()
    if not cache.is_dir():
        raise FileNotFoundError(f"Patch cache directory does not exist: {cache}")
    subset_uids = available_subset_uids(root)
    selected = set(split["train_subsets"])
    selected.add(int(split["validation_subset"]))
    if split.get("test_subset") is not None:
        selected.add(int(split["test_subset"]))
    expected_uid_subsets = {subset: subset_uids[subset] for subset in selected}

    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from training.preprocess_cache import validate_cache_manifest

    entries = validate_cache_manifest(
        cache,
        expected_patch_size=patch_size,
        expected_subsets=selected,
        expected_uid_subsets=expected_uid_subsets,
        require_both_labels=True,
    )
    summary = summarize_cache(entries)
    summary["validated"] = True
    summary["cache_dir"] = str(cache)
    return summary


def runtime_data_info() -> dict[str, Any]:
    """Return cache and data status without allowing startup info to fail."""
    result: dict[str, Any] = {
        "default_data_root": None,
        "default_cache_dir": None,
        "patch_size": DEFAULT_PATCH_SIZE,
        "cache_exists": False,
        "cache_valid": False,
        "cache_error": None,
        "cache_summary": None,
        "available_subset_ids": [],
        "default_split": None,
    }
    try:
        data_root = resolve_data_root()
        cache_dir = resolve_cache_dir()
        patch_size = int(os.getenv("PATCH_SIZE", str(DEFAULT_PATCH_SIZE)))
        result.update({
            "default_data_root": str(data_root),
            "default_cache_dir": str(cache_dir),
            "patch_size": patch_size,
            "cache_exists": cache_dir.is_dir(),
        })
        subset_uids = available_subset_uids(data_root)
        result["available_subset_ids"] = sorted(subset_uids)
        split = resolve_subset_split(subset_uids)
        result["default_split"] = {
            "train_subsets": split["train_subsets"],
            "validation_subset": split["validation_subset"],
            "test_subset": split["test_subset"],
        }
        all_subsets = sorted(subset_uids)
        full_split = {
            "train_subsets": [subset for subset in all_subsets if subset != split["validation_subset"]],
            "validation_subset": split["validation_subset"],
            "test_subset": None,
        }
        # Validate every raw subset against the default cache for a useful
        # installation-wide status. Creation validates only its selected split.
        summary = validate_cached_split(data_root, cache_dir, patch_size, full_split)
        result["cache_valid"] = True
        result["cache_summary"] = summary
    except Exception as exc:
        result["cache_error"] = str(exc)
    return result
