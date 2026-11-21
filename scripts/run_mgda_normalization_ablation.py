#!/usr/bin/env python3
"""Run the controlled raw-vs-unit MGDA comparison on the local LUNA16 cache."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torchvision

from training.dataset import build_dataloaders
from training.experiment import BASE_CONFIG, ExperimentRunner
from training.model import build_model
from training.trainer import seed_everything


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
OUTPUTS = {
    "raw": ("MGDA-raw-maskhead-10ep", "raw"),
    "normalized": ("MGDA-normalized-maskhead-10ep", "unit"),
}
TASKS = ("bce", "dice", "boundary", "focal", "hd")
TASK_LABELS = {
    "bce": "BCE", "dice": "Dice", "boundary": "Boundary",
    "focal": "Focal", "hd": "HD-distance",
}
SEED = 20261121


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _make_initial_checkpoint(config: dict, path: Path) -> str:
    if path.exists():
        raise FileExistsError(f"Shared initialization already exists: {path}")
    seed_everything(SEED)
    model = build_model(config).cpu()
    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temp_path)
    os.replace(temp_path, path)
    del model, state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _sha256(path)


def _preflight() -> tuple[dict, dict]:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"Expected Python 3.11, got {sys.version.split()[0]}")
    if torch.__version__ != "2.4.0+cu124":
        raise RuntimeError(f"Expected PyTorch 2.4.0+cu124, got {torch.__version__}")
    if torchvision.__version__ != "0.19.0+cu124":
        raise RuntimeError(f"Expected TorchVision 0.19.0+cu124, got {torchvision.__version__}")
    if np.__version__ != "1.26.4":
        raise RuntimeError(f"Expected NumPy 1.26.4, got {np.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment; refusing CPU fallback")
    device_name = torch.cuda.get_device_name(0)
    if "4050" not in device_name:
        raise RuntimeError(f"Expected RTX 4050, got {device_name}")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    hardware = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": device_name,
        "gpu_total_memory_mb": total_bytes / (1024 ** 2),
        "gpu_free_memory_before_run_mb": free_bytes / (1024 ** 2),
        "training_device": "cuda:0",
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": "enabled with warn_only=True",
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "known_nondeterminism_note": (
            "Some torchvision/CUDA detection and ROI operations may still be nondeterministic; "
            "PyTorch deterministic mode is warn-only and any warnings are preserved in logs."
        ),
    }
    logging.info("Preflight hardware: %s", hardware)
    return hardware, {"device_name": device_name, "total_memory_bytes": total_bytes}


def _verify_dataset(prior_config_path: Path, cache_path: Path, data_root: Path, batch_size: int):
    prior = json.loads(prior_config_path.read_text(encoding="utf-8"))
    prior_split = prior["data_split"]
    train_loader, validation_loader = build_dataloaders(
        data_root=data_root,
        cache_dir=cache_path,
        train_subsets=[1],
        validation_subset=0,
        batch_size=batch_size,
        num_workers=0,
        patch_size=64,
        seed=SEED,
    )
    train_uids = set(train_loader.dataset.scan_uids)
    validation_uids = set(validation_loader.dataset.scan_uids)
    if train_uids & validation_uids:
        raise RuntimeError("UID overlap found between training and validation subsets")
    if train_uids != set(prior_split["train_scan_uids"]):
        raise RuntimeError("Current train scan UIDs differ from the verified one-epoch run")
    if validation_uids != set(prior_split["validation_scan_uids"]):
        raise RuntimeError("Current validation scan UIDs differ from the verified one-epoch run")
    if len(train_uids) != 89 or len(validation_uids) != 89:
        raise RuntimeError(f"Expected 89/89 scans, got {len(train_uids)}/{len(validation_uids)}")
    if len(train_loader.dataset) != 4280 or len(validation_loader.dataset) != 3740:
        raise RuntimeError("Patch counts differ from the verified full-cache experiment")
    if len(train_loader) != 1070 or len(validation_loader) != 935:
        raise RuntimeError(
            f"Expected 1,070/935 batches per epoch, got {len(train_loader)}/{len(validation_loader)}"
        )
    manifest_hash = _sha256(cache_path / "manifest.json")
    details = {
        "cache_dir": str(cache_path),
        "cache_manifest_sha256": manifest_hash,
        "train_subsets": [1],
        "validation_subset": 0,
        "train_scan_count": len(train_uids),
        "validation_scan_count": len(validation_uids),
        "train_scan_uids": sorted(train_uids),
        "validation_scan_uids": sorted(validation_uids),
        "uid_overlap_count": len(train_uids & validation_uids),
        "train_patch_count": len(train_loader.dataset),
        "validation_patch_count": len(validation_loader.dataset),
        "train_batches_per_epoch": len(train_loader),
        "validation_batches_per_epoch": len(validation_loader),
    }
    del train_loader, validation_loader
    logging.info("Dataset controls verified: %s", {k: v for k, v in details.items() if "uids" not in k})
    return details


def _metric(histories: list[dict], key: str, mode: str):
    values = [float(row[key]) for row in histories if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))]
    if not values:
        return None
    return max(values) if mode == "max" else min(values)


def _mean(histories: list[dict], key: str):
    values = [float(row[key]) for row in histories if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))]
    return sum(values) / len(values) if values else None


def _save_plots(histories: dict[str, list[dict]], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    styles = {"raw": "#ef8354", "normalized": "#2589bd"}

    def metric_plot(key: str, ylabel: str, filename: str, lower_is_better: bool = False):
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, rows in histories.items():
            xs = [row["epoch"] for row in rows]
            ys = [row.get(key) for row in rows]
            ax.plot(xs, ys, marker="o", label=label.title(), color=styles[label])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    metric_plot("val_dice", "Validation Dice", "validation_dice.png")
    metric_plot("val_hd95", "Validation HD95 (pixels)", "hd95.png", lower_is_better=True)
    metric_plot("val_boundary_f1", "Validation Boundary F1", "boundary_f1.png")
    metric_plot("total_loss", "Training loss", "training_loss.png", lower_is_better=True)
    metric_plot("epoch_time_s", "Epoch duration (seconds)", "epoch_duration.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, rows in histories.items():
        for task in TASKS:
            ax.plot([row["epoch"] for row in rows], [row.get(f"mgda_weight_{task}") for row in rows],
                    marker="o", color=styles[label], linestyle="-" if label == "raw" else "--",
                    label=f"{label.title()} {TASK_LABELS[task]}")
    ax.set(xlabel="Epoch", ylabel="Simplex weight", ylim=(-0.03, 1.03))
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "mgda_weights.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, rows in histories.items():
        for task in TASKS:
            ax.plot([row["epoch"] for row in rows], [row.get(f"mgda_grad_norm_{task}") for row in rows],
                    marker=".", color=styles[label], linestyle="-" if label == "raw" else "--",
                    label=f"{label.title()} {TASK_LABELS[task]}")
    ax.set(xlabel="Epoch", ylabel="Original gradient L2 norm")
    ax.set_yscale("log")
    ax.grid(alpha=0.25, which="both")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "gradient_norms.png", dpi=160)
    plt.close(fig)

    cosine_keys = [f"mgda_cosine_{left}_{right}" for i, left in enumerate(TASKS) for right in TASKS[i:]]
    matrices = {}
    for label, rows in histories.items():
        matrix = np.eye(len(TASKS), dtype=np.float64)
        for i, left in enumerate(TASKS):
            for j in range(i + 1, len(TASKS)):
                key = f"mgda_cosine_{left}_{TASKS[j]}"
                value = _mean(rows, key)
                matrix[i, j] = matrix[j, i] = np.nan if value is None else value
        matrices[label] = matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, label in zip(axes, ("raw", "normalized")):
        image = ax.imshow(matrices[label], vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(f"{label.title()} MGDA")
        ax.set_xticks(range(len(TASKS)), [TASK_LABELS[t] for t in TASKS], rotation=35, ha="right")
        ax.set_yticks(range(len(TASKS)), [TASK_LABELS[t] for t in TASKS])
    fig.colorbar(image, ax=axes, label="Pairwise gradient cosine")
    fig.tight_layout()
    fig.savefig(output_dir / "gradient_cosines.png", dpi=160)
    plt.close(fig)


def _write_report(results: dict, histories: dict[str, list[dict]], metadata: dict, path: Path) -> None:
    raw = histories["raw"]
    unit = histories["normalized"]
    summary_rows = {}
    for label, rows in histories.items():
        best_dice_row = max(rows, key=lambda row: float(row["val_dice"]))
        summary_rows[label] = {
            "best_dice": _metric(rows, "val_dice", "max"),
            "best_dice_epoch": best_dice_row["epoch"],
            "best_dice_checkpoint_hd95": best_dice_row.get("val_hd95"),
            "best_dice_checkpoint_boundary_f1": best_dice_row.get("val_boundary_f1"),
            "final_dice": rows[-1].get("val_dice") if rows else None,
            "best_hd95": _metric(rows, "val_hd95", "min"),
            "final_hd95": rows[-1].get("val_hd95") if rows else None,
            "best_boundary_f1": _metric(rows, "val_boundary_f1", "max"),
            "final_boundary_f1": rows[-1].get("val_boundary_f1") if rows else None,
            "training_duration_s": sum(float(row.get("epoch_time_s", 0)) for row in rows),
            "mean_epoch_duration_s": _mean(rows, "epoch_time_s"),
            "max_epoch_duration_s": max((float(row.get("epoch_time_s", 0)) for row in rows), default=0.0),
        }
        for task in TASKS:
            summary_rows[label][f"mean_weight_{task}"] = _mean(rows, f"mgda_weight_{task}")
            summary_rows[label][f"mean_grad_norm_{task}"] = _mean(rows, f"mgda_grad_norm_{task}")
    lines = [
        "# Raw MGDA vs Unit-Normalized MGDA — Controlled Comparison",
        "",
        "## Protocol",
        "",
        f"- Seed: `{metadata['seed']}`; identical initial state SHA256: `{metadata['initial_state_sha256']}`.",
        f"- GPU: {metadata['hardware']['gpu_name']} with CUDA {metadata['hardware']['cuda_runtime']}.",
        f"- Shared cache manifest SHA256: `{metadata['dataset']['cache_manifest_sha256']}`.",
        "- Five objectives; both runs use `mask_head`, the same 89/89 UID-disjoint train/validation split, batch size 4, LR 0.005, SGD, and the same scheduler.",
        "- Both runs used 1,070 training batches and 935 validation batches per epoch. The saved per-epoch `train_sample_order_sha256` values are checked to match.",
        "- Raw MGDA is the existing `solve_mgda(..., gradient_normalization='raw')` control. Unit MGDA normalizes each objective gradient before building the Gram matrix, then applies `lambda_i / ||g_i||` to original losses, so the final backward reproduces the optimized normalized-gradient combination.",
        "- Determinism was enabled where supported. CUDA detection/ROI operations may remain nondeterministic; this is a paired run, not a statistical significance test.",
        "",
        "## Results",
        "",
        "| Metric | Raw MGDA | Unit-normalized MGDA |",
        "|---|---:|---:|",
    ]
    display = [
        ("Best validation Dice", "best_dice"), ("Best-Dice checkpoint epoch", "best_dice_epoch"),
        ("HD95 at best-Dice checkpoint", "best_dice_checkpoint_hd95"),
        ("Boundary F1 at best-Dice checkpoint", "best_dice_checkpoint_boundary_f1"),
        ("Final validation Dice", "final_dice"), ("Final HD95", "final_hd95"),
        ("Final Boundary F1", "final_boundary_f1"),
        ("Minimum HD95 across epochs", "best_hd95"),
        ("Maximum Boundary F1 across epochs", "best_boundary_f1"),
        ("Recorded wall-clock duration (s; includes outliers)", "training_duration_s"),
        ("Mean recorded epoch duration (s; includes outliers)", "mean_epoch_duration_s"),
        ("Longest recorded epoch (s)", "max_epoch_duration_s"),
    ]
    for title, key in display:
        lines.append(f"| {title} | {summary_rows['raw'][key]:.6g} | {summary_rows['normalized'][key]:.6g} |")
    for task in TASKS:
        lines.append(f"| Mean {TASK_LABELS[task]} simplex weight | {summary_rows['raw'][f'mean_weight_{task}']:.6g} | {summary_rows['normalized'][f'mean_weight_{task}']:.6g} |")
    for task in TASKS:
        lines.append(f"| Mean {TASK_LABELS[task]} original gradient norm | {summary_rows['raw'][f'mean_grad_norm_{task}']:.6g} | {summary_rows['normalized'][f'mean_grad_norm_{task}']:.6g} |")
    lines.extend(["", "## Interpretation", ""])
    raw_bce = summary_rows["raw"]["mean_weight_bce"]
    raw_focal = summary_rows["raw"]["mean_weight_focal"]
    unit_bce = summary_rows["normalized"]["mean_weight_bce"]
    unit_focal = summary_rows["normalized"]["mean_weight_focal"]
    lines.append(
        f"Raw-gradient mean weights were BCE `{raw_bce:.6g}` and Focal `{raw_focal:.6g}`; "
        f"unit-normalized mean weights were BCE `{unit_bce:.6g}` and Focal `{unit_focal:.6g}`."
    )
    raw_conflicts = [_mean(raw, f"mgda_cosine_{a}_{b}") for i, a in enumerate(TASKS) for b in TASKS[i + 1:]]
    unit_conflicts = [_mean(unit, f"mgda_cosine_{a}_{b}") for i, a in enumerate(TASKS) for b in TASKS[i + 1:]]
    raw_fraction = sum(x is not None and x < 0 for x in raw_conflicts) / len(raw_conflicts)
    unit_fraction = sum(x is not None and x < 0 for x in unit_conflicts) / len(unit_conflicts)
    lines.append(f"Negative pairwise-cosine fraction: raw `{raw_fraction:.3f}`, normalized `{unit_fraction:.3f}`.")
    dice_delta = summary_rows["normalized"]["final_dice"] - summary_rows["raw"]["final_dice"]
    hd_delta = summary_rows["normalized"]["final_hd95"] - summary_rows["raw"]["final_hd95"]
    bdy_delta = summary_rows["normalized"]["final_boundary_f1"] - summary_rows["raw"]["final_boundary_f1"]
    lines.append(f"Final validation deltas (normalized minus raw): Dice `{dice_delta:+.6g}`, HD95 `{hd_delta:+.6g}`, Boundary F1 `{bdy_delta:+.6g}`.")
    lines.append("")
    lines.append("Relative changes at each profile's best-Dice checkpoint (normalized vs raw):")
    best_relative = (
        ("Dice", "best_dice", "higher is better"),
        ("HD95", "best_dice_checkpoint_hd95", "lower is better"),
        ("Boundary F1", "best_dice_checkpoint_boundary_f1", "higher is better"),
    )
    for metric, key, direction in best_relative:
        raw_value = summary_rows["raw"][key]
        unit_value = summary_rows["normalized"][key]
        relative = (unit_value - raw_value) / abs(raw_value) * 100.0 if raw_value else float("nan")
        lines.append(f"- {metric}: `{relative:+.3f}%` ({direction}; raw `{raw_value:.6g}`, normalized `{unit_value:.6g}`).")
    raw_duration = summary_rows["raw"]["training_duration_s"]
    unit_duration = summary_rows["normalized"]["training_duration_s"]
    duration_change = (unit_duration - raw_duration) / raw_duration * 100.0 if raw_duration else float("nan")
    unit_longest_epoch = summary_rows["normalized"]["max_epoch_duration_s"]
    unit_epoch_durations = [float(row.get("epoch_time_s", 0)) for row in unit]
    second_longest_unit_epoch = sorted(unit_epoch_durations)[-2] if len(unit_epoch_durations) > 1 else 0.0
    lines.append(f"- Recorded elapsed time was `{duration_change:+.1f}%` higher for unit-normalized MGDA (raw `{raw_duration:.2f}s`, normalized `{unit_duration:.2f}s`).")
    lines.append(f"- Runtime is not a reliable throughput comparison here: normalized epoch 6 took `{unit_longest_epoch:.2f}s`, versus `{second_longest_unit_epoch:.2f}s` for its next-longest epoch. Treat the total-time difference as an observed run condition, not as a stable estimate of normalization overhead.")
    lines.append("These are observations from one paired seed. They do not establish statistical significance or independent test performance. Nodule masks are generated approximations, so segmentation metrics measure agreement with the cache masks.")
    lines.extend(["", "## Artifacts", ""])
    for label, summary in results.items():
        lines.append(f"- `{summary['config_id']}`: `{summary['run_dir']}`")
    lines.append(f"- Shared initialization: `{metadata['initial_checkpoint_path']}`")
    lines.append("- Plots: `reports/mgda_raw_vs_normalized/`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    hardware, _ = _preflight()
    prior_run = EXPERIMENTS / "MGDA-verify-all5-1ep-20261121"
    prior_config_path = prior_run / "config.json"
    if not prior_config_path.is_file():
        raise FileNotFoundError(f"Verified reference run config missing: {prior_config_path}")
    prior_config = json.loads(prior_config_path.read_text(encoding="utf-8"))
    cache_path = Path(prior_config["cache_dir"]).resolve()
    data_root = Path(prior_config["data_root"]).resolve()
    if not cache_path.is_dir() or not data_root.is_dir():
        raise FileNotFoundError("Required LUNA16 cache or raw subset metadata directory is unavailable")

    common = {
        **BASE_CONFIG,
        "data_root": str(data_root),
        "cache_dir": str(cache_path),
        "train_subsets": [1],
        "validation_subset": 0,
        "test_subset": None,
        "epochs": 10,
        "warmup_epochs": 5,
        "batch_size": 4,
        "num_workers": 0,
        "lr": 0.005,
        "min_size": 64,
        "max_size": 64,
        "patch_size": 64,
        "pretrained_backbone": True,
        "mask_loss": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "delta": 1.0, "epsilon": 1.0},
        "seed": SEED,
        "require_cuda": True,
        "hardware": hardware,
    }
    if prior_config.get("batch_size") != common["batch_size"] or prior_config.get("lr") != common["lr"]:
        raise RuntimeError("Reference run batch size or learning rate does not match requested controls")
    dataset = _verify_dataset(prior_config_path, cache_path, data_root, common["batch_size"])

    report_dir = ROOT / "reports" / "mgda_raw_vs_normalized"
    init_dir = EXPERIMENTS / "MGDA-raw-vs-normalized-initial"
    init_path = init_dir / "model_state.pth"
    for config_id, _normalization in OUTPUTS.values():
        if (EXPERIMENTS / config_id).exists():
            raise FileExistsError(f"Refusing to overwrite existing experiment directory: {EXPERIMENTS / config_id}")
    init_dir.mkdir(parents=True, exist_ok=False)
    initial_config = {**common, "training_strategy": "mgda"}
    initial_hash = _make_initial_checkpoint(initial_config, init_path)
    metadata = {
        "seed": SEED,
        "initial_checkpoint_path": str(init_path.resolve()),
        "initial_state_sha256": initial_hash,
        "hardware": hardware,
        "dataset": dataset,
        "epochs": 10,
        "batch_size": 4,
        "learning_rate": 0.005,
        "optimizer": "SGD(momentum=0.9, weight_decay=0.0001)",
        "scheduler": "linear warmup for 5 epochs then cosine annealing",
    }
    _write_json(init_dir / "metadata.json", metadata)
    logging.info("Shared initial state SHA256: %s", initial_hash)

    runner = ExperimentRunner(common, str(EXPERIMENTS))
    results: dict[str, dict] = {}
    histories: dict[str, list[dict]] = {}
    errors = {}
    for label, (config_id, normalization) in OUTPUTS.items():
        run_dir = EXPERIMENTS / config_id
        run_dir.mkdir(parents=True, exist_ok=False)
        log_handler = logging.FileHandler(run_dir / "experiment.log", encoding="utf-8")
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)
        try:
            seed_everything(SEED)
            override = {
                "config_id": config_id,
                "training_strategy": "mgda",
                "mgda_profile": "raw_mask_head" if normalization == "raw" else "unit_mask_head",
                "mgda_gradient_normalization": normalization,
                "mgda_parameter_scope": "mask_head",
                "mask_loss": common["mask_loss"],
                "initial_state_path": str(init_path.resolve()),
                "initial_state_sha256": initial_hash,
                "seed": SEED,
                "require_cuda": True,
                "hardware": hardware,
                "dataset_control": dataset,
                "expected_train_batches_per_epoch": dataset["train_batches_per_epoch"],
            }
            logging.info("Starting %s (%s-gradient MGDA), sequentially", config_id, normalization)
            summary = runner._run_single(override)
            history_path = run_dir / "history.json"
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if len(history) != 10 or [row["epoch"] for row in history] != list(range(1, 11)):
                raise RuntimeError(f"{config_id} did not complete exactly 10 epochs")
            if summary.get("train_batches_per_epoch") != 1070:
                raise RuntimeError(f"{config_id} used {summary.get('train_batches_per_epoch')} train batches, expected 1070")
            if summary.get("initial_state_sha256") != initial_hash:
                raise RuntimeError(f"{config_id} did not load the shared initialization")
            for row in history:
                simplex = [row[f"mgda_weight_{task}"] for task in TASKS]
                if not all(math.isfinite(value) and value >= -1e-10 for value in simplex) or not math.isclose(sum(simplex), 1.0, rel_tol=1e-4, abs_tol=1e-4):
                    raise RuntimeError(f"Invalid five-objective simplex at epoch {row['epoch']} in {config_id}")
                if not all(math.isfinite(row[f"mgda_grad_norm_{task}"]) for task in TASKS):
                    raise RuntimeError(f"Non-finite objective gradient norms in {config_id}")
            results[label] = {**summary, "run_dir": str(run_dir.resolve())}
            histories[label] = history
        except Exception as exc:
            errors[label] = f"{type(exc).__name__}: {exc}"
            logging.exception("Experiment %s failed; preserving its directory and log", config_id)
            _write_json(run_dir / "failure.json", {"config_id": config_id, "error": errors[label]})
        finally:
            root_logger.removeHandler(log_handler)
            log_handler.close()

    if len(histories) == 2:
        raw_hashes = [row.get("train_sample_order_sha256") for row in histories["raw"]]
        unit_hashes = [row.get("train_sample_order_sha256") for row in histories["normalized"]]
        if raw_hashes != unit_hashes or not all(raw_hashes):
            errors["sampling_control"] = "Per-epoch sample-order hashes did not match"
        _save_plots(histories, report_dir)
        _write_report(results, histories, metadata, ROOT / "reports" / "MGDA_RAW_VS_NORMALIZED_COMPARISON.md")
    _write_json(report_dir / "test_verification.json", {
        "all_experiments_completed": len(histories) == 2,
        "errors": errors,
        "initial_state_sha256": initial_hash,
        "seed": SEED,
        "required_epochs": 10,
        "required_train_batches_per_epoch": 1070,
        "completed_config_ids": [value.get("config_id") for value in results.values()],
    })
    if errors or len(results) != 2:
        logging.error("Controlled comparison incomplete: %s", errors)
        return 1
    logging.info("Both controlled MGDA runs completed and verified; report: %s", ROOT / "reports" / "MGDA_RAW_VS_NORMALIZED_COMPARISON.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
