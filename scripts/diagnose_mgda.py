#!/usr/bin/env python3
"""Measure MGDA scales/conflicts on a small set of cached positive patches.

Example:
  python scripts/diagnose_mgda.py \
    --checkpoint experiments/MGDA-all5-ep10/best_model.pth \
    --cache-dir data/patches_cache_v2 --subset 1 --max-samples 2

This is a bounded diagnostic, not a training run or an evaluation result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from checkpoint_io import load_checkpoint_weights_only


def _target_from_sample(sample: dict, device: torch.device) -> dict:
    mask = sample["mask"].to(device=device, dtype=torch.uint8)
    rows, cols = torch.where(mask > 0)
    if rows.numel() == 0:
        raise ValueError("selected positive cache entry has an empty mask")
    box = torch.stack((cols.min(), rows.min(), cols.max() + 1, rows.max() + 1))
    return {
        "boxes": box.to(dtype=torch.float32).unsqueeze(0),
        "labels": torch.ones(1, dtype=torch.int64, device=device),
        "masks": mask.unsqueeze(0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--subset", type=int, required=True)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument(
        "--parameter-scope",
        choices=("mask_head", "shared_mask_path"),
        default=None,
        help="Override the scope stored in the sibling experiment config",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None, help="Optional path to write the JSON report")
    args = parser.parse_args()
    if args.max_samples < 1:
        parser.error("--max-samples must be at least 1")

    from training.dataset import CachedLUNA16Dataset
    from training.mgda import solve_mgda
    from training.model import build_model

    checkpoint_path = Path(args.checkpoint)
    config_path = checkpoint_path.parent / "config.json"
    if not checkpoint_path.is_file() or not config_path.is_file():
        parser.error("checkpoint and sibling config.json must both exist")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint = load_checkpoint_weights_only(checkpoint_path, map_location="cpu")
    model_config = dict(config)
    model_config["pretrained_backbone"] = False
    model_config["mask_loss"] = {
        "alpha": 1.0, "beta": 1.0, "gamma": 1.0, "delta": 1.0, "epsilon": 1.0,
    }
    device = torch.device(args.device)
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.composite_mask_loss.capture_components = True
    model.train()

    dataset = CachedLUNA16Dataset(
        args.cache_dir,
        subsets=[args.subset],
        augment=False,
        expected_patch_size=int(config.get("patch_size", 64)),
    )
    positive_indices = [i for i, entry in enumerate(dataset.entries) if int(entry["label"]) == 1]
    if not positive_indices:
        parser.error(f"cache subset {args.subset} contains no positive patches")

    scope = args.parameter_scope or config.get("mgda_parameter_scope", "mask_head")
    def is_mask_parameter(name: str) -> bool:
        return "roi_heads.mask_head." in name or "roi_heads.mask_predictor." in name

    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and (
            is_mask_parameter(name)
            or (
                scope == "shared_mask_path"
                and ("backbone.body." in name or "backbone.fpn." in name)
            )
        )
    ]
    if not parameters:
        parser.error(f"no trainable parameters found for MGDA scope {scope!r}")

    report = {
        "diagnostic_only": True,
        "checkpoint": str(checkpoint_path),
        "cache_dir": str(Path(args.cache_dir)),
        "subset": args.subset,
        "parameter_scope": scope,
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "samples": [],
    }
    task_names = ("bce", "dice", "boundary", "focal", "hd")
    for index in positive_indices[:args.max_samples]:
        sample = dataset[index]
        image = sample["image"].to(device)
        target = _target_from_sample(sample, device)
        model.zero_grad(set_to_none=True)
        model([image], [target])
        objectives = model.composite_mask_loss.last_components
        if not objectives or any(name not in objectives for name in task_names):
            raise RuntimeError("model did not expose all five MGDA objectives")
        raw = solve_mgda(objectives, parameters, gradient_normalization="raw")
        unit = solve_mgda(objectives, parameters, gradient_normalization="unit")
        report["samples"].append({
            "uid": sample["uid"],
            "label": int(sample["label"]),
            "losses": {name: float(objectives[name].detach().cpu()) for name in task_names},
            "gradient_norms": raw.gradient_norms,
            "raw_weights": raw.weights,
            "unit_gradient_weights": unit.weights,
            "unit_applied_coefficients": unit.applied_coefficients,
            "gradient_cosines": {
                key.removeprefix("mgda_cosine_"): value
                for key, value in raw.diagnostics.items()
                if key.startswith("mgda_cosine_")
            },
            "raw_solver_diagnostics": raw.diagnostics,
        })

    encoded = json.dumps(report, indent=2, allow_nan=False)
    print(encoded)
    if args.out:
        Path(args.out).write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
