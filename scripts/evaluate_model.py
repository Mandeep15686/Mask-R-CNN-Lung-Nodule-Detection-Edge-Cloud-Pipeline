#!/usr/bin/env python3
"""
Full evaluation of a trained Mask R-CNN checkpoint on the LUNA16 test set.
Produces:
  - Per-metric summary (Dice, IoU, HD95, Boundary F1, Sensitivity)
  - A sampled-patch FROC proxy (not official scan-level LUNA16 FROC)
  - Per-nodule-subtype breakdown (solid / GGO / juxta-pleural)
  - Comparison table across multiple checkpoints
  - JSON + CSV reports saved to output directory

Usage:
  # Evaluate one checkpoint
  python scripts/evaluate_model.py \
    --checkpoint ./experiments/G-a03-b05-g05/best_model.pth \
    --data ./data/luna16 --fold 9 --out ./eval_results

  # Compare multiple checkpoints
  python scripts/evaluate_model.py \
    --checkpoints ./experiments/*/best_model.pth \
    --data ./data/luna16 --fold 9 --out ./eval_results
"""
import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from glob import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [EVAL] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Nodule subtype classifier (by size)
# ---------------------------------------------------------------------------
def classify_nodule_subtype(diameter_mm: float) -> str:
    """
    Approximate subtype from diameter:
      < 6mm   → small solid
      6–15mm  → solid
      > 15mm  → large / possibly GGO
    In real research this requires radiologist labels; use diameter as proxy.
    """
    if diameter_mm < 6.0:
        return "small_solid"
    elif diameter_mm < 15.0:
        return "solid"
    else:
        return "large"


# ---------------------------------------------------------------------------
# Single checkpoint evaluation
# ---------------------------------------------------------------------------
def evaluate_checkpoint(
    checkpoint_path: str,
    data_root: str,
    fold: int = 9,
    batch_size: int = 4,
    patch_size: int = 64,
    score_threshold: float = 0.3,
    device_str: str = "auto",
) -> dict:
    """
    Load a Mask R-CNN checkpoint and run full evaluation on the given fold.

    Returns a dict of aggregate metrics + per-subtype breakdown.
    """
    import torch
    import numpy as np
    from training.model import LungNoduleMaskRCNN
    from training.dataset import LUNA16Dataset
    from training.metrics import (
        evaluate_batch, compute_froc, dice_score,
        iou_score, hausdorff_distance_95, boundary_f1, volume_error_percent,
    )
    from torch.utils.data import DataLoader

    # Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info(f"Device: {device}")

    data_path = Path(data_root).expanduser().resolve()
    available_folds = sorted(
        int(path.name.removeprefix("subset"))
        for path in data_path.glob("subset*")
        if path.is_dir() and path.name.removeprefix("subset").isdigit()
    )
    if fold not in available_folds:
        raise ValueError(
            f"Evaluation subset {fold} is unavailable under {data_path}. "
            f"Available subsets: {available_folds}"
        )
    logger.warning(
        "FROC is computed over sampled patches from the selected subset; "
        "this is a proxy and is not official scan-level LUNA16 FROC."
    )

    # Load checkpoint metadata
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = ckpt.get("config", {})
    mask_loss_cfg = config.get("mask_loss", {"alpha": 1.0})
    training_fold = config.get("fold")
    if training_fold is None:
        evaluation_data_role = "unknown_training_split"
        logger.warning(
            "Checkpoint has no training fold metadata; independence of this "
            "evaluation subset cannot be established."
        )
    elif fold == int(training_fold):
        evaluation_data_role = "validation_fold_used_during_training"
        logger.warning(
            "Selected subset is the checkpoint's validation fold and may have "
            "been used for model selection; it is not an independent test set."
        )
    else:
        evaluation_data_role = "subset_in_training_data"
        logger.warning(
            "Selected subset was included in training because training uses all "
            "subsets except its validation fold; this is not an independent test set."
        )

    # Build model
    model = LungNoduleMaskRCNN(num_classes=2, mask_loss_cfg=mask_loss_cfg)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.to(device)
    model.eval()
    logger.info(f"Loaded: {checkpoint_path}")

    # Dataset
    test_ds = LUNA16Dataset(
        data_root=data_root,
        subsets=[fold],
        patch_size=patch_size,
        augment=False,
    )
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    logger.info(f"Test samples: {len(test_ds)}")

    all_preds, all_targets, all_labels, all_diameters = [], [], [], []
    all_uids, false_positive_detections = [], []
    all_scores, detection_labels = [], []
    inference_times = []
    subtype_metrics: dict[str, list] = {"small_solid": [], "solid": [], "large": []}

    with torch.no_grad():
        for batch in loader:
            images   = batch["image"].to(device)
            masks    = batch["mask"]
            labels   = batch["label"]
            diams    = batch["diameter_mm"]

            t0 = time.perf_counter()
            outputs = model(list(images))
            inference_times.append((time.perf_counter() - t0) * 1000 / len(images))

            for i, out in enumerate(outputs):
                h, w = masks[i].shape
                pred_mask = np.zeros((h, w), dtype=np.float32)

                scores_i = out.get("scores", torch.zeros(0)).cpu().numpy()
                boxes_i  = out.get("boxes",  torch.zeros(0, 4)).cpu().numpy()
                masks_i  = out.get("masks",  torch.zeros(0, 1, h, w)).cpu().numpy()
                false_positive_detections.append(
                    int(np.any(masks_i > 0.5, axis=(1, 2, 3)).sum())
                    if len(masks_i) else 0
                )

                if len(scores_i) > 0:
                    best_idx = int(np.argmax(scores_i))
                    pred_mask = masks_i[best_idx, 0]
                    all_scores.append(float(scores_i[best_idx]))
                else:
                    all_scores.append(0.0)

                # Detection label: 1 if GT has nodule and predicted score > threshold
                is_nodule_gt = int(labels[i].item())
                detection_labels.append(is_nodule_gt)

                gt_mask = masks[i].numpy()
                all_preds.append(pred_mask)
                all_targets.append(gt_mask)
                all_uids.append(str(batch["uid"][i]))
                all_labels.append(is_nodule_gt)
                all_diameters.append(float(diams[i].item()))

                # Per-subtype
                if is_nodule_gt == 1:
                    diam = float(diams[i].item())
                    sub  = classify_nodule_subtype(diam)
                    pred_bin = (pred_mask > score_threshold).astype(np.uint8)
                    subtype_metrics[sub].append({
                        "dice":        dice_score(pred_bin, gt_mask),
                        "iou":         iou_score(pred_bin, gt_mask),
                        "hd95":        hausdorff_distance_95(pred_bin, gt_mask),
                        "boundary_f1": boundary_f1(pred_bin, gt_mask),
                        "vol_err":     volume_error_percent(pred_bin, gt_mask),
                    })

    # Aggregate segmentation metrics
    preds_arr   = np.stack(all_preds)
    targets_arr = np.stack(all_targets)
    seg_metrics = evaluate_batch(
        preds_arr,
        targets_arr,
        threshold=score_threshold,
        scan_uids=all_uids,
        false_positive_detections=false_positive_detections,
    )

    # FROC
    n_scans = len({sample["uid"] for sample in test_ds.all_samples})
    froc = compute_froc(all_scores, detection_labels, n_scans=n_scans)

    # Per-subtype aggregation
    subtype_summary = {}
    for sub, records in subtype_metrics.items():
        if records:
            subtype_summary[sub] = {
                k: float(np.nanmean([r[k] for r in records]))
                for k in records[0]
            }
            subtype_summary[sub]["n"] = len(records)
        else:
            subtype_summary[sub] = {"n": 0}

    return {
        "checkpoint":           checkpoint_path,
        "config_id":            config.get("config_id", Path(checkpoint_path).parent.name),
        "loss_description":     config.get("mask_loss", {}),
        "fold":                 fold,
        "evaluation_data_role": evaluation_data_role,
        "n_samples":            len(test_ds),
        "avg_inference_ms":     float(np.mean(inference_times)) if inference_times else 0.0,
        **seg_metrics,
        "patch_sample_froc_score": froc["froc_score"],
        "patch_sample_froc_per_point": froc["froc_per_point"],
        "froc_protocol": "sampled_patch_proxy_not_official_scan_level_luna16",
        "subtype_metrics":      subtype_summary,
    }


# ---------------------------------------------------------------------------
# Multi-checkpoint comparison
# ---------------------------------------------------------------------------
def compare_checkpoints(
    checkpoint_paths: list[str],
    data_root: str,
    fold: int,
    output_dir: str,
    **eval_kwargs,
) -> list[dict]:
    results = []
    for ckpt_path in checkpoint_paths:
        logger.info(f"\n{'='*60}\nEvaluating: {ckpt_path}\n{'='*60}")
        try:
            result = evaluate_checkpoint(ckpt_path, data_root, fold=fold, **eval_kwargs)
            results.append(result)
            logger.info(
                f"  Dice={result['dice']:.4f}  IoU={result['iou']:.4f}  "
                f"HD95={result['hd95']:.2f}  BF1={result['boundary_f1']:.4f}  "
                f"patch-FROC proxy={result['patch_sample_froc_score']:.4f}"
            )
        except Exception as e:
            logger.error(f"  FAILED: {e}")

    # Sort by Dice descending
    results.sort(key=lambda r: r.get("dice", 0), reverse=True)

    # Save JSON
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save CSV
    if results:
        flat_keys = [
            "config_id", "checkpoint", "fold", "n_samples",
            "dice", "iou", "sensitivity", "specificity",
            "hd95", "boundary_f1", "asd", "vol_err",
            "patch_sample_froc_score", "avg_inference_ms",
        ]
        with open(out / "evaluation_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=flat_keys, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                flat = {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}
                writer.writerow(flat)

    # Print summary table
    print(f"\n{'='*90}")
    print(f"{'Config ID':<30} {'Dice':>6} {'IoU':>6} {'HD95':>7} {'BF1':>6} {'pFROC':>6} {'ms/img':>7}")
    print(f"{'-'*90}")
    for r in results:
        print(
            f"{r.get('config_id','?'):<30} "
            f"{r.get('dice',0):>6.4f} "
            f"{r.get('iou',0):>6.4f} "
            f"{r.get('hd95',999):>7.2f} "
            f"{r.get('boundary_f1',0):>6.4f} "
            f"{r.get('patch_sample_froc_score',0):>6.4f} "
            f"{r.get('avg_inference_ms',0):>7.1f}"
        )
    print(f"{'='*90}")
    print(f"\nFull results saved to: {output_dir}/")

    # Print subtype breakdown for best model
    if results:
        best = results[0]
        print(f"\nSubtype breakdown — {best['config_id']}:")
        for sub, sm in best.get("subtype_metrics", {}).items():
            if sm.get("n", 0) > 0:
                print(
                    f"  {sub:<15} n={sm['n']:>3}  "
                    f"Dice={sm.get('dice',0):.4f}  "
                    f"HD95={sm.get('hd95',999):.2f}  "
                    f"BF1={sm.get('boundary_f1',0):.4f}"
                )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Mask R-CNN on LUNA16")
    # Single checkpoint
    parser.add_argument("--checkpoint",  type=str, default=None,
                        help="Path to a single .pth checkpoint")
    # Multiple checkpoints
    parser.add_argument("--checkpoints", type=str, nargs="+", default=None,
                        help="Glob pattern or list of checkpoint paths")
    parser.add_argument("--data",         required=True, help="LUNA16 data root")
    parser.add_argument("--fold",         type=int, required=True,
                        help="Explicit evaluation subset (must be independent of training/model selection)")
    parser.add_argument("--out",          default="./eval_results")
    parser.add_argument("--batch-size",   type=int,   default=4)
    parser.add_argument("--patch-size",   type=int,   default=64)
    parser.add_argument("--threshold",    type=float, default=0.3)
    parser.add_argument("--device",       default="auto")
    args = parser.parse_args()

    # Collect checkpoint paths
    ckpt_paths = []
    if args.checkpoint:
        ckpt_paths = [args.checkpoint]
    elif args.checkpoints:
        for pattern in args.checkpoints:
            expanded = glob(pattern)
            ckpt_paths.extend(expanded if expanded else [pattern])
    else:
        # Auto-discover in experiments/
        ckpt_paths = glob("./experiments/*/best_model.pth")
        if not ckpt_paths:
            parser.error("Provide --checkpoint or --checkpoints, or place checkpoints in ./experiments/")

    if not ckpt_paths:
        logger.error("No checkpoint files found.")
        sys.exit(1)

    logger.info(f"Evaluating {len(ckpt_paths)} checkpoint(s) on fold {args.fold}...")

    compare_checkpoints(
        checkpoint_paths=ckpt_paths,
        data_root=args.data,
        fold=args.fold,
        output_dir=args.out,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        score_threshold=args.threshold,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
