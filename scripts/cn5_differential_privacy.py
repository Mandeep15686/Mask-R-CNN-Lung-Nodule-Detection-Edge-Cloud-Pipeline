#!/usr/bin/env python3
"""
CN5 prototype: Gaussian perturbation of edge features.

This script uses the same cloud continuation as live split inference and a
jointly clipped Gaussian feature-release mechanism. Its formal guarantee is
conditional on a fixed checkpoint and covers numeric feature values only;
metadata, model training, and clinical utility are outside its scope. MIA is
only an empirical attack diagnostic, not a privacy proof.

Applies Gaussian noise ((ε, δ)-DP under the stated release assumptions) to intermediate feature maps at the split point
and measures the privacy–utility tradeoff:
  - Privacy:  Membership Inference Attack (MIA) accuracy (lower = more private)
  - Utility:  Dice score degradation vs baseline (lower = better)

Usage:
  python scripts/cn5_differential_privacy.py \
    --checkpoint ./experiments/G-a03-b05-g05/best_model.pth \
    --data ./data/luna16 \
    --split layer2 \
    --epsilons 0.1 0.5 1.0 \
    --out ./experiments/cn5_dp
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from checkpoint_io import load_checkpoint_weights_only

logging.basicConfig(level=logging.INFO, format="%(asctime)s [CN5-DP] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DP noise injection
# ---------------------------------------------------------------------------
def add_gaussian_dp_noise(
    features: dict,
    epsilon: float,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
    accountant=None,
) -> dict:
    """Apply the shared, jointly clipped Gaussian mechanism to one scan."""
    from edge.dp import BasicCompositionAccountant, gaussian_feature_release

    if accountant is None:
        accountant = BasicCompositionAccountant(epsilon_budget=epsilon, delta_budget=delta)
    noised, _metadata = gaussian_feature_release(
        features,
        epsilon=epsilon,
        delta=delta,
        clip_norm=sensitivity,
        accountant=accountant,
    )
    return noised


# ---------------------------------------------------------------------------
# Membership Inference Attack (shadow model)
# ---------------------------------------------------------------------------
class MembershipInferenceAttack:
    """
    Simple black-box MIA: trains a binary classifier on
    (feature_norm, prediction_confidence) → member / non-member.

    Lower attack accuracy = better privacy (ideal: ~0.5, random guessing).
    """

    def __init__(self):
        self.classifier = None

    def fit(
        self,
        member_features:    list[np.ndarray],
        nonmember_features: list[np.ndarray],
    ):
        """
        Train the attack classifier.
        member_features:    features extracted from training samples
        nonmember_features: features extracted from held-out samples
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("Install scikit-learn: pip install scikit-learn")

        X_m  = np.array([self._feature_to_vec(f) for f in member_features])
        X_nm = np.array([self._feature_to_vec(f) for f in nonmember_features])
        X    = np.vstack([X_m, X_nm])
        y    = np.array([1] * len(X_m) + [0] * len(X_nm))

        self.scaler     = StandardScaler().fit(X)
        self.classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        self.classifier.fit(self.scaler.transform(X), y)

    def attack_accuracy(self, member_features: list, nonmember_features: list) -> float:
        """Return balanced accuracy of the attack (0.5 = random, 1.0 = perfect)."""
        if self.classifier is None:
            raise RuntimeError("Fit the attack first")
        from sklearn.metrics import balanced_accuracy_score
        X_m  = np.array([self._feature_to_vec(f) for f in member_features])
        X_nm = np.array([self._feature_to_vec(f) for f in nonmember_features])
        X    = np.vstack([X_m, X_nm])
        y    = np.array([1] * len(X_m) + [0] * len(X_nm))
        pred = self.classifier.predict(self.scaler.transform(X))
        return float(balanced_accuracy_score(y, pred))

    @staticmethod
    def _feature_to_vec(feat_dict: dict) -> np.ndarray:
        """Summarise feature dict → fixed-length vector for the classifier."""
        vecs = []
        for v in feat_dict.values():
            if hasattr(v, "numpy"):
                arr = v.detach().cpu().numpy().flatten()
            else:
                arr = np.asarray(v).flatten()
            # Use statistical summary to keep vector small
            vecs.extend([
                float(arr.mean()),
                float(arr.std()),
                float(np.percentile(arr, 25)),
                float(np.percentile(arr, 75)),
                float(arr.max()),
                float(arr.min()),
            ])
        return np.array(vecs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
def run_dp_experiment(
    checkpoint_path: str,
    data_root:       str,
    split_layer:     str = "layer2",
    epsilons:        list[float] = None,
    delta:           float = 1e-5,
    n_samples:       int = 200,
    output_dir:      str = "./experiments/cn5_dp",
) -> list[dict]:
    """
    For each ε in epsilons:
      1. Extract features (edge step) with DP noise level σ(ε)
      2. Run remaining Mask R-CNN (cloud step) → measure Dice
      3. Run Membership Inference Attack → measure attack accuracy
      4. Log privacy-utility tradeoff point
    """
    if epsilons is None:
        epsilons = [0.05, 0.1, 0.25, 0.5, 1.0]
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    if not np.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta must be between 0 and 1")
    if not epsilons or any(not np.isfinite(eps) or not 0 < eps <= 1 for eps in epsilons):
        raise ValueError("Per-scan epsilons must be in (0, 1] for the selected Gaussian calibration")

    import torch
    from training.dataset import LUNA16Dataset
    from edge.feature_extractor import EdgeFeatureExtractor
    from edge.dp import BasicCompositionAccountant
    from backend.app.ml.inference import InferenceEngine
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | Split: {split_layer}")

    # Load the checkpoint and require recorded scan-level train/validation roles.
    ckpt  = load_checkpoint_weights_only(checkpoint_path, map_location=device)
    cfg = dict(ckpt.get("config", {}))
    split = cfg.get("data_split") or {}
    train_subsets = split.get("train_subsets")
    validation_subset = split.get("validation_subset")
    if not train_subsets or validation_subset is None or validation_subset in train_subsets:
        raise ValueError(
            "Checkpoint must record disjoint train_subsets and validation_subset; "
            "this script will not invent membership labels or an independent test set"
        )
    inference_engine = InferenceEngine(default_checkpoint_path=checkpoint_path, device=device)
    if inference_engine.model is None:
        raise RuntimeError(f"Could not load checkpoint for split continuation: {inference_engine.model_loading_error}")
    loaded = inference_engine._get_checkpoint(checkpoint_path)
    transform = loaded.model.model.transform

    # Feature extractor (edge)
    extractor = EdgeFeatureExtractor(
        split_layer=split_layer,
        device=device,
        checkpoint_path=checkpoint_path,
        image_mean=tuple(transform.image_mean),
        image_std=tuple(transform.image_std),
    )

    # Membership examples are taken from the recorded training and validation
    # scan subsets, then kept as paired features/targets for actual inference.
    patch_size = int(cfg.get("patch_size", 64))
    member_ds = LUNA16Dataset(data_root, subsets=train_subsets, patch_size=patch_size, augment=False)
    nonmember_ds = LUNA16Dataset(data_root, subsets=[validation_subset], patch_size=patch_size, augment=False)
    member_loader = DataLoader(member_ds, batch_size=1, shuffle=False, num_workers=0)
    nonmember_loader = DataLoader(nonmember_ds, batch_size=1, shuffle=False, num_workers=0)

    # Collect training (member) and test (non-member) features
    member_feats, nonmember_feats = [], []

    def collect(loader, destination, target_masks):
        with torch.no_grad():
            for idx, batch in enumerate(loader):
                if idx >= n_samples:
                    break
                feat = extractor.extract(batch["image"].to(device))
                destination.append({k: v.cpu() for k, v in feat.items()})
                target_masks.append(batch["mask"][0].cpu().numpy())

    member_masks, nonmember_masks = [], []
    collect(member_loader, member_feats, member_masks)
    collect(nonmember_loader, nonmember_feats, nonmember_masks)
    if not member_feats or not nonmember_feats:
        raise ValueError("Both recorded train and validation subsets need at least one sample")

    # Baseline Dice (no DP noise, ε = ∞)
    baseline_dice = _measure_dice_from_features(
        inference_engine, member_feats[:20], member_masks[:20], split_layer,
        loaded.sha256, extractor.image_mean, extractor.image_std,
    )
    logger.info(f"Baseline pseudo-mask agreement Dice (no DP): {baseline_dice:.4f}")

    # Train MIA on baseline (no noise) features
    mia = MembershipInferenceAttack()
    try:
        mia.fit(member_feats[:50], nonmember_feats[:50])
        baseline_mia = mia.attack_accuracy(member_feats[50:], nonmember_feats[50:])
        logger.info(f"Baseline MIA accuracy (no DP): {baseline_mia:.4f}")
    except Exception as e:
        logger.warning(f"MIA failed: {e} — skipping attack accuracy")
        baseline_mia = None

    results = []
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for eps in epsilons:
        logger.info(f"\n--- epsilon = {eps} ---")
        n_releases = min(50, len(member_feats)) + min(50, len(nonmember_feats))
        if delta * n_releases >= 1.0:
            raise ValueError("Basic-composition total delta must remain below 1")
        accountant = BasicCompositionAccountant(
            epsilon_budget=eps * n_releases,
            delta_budget=delta * n_releases,
        )
        noised_member = [
            add_gaussian_dp_noise(f, eps, delta, accountant=accountant)
            for f in member_feats[:50]
        ]
        noised_nonmember = [
            add_gaussian_dp_noise(f, eps, delta, accountant=accountant)
            for f in nonmember_feats[:50]
        ]
        sigma = (2.0 / eps) * np.sqrt(2 * np.log(1.25 / delta))

        # Measure Dice after noise
        dice_noised = _measure_dice_from_features(
            inference_engine, noised_member[:20], member_masks[:20], split_layer,
            loaded.sha256, extractor.image_mean, extractor.image_std,
        )
        dice_drop   = baseline_dice - dice_noised

        # MIA on noised features
        mia_noised  = None
        try:
            mia.fit(noised_member[:40], noised_nonmember[:40])
            mia_noised = mia.attack_accuracy(noised_member[40:], noised_nonmember[40:])
        except Exception:
            pass

        # Privacy leakage reduction (from baseline)
        mia_drop = None
        if baseline_mia is not None and mia_noised is not None:
            mia_drop = baseline_mia - mia_noised

        entry = {
            "epsilon":           eps,
            "delta":             delta,
            "sigma":             sigma,
            "dice":              round(dice_noised, 4),
            "dice_drop":         round(dice_drop, 4),
            "mia_accuracy":      round(mia_noised, 4) if mia_noised else None,
            "mia_drop_from_base":round(mia_drop, 4)   if mia_drop   else None,
            "split_layer":       split_layer,
            "epsilon_spent":     accountant.epsilon_spent,
            "delta_spent":       accountant.delta_spent,
            "mechanism":         "joint_l2_clip_gaussian_replacement_adjacency_basic_composition",
        }
        results.append(entry)
        logger.info(
            f"  sigma={sigma:.3f}  Dice={dice_noised:.4f} (drop={dice_drop:.4f})  "
            f"MIA={mia_noised if mia_noised is not None else 'unavailable'}"
        )

    # Save results
    full_report = {
        "checkpoint":        checkpoint_path,
        "evaluation_scope":  "sampled patches; masks are generated pseudo-masks",
        "privacy_scope":     "numeric feature values conditional on fixed checkpoint; metadata and training are excluded",
        "data_split":        split,
        "split_layer":       split_layer,
        "baseline_dice":     round(baseline_dice, 4),
        "baseline_mia_acc":  round(baseline_mia, 4) if baseline_mia is not None else None,
        "delta":             delta,
        "results":           results,
    }
    report_path = out_dir / "cn5_dp_results.json"
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)

    # Print table
    print(f"\n{'='*72}")
    print("CN5 - Differential Privacy - Privacy-Utility Tradeoff")
    print(f"Split: {split_layer} | Checkpoint: {Path(checkpoint_path).parent.name}")
    print(f"Baseline: Dice={baseline_dice:.4f}  MIA={baseline_mia if baseline_mia is not None else 'unavailable'}")
    print(f"{'-'*72}")
    print(f"{'epsilon':>8}  {'sigma':>8}  {'Dice':>7}  {'delta Dice':>10}  {'MIA acc':>8}  {'delta MIA':>9}")
    print(f"{'-'*72}")
    for r in results:
        print(
            f"{str(r['epsilon']):>8}  "
            f"{r['sigma']:>8.3f}  "
            f"{r['dice']:>7.4f}  "
            f"{r['dice_drop']:>7.4f}  "
            f"{str(r['mia_accuracy'] or '—'):>8}  "
            f"{str(r['mia_drop_from_base'] or '—'):>7}"
        )
    print(f"{'='*72}")
    print(f"Saved to: {report_path}")
    return results


# ---------------------------------------------------------------------------
# Helper: score the actual cloud continuation from (noisy) features
# ---------------------------------------------------------------------------
def _measure_dice_from_features(
    inference_engine,
    feature_list: list,
    target_masks: list,
    split_layer: str,
    checkpoint_sha256: str,
    image_mean: list[float],
    image_std: list[float],
) -> float:
    """Run the same remaining-backbone/FPN/RPN/RoI continuation as the live API."""
    from backend.app.ml.inference import _rle_decode
    from training.metrics import dice_score

    dices = []
    for feat_dict, gt_mask in zip(feature_list[:20], target_masks[:20]):
        detections = inference_engine.predict_from_edge_features(
            feat_dict,
            split_layer,
            "cn5-dp-evaluation",
            expected_checkpoint_sha256=checkpoint_sha256,
            edge_image_mean=image_mean,
            edge_image_std=image_std,
        )
        if detections:
            best = max(detections, key=lambda detection: detection.score)
            pred_bin = (
                _rle_decode(best.mask_rle, tuple(gt_mask.shape)).astype(np.uint8)
                if best.mask_rle
                else np.zeros_like(gt_mask, dtype=np.uint8)
            )
        else:
            pred_bin = np.zeros_like(gt_mask, dtype=np.uint8)
        dices.append(dice_score(pred_bin, gt_mask))

    return float(np.mean(dices)) if dices else 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CN5: Differential Privacy Experiment")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data",       required=True)
    parser.add_argument("--split",      default="layer2",
                        choices=["layer1", "layer2", "layer3"])
    parser.add_argument("--epsilons",   type=float, nargs="+",
                        default=[0.05, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--delta",      type=float, default=1e-5)
    parser.add_argument("--n-samples",  type=int,   default=100)
    parser.add_argument("--out",        default="./experiments/cn5_dp")
    args = parser.parse_args()

    run_dp_experiment(
        checkpoint_path=args.checkpoint,
        data_root=args.data,
        split_layer=args.split,
        epsilons=args.epsilons,
        delta=args.delta,
        n_samples=args.n_samples,
        output_dir=args.out,
    )


if __name__ == "__main__":
    main()
