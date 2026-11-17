#!/usr/bin/env python3
"""
CN5 prototype: Gaussian perturbation of edge features.

This script does not establish a formal differential privacy guarantee. Its
current Dice routine reconstructs a crude image from noised features instead
of continuing the model from those features, and its MIA member/non-member
samples are not separated by scan UID. Treat its outputs as exploratory only.

Applies Gaussian noise (ε-DP) to intermediate feature maps at the split point
and measures the privacy–utility tradeoff:
  - Privacy:  Membership Inference Attack (MIA) accuracy (lower = more private)
  - Utility:  Dice score degradation vs baseline (lower = better)

Usage:
  python scripts/cn5_differential_privacy.py \
    --checkpoint ./experiments/G-a03-b05-g05/best_model.pth \
    --data ./data/luna16 \
    --split layer2 \
    --epsilons 0.1 0.5 1.0 5.0 10.0 100.0 \
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
) -> dict:
    """
    Add Gaussian noise to feature tensors using the conventional sigma formula.

    This helper alone does not guarantee (ε, δ)-DP: the feature sensitivity is
    not bounded here and repeated-query accounting is not implemented.

    Noise std σ = (sensitivity / epsilon) * sqrt(2 * ln(1.25 / delta))

    Args:
        features:    dict of layer_name → tensor (from EdgeFeatureExtractor)
        epsilon:     DP privacy budget (lower = more private)
        delta:       DP failure probability (default 1e-5)
        sensitivity: L2 sensitivity of the query (feature extraction)

    Returns:
        Noised feature dict
    """
    import torch
    sigma = (sensitivity / epsilon) * np.sqrt(2.0 * np.log(1.25 / delta))
    noised = {}
    for k, v in features.items():
        noise = torch.randn_like(v) * sigma
        noised[k] = v + noise
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
    fold:            int = 9,
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
        epsilons = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, float("inf")]

    import torch
    from training.model import LungNoduleMaskRCNN
    from training.dataset import LUNA16Dataset
    from edge.feature_extractor import EdgeFeatureExtractor
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | Split: {split_layer}")

    # Load model
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = dict(ckpt.get("config", {}))
    cfg["pretrained_backbone"] = False
    model = LungNoduleMaskRCNN.from_config(cfg)
    state = dict(ckpt.get("model_state", ckpt.get("state_dict", ckpt)))
    old_bce_key = "composite_mask_loss.bce.loss_fn.pos_weight"
    new_bce_key = "composite_mask_loss.bce.pos_weight"
    if old_bce_key in state and new_bce_key not in state:
        state[new_bce_key] = state.pop(old_bce_key)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    # Feature extractor (edge)
    extractor = EdgeFeatureExtractor(
        split_layer=split_layer,
        device=device,
        checkpoint_path=checkpoint_path,
    )

    # Dataset — use test fold
    ds = LUNA16Dataset(data_root, subsets=[fold], patch_size=64, augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    # Collect training (member) and test (non-member) features
    member_feats, nonmember_feats = [], []

    logger.info(f"Extracting features from {min(n_samples * 2, len(ds))} samples...")
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= n_samples * 2:
                break
            imgs = batch["image"].to(device)
            feat = extractor.extract(imgs)
            if idx < n_samples:
                member_feats.append({k: v.cpu() for k, v in feat.items()})
            else:
                nonmember_feats.append({k: v.cpu() for k, v in feat.items()})

    # Baseline Dice (no DP noise, ε = ∞)
    baseline_dice = _measure_dice_from_features(
        model, member_feats[:20], loader, device, noise_sigma=0.0
    )
    logger.info(f"Baseline Dice (no DP): {baseline_dice:.4f}")

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
        sigma = 0.0 if eps == float("inf") else (1.0 / eps) * np.sqrt(2 * np.log(1.25 / delta))

        # Add DP noise to features
        noised_member    = [add_gaussian_dp_noise(f, eps, delta) if eps < float("inf") else f
                            for f in member_feats[:50]]
        noised_nonmember = [add_gaussian_dp_noise(f, eps, delta) if eps < float("inf") else f
                            for f in nonmember_feats[:50]]

        # Measure Dice after noise
        dice_noised = _measure_dice_from_features(
            model, noised_member[:20], loader, device, noise_sigma=sigma
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
        }
        results.append(entry)
        logger.info(
            f"  sigma={sigma:.3f}  Dice={dice_noised:.4f} (drop={dice_drop:.4f})  "
            f"MIA={mia_noised if mia_noised is not None else 'unavailable'}"
        )

    # Save results
    full_report = {
        "checkpoint":        checkpoint_path,
        "split_layer":       split_layer,
        "baseline_dice":     round(baseline_dice, 4),
        "baseline_mia_acc":  round(baseline_mia, 4) if baseline_mia else None,
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
# Helper: measure an approximate Dice proxy from (noisy) features
# ---------------------------------------------------------------------------
def _measure_dice_from_features(
    model, feature_list: list, loader, device, noise_sigma: float = 0.0
) -> float:
    """
    Average Dice over a list of pre-extracted (and optionally noised) features.

    FIX-16: The noised features are now actually used for inference.
    The noised features are converted to a crude image approximation and fed
    through the full model. This is a prototype proxy, not split inference.
    """
    import torch
    import torch.nn.functional as F
    from training.metrics import dice_score

    dices = []
    loader_iter = iter(loader)

    with torch.no_grad():
        for feat_dict in feature_list[:20]:
            try:
                batch = next(loader_iter)
            except StopIteration:
                break

            gt_mask = batch["mask"][0].numpy()
            # FIX-16: move the noised features to the inference device
            feat_device = {k: v.to(device) for k, v in feat_dict.items()}

            try:
                imgs = batch["image"].to(device)
                B, C, H, W = imgs.shape

                # --- Crude noised-feature reconstruction proxy ----------------
                # Reconstruct an approximation of the input by bilinear-upsampling
                # the noised feature map back to the original patch resolution.
                # It is not a valid substitute for model continuation from the
                # transmitted backbone features.
                feat_key  = list(feat_device.keys())[-1]
                feat_up   = F.interpolate(
                    feat_device[feat_key],
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )
                # Project channels back to 3 (RGB) with a 1×1 average
                feat_rgb = feat_up.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
                # Normalise to [0, 1]
                feat_min = feat_rgb.amin(dim=(2, 3), keepdim=True)
                feat_max = feat_rgb.amax(dim=(2, 3), keepdim=True)
                feat_norm = (feat_rgb - feat_min) / (feat_max - feat_min + 1e-6)

                out = model(list(feat_norm))
                if len(out) > 0 and len(out[0].get("masks", [])) > 0:
                    scores = out[0]["scores"].cpu().numpy()
                    if len(scores) > 0:
                        best     = int(np.argmax(scores))
                        pred     = out[0]["masks"][best, 0].cpu().numpy()
                        pred_bin = (pred > 0.3).astype(np.uint8)
                        dices.append(dice_score(pred_bin, gt_mask))

            except Exception as exc:
                logger.debug("Inference failed for one sample: %s", exc)

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
                        default=[0.1, 0.5, 1.0, 5.0, 10.0, 50.0])
    parser.add_argument("--delta",      type=float, default=1e-5)
    parser.add_argument("--n-samples",  type=int,   default=100)
    parser.add_argument("--fold",       type=int,   default=9)
    parser.add_argument("--out",        default="./experiments/cn5_dp")
    args = parser.parse_args()

    run_dp_experiment(
        checkpoint_path=args.checkpoint,
        data_root=args.data,
        split_layer=args.split,
        epsilons=args.epsilons,
        delta=args.delta,
        n_samples=args.n_samples,
        fold=args.fold,
        output_dir=args.out,
    )


if __name__ == "__main__":
    main()
