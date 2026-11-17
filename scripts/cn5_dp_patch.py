"""
Patch for scripts/cn5_differential_privacy.py

FIXES APPLIED
=============
FIX-16  Noised features not passed to model inference:
  BEFORE: _measure_dice_from_features() received feat_device (the noised features)
          but then ran `out = model(list(imgs))` using the *original* raw images.
          Changing epsilon had zero effect on the Dice score because the noise
          was computed and then discarded.
  AFTER:  The function now uses the noised features to build a crude image
          approximation and runs the full model on it. This is only a proxy;
          it is not cloud split inference or a valid segmentation utility score.

NOTE: A truly valid (ε,δ)-DP guarantee also requires:
  1. Bounded L2 sensitivity (clip features before adding noise).
  2. A privacy accountant for repeated queries.
  3. True train/test membership separation for the MIA evaluation.
  These are research-level requirements not addressed by this patch alone.
  Do not claim formal DP guarantees without those steps.

Usage: replace _measure_dice_from_features in cn5_differential_privacy.py
       with the version below.
"""

# ---------------------------------------------------------------------------
# Drop-in replacement for _measure_dice_from_features
# ---------------------------------------------------------------------------

REPLACEMENT_FUNCTION = '''
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
'''

# =============================================================================
# How to apply this patch
# =============================================================================
# In scripts/cn5_differential_privacy.py, find and replace the full function:
#
#   def _measure_dice_from_features(
#       model, feature_list: list, loader, device, noise_sigma: float = 0.0
#   ) -> float:
#       ...
#
# with the string in REPLACEMENT_FUNCTION above.
#
# Alternatively run from the repo root:
#   python scripts/cn5_dp_patch.py --apply
#

if __name__ == "__main__":
    import argparse, pathlib, textwrap

    parser = argparse.ArgumentParser(description="Apply FIX-16 to cn5_differential_privacy.py")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--target",
        default="scripts/cn5_differential_privacy.py",
        help="Path to the file to patch",
    )
    args = parser.parse_args()

    if not args.apply:
        print("Dry run — pass --apply to actually patch the file.")
        print("Replacement function:\n")
        print(REPLACEMENT_FUNCTION)
    else:
        target = pathlib.Path(args.target)
        if not target.exists():
            raise FileNotFoundError(f"Target not found: {target}")

        src = target.read_text(encoding="utf-8")
        MARKER_START = "def _measure_dice_from_features("
        MARKER_END_SIGNATURE = "    return float(np.mean(dices)) if dices else 0.0\n"

        start = src.find(MARKER_START)
        end   = src.find(MARKER_END_SIGNATURE, start)
        if start == -1 or end == -1:
            raise ValueError("Could not locate _measure_dice_from_features in the file.")

        current_function = src[start:end]
        if "FIX-16: The noised features are now actually used for inference." in current_function:
            print(f"FIX-16 already present in {target}; no changes written.")
            raise SystemExit(0)

        end   += len(MARKER_END_SIGNATURE)
        new_src = src[:start] + textwrap.dedent(REPLACEMENT_FUNCTION).lstrip("\n") + src[end:]
        target.write_text(new_src, encoding="utf-8")
        print(f"Patched {target}")
