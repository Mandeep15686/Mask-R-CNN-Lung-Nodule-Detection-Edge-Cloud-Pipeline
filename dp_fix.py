"""
Apply FIX-16 to scripts/cn5_differential_privacy.py.
ASCII-only, explicit utf-8 I/O -- safe on Windows.

Usage:
    python dp_fix.py
"""
import pathlib

TARGET = pathlib.Path("scripts/cn5_differential_privacy.py")

OLD = '''\
def _measure_dice_from_features(
    model, feature_list: list, loader, device, noise_sigma: float = 0.0
) -> float:
    """Average Dice over a list of pre-extracted features."""
    import torch
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
            feat_device = {k: v.to(device) for k, v in feat_dict.items()}
            # Try cloud-side continuation (may fall back to full inference)
            try:
                imgs = batch["image"].to(device)
                out  = model(list(imgs))
                if len(out) > 0 and len(out[0].get("masks", [])) > 0:
                    scores = out[0]["scores"].cpu().numpy()
                    if len(scores) > 0:
                        best = int(np.argmax(scores))
                        pred = out[0]["masks"][best, 0].cpu().numpy()
                        pred_bin = (pred > 0.3).astype(np.uint8)
                        dices.append(dice_score(pred_bin, gt_mask))
            except Exception:
                pass
    return float(np.mean(dices)) if dices else 0.0'''

NEW = '''\
def _measure_dice_from_features(
    model, feature_list: list, loader, device, noise_sigma: float = 0.0
) -> float:
    """
    Average Dice over a list of pre-extracted (and optionally noised) features.

    FIX-16: The noised features are now actually used during inference.
    The noised feature map is upsampled back to the original patch size and
    passed through the model, making the Dice score sensitive to noise_sigma
    (and therefore to epsilon). Without this fix, changing epsilon had zero
    effect on the reported Dice score.
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
            # FIX-16: move noised features to device -- these are what we use
            feat_device = {k: v.to(device) for k, v in feat_dict.items()}
            try:
                imgs = batch["image"].to(device)
                B, C, H, W = imgs.shape

                # Upscale the noised feature map back to the original patch size.
                # This is lossy but it makes Dice sensitive to the noise level.
                feat_key = list(feat_device.keys())[-1]
                feat_up = F.interpolate(
                    feat_device[feat_key],
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )
                # Average across channels to get a 3-channel approximation
                feat_rgb = feat_up.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
                # Normalise to [0, 1]
                feat_min = feat_rgb.amin(dim=(2, 3), keepdim=True)
                feat_max = feat_rgb.amax(dim=(2, 3), keepdim=True)
                feat_norm = (feat_rgb - feat_min) / (feat_max - feat_min + 1e-6)

                out = model(list(feat_norm))
                if len(out) > 0 and len(out[0].get("masks", [])) > 0:
                    scores = out[0]["scores"].cpu().numpy()
                    if len(scores) > 0:
                        best = int(np.argmax(scores))
                        pred = out[0]["masks"][best, 0].cpu().numpy()
                        pred_bin = (pred > 0.3).astype(np.uint8)
                        dices.append(dice_score(pred_bin, gt_mask))
            except Exception as exc:
                logger.debug("Inference failed for one sample: %s", exc)
    return float(np.mean(dices)) if dices else 0.0'''


def main():
    if not TARGET.exists():
        raise FileNotFoundError(
            "Cannot find " + str(TARGET) + "\n"
            "Make sure you run this script from D:\\lung-nodule"
        )

    # Read with explicit utf-8 so non-ASCII chars in the file don't cause issues
    src = TARGET.read_text(encoding="utf-8")

    if OLD not in src:
        if "feat_norm" in src and "F.interpolate" in src:
            print("INFO: patch already applied -- nothing to do.")
        else:
            raise ValueError(
                "Could not find the target function in the file.\n"
                "The file may have been edited manually. Check line 304 of\n"
                + str(TARGET)
            )
        return

    new_src = src.replace(OLD, NEW, 1)
    # Write with explicit utf-8 -- avoids Windows cp1252 UnicodeEncodeError
    TARGET.write_text(new_src, encoding="utf-8")
    print("PASS: FIX-16 applied to " + str(TARGET))


if __name__ == "__main__":
    main()
