"""Run the upload-equivalent inference path on cached positive/negative patches."""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.ml.inference import InferenceEngine


def _patch_png(path: Path) -> bytes:
    with np.load(path) as patch:
        image = patch["image"]
    if image.shape != (3, 64, 64):
        raise ValueError(f"Expected cached patch shape (3,64,64), got {image.shape} at {path}")
    rgb = np.rint(np.clip(image.transpose(1, 2, 0), 0.0, 1.0) * 255.0).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _select_patch(manifest: list[dict], cache_dir: Path, label: int, subset: int) -> tuple[dict, Path]:
    entry = next(
        (item for item in manifest if int(item["label"]) == label and int(item["subset"]) == subset),
        None,
    )
    if entry is None:
        raise FileNotFoundError(f"No label={label} patch is available for subset{subset}")
    path = cache_dir / entry["file"]
    if not path.is_file():
        raise FileNotFoundError(f"Cache manifest patch is missing: {path}")
    return entry, path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="./experiments/best_model.pth")
    parser.add_argument("--cache-dir", default="./data/patches_cache_v3_smoke")
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Cache manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError("Cache manifest must contain a JSON list")

    positive, positive_path = _select_patch(manifest, cache_dir, 1, args.subset)
    negative, negative_path = _select_patch(manifest, cache_dir, 0, args.subset)
    engine = InferenceEngine(default_checkpoint_path=checkpoint)
    if not engine.status["model_loaded"]:
        raise RuntimeError(engine.status["model_loading_error"] or "Trained checkpoint failed to load")

    for label_name, entry, path in (
        ("positive", positive, positive_path),
        ("negative", negative, negative_path),
    ):
        detections = engine.predict_from_bytes(_patch_png(path))
        if args.threshold is not None:
            detections = [detection for detection in detections if detection.score >= args.threshold]
        print(
            f"{label_name}: subset={entry['subset']} label={entry['label']} "
            f"detections={len(detections)} "
            f"scores={[round(item.score, 4) for item in detections]} "
            f"boxes={[item.box for item in detections]}"
        )
    print(f"checkpoint={checkpoint}")
    metadata = engine.checkpoint_info()
    print(f"config_id={metadata['config_id']} epoch={metadata['epoch']}")
    print("Note: these are sampled-patch checks, not whole-volume or clinical validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
