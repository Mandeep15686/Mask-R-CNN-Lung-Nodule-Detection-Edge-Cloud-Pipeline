#!/usr/bin/env python3
"""Train the learned edge feature bottleneck for a specific trained backbone.

The codec is a channel bottleneck, not an entropy coder. Run validation with
the held-out subset and compare detection quality/rate before enabling it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edge.compressor import _LearnedCodecBank  # noqa: E402
from edge.feature_extractor import EdgeFeatureExtractor  # noqa: E402
from training.dataset import LUNA16Dataset  # noqa: E402
from backend.app.ml.inference import InferenceEngine  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_codec(
    checkpoint: str,
    data_root: str,
    train_subset: int,
    out_path: str,
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-3,
    max_batches: int = 0,
    rate_weight: float = 1e-4,
) -> dict:
    if epochs < 1 or batch_size < 1 or lr <= 0 or rate_weight < 0:
        raise ValueError("epochs/batch_size/lr/rate_weight are outside valid ranges")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    engine = InferenceEngine(default_checkpoint_path=checkpoint_path, device=device)
    if engine.model is None:
        raise RuntimeError(f"Could not load trained detector: {engine.model_loading_error}")
    loaded = engine._get_checkpoint(checkpoint_path)
    transform = loaded.model.model.transform
    extractor = EdgeFeatureExtractor(
        split_layer="layer3",
        device=device,
        checkpoint_path=checkpoint_path,
        image_mean=tuple(transform.image_mean),
        image_std=tuple(transform.image_std),
    ).eval()
    dataset = LUNA16Dataset(
        data_root,
        subsets=[train_subset],
        patch_size=int(loaded.config.get("patch_size", 64)),
        augment=True,
    )
    if not dataset:
        raise ValueError(f"No training patches found in subset{train_subset}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    codec = _LearnedCodecBank().to(device).train()
    optimizer = torch.optim.Adam(codec.parameters(), lr=lr)
    history = []

    for epoch in range(1, epochs + 1):
        losses = []
        for batch_index, batch in enumerate(loader, start=1):
            if max_batches and batch_index > max_batches:
                break
            images = batch["image"].to(device)
            with torch.no_grad():
                targets = extractor.extract(images)
            optimizer.zero_grad(set_to_none=True)
            objective = torch.zeros((), device=device)
            for key, value in targets.items():
                latent = codec.encode(key, value.detach())
                restored = codec.decode(key, latent)
                objective = objective + torch.nn.functional.mse_loss(restored, value.detach())
                objective = objective + rate_weight * latent.abs().mean()
            objective.backward()
            optimizer.step()
            losses.append(float(objective.detach().item()))
        if not losses:
            raise RuntimeError("Codec training produced no batches")
        history.append({"epoch": epoch, "mean_objective": sum(losses) / len(losses), "batches": len(losses)})

    output = Path(out_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "metadata": {
            "format": "learned_feature_codec_v1",
            "backbone_checkpoint_sha256": loaded.sha256,
            "train_subset": int(train_subset),
            "train_samples": len(dataset),
            "objective": "sum_per_level_mse_plus_rate_weighted_latent_l1",
            "rate_weight": float(rate_weight),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "latent_channels": dict(_LearnedCodecBank.LATENT_CHANNELS),
            "history": history,
            "interpretation": "Training fit only; not validated rate-distortion or clinical utility evidence",
        },
        "codec_state": {key: value.detach().cpu() for key, value in codec.state_dict().items()},
    }
    torch.save(artifact, output)
    report = {**artifact["metadata"], "codec_checkpoint": str(output), "codec_sha256": sha256_file(output)}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Trained Mask R-CNN checkpoint")
    parser.add_argument("--data", required=True)
    parser.add_argument("--train-subset", required=True, type=int)
    parser.add_argument("--out", required=True, help="Output .pth codec checkpoint")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means all batches")
    parser.add_argument("--rate-weight", type=float, default=1e-4)
    args = parser.parse_args()
    report = train_codec(**vars(args))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
