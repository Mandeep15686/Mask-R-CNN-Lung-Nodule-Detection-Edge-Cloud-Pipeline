import pytest
import torch
from torch import nn

from scripts.evaluate_model import load_checkpoint_state_dict


def test_checkpoint_loads_strictly_when_keys_match():
    source = nn.Linear(4, 2)
    target = nn.Linear(4, 2)
    load_checkpoint_state_dict(target, {"model_state": source.state_dict()})
    for left, right in zip(source.parameters(), target.parameters()):
        assert torch.equal(left, right)


def test_checkpoint_reports_missing_and_unexpected_keys():
    target = nn.Linear(4, 2)
    with pytest.raises(RuntimeError, match="Strict load failed"):
        load_checkpoint_state_dict(
            target,
            {"model_state": {"weight": torch.zeros(2, 4), "extra": torch.zeros(1)}},
        )


@pytest.mark.slow
def test_mask_rcnn_positive_and_negative_forward_backward():
    from training.model import LungNoduleMaskRCNN

    model = LungNoduleMaskRCNN(
        pretrained_backbone=False,
        trainable_backbone_layers=3,
        min_size=64,
        max_size=64,
        box_detections_per_img=8,
        mask_roi_pool_output_size=7,
        hidden_dim=32,
        mask_loss_cfg={"alpha": 1.0, "beta": 0.0, "gamma": 0.0},
    )
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    images = [torch.rand(3, 64, 64, device=device) for _ in range(2)]
    positive_mask = torch.zeros((1, 64, 64), dtype=torch.uint8, device=device)
    positive_mask[:, 24:40, 24:40] = 1
    targets = [
        {
            "boxes": torch.tensor([[24, 24, 40, 40]], dtype=torch.float32, device=device),
            "labels": torch.tensor([1], dtype=torch.int64, device=device),
            "masks": positive_mask,
        },
        {
            "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
            "labels": torch.zeros((0,), dtype=torch.int64, device=device),
            "masks": torch.zeros((0, 64, 64), dtype=torch.uint8, device=device),
        },
    ]

    losses = model(images, targets)
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    # A saved trained state must reload strictly into the same architecture.
    restored = LungNoduleMaskRCNN(
        pretrained_backbone=False,
        trainable_backbone_layers=3,
        min_size=64,
        max_size=64,
        box_detections_per_img=8,
        mask_roi_pool_output_size=7,
        hidden_dim=32,
        mask_loss_cfg={"alpha": 1.0, "beta": 0.0, "gamma": 0.0},
    ).to(device)
    load_checkpoint_state_dict(restored, {"model_state": model.state_dict()})
