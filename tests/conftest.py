"""Shared pytest fixtures."""
import os
import sys
import pytest
import numpy as np
import torch

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force CPU for all tests unless GPU explicitly requested
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture
def binary_mask_pair():
    """Return (pred_mask, gt_mask) as numpy arrays with partial overlap."""
    rng = np.random.default_rng(seed=0)
    gt   = np.zeros((64, 64), dtype=np.uint8)
    pred = np.zeros((64, 64), dtype=np.uint8)
    gt[20:40, 20:40]   = 1
    pred[22:42, 22:42] = 1   # ~75% overlap with gt
    return pred, gt


@pytest.fixture
def ct_patch_tensor():
    """Synthetic 3-channel (2.5D) CT patch tensor."""
    torch.manual_seed(0)
    return torch.rand(1, 3, 64, 64)
