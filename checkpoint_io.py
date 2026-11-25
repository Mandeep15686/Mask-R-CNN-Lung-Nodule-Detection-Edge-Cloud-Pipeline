"""Restricted PyTorch checkpoint loading shared by app, training, and edge code."""
from __future__ import annotations

from os import PathLike
from typing import Any

import torch
from torch.torch_version import TorchVersion

# ``safe_globals`` as a context manager is only available in newer PyTorch
# releases. ``add_safe_globals`` is supported by the project's pinned 2.4
# release and registers this one harmless metadata class process-wide.
torch.serialization.add_safe_globals([TorchVersion])


def load_checkpoint_weights_only(
    checkpoint_path: str | PathLike[str],
    *,
    map_location: Any = "cpu",
) -> Any:
    """Load tensor checkpoint data while allowing PyTorch's version metadata.

    Some checkpoints serialize ``torch.__version__`` as ``TorchVersion``.
    Allowlist only that simple PyTorch metadata type and keep the restricted
    ``weights_only`` unpickler enabled. Arbitrary checkpoint globals remain
    rejected.
    """
    return torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=True,
    )
