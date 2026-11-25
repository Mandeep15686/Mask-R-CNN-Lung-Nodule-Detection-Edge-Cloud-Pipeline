import pickle

import pytest
import torch

from checkpoint_io import load_checkpoint_weights_only


class UntrustedCheckpointMetadata:
    pass


def test_safe_loader_accepts_torch_version_metadata(tmp_path):
    checkpoint_path = tmp_path / "torch-version.pth"
    tensor = torch.arange(4)
    torch.save({"model_state": {"weight": tensor}, "torch_version": torch.__version__}, checkpoint_path)

    checkpoint = load_checkpoint_weights_only(checkpoint_path, map_location="cpu")

    assert checkpoint["torch_version"] == torch.__version__
    torch.testing.assert_close(checkpoint["model_state"]["weight"], tensor)
    assert torch.serialization.get_safe_globals().count(torch.torch_version.TorchVersion) == 1


def test_safe_loader_still_rejects_other_untrusted_globals(tmp_path):
    checkpoint_path = tmp_path / "untrusted-metadata.pth"
    torch.save({"metadata": UntrustedCheckpointMetadata()}, checkpoint_path)

    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        load_checkpoint_weights_only(checkpoint_path, map_location="cpu")
