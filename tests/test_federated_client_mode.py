import pytest

from training.federated import _resolve_client_subset


def test_distributed_client_can_pin_its_local_subset():
    assert _resolve_client_subset(0, [0, 1], requested_subset=1) == 1


def test_distributed_client_rejects_unavailable_local_subset():
    with pytest.raises(ValueError, match="unavailable"):
        _resolve_client_subset(0, [0, 1], requested_subset=4)


def test_simulation_subset_assignment_is_deterministic():
    available = [2, 5, 8]
    assert [_resolve_client_subset(i, available) for i in range(6)] == [2, 5, 8, 2, 5, 8]


def test_client_cannot_start_without_local_data():
    with pytest.raises(ValueError, match="No LUNA16 subsets"):
        _resolve_client_subset(0, [])
