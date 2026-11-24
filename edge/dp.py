"""Patient-level Gaussian mechanism for numeric edge feature releases.

The guarantee is conditional on a fixed model and identical public metadata.
It covers one patient's transmitted feature vector, not model training,
identifiers, timing/traffic metadata, or other side channels.
"""
from __future__ import annotations

import math
import json
import os
import secrets
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping

import torch
import numpy as np


def _secure_gaussian_like(reference: torch.Tensor) -> torch.Tensor:
    """Draw normal values from OS-backed CSPRNG bytes via Box-Muller."""
    count = reference.numel()
    pair_count = (count + 1) // 2
    words = np.frombuffer(secrets.token_bytes(pair_count * 16), dtype="<u8").reshape(-1, 2)
    mask = (1 << 52) - 1
    denominator = float((1 << 52) + 2)
    uniforms = (np.bitwise_and(words, mask).astype(np.float64) + 1.0) / denominator
    radius = np.sqrt(-2.0 * np.log(uniforms[:, 0]))
    angle = 2.0 * np.pi * uniforms[:, 1]
    normal = np.empty(pair_count * 2, dtype=np.float32)
    normal[0::2] = (radius * np.cos(angle)).astype(np.float32)
    normal[1::2] = (radius * np.sin(angle)).astype(np.float32)
    values = torch.from_numpy(normal[:count].reshape(tuple(reference.shape)))
    return values.to(device=reference.device, dtype=reference.dtype)


class BasicCompositionAccountant:
    """Conservative sequential accountant with an optional durable ledger.

    A durable ledger is required for a live edge deployment: every release is
    reserved and written before noise is returned, so process restarts do not
    reset the privacy budget. The ledger is protected by an OS file lock for
    multiple local worker processes. Keep it on persistent, access-controlled
    storage and do not delete it while reusing the same privacy budget.
    """

    def __init__(self, epsilon_budget: float, delta_budget: float, ledger_path: str | Path | None = None):
        if not math.isfinite(epsilon_budget) or epsilon_budget <= 0:
            raise ValueError("epsilon_budget must be finite and positive")
        if not math.isfinite(delta_budget) or not 0 < delta_budget < 1:
            raise ValueError("delta_budget must be between 0 and 1")
        self.epsilon_budget = float(epsilon_budget)
        self.delta_budget = float(delta_budget)
        self.epsilon_spent = 0.0
        self.delta_spent = 0.0
        self.releases = 0
        self._lock = threading.Lock()
        self.ledger_path = Path(ledger_path).expanduser().resolve() if ledger_path else None
        if self.ledger_path is not None:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self._file_lock():
                state = self._read_ledger()
                if state is None:
                    self._write_ledger()
                else:
                    self._load_state(state)

    @contextmanager
    def _file_lock(self):
        """Lock a separate stable lock file while atomically replacing ledger."""
        assert self.ledger_path is not None
        lock_path = self.ledger_path.with_suffix(self.ledger_path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock_file:
            if os.name == "nt":
                import msvcrt

                lock_file.seek(0)
                if not lock_file.read(1):
                    lock_file.seek(0)
                    lock_file.write(b"0")
                    lock_file.flush()
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _read_ledger(self) -> dict | None:
        assert self.ledger_path is not None
        if not self.ledger_path.exists():
            return None
        try:
            state = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"DP accountant ledger is unreadable; refusing releases: {exc}") from exc
        if not isinstance(state, dict) or state.get("version") != 1:
            raise RuntimeError("DP accountant ledger has an unsupported format; refusing releases")
        return state

    def _load_state(self, state: dict) -> None:
        try:
            budgets_match = (
                math.isclose(float(state["epsilon_budget"]), self.epsilon_budget, rel_tol=0, abs_tol=1e-15)
                and math.isclose(float(state["delta_budget"]), self.delta_budget, rel_tol=0, abs_tol=1e-18)
            )
            epsilon_spent = float(state["epsilon_spent"])
            delta_spent = float(state["delta_spent"])
            releases = int(state["releases"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("DP accountant ledger is incomplete; refusing releases") from exc
        if not budgets_match:
            raise RuntimeError("Configured DP budgets differ from the persisted ledger; refusing releases")
        if (
            not math.isfinite(epsilon_spent)
            or not math.isfinite(delta_spent)
            or epsilon_spent < 0
            or delta_spent < 0
            or releases < 0
            or epsilon_spent > self.epsilon_budget
            or delta_spent > self.delta_budget
        ):
            raise RuntimeError("DP accountant ledger contains invalid spend; refusing releases")
        self.epsilon_spent = epsilon_spent
        self.delta_spent = delta_spent
        self.releases = releases

    def _write_ledger(self) -> None:
        assert self.ledger_path is not None
        payload = {
            "version": 1,
            "epsilon_budget": self.epsilon_budget,
            "delta_budget": self.delta_budget,
            "epsilon_spent": self.epsilon_spent,
            "delta_spent": self.delta_spent,
            "releases": self.releases,
        }
        fd, temporary_name = tempfile.mkstemp(
            prefix=self.ledger_path.name + ".", suffix=".tmp", dir=self.ledger_path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.ledger_path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def reserve(self, epsilon: float, delta: float) -> tuple[float, float, int]:
        """Atomically account a release before producing its output."""
        with self._lock:
            if self.ledger_path is not None:
                with self._file_lock():
                    state = self._read_ledger()
                    if state is None:
                        raise RuntimeError("DP accountant ledger disappeared; refusing releases")
                    self._load_state(state)
                    result = self._reserve_in_memory(epsilon, delta)
                    self._write_ledger()
                    return result
            return self._reserve_in_memory(epsilon, delta)

    def _reserve_in_memory(self, epsilon: float, delta: float) -> tuple[float, float, int]:
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("release epsilon must be finite and positive")
        if not math.isfinite(delta) or not 0 < delta < 1:
            raise ValueError("release delta must be between 0 and 1")
        next_epsilon = self.epsilon_spent + epsilon
        next_delta = self.delta_spent + delta
        if next_epsilon > self.epsilon_budget or next_delta > self.delta_budget:
            raise RuntimeError(
                "DP release refused: sequential composition would exceed budget "
                f"(ε={next_epsilon:.6g}/{self.epsilon_budget:.6g}, "
                f"δ={next_delta:.6g}/{self.delta_budget:.6g})"
            )
        self.epsilon_spent = next_epsilon
        self.delta_spent = next_delta
        self.releases += 1
        return self.epsilon_spent, self.delta_spent, self.releases


def gaussian_feature_release(
    features: Mapping[str, torch.Tensor],
    *,
    epsilon: float,
    delta: float,
    clip_norm: float,
    accountant: BasicCompositionAccountant,
) -> tuple[dict[str, torch.Tensor], dict[str, float | int | str]]:
    """Clip the whole one-patient feature vector and apply calibrated Gaussian noise.

    Uses replacement adjacency, so sensitivity is bounded by ``2 * clip_norm``.
    To keep the classical Gaussian calibration valid, this implementation
    restricts per-release epsilon to (0, 1]. Gaussian samples are drawn from
    the operating system CSPRNG rather than a potentially seeded ML PRNG.
    """
    if not features:
        raise ValueError("At least one feature tensor is required")
    if not math.isfinite(epsilon) or not 0 < epsilon <= 1:
        raise ValueError("Per-release epsilon must be in (0, 1]")
    if not math.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("Per-release delta must be in (0, 1)")
    if not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("clip_norm must be finite and positive")

    values: dict[str, torch.Tensor] = {}
    for name, tensor in features.items():
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
            raise ValueError(f"Feature {name!r} must be a batched 4D tensor")
        if tensor.shape[0] != 1:
            raise ValueError("DP feature release requires exactly one patient per request")
        value = tensor.dequantize().float() if tensor.is_quantized else tensor.float()
        if not torch.isfinite(value).all():
            raise ValueError(f"Feature {name!r} contains non-finite values")
        values[name] = value

    device = next(iter(values.values())).device
    if any(value.device != device for value in values.values()):
        raise ValueError("All feature tensors must be on the same device")
    norm_sq = sum(value.double().square().sum() for value in values.values())
    joint_norm = float(norm_sq.sqrt().item())
    sensitivity = 2.0 * clip_norm
    sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon

    # Reserve first: any failure after this point only overcounts privacy spend.
    epsilon_spent, delta_spent, releases = accountant.reserve(epsilon, delta)
    scale = min(1.0, clip_norm / max(joint_norm, 1e-12))
    released = {
        name: value.mul(scale).add(_secure_gaussian_like(value), alpha=sigma)
        for name, value in values.items()
    }
    metadata: dict[str, float | int | str] = {
        "mechanism": "joint_l2_clip_gaussian_replacement_adjacency_basic_composition",
        "scope": "numeric_feature_values_only_fixed_checkpoint_public_metadata",
        "adjacency": "replacement",
        "randomness": "os_csprng_box_muller",
        "unit": "one_patient_feature_bundle_per_request",
        "epsilon_release": float(epsilon),
        "delta_release": float(delta),
        "epsilon_spent": epsilon_spent,
        "delta_spent": delta_spent,
        "releases": releases,
        "clip_norm": float(clip_norm),
        "sensitivity_bound": sensitivity,
        "noise_sigma": sigma,
    }
    return released, metadata
