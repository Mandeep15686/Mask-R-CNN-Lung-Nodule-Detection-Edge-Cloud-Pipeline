"""Numerically stable MGDA utilities for combining differentiable objectives."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch

MGDA_NORM_EPSILON = 1e-12


@dataclass(frozen=True)
class MGDASolution:
    """Weights and diagnostics from one MGDA solve."""

    weights: dict[str, float]
    gradient_norms: dict[str, float]
    applied_coefficients: dict[str, float]
    diagnostics: dict[str, float]


@dataclass(frozen=True)
class MGDAResult:
    """MGDA solver result, including coefficients used by the final backward."""

    lambdas: torch.Tensor
    grad_norm_sq: float
    n_iters: int
    task_names: list[str]
    gradient_norms: dict[str, float]
    applied_coefficients: dict[str, float]
    diagnostics: dict[str, float]


def project_simplex(values: torch.Tensor) -> torch.Tensor:
    """Euclidean projection of a vector onto the probability simplex."""
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("values must be a non-empty 1-D tensor")
    sorted_values, _ = torch.sort(values, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=0) - 1.0
    indices = torch.arange(1, values.numel() + 1, device=values.device, dtype=values.dtype)
    positive = sorted_values - cumulative / indices > 0
    if not bool(positive.any()):
        return torch.full_like(values, 1.0 / values.numel())
    rho = int(torch.nonzero(positive, as_tuple=False)[-1].item())
    threshold = cumulative[rho] / (rho + 1)
    return torch.clamp(values - threshold, min=0.0)


def min_norm_simplex_weights(
    gram: torch.Tensor,
    *,
    max_iter: int = 5000,
    tolerance: float = 1e-9,
) -> torch.Tensor:
    """Solve ``min_w w.T @ gram @ w`` subject to ``w`` being on the simplex.

    The projected-gradient solver uses a Lipschitz step based on the largest
    Gram eigenvalue, and stops using the Frank-Wolfe duality gap. Scaling the
    matrix by a positive constant improves conditioning without changing its
    minimizer.
    """
    matrix = torch.as_tensor(gram, dtype=torch.float64, device="cpu")
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("gram must be a non-empty square matrix")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("gram matrix must contain only finite values")
    if max_iter < 1 or tolerance <= 0:
        raise ValueError("max_iter and tolerance must be positive")

    matrix = (matrix + matrix.T) * 0.5
    scale = matrix.abs().max()
    count = matrix.shape[0]
    if count == 1:
        return torch.ones(1, dtype=torch.float64)
    if scale <= torch.finfo(matrix.dtype).tiny:
        return torch.full((count,), 1.0 / count, dtype=matrix.dtype)
    matrix = matrix / scale

    largest_eigenvalue = torch.linalg.eigvalsh(matrix).max().clamp_min(0.0)
    if largest_eigenvalue <= torch.finfo(matrix.dtype).eps:
        return torch.full((count,), 1.0 / count, dtype=matrix.dtype)

    step_size = 1.0 / (2.0 * largest_eigenvalue)
    weights = torch.full((count,), 1.0 / count, dtype=matrix.dtype)
    for _ in range(max_iter):
        gradient_half = matrix @ weights
        updated = project_simplex(weights - step_size * (2.0 * gradient_half))
        # For f(w)=w'Gw, this gap is max_{s in simplex}<w-s,Gw>.
        objective = torch.dot(weights, gradient_half)
        gap = objective - gradient_half.min()
        weights = updated
        if gap <= tolerance:
            break

    weights = weights.clamp_min(0.0)
    return weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)


def solve_mgda(
    objectives: Mapping[str, torch.Tensor],
    shared_parameters: Sequence[torch.nn.Parameter],
    *,
    gradient_normalization: str = "raw",
    zero_gradient_tolerance: float = 1e-12,
) -> MGDASolution:
    """Measure task gradients and solve the MGDA simplex QP.

    ``raw`` minimizes the norm of the ordinary weighted gradient. ``unit``
    first normalizes each task gradient, solves the same simplex problem, then
    returns coefficients ``lambda_i / ||g_i||`` so applying the returned
    coefficients to the original losses produces that normalized-gradient
    combination on ``shared_parameters``.

    Objectives with a zero local gradient remain part of the input and their
    losses are still computed and logged, but they are excluded from this
    step's simplex QP because they have no local descent direction to combine.
    Their reported lambda/coefficient is zero; no arbitrary floor is applied.
    """
    if gradient_normalization not in {"raw", "unit"}:
        raise ValueError("gradient_normalization must be 'raw' or 'unit'")
    names = list(objectives)
    parameters = [parameter for parameter in shared_parameters if parameter.requires_grad]
    if not names:
        raise ValueError("at least one objective is required")
    if not parameters:
        raise ValueError("MGDA requires at least one trainable shared parameter")
    if zero_gradient_tolerance < 0:
        raise ValueError("zero_gradient_tolerance must be non-negative")

    flat_gradients: list[torch.Tensor] = []
    norms: dict[str, float] = {}
    for name in names:
        loss = objectives[name]
        if loss.numel() != 1 or not bool(torch.isfinite(loss.detach()).all()):
            raise ValueError(f"MGDA objective {name!r} must be a finite scalar")
        if not loss.requires_grad:
            raise ValueError(f"MGDA objective {name!r} is not differentiable")
        gradients = torch.autograd.grad(
            loss, parameters, retain_graph=True, allow_unused=True
        )
        pieces = [
            (gradient.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
             if gradient is not None
             else torch.zeros(parameter.numel(), dtype=torch.float64))
            for parameter, gradient in zip(parameters, gradients)
        ]
        flat = torch.cat(pieces)
        if not bool(torch.isfinite(flat).all()):
            raise FloatingPointError(f"MGDA gradient for objective {name!r} is non-finite")
        flat_gradients.append(flat)
        norms[name] = float(torch.linalg.vector_norm(flat).item())

    diagnostics: dict[str, float] = {
        "mgda_shared_parameter_count": float(len(parameters)),
        "mgda_shared_parameter_numel": float(sum(p.numel() for p in parameters)),
        "mgda_zero_gradient_threshold": max(
            zero_gradient_tolerance, MGDA_NORM_EPSILON
        ),
    }
    for name, norm in norms.items():
        diagnostics[f"mgda_normalized_grad_norm_{name}"] = (
            0.0 if norm <= max(zero_gradient_tolerance, MGDA_NORM_EPSILON) else 1.0
        )

    active_threshold = max(zero_gradient_tolerance, MGDA_NORM_EPSILON)
    active_indices = [
        i for i, name in enumerate(names) if norms[name] > active_threshold
    ]
    weights = {name: 0.0 for name in names}
    coefficients = {name: 0.0 for name in names}
    if active_indices:
        active_grads = torch.stack([flat_gradients[i] for i in active_indices])
        active_norms = torch.tensor([norms[names[i]] for i in active_indices], dtype=torch.float64)
        if gradient_normalization == "unit":
            safe_norms = active_norms.clamp_min(MGDA_NORM_EPSILON)
            qp_grads = active_grads / safe_norms[:, None]
        else:
            qp_grads = active_grads
        gram = qp_grads @ qp_grads.T
        active_weights = min_norm_simplex_weights(gram)
        for local_i, original_i in enumerate(active_indices):
            name = names[original_i]
            lam = float(active_weights[local_i].item())
            weights[name] = lam
            coefficients[name] = (
                lam / max(norms[name], MGDA_NORM_EPSILON)
                if gradient_normalization == "unit"
                else lam
            )

        norms_active = active_grads.norm(dim=1).clamp_min(MGDA_NORM_EPSILON)
        cosine = (active_grads @ active_grads.T) / (
            norms_active[:, None] * norms_active[None, :]
        )
        for i, original_i in enumerate(active_indices):
            for j in range(i, len(active_indices)):
                original_j = active_indices[j]
                key = f"{names[original_i]}_{names[original_j]}"
                diagnostics[f"mgda_cosine_{key}"] = float(cosine[i, j].item())
                diagnostics[f"mgda_gram_{key}"] = float(gram[i, j].item())
        qp_norm_sq = max(0.0, float((active_weights @ gram @ active_weights).item()))
        diagnostics["mgda_qp_gradient_norm_sq"] = qp_norm_sq
        diagnostics["mgda_qp_gradient_norm"] = qp_norm_sq ** 0.5
        diagnostics["mgda_active_objective_count"] = float(len(active_indices))
    else:
        diagnostics["mgda_active_objective_count"] = 0.0

    return MGDASolution(weights, norms, coefficients, diagnostics)


def compute_mgda_weights(
    objectives: Mapping[str, torch.Tensor],
    shared_parameters: Sequence[torch.nn.Parameter],
) -> tuple[dict[str, float], dict[str, float]]:
    """Backward-compatible raw MGDA interface returning weights and norms."""
    result = solve_mgda(objectives, shared_parameters)
    return result.weights, result.gradient_norms


def _gram_matrix(flat_grads: torch.Tensor) -> torch.Tensor:
    """Return the symmetric task-gradient Gram matrix in float64 CPU space."""
    gradients = torch.as_tensor(flat_grads, dtype=torch.float64, device="cpu")
    if gradients.ndim != 2 or gradients.shape[0] == 0:
        raise ValueError("flat_grads must be a non-empty (tasks, parameters) matrix")
    if not bool(torch.isfinite(gradients).all()):
        raise ValueError("flat_grads must contain only finite values")
    gram = gradients @ gradients.T
    return (gram + gram.T) * 0.5


def frank_wolfe_min_norm(
    flat_grads: torch.Tensor, *, max_iter: int = 200, tol: float = 1e-8
) -> tuple[torch.Tensor, float, int]:
    """Minimize the norm of a convex combination of task gradients.

    This is the simplex-constrained MGDA QP solved by Frank-Wolfe with exact
    line search. Returns ``(weights, squared_norm, iterations)``.
    """
    if max_iter < 1 or tol <= 0:
        raise ValueError("max_iter and tol must be positive")
    gram = _gram_matrix(flat_grads)
    count = gram.shape[0]
    if count == 1:
        return torch.ones(1, dtype=gram.dtype), float(gram[0, 0].clamp_min(0)), 0
    gram_scale = gram.abs().max()
    if float(gram_scale) <= torch.finfo(gram.dtype).tiny:
        weights = torch.full((count,), 1.0 / count, dtype=gram.dtype)
        return weights, 0.0, 0
    # The QP minimizer is unchanged by positive scaling. Normalize so the
    # Frank-Wolfe dual-gap tolerance behaves identically for small and large
    # gradient magnitudes, then report the norm in the original scale below.
    scaled_gram = gram / gram_scale

    # Start at the best vertex. Convex quadratic objective is smooth over the simplex.
    weights = torch.zeros(count, dtype=gram.dtype)
    weights[int(torch.diagonal(scaled_gram).argmin())] = 1.0
    iterations = 0
    for iterations in range(1, max_iter + 1):
        grad = scaled_gram @ weights
        vertex = int(grad.argmin().item())
        gap = float((weights @ grad - grad[vertex]).clamp_min(0.0).item())
        if gap <= tol * max(1.0, abs(float((weights @ grad).item()))):
            iterations -= 1
            break
        direction = -weights
        direction[vertex] += 1.0
        curvature = float((direction @ scaled_gram @ direction).item())
        slope = float((direction @ grad).item())
        step = min(1.0, max(0.0, -slope / curvature)) if curvature > 0 else 1.0
        weights = weights + step * direction
        weights.clamp_(min=0.0)
        weights /= weights.sum().clamp_min(torch.finfo(weights.dtype).eps)

    norm_sq = max(0.0, float((weights @ gram @ weights).item()))
    return weights, norm_sq, iterations


def collect_flat_gradients(
    losses: Mapping[str, torch.Tensor],
    params: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool = True,
) -> tuple[list[str], torch.Tensor]:
    """Collect each scalar task gradient on the same parameters, without .grad writes."""
    names = list(losses)
    parameters = [p for p in params if p.requires_grad]
    if not names or not parameters:
        raise ValueError("losses and trainable parameters must both be non-empty")
    rows = []
    for index, name in enumerate(names):
        loss = losses[name]
        if loss.numel() != 1 or not loss.requires_grad:
            raise ValueError(f"loss {name!r} must be a differentiable scalar")
        grads = torch.autograd.grad(
            loss, parameters,
            retain_graph=retain_graph or index < len(names) - 1,
            allow_unused=True,
        )
        pieces = [
            gradient.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
            if gradient is not None else torch.zeros(parameter.numel(), dtype=torch.float64)
            for parameter, gradient in zip(parameters, grads)
        ]
        row = torch.cat(pieces)
        if not bool(torch.isfinite(row).all()):
            raise FloatingPointError(f"gradient for task {name!r} is non-finite")
        rows.append(row)
    return names, torch.stack(rows)


def pairwise_cosine_similarity(flat_grads: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine matrix with zero-safe normalization."""
    gradients = torch.as_tensor(flat_grads, dtype=torch.float64, device="cpu")
    norms = torch.linalg.vector_norm(gradients, dim=1).clamp_min(MGDA_NORM_EPSILON)
    cosine = (gradients @ gradients.T) / (norms[:, None] * norms[None, :])
    return cosine.clamp(-1.0, 1.0)


def compute_gradient_conflict_stats(flat_grads: torch.Tensor) -> dict[str, float]:
    cosine = pairwise_cosine_similarity(flat_grads)
    if cosine.shape[0] < 2:
        return {"conflict_fraction": 0.0, "mean_pairwise_cosine": 1.0}
    upper = cosine[torch.triu_indices(cosine.shape[0], cosine.shape[1], offset=1).unbind()]
    return {
        "conflict_fraction": float((upper < 0).double().mean().item()),
        "mean_pairwise_cosine": float(upper.mean().item()),
    }


class MGDASolver:
    """MGDA solver that fixes one shared parameter vector for all objectives."""

    def __init__(
        self, model: torch.nn.Module, param_scope: str = "mask_head", *,
        max_iter: int = 200, tol: float = 1e-8,
        gradient_normalization: str = "raw",
    ):
        if param_scope not in {"mask_head", "full"}:
            raise ValueError("param_scope must be 'mask_head' or 'full'")
        if gradient_normalization not in {"raw", "unit"}:
            raise ValueError("gradient_normalization must be 'raw' or 'unit'")
        self.param_scope = param_scope
        self.max_iter = max_iter
        self.tol = tol
        self.gradient_normalization = gradient_normalization
        selected = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if param_scope == "full" or "roi_heads.mask_head." in name or "roi_heads.mask_predictor." in name:
                selected.append(parameter)
        if not selected:
            raise ValueError(f"No trainable parameters selected for MGDA scope {param_scope!r}")
        self.parameters = selected

    def solve(self, task_losses: Mapping[str, torch.Tensor]) -> MGDAResult:
        names, gradients = collect_flat_gradients(task_losses, self.parameters)
        norms_tensor = torch.linalg.vector_norm(gradients, dim=1)
        norms = {name: float(norm.item()) for name, norm in zip(names, norms_tensor)}
        active = [i for i, norm in enumerate(norms_tensor) if float(norm) > MGDA_NORM_EPSILON]
        lambdas = torch.zeros(len(names), dtype=torch.float64)
        applied = {name: 0.0 for name in names}
        iters = 0
        grad_norm_sq = 0.0
        if active:
            active_grads = gradients[active]
            active_norms = norms_tensor[active]
            qp_grads = (
                active_grads / active_norms.clamp_min(MGDA_NORM_EPSILON)[:, None]
                if self.gradient_normalization == "unit" else active_grads
            )
            active_lambdas, grad_norm_sq, iters = frank_wolfe_min_norm(
                qp_grads, max_iter=self.max_iter, tol=self.tol
            )
            for local_i, original_i in enumerate(active):
                name = names[original_i]
                value = float(active_lambdas[local_i].item())
                lambdas[original_i] = value
                applied[name] = (
                    value / max(norms[name], MGDA_NORM_EPSILON)
                    if self.gradient_normalization == "unit" else value
                )
        stats = compute_gradient_conflict_stats(gradients)
        stats.update({
            "mgda_shared_parameter_count": float(len(self.parameters)),
            "mgda_shared_parameter_numel": float(sum(p.numel() for p in self.parameters)),
            "mgda_active_objective_count": float(len(active)),
            "mgda_qp_gradient_norm_sq": grad_norm_sq,
        })
        cosines = pairwise_cosine_similarity(gradients)
        for i, left in enumerate(names):
            for j in range(i, len(names)):
                stats[f"mgda_cosine_{left}_{names[j]}"] = float(cosines[i, j].item())
        return MGDAResult(lambdas, grad_norm_sq, iters, names, norms, applied, stats)


class LambdaHistory:
    """Per-step simplex weight tracker with epoch averages."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._values: list[dict[str, float]] = []

    def record(self, values: Mapping[str, float]) -> None:
        self._values.append({key: float(value) for key, value in values.items()})

    def epoch_averages(self) -> dict[str, float]:
        if not self._values:
            return {}
        names = self._values[0]
        return {name: sum(row.get(name, 0.0) for row in self._values) / len(self._values) for name in names}


@contextmanager
def mgda_capture_context(cache: dict) -> Iterator[dict]:
    """Activate the model's optional MGDA mask-logit capture context."""
    from .model import _MGDA_CACHE

    token = _MGDA_CACHE.set(cache)
    try:
        yield cache
    finally:
        _MGDA_CACHE.reset(token)
