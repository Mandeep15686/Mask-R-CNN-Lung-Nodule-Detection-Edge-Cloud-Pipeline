"""Pydantic v2 schemas for all API endpoints."""
from __future__ import annotations
from datetime import datetime
from typing import Optional, Any, Literal
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Experiment schemas
# ---------------------------------------------------------------------------

class LossWeights(BaseModel):
    alpha: float = Field(1.0, ge=0.0, le=1.0, description="BCE weight")
    beta:  float = Field(0.0, ge=0.0, le=1.0, description="Dice weight")
    gamma: float = Field(0.0, ge=0.0, le=1.0, description="Boundary weight")
    delta: float = Field(0.0, ge=0.0, le=1.0, description="Focal weight")
    epsilon: float = Field(0.0, ge=0.0, le=1.0, description="HD95 weight")

    @model_validator(mode="after")
    def require_active_weight(self):
        if not self.is_valid():
            raise ValueError("At least one loss weight must be > 0")
        return self

    def is_valid(self) -> bool:
        return (self.alpha + self.beta + self.gamma + self.delta + self.epsilon) > 0


class ExperimentCreate(BaseModel):
    config_id: str = Field(..., min_length=1, max_length=64)
    phase: int = Field(1, ge=1, le=3)
    loss_weights: LossWeights
    epochs: int = Field(30, ge=1, le=200)
    batch_size: int = Field(4, ge=1, le=32)
    fold: int = Field(0, ge=0, le=9)
    lr: float = Field(0.005, gt=0)
    data_root: Optional[str] = Field(None, max_length=1024)
    cache_mode: Literal["cached", "raw"] = "cached"
    patch_size: int = Field(64, ge=8, le=512)
    train_subsets: Optional[list[int]] = None
    validation_subset: Optional[int] = Field(None, ge=0, le=9)
    test_subset: Optional[int] = Field(None, ge=0, le=9)
    cache_dir: Optional[str] = Field(
        None,
        max_length=1024,
        description="Pre-extracted LUNA16 patch cache directory",
    )

    @model_validator(mode="after")
    def validate_subset_split(self):
        explicit = any(
            value is not None
            for value in (self.train_subsets, self.validation_subset, self.test_subset)
        )
        if not explicit:
            return self
        if not self.train_subsets or self.validation_subset is None:
            raise ValueError(
                "Explicit splits require non-empty train_subsets and validation_subset"
            )
        values = list(self.train_subsets)
        if any(subset < 0 or subset > 9 for subset in values):
            raise ValueError("train_subsets values must be between 0 and 9")
        if len(values) != len(set(values)):
            raise ValueError("train_subsets cannot contain duplicate subset IDs")
        groups = [set(values), {self.validation_subset}]
        if self.test_subset is not None:
            groups.append({self.test_subset})
        for index, group in enumerate(groups):
            if any(group & other for other in groups[index + 1:]):
                raise ValueError("Training, validation, and test subsets must be disjoint")
        return self


class ExperimentResponse(BaseModel):
    id: int
    config_id: str
    phase: int
    alpha: float
    beta: float
    gamma: float
    delta: float
    epsilon: float
    loss_description: Optional[str]
    status: str
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    config_json: Optional[dict[str, Any]] = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Training run schemas
# ---------------------------------------------------------------------------

class TrainingRunResponse(BaseModel):
    id: int
    experiment_id: int
    fold: int
    epoch: int
    train_total_loss: Optional[float]
    train_bce_loss: Optional[float]
    train_dice_loss: Optional[float]
    train_boundary_loss: Optional[float]
    train_focal_loss: Optional[float]
    train_hd_loss: Optional[float]
    val_dice: Optional[float]
    val_iou: Optional[float]
    val_hd95: Optional[float]
    val_boundary_f1: Optional[float]
    val_sensitivity: Optional[float]
    val_specificity: Optional[float]
    val_asd: Optional[float]
    val_vol_err: Optional[float]
    val_total_loss: Optional[float] = None
    lr: Optional[float]
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Inference schemas
# ---------------------------------------------------------------------------

class NoduleDetection(BaseModel):
    box: list[float]           # [x1, y1, x2, y2]
    score: float               # detection confidence
    mask_rle: Optional[str]    # run-length encoded mask (optional)
    diameter_estimate_mm: Optional[float]


class InferenceRequest(BaseModel):
    scan_uid: str
    image_b64: str             # base64-encoded CT patch (PNG)
    experiment_id: Optional[int] = None
    source: str = "cloud"      # "edge" or "cloud"


class InferenceResponse(BaseModel):
    scan_uid: str
    nodules: list[NoduleDetection]
    inference_time_ms: float
    source: str
    result_id: int
    checkpoint_path: Optional[str] = None
    config_id: Optional[str] = None
    score_threshold: Optional[float] = None


class InferenceResultResponse(BaseModel):
    id: int
    scan_uid: str
    experiment_id: Optional[int] = None
    nodules: list[Any]
    dice: Optional[float] = None
    iou: Optional[float] = None
    hd95: Optional[float] = None
    sensitivity: Optional[float] = None
    boundary_f1: Optional[float] = None
    source: str
    inference_time_ms: Optional[float] = None
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Edge / network schemas
# ---------------------------------------------------------------------------

class EdgeNodeRegister(BaseModel):
    node_id: str
    hostname: str
    ip_address: str
    split_layer: str = "layer2"
    compression_method: str = "none"


class EdgeNodeResponse(BaseModel):
    id: int
    node_id: str
    hostname: str
    ip_address: str
    is_online: bool
    last_seen: Optional[datetime]
    avg_latency_ms: Optional[float]
    avg_bandwidth_mbps: Optional[float]
    split_layer: str
    compression_ratio: float

    model_config = {"from_attributes": True}


class NetworkLogCreate(BaseModel):
    node_id: str
    latency_ms: float
    bandwidth_mbps: float
    payload_size_kb: float
    inference_time_ms: float
    split_layer: str
    compression_method: str
    compression_ratio: float


class NetworkLogResponse(BaseModel):
    id: int
    node_id: str
    timestamp: datetime
    latency_ms: float
    bandwidth_mbps: float
    payload_size_kb: float
    inference_time_ms: float
    split_layer: str
    compression_method: str
    compression_ratio: float

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Summary / analytics schemas
# ---------------------------------------------------------------------------

class ExperimentSummaryResponse(BaseModel):
    experiment_id: int
    config_id: str
    loss_description: str
    best_dice: float
    best_epoch: int
    final_hd95: Optional[float]
    final_boundary_f1: Optional[float]
    total_epochs: int
    status: str


class DashboardHealthResponse(BaseModel):
    overall: Literal["healthy", "degraded", "unavailable", "unknown"]
    updated_at: datetime
    architecture: str
    device: Optional[str] = None
    gpu_name: Optional[str] = None
    statuses: dict[str, dict[str, Any]]


class DashboardSummaryResponse(BaseModel):
    experiment_counts: dict[str, int]
    inference_requests: int
    active_edge_nodes: int
    recorded_network_requests: int
    best_experiment: Optional[dict[str, Any]] = None
    recent_experiments: list[dict[str, Any]]


class DashboardActivityEvent(BaseModel):
    timestamp: datetime
    event: str
    detail: str
    source: str
    entity_id: Optional[str] = None


class SystemMetricsResponse(BaseModel):
    cpu_percent: Optional[float] = None
    ram_used_bytes: Optional[int] = None
    ram_total_bytes: Optional[int] = None
    gpu_name: Optional[str] = None
    gpu_memory_used_bytes: Optional[int] = None
    gpu_memory_total_bytes: Optional[int] = None
    gpu_utilization_percent: Optional[float] = None
    gpu_temperature_c: Optional[float] = None
    updated_at: datetime
