"""Pydantic v2 schemas for all API endpoints."""
from __future__ import annotations
from datetime import datetime
from typing import Optional, Any
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
    data_root: str = Field("./data/luna16")


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


class InferenceResultResponse(BaseModel):
    id: int
    scan_uid: str
    nodules: list[Any]
    dice: Optional[float]
    iou: Optional[float]
    hd95: Optional[float]
    sensitivity: Optional[float]
    source: str
    inference_time_ms: Optional[float]
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
