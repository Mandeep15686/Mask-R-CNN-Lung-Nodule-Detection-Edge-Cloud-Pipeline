"""Edge node management and network monitoring routes."""
import base64
import hmac
import io
import time
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from ..database import get_db, EdgeNode, NetworkLog
from ..schemas import (
    EdgeNodeRegister, EdgeNodeResponse,
    NetworkLogCreate, NetworkLogResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Edge node registration
# ---------------------------------------------------------------------------
@router.post("/nodes/register", response_model=EdgeNodeResponse)
async def register_node(payload: EdgeNodeRegister, db: Session = Depends(get_db)):
    node = db.query(EdgeNode).filter(EdgeNode.node_id == payload.node_id).first()
    if node:
        node.hostname = payload.hostname
        node.ip_address = payload.ip_address
        node.split_layer = payload.split_layer
        node.is_online = True
        node.last_seen = datetime.now(timezone.utc)
    else:
        node = EdgeNode(
            node_id=payload.node_id,
            hostname=payload.hostname,
            ip_address=payload.ip_address,
            split_layer=payload.split_layer,
            is_online=True,
            last_seen=datetime.now(timezone.utc),
        )
        db.add(node)
    db.commit()
    db.refresh(node)
    return node


@router.get("/nodes", response_model=list[EdgeNodeResponse])
async def list_nodes(db: Session = Depends(get_db)):
    return db.query(EdgeNode).all()


@router.get("/nodes/{node_id}", response_model=EdgeNodeResponse)
async def get_node(node_id: str, db: Session = Depends(get_db)):
    n = db.query(EdgeNode).filter(EdgeNode.node_id == node_id).first()
    if not n:
        raise HTTPException(status_code=404, detail="Node not found")
    return n


@router.post("/nodes/{node_id}/heartbeat")
async def heartbeat(node_id: str, db: Session = Depends(get_db)):
    n = db.query(EdgeNode).filter(EdgeNode.node_id == node_id).first()
    if not n:
        raise HTTPException(status_code=404, detail="Node not found")
    n.is_online = True
    n.last_seen = datetime.now(timezone.utc)
    db.commit()
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Edge-side: receive compressed features and run remaining model
# ---------------------------------------------------------------------------
@router.post("/infer-features")
async def infer_from_features(
    request: Request,
    node_id: str,
    scan_uid: str,
    split_layer: str = "layer2",
    db: Session = Depends(get_db),
):
    """
    Receive compressed intermediate feature maps from an edge node
    and complete the remaining Mask R-CNN computation (RPN + RoI + Mask head).
    """
    t0 = time.perf_counter()
    expected_token = os.getenv("EDGE_API_TOKEN", "")
    supplied_token = request.headers.get("x-edge-token", "")
    if not expected_token:
        raise HTTPException(status_code=503, detail="Edge feature endpoint is not configured")
    if not hmac.compare_digest(supplied_token, expected_token):
        raise HTTPException(status_code=401, detail="Invalid edge node credentials")
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > 64 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Feature payload exceeds 64 MiB")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header") from exc
    body = await request.body()
    if len(body) > 64 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Feature payload exceeds 64 MiB")
    payload_kb = len(body) / 1024.0

    try:
        import torch
        feature_data = torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)
        if not isinstance(feature_data, dict) or not isinstance(feature_data.get("features"), dict):
            raise ValueError("Payload must contain a feature tensor mapping")
        features = feature_data["features"]
        if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in features.items()):
            raise ValueError("Feature mapping must contain only named tensors")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid feature payload: {e}")

    engine = request.app.state.inference_engine
    required_fpn_keys = {"0", "1", "2", "3", "pool"}
    if set(features) != required_fpn_keys:
        raise HTTPException(
            status_code=501,
            detail=(
                "This edge extractor sends a single intermediate feature map, but cloud "
                "inference currently requires all five FPN feature maps. No prediction was run."
            ),
        )
    try:
        nodules = engine.predict_from_features(features, scan_uid)
    except Exception as e:
        logger.exception("Cloud-side inference from features failed")
        raise HTTPException(status_code=500, detail=str(e))

    inference_ms = (time.perf_counter() - t0) * 1000

    # Log network event
    log = NetworkLog(
        node_id=node_id,
        latency_ms=feature_data.get("edge_processing_ms", 0),
        bandwidth_mbps=feature_data.get("bandwidth_mbps", 0),
        payload_size_kb=payload_kb,
        inference_time_ms=inference_ms,
        split_layer=split_layer,
        compression_method=feature_data.get("compression_method", "none"),
        compression_ratio=feature_data.get("compression_ratio", 1.0),
    )
    db.add(log)
    db.commit()

    return {
        "scan_uid": scan_uid,
        "nodules": [n.model_dump() for n in nodules],
        "cloud_inference_ms": inference_ms,
        "payload_kb": payload_kb,
    }


# ---------------------------------------------------------------------------
# Network metrics logging
# ---------------------------------------------------------------------------
@router.post("/network-log", response_model=NetworkLogResponse)
async def log_network(payload: NetworkLogCreate, db: Session = Depends(get_db)):
    log = NetworkLog(**payload.model_dump())
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


@router.get("/network-log", response_model=list[NetworkLogResponse])
async def get_network_logs(
    node_id: Optional[str] = None,
    limit: int = 200,
    db: Session = Depends(get_db),
):
    q = db.query(NetworkLog)
    if node_id:
        q = q.filter(NetworkLog.node_id == node_id)
    return q.order_by(NetworkLog.timestamp.desc()).limit(limit).all()


@router.get("/network-stats")
async def network_stats(db: Session = Depends(get_db)):
    """Aggregate edge-cloud network performance statistics."""
    from sqlalchemy import func
    rows = (
        db.query(
            NetworkLog.split_layer,
            NetworkLog.compression_method,
            func.avg(NetworkLog.latency_ms).label("avg_latency"),
            func.avg(NetworkLog.bandwidth_mbps).label("avg_bw"),
            func.avg(NetworkLog.payload_size_kb).label("avg_payload_kb"),
            func.avg(NetworkLog.inference_time_ms).label("avg_inference_ms"),
            func.avg(NetworkLog.compression_ratio).label("avg_compression"),
            func.count(NetworkLog.id).label("n_requests"),
        )
        .group_by(NetworkLog.split_layer, NetworkLog.compression_method)
        .all()
    )
    return [
        {
            "split_layer": r.split_layer,
            "compression_method": r.compression_method,
            "avg_latency_ms": round(r.avg_latency or 0, 2),
            "avg_bandwidth_mbps": round(r.avg_bw or 0, 2),
            "avg_payload_kb": round(r.avg_payload_kb or 0, 2),
            "avg_inference_ms": round(r.avg_inference_ms or 0, 2),
            "avg_compression_ratio": round(r.avg_compression or 1, 3),
            "n_requests": r.n_requests,
        }
        for r in rows
    ]


@router.get("/split-latency-analysis")
async def split_latency_analysis(db: Session = Depends(get_db)):
    """
    CN Research: Compare total latency across split points.
    Returns data for the split-point optimisation plot.
    """
    from sqlalchemy import func
    rows = (
        db.query(
            NetworkLog.split_layer,
            func.avg(NetworkLog.latency_ms + NetworkLog.inference_time_ms).label("total_latency"),
            func.avg(NetworkLog.payload_size_kb).label("payload_kb"),
            func.count(NetworkLog.id).label("n"),
        )
        .group_by(NetworkLog.split_layer)
        .all()
    )
    return [
        {
            "split_layer": r.split_layer,
            "avg_total_latency_ms": round(r.total_latency or 0, 2),
            "avg_payload_kb": round(r.payload_kb or 0, 2),
            "n_samples": r.n,
        }
        for r in rows
    ]
