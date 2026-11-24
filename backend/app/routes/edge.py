"""Edge node management and network monitoring routes."""
import hmac
import json
import time
import logging
import os
from functools import lru_cache
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
from ..ml.inference import InferenceUnavailableError

router = APIRouter()
logger = logging.getLogger(__name__)
_MAX_PROBE_BYTES = 1024 * 1024
_BACKBONE_PREFIXES = {
    "layer1": {"layer1"},
    "layer2": {"layer1", "layer2"},
    "layer3": {"layer1", "layer2", "layer3"},
}
_DEVELOPMENT_ENVS = {"dev", "development", "test", "testing"}


@lru_cache(maxsize=4)
def _load_learned_decoder(
    checkpoint_path: str,
    codec_sha256: str,
    backbone_sha256: str,
):
    """Cache a decoder by content hash and its bound backbone checkpoint."""
    import torch
    from edge.compressor import FeatureCompressor, _sha256_file

    if _sha256_file(checkpoint_path) != codec_sha256:
        raise ValueError("Learned codec file changed while loading")

    decoder = FeatureCompressor(
        method="learned",
        device=torch.device("cpu"),
        expected_backbone_sha256=backbone_sha256,
        checkpoint_path=checkpoint_path,
    )
    if decoder.codec_sha256 != codec_sha256:
        raise ValueError("Loaded learned codec does not match the payload fingerprint")
    return decoder


def _validate_edge_token(supplied_token: str, *, expected_token: str | None = None,
                         app_env: str | None = None) -> None:
    """Require a configured token outside explicitly selected dev/test modes."""
    expected = os.getenv("EDGE_API_TOKEN", "") if expected_token is None else expected_token
    environment = (os.getenv("APP_ENV", "production") if app_env is None else app_env).strip().lower()
    if not expected:
        if environment in _DEVELOPMENT_ENVS:
            return
        raise HTTPException(
            status_code=503,
            detail="Edge feature inference is disabled until EDGE_API_TOKEN is configured",
        )
    if not hmac.compare_digest(supplied_token, expected):
        raise HTTPException(status_code=401, detail="Invalid edge node credentials")


# ---------------------------------------------------------------------------
# Edge node registration
# ---------------------------------------------------------------------------
@router.post("/network-probe")
async def network_probe(request: Request):
    """Consume a bounded probe body and report the number of bytes received."""
    declared_length = request.headers.get("content-length")
    if declared_length:
        try:
            declared = int(declared_length)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header") from exc
        if declared < 0:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header")
        if declared > _MAX_PROBE_BYTES:
            raise HTTPException(status_code=413, detail="Network probe exceeds 1 MiB")

    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > _MAX_PROBE_BYTES:
            raise HTTPException(status_code=413, detail="Network probe exceeds 1 MiB")
    return {"status": "ok", "bytes_received": total}


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
    supplied_token = request.headers.get("x-edge-token", "")
    _validate_edge_token(supplied_token)
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
        header, separator, raw = body.partition(b"\x00")
        if not separator or not header or len(header) > 64 * 1024:
            raise ValueError("Missing or oversized JSON feature header")
        feature_data = json.loads(header.decode("utf-8"))
        if not isinstance(feature_data, dict):
            raise ValueError("Feature header must be a JSON object")
        version = feature_data.get("version")
        keys = feature_data.get("keys")
        shapes = feature_data.get("shapes")
        if (
            version not in (1, 2)
            or not isinstance(keys, list)
            or not isinstance(shapes, list)
            or not keys
            or len(keys) != len(shapes)
            or len(keys) > 16
            or any(not isinstance(key, str) or not key or len(key) > 64 for key in keys)
            or len(set(keys)) != len(keys)
        ):
            raise ValueError("Invalid feature header schema")

        # Version 2 carries per-key dtypes + quantisation params; version 1
        # was float32-only (the "dtype" field).  Normalise to per-key lists.
        if version == 2:
            dtypes_list = feature_data.get("dtypes")
            scales_list = feature_data.get("scales")
            zps_list    = feature_data.get("zero_points")
            if (
                not isinstance(dtypes_list, list)
                or len(dtypes_list) != len(keys)
                or not isinstance(scales_list, list)
                or len(scales_list) != len(keys)
                or not isinstance(zps_list, list)
                or len(zps_list) != len(keys)
            ):
                raise ValueError("Invalid version-2 feature header (dtypes/scales/zero_points)")
        else:
            # version 1 — all float32, no quantisation
            if feature_data.get("dtype") != "float32":
                raise ValueError("Version-1 feature header must have dtype='float32'")
            dtypes_list = ["float32"] * len(keys)
            scales_list = [1.0] * len(keys)
            zps_list    = [0]   * len(keys)

        features: dict = {}
        expected_bytes = 0
        for key, shape, dtype_str, scale, zp in zip(keys, shapes, dtypes_list, scales_list, zps_list):
            if (
                not isinstance(shape, list)
                or not shape
                or len(shape) > 8
                or any(type(dim) is not int or dim < 1 for dim in shape)
            ):
                raise ValueError(f"Invalid shape for feature {key!r}")
            if dtype_str not in ("float32", "int8"):
                raise ValueError(f"Unsupported dtype {dtype_str!r} for feature {key!r}")
            if dtype_str == "int8":
                if (
                    isinstance(scale, bool)
                    or not isinstance(scale, (int, float))
                    or not np.isfinite(scale)
                    or scale <= 0
                    or type(zp) is not int
                    or not -128 <= zp <= 127
                ):
                    raise ValueError(f"Invalid quantisation parameters for feature {key!r}")
            np_dtype  = np.dtype("<f4") if dtype_str == "float32" else np.dtype("int8")
            item_size = np_dtype.itemsize

            element_count = 1
            for dim in shape:
                if element_count > (64 * 1024 * 1024 // item_size) // dim:
                    raise ValueError("Decoded feature data exceeds 64 MiB")
                element_count *= dim
            byte_count = element_count * item_size
            expected_bytes += byte_count
            if expected_bytes > 64 * 1024 * 1024:
                raise ValueError("Decoded feature data exceeds 64 MiB")

            arr = np.frombuffer(raw[:byte_count], dtype=np_dtype)
            if arr.size != element_count:
                raise ValueError(f"Feature {key!r} data is truncated")
            arr = arr.copy().reshape(shape)
            raw = raw[byte_count:]

            if dtype_str == "int8":
                # Reconstruct the quantised tensor so downstream can dequantise
                t_int8 = torch.from_numpy(arr).to(torch.int8)
                features[key] = torch._make_per_tensor_quantized_tensor(
                    t_int8, float(scale), int(zp)
                )
            else:
                if not np.isfinite(arr).all():
                    raise ValueError(f"Feature {key!r} contains non-finite values")
                features[key] = torch.from_numpy(arr)

        if raw:
            raise ValueError("Feature data contains unexpected trailing bytes")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid feature payload: {e}")

    compression_method = feature_data.get("compression_method", "none")
    if compression_method == "learned":
        codec_path = os.getenv("LEARNED_COMPRESSOR_CHECKPOINT", "").strip()
        if not codec_path or not os.path.isfile(codec_path):
            raise HTTPException(
                status_code=503,
                detail="Learned feature decoder is not configured on the cloud",
            )
        if not feature_data.get("checkpoint_sha256"):
            raise HTTPException(status_code=400, detail="Learned payload is missing backbone fingerprint")
        codec_sha256 = feature_data.get("codec_sha256")
        if not codec_sha256:
            raise HTTPException(status_code=400, detail="Learned payload is missing codec fingerprint")
        try:
            from edge.compressor import _sha256_file

            cloud_codec_sha256 = _sha256_file(codec_path)
            if codec_sha256 != cloud_codec_sha256:
                raise HTTPException(status_code=503, detail="Edge/cloud learned codec checkpoints differ")
            decoder = _load_learned_decoder(
                os.path.abspath(codec_path), cloud_codec_sha256,
                str(feature_data["checkpoint_sha256"]),
            )
            expected_layers = _BACKBONE_PREFIXES.get(split_layer)
            if expected_layers is None or set(features) != expected_layers:
                raise HTTPException(status_code=400, detail="Learned feature levels do not match split_layer")
            expanded_bytes = 0
            for name, latent in features.items():
                expected_latent_channels = decoder._autoencoder.LATENT_CHANNELS[name]
                if latent.ndim != 4 or latent.shape[0] != 1 or latent.shape[1] != expected_latent_channels:
                    raise HTTPException(status_code=400, detail=f"Invalid learned latent shape for {name}")
                expanded_bytes += latent.shape[0] * decoder._autoencoder.CHANNELS[name] * latent.shape[2] * latent.shape[3] * 4
            if expanded_bytes > 64 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Decoded learned features exceed 64 MiB")
            features = decoder.decompress(features)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Could not decode learned edge features")
            raise HTTPException(status_code=400, detail=f"Invalid learned feature payload: {exc}") from exc
    elif compression_method not in {"none", "int8"}:
        raise HTTPException(status_code=400, detail=f"Unsupported compression method: {compression_method}")

    engine = request.app.state.inference_engine
    # BUG-FIX: previously a strict equality check against FPN output keys
    # {"0","1","2","3","pool"} raised HTTP 501 for every real edge request,
    # because the edge sends backbone-layer keys ("layer1"/"layer2"/"layer3").
    # Now we route by key set:
    #  • FPN keys  → legacy direct-FPN path (predict_from_features)
    #  • layer keys → edge-split path: cloud runs remaining layers + FPN
    #                 (predict_from_edge_features)
    feature_keys = set(features)
    try:
        fpn_keys = {"0", "1", "2", "3", "pool"}
        if feature_keys == fpn_keys:
            # Direct FPN features (e.g. from a future full-FPN edge extractor)
            nodules = engine.predict_from_features(features, scan_uid)
        elif split_layer in _BACKBONE_PREFIXES and feature_keys == _BACKBONE_PREFIXES[split_layer]:
            # Backbone layer features from edge split — normal production path
            nodules = engine.predict_from_edge_features(
                features,
                split_layer,
                scan_uid,
                expected_checkpoint_sha256=feature_data.get("checkpoint_sha256"),
                edge_image_mean=feature_data.get("image_mean"),
                edge_image_std=feature_data.get("image_std"),
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Feature keys {sorted(feature_keys)!r} do not match split_layer="
                    f"{split_layer!r}. Expected the cumulative backbone features "
                    f"{sorted(_BACKBONE_PREFIXES.get(split_layer, set()))!r} or FPN keys."
                ),
            )
    except HTTPException:
        raise
    except InferenceUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.exception("Cloud-side inference from features failed")
        raise HTTPException(status_code=500, detail="Cloud-side inference failed") from e

    inference_ms = (time.perf_counter() - t0) * 1000

    # Store each successful edge/cloud scan once with distinct component timings.
    # latency_ms is the edge's measured health-probe RTT, never edge compute time.
    log = NetworkLog(
        node_id=node_id,
        latency_ms=feature_data.get("probe_latency_ms"),
        edge_processing_ms=feature_data.get("edge_processing_ms"),
        bandwidth_mbps=feature_data.get("bandwidth_mbps", 0),
        payload_size_kb=payload_kb,
        inference_time_ms=inference_ms,
        split_layer=split_layer,
        compression_method=feature_data.get("compression_method", "none"),
        compression_ratio=feature_data.get("compression_ratio", 1.0),
        timing_schema_version=(
            2 if feature_data.get("probe_latency_ms") is not None else None
        ),
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
            func.avg(NetworkLog.edge_processing_ms).label("avg_edge_processing"),
            func.avg(NetworkLog.bandwidth_mbps).label("avg_bw"),
            func.avg(NetworkLog.payload_size_kb).label("avg_payload_kb"),
            func.avg(NetworkLog.inference_time_ms).label("avg_inference_ms"),
            func.avg(NetworkLog.total_pipeline_ms).label("avg_total_pipeline_ms"),
            func.avg(NetworkLog.compression_ratio).label("avg_compression"),
            func.count(NetworkLog.id).label("n_requests"),
        )
        .group_by(NetworkLog.split_layer, NetworkLog.compression_method)
        .filter(NetworkLog.timing_schema_version == 2)
        .all()
    )
    return [
        {
            "split_layer": r.split_layer,
            "compression_method": r.compression_method,
            "avg_latency_ms": round(r.avg_latency, 2) if r.avg_latency is not None else None,
            "avg_network_rtt_ms": round(r.avg_latency, 2) if r.avg_latency is not None else None,
            "avg_edge_processing_ms": round(r.avg_edge_processing, 2) if r.avg_edge_processing is not None else None,
            "avg_bandwidth_mbps": round(r.avg_bw or 0, 2),
            "avg_payload_kb": round(r.avg_payload_kb or 0, 2),
            "avg_inference_ms": round(r.avg_inference_ms, 2) if r.avg_inference_ms is not None else None,
            "avg_total_pipeline_ms": round(r.avg_total_pipeline_ms, 2) if r.avg_total_pipeline_ms is not None else None,
            "avg_compression_ratio": round(r.avg_compression or 1, 3),
            "n_requests": r.n_requests,
        }
        for r in rows
    ]


@router.get("/split-latency-analysis")
async def split_latency_analysis(db: Session = Depends(get_db)):
    """
    Return versioned timing components grouped by split point. Total pipeline
    time is nullable when the edge did not report a complete measurement.
    """
    from sqlalchemy import func
    rows = (
        db.query(
            NetworkLog.split_layer,
            func.avg(NetworkLog.total_pipeline_ms).label("total_latency"),
            func.avg(NetworkLog.latency_ms).label("network_rtt"),
            func.avg(NetworkLog.edge_processing_ms).label("edge_processing"),
            func.avg(NetworkLog.inference_time_ms).label("cloud_inference"),
            func.avg(NetworkLog.payload_size_kb).label("payload_kb"),
            func.count(NetworkLog.id).label("n"),
        )
        .group_by(NetworkLog.split_layer)
        .filter(NetworkLog.timing_schema_version == 2)
        .all()
    )
    return [
        {
            "split_layer": r.split_layer,
            "avg_total_latency_ms": round(r.total_latency, 2) if r.total_latency is not None else None,
            "avg_network_rtt_ms": round(r.network_rtt, 2) if r.network_rtt is not None else None,
            "avg_edge_processing_ms": round(r.edge_processing, 2) if r.edge_processing is not None else None,
            "avg_cloud_inference_ms": round(r.cloud_inference, 2) if r.cloud_inference is not None else None,
            "avg_payload_kb": round(r.payload_kb or 0, 2),
            "n_samples": r.n,
        }
        for r in rows
    ]
