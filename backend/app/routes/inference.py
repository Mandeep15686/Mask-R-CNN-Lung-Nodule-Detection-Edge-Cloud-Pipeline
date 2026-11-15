"""Inference API routes with explicit checkpoint selection and error handling."""
import base64
import binascii
import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from ..database import Experiment, InferenceResult, get_db
from ..ml.inference import (
    CheckpointNotFoundError,
    InferenceEngine,
    InferenceUnavailableError,
)
from ..schemas import (
    InferenceRequest,
    InferenceResponse,
    InferenceResultResponse,
    NoduleDetection,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def get_engine(request: Request) -> InferenceEngine:
    return request.app.state.inference_engine


def _checkpoint_for_request(
    experiment_id: Optional[int], db: Session, engine: InferenceEngine
) -> tuple[Optional[str], Optional[str]]:
    if experiment_id is None:
        return None, None
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} was not found")
    try:
        checkpoint = engine.checkpoint_for_experiment(experiment.config_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return str(checkpoint), experiment.config_id


def _execute_inference(
    image_bytes: bytes,
    experiment_id: Optional[int],
    db: Session,
    engine: InferenceEngine,
) -> tuple[list[NoduleDetection], dict, Optional[str]]:
    checkpoint_path, config_id = _checkpoint_for_request(experiment_id, db, engine)
    try:
        detections = engine.predict_from_bytes(image_bytes, checkpoint_path=checkpoint_path)
        metadata = engine.checkpoint_info(checkpoint_path)
    except CheckpointNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InferenceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Inference execution failed")
        raise HTTPException(status_code=500, detail="Inference execution failed") from exc
    if config_id is not None:
        metadata["config_id"] = config_id
    logger.info(
        "Inference completed config_id=%s checkpoint=%s detections=%d",
        metadata.get("config_id"),
        metadata.get("checkpoint_path"),
        len(detections),
    )
    return detections, metadata, config_id


@router.get("/status")
async def inference_status(request: Request):
    """Readiness and default checkpoint information for the UI."""
    return request.app.state.inference_engine.status


@router.post("/predict", response_model=InferenceResponse)
async def predict(
    payload: InferenceRequest,
    db: Session = Depends(get_db),
    engine: InferenceEngine = Depends(get_engine),
):
    """Run inference on one exported RGB 2.5D PNG/JPEG patch."""
    try:
        image_bytes = base64.b64decode(payload.image_b64, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image data") from exc

    started = time.perf_counter()
    nodules, metadata, _ = _execute_inference(
        image_bytes, payload.experiment_id, db, engine
    )
    inference_ms = (time.perf_counter() - started) * 1000
    result = InferenceResult(
        scan_uid=payload.scan_uid,
        experiment_id=payload.experiment_id,
        nodules=[n.model_dump() for n in nodules],
        source=payload.source,
        inference_time_ms=inference_ms,
    )
    db.add(result)
    db.commit()
    db.refresh(result)
    return InferenceResponse(
        scan_uid=payload.scan_uid,
        nodules=nodules,
        inference_time_ms=inference_ms,
        source=payload.source,
        result_id=result.id,
        checkpoint_path=metadata["checkpoint_path"],
        config_id=metadata.get("config_id"),
        score_threshold=metadata["score_threshold"],
    )


@router.post("/upload", response_model=InferenceResponse)
async def predict_upload(
    scan_uid: str = Form(...),
    source: str = Form("cloud"),
    experiment_id: Optional[int] = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    engine: InferenceEngine = Depends(get_engine),
):
    """Run inference on one RGB 2.5D patch upload."""
    if file.content_type not in {"image/png", "image/jpeg", "application/octet-stream"}:
        raise HTTPException(status_code=415, detail="Upload a PNG or JPEG 2.5D patch")
    image_bytes = await file.read()
    started = time.perf_counter()
    nodules, metadata, _ = _execute_inference(image_bytes, experiment_id, db, engine)
    inference_ms = (time.perf_counter() - started) * 1000
    result = InferenceResult(
        scan_uid=scan_uid,
        experiment_id=experiment_id,
        nodules=[n.model_dump() for n in nodules],
        source=source,
        inference_time_ms=inference_ms,
    )
    db.add(result)
    db.commit()
    db.refresh(result)
    return InferenceResponse(
        scan_uid=scan_uid,
        nodules=nodules,
        inference_time_ms=inference_ms,
        source=source,
        result_id=result.id,
        checkpoint_path=metadata["checkpoint_path"],
        config_id=metadata.get("config_id"),
        score_threshold=metadata["score_threshold"],
    )


@router.get("/results", response_model=list[InferenceResultResponse])
async def list_results(
    scan_uid: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    query = db.query(InferenceResult)
    if scan_uid:
        query = query.filter(InferenceResult.scan_uid == scan_uid)
    return query.order_by(InferenceResult.created_at.desc()).offset(offset).limit(limit).all()


@router.get("/results/{result_id}", response_model=InferenceResultResponse)
async def get_result(result_id: int, db: Session = Depends(get_db)):
    result = db.query(InferenceResult).filter(InferenceResult.id == result_id).first()
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")
    return result
