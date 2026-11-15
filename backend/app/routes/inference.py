"""Inference API routes."""
import base64
import io
import time
import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request
from sqlalchemy.orm import Session

from ..database import get_db, InferenceResult
from ..schemas import InferenceRequest, InferenceResponse, NoduleDetection, InferenceResultResponse
from ..ml.inference import InferenceEngine

router = APIRouter()
logger = logging.getLogger(__name__)


def get_engine(request: Request) -> InferenceEngine:
    return request.app.state.inference_engine


# ---------------------------------------------------------------------------
# POST /api/inference/predict — base64 image input
# ---------------------------------------------------------------------------
@router.post("/predict", response_model=InferenceResponse)
async def predict(
    payload: InferenceRequest,
    db: Session = Depends(get_db),
    engine: InferenceEngine = Depends(get_engine),
):
    """
    Run Mask R-CNN inference on a CT patch.
    Accepts base64-encoded PNG image.
    """
    try:
        image_bytes = base64.b64decode(payload.image_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    t0 = time.perf_counter()
    try:
        nodules = engine.predict_from_bytes(image_bytes, experiment_id=payload.experiment_id)
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(e))
    inference_ms = (time.perf_counter() - t0) * 1000

    # Persist result
    result = InferenceResult(
        scan_uid=payload.scan_uid,
        experiment_id=payload.experiment_id,
        nodules=[n.model_dump() for n in nodules],
        source=payload.source,
        inference_time_ms=inference_ms,
        # sensitivity here = fraction of detections with score >= 0.5 (a proxy metric,
        # not the true sensitivity which requires ground-truth masks)
        sensitivity=float(np.mean([n.score >= 0.5 for n in nodules])) if nodules else None,
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
    )


# ---------------------------------------------------------------------------
# POST /api/inference/upload — multipart file upload
# ---------------------------------------------------------------------------
@router.post("/upload", response_model=InferenceResponse)
async def predict_upload(
    scan_uid: str = Form(...),
    source: str = Form("cloud"),
    experiment_id: Optional[int] = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    engine: InferenceEngine = Depends(get_engine),
):
    """Upload a CT patch file and run inference."""
    if file.content_type not in ("image/png", "image/jpeg", "application/octet-stream"):
        raise HTTPException(status_code=415, detail="Unsupported file type")

    image_bytes = await file.read()
    t0 = time.perf_counter()
    try:
        nodules = engine.predict_from_bytes(image_bytes, experiment_id=experiment_id)
    except Exception as e:
        logger.exception("Inference failed on upload")
        raise HTTPException(status_code=500, detail=str(e))
    inference_ms = (time.perf_counter() - t0) * 1000

    result = InferenceResult(
        scan_uid=scan_uid,
        experiment_id=experiment_id,
        nodules=[n.model_dump() for n in nodules],
        source=source,
        inference_time_ms=inference_ms,
        sensitivity=float(np.mean([n.score >= 0.5 for n in nodules])) if nodules else None,
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
    )


# ---------------------------------------------------------------------------
# GET /api/inference/results
# ---------------------------------------------------------------------------
@router.get("/results", response_model=list[InferenceResultResponse])
async def list_results(
    scan_uid: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    q = db.query(InferenceResult)
    if scan_uid:
        q = q.filter(InferenceResult.scan_uid == scan_uid)
    return q.order_by(InferenceResult.created_at.desc()).offset(offset).limit(limit).all()


@router.get("/results/{result_id}", response_model=InferenceResultResponse)
async def get_result(result_id: int, db: Session = Depends(get_db)):
    r = db.query(InferenceResult).filter(InferenceResult.id == result_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Result not found")
    return r
