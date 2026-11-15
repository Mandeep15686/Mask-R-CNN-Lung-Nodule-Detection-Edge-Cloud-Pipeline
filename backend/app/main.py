"""
FastAPI backend: inference, experiment management, results storage.
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .database import create_tables
from .routes import inference, experiments, results, edge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("Starting Lung Nodule Detection API...")
    create_tables()
    # Pre-load inference engine
    from .ml.inference import InferenceEngine
    app.state.inference_engine = InferenceEngine()
    logger.info("API ready.")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Lung Nodule Detection API",
    description="Mask R-CNN Edge–Cloud Pipeline for LUNA16",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "CORS_ALLOW_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true",
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(inference.router,   prefix="/api/inference",   tags=["Inference"])
app.include_router(experiments.router, prefix="/api/experiments", tags=["Experiments"])
app.include_router(results.router,     prefix="/api/results",     tags=["Results"])
app.include_router(edge.router,        prefix="/api/edge",        tags=["Edge"])


@app.get("/health")
async def health(request: Request):
    engine = getattr(request.app.state, "inference_engine", None)
    model_status = engine.status if engine is not None else {
        "model_loaded": False,
        "checkpoint_path": None,
        "device": None,
        "score_threshold": None,
        "model_loading_error": "Inference engine has not initialized",
    }
    return {"status": "ok", "version": "1.0.0", **model_status}


@app.get("/api/info")
async def info():
    return {
        "model": "Mask R-CNN ResNet-50-FPN",
        "dataset": "LUNA16",
        "task": "Lung Nodule Detection + Segmentation",
        "loss_functions": ["BCE", "Dice", "Boundary", "Focal", "HD95"],
    }
