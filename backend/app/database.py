"""
SQLAlchemy models + session factory.
Uses SQLite by default; swap DATABASE_URL env var for PostgreSQL.
"""
import os
from datetime import datetime, timezone
from typing import Generator

from sqlalchemy import (
    Column, Integer, Float, String, Text, DateTime,
    Boolean, JSON, ForeignKey, create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./lung_nodule.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30} if "sqlite" in DATABASE_URL else {},
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    config_id = Column(String(64), unique=True, index=True)
    phase = Column(Integer, default=1)                     # 1, 2, or 3
    alpha = Column(Float, default=1.0)                     # BCE weight
    beta = Column(Float, default=0.0)                      # Dice weight
    gamma = Column(Float, default=0.0)                     # Boundary weight
    delta = Column(Float, default=0.0)                     # Focal weight
    epsilon = Column(Float, default=0.0)                   # HD95 weight
    loss_description = Column(String(256), nullable=True)
    status = Column(String(32), default="pending")         # pending/running/done/failed
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    config_json = Column(JSON, nullable=True)
    runs = relationship("TrainingRun", back_populates="experiment", cascade="all, delete-orphan")


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), index=True)
    fold = Column(Integer, default=0)
    epoch = Column(Integer)
    # Train losses
    train_total_loss = Column(Float, nullable=True)
    train_bce_loss = Column(Float, nullable=True)
    train_dice_loss = Column(Float, nullable=True)
    train_boundary_loss = Column(Float, nullable=True)
    train_focal_loss = Column(Float, nullable=True)
    train_hd_loss = Column(Float, nullable=True)
    # Val metrics
    val_dice = Column(Float, nullable=True)
    val_iou = Column(Float, nullable=True)
    val_hd95 = Column(Float, nullable=True)
    val_boundary_f1 = Column(Float, nullable=True)
    val_sensitivity = Column(Float, nullable=True)
    val_specificity = Column(Float, nullable=True)
    val_asd = Column(Float, nullable=True)
    val_vol_err = Column(Float, nullable=True)
    val_total_loss = Column(Float, nullable=True)
    lr = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    experiment = relationship("Experiment", back_populates="runs")


class InferenceResult(Base):
    __tablename__ = "inference_results"

    id = Column(Integer, primary_key=True, index=True)
    scan_uid = Column(String(128), index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=True)
    # Detected nodules (stored as JSON list)
    nodules = Column(JSON, default=list)
    # Aggregate metrics (if GT available)
    dice = Column(Float, nullable=True)
    iou = Column(Float, nullable=True)
    hd95 = Column(Float, nullable=True)
    boundary_f1 = Column(Float, nullable=True)
    sensitivity = Column(Float, nullable=True)
    # Pipeline info
    source = Column(String(32), default="cloud")          # "edge" or "cloud"
    inference_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    image_path = Column(String(512), nullable=True)
    mask_path = Column(String(512), nullable=True)


class EdgeNode(Base):
    __tablename__ = "edge_nodes"

    id = Column(Integer, primary_key=True, index=True)
    node_id = Column(String(64), unique=True, index=True)
    hostname = Column(String(128))
    ip_address = Column(String(64))
    is_online = Column(Boolean, default=False)
    last_seen = Column(DateTime(timezone=True), nullable=True)
    # Metrics
    avg_latency_ms = Column(Float, nullable=True)
    avg_bandwidth_mbps = Column(Float, nullable=True)
    split_layer = Column(String(64), default="layer2")
    compression_ratio = Column(Float, default=1.0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class NetworkLog(Base):
    __tablename__ = "network_logs"

    id = Column(Integer, primary_key=True, index=True)
    node_id = Column(String(64), index=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    latency_ms = Column(Float)
    bandwidth_mbps = Column(Float)
    payload_size_kb = Column(Float)
    inference_time_ms = Column(Float)
    split_layer = Column(String(64))
    compression_method = Column(String(32))
    compression_ratio = Column(Float)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_tables():
    Base.metadata.create_all(bind=engine)
    # Additive SQLite migration for validation loss; preserves existing runs.
    if "sqlite" in DATABASE_URL:
        from sqlalchemy import inspect, text
        columns = {column["name"] for column in inspect(engine).get_columns("training_runs")}
        if "val_total_loss" not in columns:
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE training_runs ADD COLUMN val_total_loss FLOAT"))


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
