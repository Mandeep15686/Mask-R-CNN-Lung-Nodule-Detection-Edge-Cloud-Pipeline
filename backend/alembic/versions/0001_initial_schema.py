"""Initial database schema.

Revision ID: 0001
Revises:
Create Date: 2024-01-01 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Experiments table
    op.create_table(
        "experiments",
        sa.Column("id",               sa.Integer(),     nullable=False, primary_key=True),
        sa.Column("config_id",        sa.String(64),    nullable=False, unique=True, index=True),
        sa.Column("phase",            sa.Integer(),     nullable=True,  default=1),
        sa.Column("alpha",            sa.Float(),       nullable=True,  default=1.0),
        sa.Column("beta",             sa.Float(),       nullable=True,  default=0.0),
        sa.Column("gamma",            sa.Float(),       nullable=True,  default=0.0),
        sa.Column("delta",            sa.Float(),       nullable=True,  default=0.0),
        sa.Column("epsilon",          sa.Float(),       nullable=True,  default=0.0),
        sa.Column("loss_description", sa.String(256),   nullable=True),
        sa.Column("status",           sa.String(32),    nullable=True,  default="pending"),
        sa.Column("created_at",       sa.DateTime(),    nullable=True),
        sa.Column("started_at",       sa.DateTime(),    nullable=True),
        sa.Column("finished_at",      sa.DateTime(),    nullable=True),
        sa.Column("config_json",      sa.JSON(),        nullable=True),
    )

    # Training runs table
    op.create_table(
        "training_runs",
        sa.Column("id",                 sa.Integer(), nullable=False, primary_key=True),
        sa.Column("experiment_id",      sa.Integer(), sa.ForeignKey("experiments.id"), index=True),
        sa.Column("fold",               sa.Integer(), nullable=True),
        sa.Column("epoch",              sa.Integer(), nullable=True),
        sa.Column("train_total_loss",   sa.Float(),   nullable=True),
        sa.Column("train_bce_loss",     sa.Float(),   nullable=True),
        sa.Column("train_dice_loss",    sa.Float(),   nullable=True),
        sa.Column("train_boundary_loss",sa.Float(),   nullable=True),
        sa.Column("train_focal_loss",   sa.Float(),   nullable=True),
        sa.Column("train_hd_loss",      sa.Float(),   nullable=True),
        sa.Column("val_dice",           sa.Float(),   nullable=True),
        sa.Column("val_iou",            sa.Float(),   nullable=True),
        sa.Column("val_hd95",           sa.Float(),   nullable=True),
        sa.Column("val_boundary_f1",    sa.Float(),   nullable=True),
        sa.Column("val_sensitivity",    sa.Float(),   nullable=True),
        sa.Column("val_specificity",    sa.Float(),   nullable=True),
        sa.Column("val_asd",            sa.Float(),   nullable=True),
        sa.Column("val_vol_err",        sa.Float(),   nullable=True),
        sa.Column("lr",                 sa.Float(),   nullable=True),
        sa.Column("created_at",         sa.DateTime(),nullable=True),
    )

    # Inference results table
    op.create_table(
        "inference_results",
        sa.Column("id",               sa.Integer(),  nullable=False, primary_key=True),
        sa.Column("scan_uid",         sa.String(128),nullable=True,  index=True),
        sa.Column("experiment_id",    sa.Integer(),  sa.ForeignKey("experiments.id"), nullable=True),
        sa.Column("nodules",          sa.JSON(),     nullable=True),
        sa.Column("dice",             sa.Float(),    nullable=True),
        sa.Column("iou",              sa.Float(),    nullable=True),
        sa.Column("hd95",             sa.Float(),    nullable=True),
        sa.Column("boundary_f1",      sa.Float(),    nullable=True),
        sa.Column("sensitivity",      sa.Float(),    nullable=True),
        sa.Column("source",           sa.String(32), nullable=True),
        sa.Column("inference_time_ms",sa.Float(),    nullable=True),
        sa.Column("created_at",       sa.DateTime(), nullable=True),
        sa.Column("image_path",       sa.String(512),nullable=True),
        sa.Column("mask_path",        sa.String(512),nullable=True),
    )

    # Edge nodes table
    op.create_table(
        "edge_nodes",
        sa.Column("id",              sa.Integer(),   nullable=False, primary_key=True),
        sa.Column("node_id",         sa.String(64),  nullable=False, unique=True, index=True),
        sa.Column("hostname",        sa.String(128), nullable=True),
        sa.Column("ip_address",      sa.String(64),  nullable=True),
        sa.Column("is_online",       sa.Boolean(),   nullable=True),
        sa.Column("last_seen",       sa.DateTime(),  nullable=True),
        sa.Column("avg_latency_ms",  sa.Float(),     nullable=True),
        sa.Column("avg_bandwidth_mbps",sa.Float(),   nullable=True),
        sa.Column("split_layer",     sa.String(64),  nullable=True),
        sa.Column("compression_ratio",sa.Float(),    nullable=True),
        sa.Column("created_at",      sa.DateTime(),  nullable=True),
    )

    # Network logs table
    op.create_table(
        "network_logs",
        sa.Column("id",               sa.Integer(),  nullable=False, primary_key=True),
        sa.Column("node_id",          sa.String(64), nullable=True,  index=True),
        sa.Column("timestamp",        sa.DateTime(), nullable=True),
        sa.Column("latency_ms",       sa.Float(),    nullable=True),
        sa.Column("bandwidth_mbps",   sa.Float(),    nullable=True),
        sa.Column("payload_size_kb",  sa.Float(),    nullable=True),
        sa.Column("inference_time_ms",sa.Float(),    nullable=True),
        sa.Column("split_layer",      sa.String(64), nullable=True),
        sa.Column("compression_method",sa.String(32),nullable=True),
        sa.Column("compression_ratio",sa.Float(),    nullable=True),
    )


def downgrade() -> None:
    op.drop_table("network_logs")
    op.drop_table("edge_nodes")
    op.drop_table("inference_results")
    op.drop_table("training_runs")
    op.drop_table("experiments")
