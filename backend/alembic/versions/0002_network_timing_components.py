"""Add separate edge, network, and total timing fields to network logs.

Revision ID: 0002
Revises: 0001
"""
from alembic import op
import sqlalchemy as sa


revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("network_logs", sa.Column("edge_processing_ms", sa.Float(), nullable=True))
    op.add_column("network_logs", sa.Column("total_pipeline_ms", sa.Float(), nullable=True))
    # Historical latency_ms values may be either RTT or edge processing time.
    # Keep old rows unversioned because their true semantics cannot be recovered.
    op.add_column("network_logs", sa.Column("timing_schema_version", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("network_logs", "timing_schema_version")
    op.drop_column("network_logs", "total_pipeline_ms")
    op.drop_column("network_logs", "edge_processing_ms")
