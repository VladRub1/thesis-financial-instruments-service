"""Initial schema — jobs, artifacts, extractions, corrections

Revision ID: 001
Revises: None
Create Date: 2025-01-01 00:00:00.000000
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("status", sa.String(20), nullable=False, index=True, server_default="queued"),
        sa.Column("pipeline", sa.String(20), server_default="ocr+extract"),
        sa.Column("engine_ocr", sa.String(30), server_default="tesseract"),
        sa.Column("engine_llm", sa.String(30), server_default="llama_cpp"),
        sa.Column("lang", sa.String(20), server_default="rus+eng"),
        sa.Column("schema_version", sa.String(10), server_default="v1"),
        sa.Column("input_type", sa.String(10), nullable=False),
        sa.Column("original_filename", sa.String(500)),
        sa.Column("source_path", sa.String(1000)),
        sa.Column("sha256", sa.String(64), index=True),
        sa.Column("page_count", sa.Integer),
        sa.Column("progress_pages", sa.Integer, server_default="0"),
        sa.Column("duration_ms_ocr", sa.Integer),
        sa.Column("duration_ms_llm", sa.Integer),
        sa.Column("duration_ms_total", sa.Integer),
        sa.Column("error_code", sa.String(50)),
        sa.Column("error_message", sa.Text),
        sa.Column("trace_id", sa.String(36)),
    )

    op.create_table(
        "artifacts",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("job_id", UUID(as_uuid=True), sa.ForeignKey("jobs.id", ondelete="CASCADE"), unique=True),
        sa.Column("ocr_json_path", sa.String(1000)),
        sa.Column("ocr_md_path", sa.String(1000)),
        sa.Column("extraction_json_path", sa.String(1000)),
        sa.Column("meta_path", sa.String(1000)),
    )

    op.create_table(
        "extractions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("job_id", UUID(as_uuid=True), sa.ForeignKey("jobs.id", ondelete="CASCADE"), index=True),
        sa.Column("schema_version", sa.String(10), server_default="v1"),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("json_validated", JSONB),
        sa.Column("json_raw", sa.Text),
        sa.Column("confidence", sa.Float),
        sa.Column("warnings", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "corrections",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("job_id", UUID(as_uuid=True), sa.ForeignKey("jobs.id", ondelete="CASCADE"), index=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("payload_json", JSONB, nullable=False),
        sa.Column("previous_payload_json", JSONB),
        sa.Column("comment", sa.Text),
        sa.Column("submitted_by", sa.String(200)),
        sa.Column("version", sa.Integer, server_default="1"),
    )


def downgrade() -> None:
    op.drop_table("corrections")
    op.drop_table("extractions")
    op.drop_table("artifacts")
    op.drop_table("jobs")
