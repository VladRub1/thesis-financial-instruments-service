from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)

    pipeline: Mapped[str] = mapped_column(String(20), default="ocr+extract")
    engine_ocr: Mapped[str] = mapped_column(String(30), default="tesseract")
    engine_llm: Mapped[str] = mapped_column(String(30), default="llama_cpp")
    lang: Mapped[str] = mapped_column(String(20), default="rus+eng")
    schema_version: Mapped[str] = mapped_column(String(10), default="v1")

    input_type: Mapped[str] = mapped_column(String(10))  # upload | path
    original_filename: Mapped[str | None] = mapped_column(String(500))
    source_path: Mapped[str | None] = mapped_column(String(1000))
    sha256: Mapped[str | None] = mapped_column(String(64), index=True)

    page_count: Mapped[int | None] = mapped_column(Integer)
    progress_pages: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms_ocr: Mapped[int | None] = mapped_column(Integer)
    duration_ms_llm: Mapped[int | None] = mapped_column(Integer)
    duration_ms_total: Mapped[int | None] = mapped_column(Integer)

    error_code: Mapped[str | None] = mapped_column(String(50))
    error_message: Mapped[str | None] = mapped_column(Text)
    trace_id: Mapped[str | None] = mapped_column(String(36))

    artifact: Mapped[Artifact | None] = relationship(back_populates="job", uselist=False)
    extractions: Mapped[list[Extraction]] = relationship(back_populates="job")
    corrections: Mapped[list[Correction]] = relationship(back_populates="job")


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("jobs.id", ondelete="CASCADE"), unique=True
    )
    ocr_json_path: Mapped[str | None] = mapped_column(String(1000))
    ocr_md_path: Mapped[str | None] = mapped_column(String(1000))
    extraction_json_path: Mapped[str | None] = mapped_column(String(1000))
    meta_path: Mapped[str | None] = mapped_column(String(1000))

    job: Mapped[Job] = relationship(back_populates="artifact")


class Extraction(Base):
    __tablename__ = "extractions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    schema_version: Mapped[str] = mapped_column(String(10), default="v1")
    status: Mapped[str] = mapped_column(String(30))  # succeeded | failed_validation | failed_runtime
    json_validated: Mapped[dict | None] = mapped_column(JSON)
    json_raw: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    warnings: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    job: Mapped[Job] = relationship(back_populates="extractions")


class Correction(Base):
    __tablename__ = "corrections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    payload_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    previous_payload_json: Mapped[dict | None] = mapped_column(JSON)
    comment: Mapped[str | None] = mapped_column(Text)
    submitted_by: Mapped[str | None] = mapped_column(String(200))
    version: Mapped[int] = mapped_column(Integer, default=1)

    job: Mapped[Job] = relationship(back_populates="corrections")
