"""Request / response models for the Jobs API."""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class PipelineType(str, Enum):
    ocr_only = "ocr_only"
    ocr_extract = "ocr+extract"


class OCREngineType(str, Enum):
    tesseract = "tesseract"
    paddleocr = "paddleocr"


class LLMEngineType(str, Enum):
    llama_cpp = "llama_cpp"


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class JobCreateByPath(BaseModel):
    file_path: str
    pipeline: PipelineType = PipelineType.ocr_extract
    engine_ocr: OCREngineType = OCREngineType.tesseract
    engine_llm: LLMEngineType = LLMEngineType.llama_cpp
    lang: str = "rus+eng"
    schema_version: str = "v1"


class JobCreateResponse(BaseModel):
    job_id: uuid.UUID
    status: JobStatus
    poll_url: str


class JobStatusResponse(BaseModel):
    job_id: uuid.UUID
    status: JobStatus
    progress_pages: int = 0
    page_count: int | None = None
    created_at: datetime
    updated_at: datetime
    error_code: str | None = None
    error_message: str | None = None


class ArtifactPaths(BaseModel):
    ocr_json: str | None = None
    ocr_md: str | None = None
    extraction_json: str | None = None
    meta: str | None = None


class ExtractionPayload(BaseModel):
    status: str | None = None
    json_validated: dict | None = None
    confidence: float | None = None
    warnings: list[str] = Field(default_factory=list)


class JobResultResponse(BaseModel):
    job_id: uuid.UUID
    status: JobStatus
    artifacts: ArtifactPaths | None = None
    extraction: ExtractionPayload | None = None


class CorrectionSubmit(BaseModel):
    fields: dict
    comment: str | None = None
    submitted_by: str | None = None


class CorrectionResponse(BaseModel):
    id: int
    job_id: uuid.UUID
    version: int
    created_at: datetime


class AdminJobFilter(BaseModel):
    status: JobStatus | None = None
    engine_ocr: OCREngineType | None = None
    limit: int = Field(50, ge=1, le=500)
    offset: int = Field(0, ge=0)


class HealthResponse(BaseModel):
    db: bool
    redis: bool
    llm_model: bool
