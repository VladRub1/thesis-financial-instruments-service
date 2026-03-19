"""Job endpoints — create, poll, result, corrections."""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from arq import ArqRedis
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.core.security import validate_file_path
from app.schemas.jobs import (
    ArtifactPaths,
    CorrectionResponse,
    CorrectionSubmit,
    ExtractionPayload,
    JobCreateByPath,
    JobCreateResponse,
    JobResultResponse,
    JobStatus,
    JobStatusResponse,
)
from app.services.jobs import compute_sha256, create_job, get_job, get_job_with_relations, submit_correction

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


def _arq(request: Request) -> ArqRedis:
    pool: ArqRedis | None = request.app.state.arq_pool
    if pool is None:
        raise HTTPException(503, "Job queue unavailable")
    return pool


# ── Create job ──────────────────────────────────────────────

@router.post("", response_model=JobCreateResponse, status_code=status.HTTP_201_CREATED,
             summary="Create a new OCR / extraction job")
async def create_job_upload(
    request: Request,
    db: AsyncSession = Depends(get_db),
    file: UploadFile | None = File(None),
    pipeline: str = Form("ocr+extract"),
    engine_ocr: str = Form("tesseract"),
    engine_llm: str = Form("llama_cpp"),
    lang: str = Form("rus+eng"),
    schema_version: str = Form("v1"),
):
    """Upload a PDF to start a processing job."""
    if file is None:
        raise HTTPException(422, "Provide a file upload")

    from app.core.config import settings

    _ALLOWED_EXT = {".pdf", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if not file.filename or ext not in _ALLOWED_EXT:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Allowed file types: {', '.join(sorted(_ALLOWED_EXT))}",
        )

    contents = await file.read()
    sha = hashlib.sha256(contents).hexdigest()

    upload_dir = settings.upload_path
    dest = upload_dir / f"{sha[:16]}_{file.filename}"
    dest.write_bytes(contents)

    job = await create_job(
        db,
        input_type="upload",
        original_filename=file.filename,
        source_path=str(dest),
        sha256=sha,
        pipeline=pipeline,
        engine_ocr=engine_ocr,
        engine_llm=engine_llm,
        lang=lang,
        schema_version=schema_version,
    )

    pool = _arq(request)
    await pool.enqueue_job("process_job", str(job.id))

    return JobCreateResponse(
        job_id=job.id,
        status=JobStatus.queued,
        poll_url=f"/v1/jobs/{job.id}",
    )


@router.post("/by-path", response_model=JobCreateResponse, status_code=status.HTTP_201_CREATED,
              summary="Create a job from a server-side file path")
async def create_job_by_path(
    body: JobCreateByPath,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    resolved = validate_file_path(body.file_path)
    _ALLOWED_EXT = {".pdf", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    if resolved.suffix.lower() not in _ALLOWED_EXT:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Allowed file types: {', '.join(sorted(_ALLOWED_EXT))}",
        )

    sha = compute_sha256(resolved)
    job = await create_job(
        db,
        input_type="path",
        original_filename=resolved.name,
        source_path=str(resolved),
        sha256=sha,
        pipeline=body.pipeline.value,
        engine_ocr=body.engine_ocr.value,
        engine_llm=body.engine_llm.value,
        lang=body.lang,
        schema_version=body.schema_version,
    )

    pool = _arq(request)
    await pool.enqueue_job("process_job", str(job.id))

    return JobCreateResponse(
        job_id=job.id,
        status=JobStatus.queued,
        poll_url=f"/v1/jobs/{job.id}",
    )


# ── Poll status ─────────────────────────────────────────────

@router.get("/{job_id}", response_model=JobStatusResponse, summary="Get job status")
async def get_job_status(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    job = await get_job(db, job_id)
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    return JobStatusResponse(
        job_id=job.id,
        status=JobStatus(job.status),
        progress_pages=job.progress_pages,
        page_count=job.page_count,
        created_at=job.created_at,
        updated_at=job.updated_at,
        error_code=job.error_code,
        error_message=job.error_message,
    )


# ── Result ───────────────────────────────────────────────────

@router.get("/{job_id}/result", response_model=JobResultResponse, summary="Get job result")
async def get_job_result(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    job = await get_job_with_relations(db, job_id)
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    if job.status not in ("succeeded", "failed"):
        raise HTTPException(status.HTTP_409_CONFLICT, "Job not finished yet")

    artifacts = None
    if job.artifact:
        artifacts = ArtifactPaths(
            ocr_json=job.artifact.ocr_json_path,
            ocr_md=job.artifact.ocr_md_path,
            extraction_json=job.artifact.extraction_json_path,
            meta=job.artifact.meta_path,
        )

    extraction = None
    if job.extractions:
        latest = max(job.extractions, key=lambda e: e.created_at)
        extraction = ExtractionPayload(
            status=latest.status,
            json_validated=latest.json_validated,
            confidence=latest.confidence,
            warnings=latest.warnings or [],
        )

    return JobResultResponse(
        job_id=job.id,
        status=JobStatus(job.status),
        artifacts=artifacts,
        extraction=extraction,
    )


# ── Corrections ──────────────────────────────────────────────

@router.post("/{job_id}/corrections", response_model=CorrectionResponse,
             status_code=status.HTTP_201_CREATED, summary="Submit corrections")
async def post_correction(
    job_id: uuid.UUID,
    body: CorrectionSubmit,
    db: AsyncSession = Depends(get_db),
):
    job = await get_job(db, job_id)
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")

    corr = await submit_correction(db, job_id, body.fields, body.comment, body.submitted_by)
    return CorrectionResponse(
        id=corr.id,
        job_id=corr.job_id,
        version=corr.version,
        created_at=corr.created_at,
    )
