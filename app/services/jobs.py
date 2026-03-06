"""Job lifecycle service — create, query, store results."""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Artifact, Correction, Extraction, Job


async def create_job(
    session: AsyncSession,
    *,
    input_type: str,
    original_filename: str | None,
    source_path: str | None,
    sha256: str | None,
    pipeline: str = "ocr+extract",
    engine_ocr: str = "tesseract",
    engine_llm: str = "llama_cpp",
    lang: str = "rus+eng",
    schema_version: str = "v1",
) -> Job:
    job = Job(
        id=uuid.uuid4(),
        input_type=input_type,
        original_filename=original_filename,
        source_path=source_path,
        sha256=sha256,
        pipeline=pipeline,
        engine_ocr=engine_ocr,
        engine_llm=engine_llm,
        lang=lang,
        schema_version=schema_version,
        status="queued",
        trace_id=str(uuid.uuid4()),
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


async def get_job(session: AsyncSession, job_id: uuid.UUID) -> Job | None:
    return await session.get(Job, job_id)


async def get_job_with_relations(session: AsyncSession, job_id: uuid.UUID) -> Job | None:
    """Load job with artifact, extractions, and corrections eagerly."""
    q = (
        select(Job)
        .where(Job.id == job_id)
        .options(
            selectinload(Job.artifact),
            selectinload(Job.extractions),
            selectinload(Job.corrections),
        )
    )
    result = await session.execute(q)
    return result.scalar_one_or_none()


async def list_jobs(
    session: AsyncSession,
    *,
    status: str | None = None,
    engine_ocr: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Job]:
    q = select(Job).order_by(Job.created_at.desc()).offset(offset).limit(limit)
    if status:
        q = q.where(Job.status == status)
    if engine_ocr:
        q = q.where(Job.engine_ocr == engine_ocr)
    result = await session.execute(q)
    return list(result.scalars().all())


async def update_job(session: AsyncSession, job: Job, **kwargs) -> Job:
    for k, v in kwargs.items():
        setattr(job, k, v)
    await session.commit()
    await session.refresh(job)
    return job


async def store_artifact(session: AsyncSession, job_id: uuid.UUID, **paths: str | None) -> Artifact:
    art = Artifact(job_id=job_id, **paths)
    session.add(art)
    await session.commit()
    return art


async def store_extraction(
    session: AsyncSession,
    job_id: uuid.UUID,
    *,
    status: str,
    json_validated: dict | None,
    json_raw: str,
    confidence: float | None,
    warnings: list | None,
    schema_version: str = "v1",
) -> Extraction:
    ext = Extraction(
        job_id=job_id,
        schema_version=schema_version,
        status=status,
        json_validated=json_validated,
        json_raw=json_raw,
        confidence=confidence,
        warnings=warnings or [],
    )
    session.add(ext)
    await session.commit()
    return ext


async def submit_correction(
    session: AsyncSession,
    job_id: uuid.UUID,
    payload: dict,
    comment: str | None,
    submitted_by: str | None,
) -> Correction:
    last_q = (
        select(Correction)
        .where(Correction.job_id == job_id)
        .order_by(Correction.version.desc())
        .limit(1)
    )
    result = await session.execute(last_q)
    last = result.scalar_one_or_none()

    prev_payload = last.payload_json if last else None
    new_version = (last.version + 1) if last else 1

    corr = Correction(
        job_id=job_id,
        payload_json=payload,
        previous_payload_json=prev_payload,
        comment=comment,
        submitted_by=submitted_by,
        version=new_version,
    )
    session.add(corr)
    await session.commit()
    await session.refresh(corr)
    return corr


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
