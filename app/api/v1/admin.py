"""Admin endpoints — API-key protected."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.core.security import verify_admin_key
from app.schemas.jobs import HealthResponse, JobStatus, JobStatusResponse
from app.services.jobs import list_jobs

router = APIRouter(prefix="/v1/admin", tags=["admin"], dependencies=[Depends(verify_admin_key)])


@router.get("/jobs", response_model=list[JobStatusResponse], summary="List jobs (admin)")
async def admin_list_jobs(
    db: AsyncSession = Depends(get_db),
    status: str | None = Query(None),
    engine_ocr: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    jobs = await list_jobs(db, status=status, engine_ocr=engine_ocr, limit=limit, offset=offset)
    return [
        JobStatusResponse(
            job_id=j.id,
            status=JobStatus(j.status),
            progress_pages=j.progress_pages,
            page_count=j.page_count,
            created_at=j.created_at,
            updated_at=j.updated_at,
            error_code=j.error_code,
            error_message=j.error_message,
        )
        for j in jobs
    ]


@router.get("/health", response_model=HealthResponse, summary="Service health")
async def admin_health(request: Request, db: AsyncSession = Depends(get_db)):
    db_ok = False
    try:
        await db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    redis_ok = False
    try:
        pool = request.app.state.arq_pool
        if pool:
            await pool.ping()
            redis_ok = True
    except Exception:
        pass

    from pathlib import Path
    from app.core.config import settings
    llm_ok = Path(settings.LLM_MODEL_PATH).is_file()

    return HealthResponse(db=db_ok, redis=redis_ok, llm_model=llm_ok)
