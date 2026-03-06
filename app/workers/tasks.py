"""arq worker tasks — OCR + LLM processing runs here, never in API handlers."""
from __future__ import annotations

import uuid
from pathlib import Path

from arq import cron
from arq.connections import RedisSettings

from app.core.config import settings
from app.core.logging import get_logger, setup_logging

log = get_logger(__name__)


async def startup(ctx: dict) -> None:
    setup_logging(settings.DEBUG)
    log.info("Worker starting up …")

    from app.db.session import async_session_factory
    ctx["session_factory"] = async_session_factory

    from app.llm.engine import get_engine
    engine = get_engine()
    try:
        engine.load()
    except Exception:
        log.warning("LLM model not found at %s — extraction will be unavailable", settings.LLM_MODEL_PATH)
    ctx["llm_engine"] = engine


async def shutdown(ctx: dict) -> None:
    log.info("Worker shutting down …")
    engine = ctx.get("llm_engine")
    if engine:
        engine.unload()


async def process_job(ctx: dict, job_id_str: str) -> None:
    """Main job processing task."""
    from app.services.jobs import get_job, store_artifact, store_extraction, update_job
    from app.services.pipeline import run_full_pipeline

    job_id = uuid.UUID(job_id_str)
    session_factory = ctx["session_factory"]
    llm_engine = ctx.get("llm_engine")

    async with session_factory() as session:
        job = await get_job(session, job_id)
        if not job:
            log.error("Job %s not found in DB", job_id)
            return

        await update_job(session, job, status="running")

        try:
            pdf_path = Path(job.source_path)  # type: ignore[arg-type]

            result = run_full_pipeline(
                pdf_path,
                engine_ocr_name=job.engine_ocr,
                lang=job.lang,
                pipeline=job.pipeline,
                llm_engine=llm_engine,
            )

            await update_job(
                session, job,
                status="succeeded",
                page_count=result["page_count"],
                progress_pages=result["page_count"],
                duration_ms_ocr=result["duration_ms_ocr"],
                duration_ms_llm=result.get("duration_ms_llm"),
                duration_ms_total=(result["duration_ms_ocr"] + (result.get("duration_ms_llm") or 0)),
            )

            await store_artifact(
                session, job_id,
                ocr_json_path=result["ocr_json_path"],
                ocr_md_path=result["ocr_md_path"],
                extraction_json_path=result.get("extraction_json_path"),
                meta_path=result.get("meta_path"),
            )

            ext_result = result.get("extraction_result")
            if ext_result:
                validated_dict = ext_result.validated.model_dump(mode="json") if ext_result.validated else None
                await store_extraction(
                    session, job_id,
                    status=ext_result.status,
                    json_validated=validated_dict,
                    json_raw=ext_result.raw,
                    confidence=ext_result.confidence,
                    warnings=ext_result.warnings,
                )

        except Exception as exc:
            log.exception("Job %s failed", job_id)
            await update_job(
                session, job,
                status="failed",
                error_code="PIPELINE_ERROR",
                error_message=str(exc)[:2000],
            )


class WorkerSettings:
    functions = [process_job]
    on_startup = startup
    on_shutdown = shutdown
    max_jobs = settings.WORKER_MAX_JOBS
    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)
    job_timeout = 600
