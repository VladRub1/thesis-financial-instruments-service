"""FastAPI application entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.admin import router as admin_router
from app.api.v1.jobs import router as jobs_router
from app.core.config import settings
from app.core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.DEBUG)

    # Connect to Redis for arq job queue
    from arq import create_pool
    from arq.connections import RedisSettings

    try:
        app.state.arq_pool = await create_pool(RedisSettings.from_dsn(settings.REDIS_URL))
    except Exception:
        app.state.arq_pool = None

    yield

    if app.state.arq_pool:
        await app.state.arq_pool.close()


app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    description="Document AI PoC — OCR + LLM extraction for Russian bank guarantee scans",
    lifespan=lifespan,
)

app.include_router(jobs_router)
app.include_router(admin_router)


@app.get("/", include_in_schema=False)
async def root():
    return {"service": settings.APP_NAME, "docs": "/docs"}
