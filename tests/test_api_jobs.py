"""API tests for job endpoints."""
from __future__ import annotations

import io
import uuid

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_job_upload_pdf_success(client: AsyncClient):
    pdf_bytes = b"%PDF-1.4 fake"
    resp = await client.post(
        "/v1/jobs",
        files={"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={"pipeline": "ocr_only"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert "job_id" in body
    assert body["status"] == "queued"
    assert body["poll_url"].startswith("/v1/jobs/")


@pytest.mark.asyncio
async def test_create_job_invalid_extension_415(client: AsyncClient):
    resp = await client.post(
        "/v1/jobs",
        files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert resp.status_code == 415


@pytest.mark.asyncio
async def test_create_job_missing_payload_422(client: AsyncClient):
    resp = await client.post("/v1/jobs")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_job_not_found_404(client: AsyncClient):
    fake_id = uuid.uuid4()
    resp = await client.get(f"/v1/jobs/{fake_id}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_result_before_done_409(client: AsyncClient):
    pdf_bytes = b"%PDF-1.4 fake"
    create_resp = await client.post(
        "/v1/jobs",
        files={"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={"pipeline": "ocr_only"},
    )
    job_id = create_resp.json()["job_id"]
    resp = await client.get(f"/v1/jobs/{job_id}/result")
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_create_job_path_forbidden_no_roots(client: AsyncClient):
    import os
    os.environ["ALLOWED_INPUT_ROOTS"] = ""
    from app.core.config import Settings
    # Reset singleton for this test
    resp = await client.post("/v1/jobs/by-path", json={"file_path": "/etc/passwd"})
    assert resp.status_code in (403, 404)
    os.environ["ALLOWED_INPUT_ROOTS"] = "/tmp"
