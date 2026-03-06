"""Tests for correction submission."""
from __future__ import annotations

import io
import uuid

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_corrections_submit_success(client: AsyncClient):
    pdf_bytes = b"%PDF-1.4 fake"
    create_resp = await client.post(
        "/v1/jobs",
        files={"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={"pipeline": "ocr_only"},
    )
    job_id = create_resp.json()["job_id"]

    resp = await client.post(
        f"/v1/jobs/{job_id}/corrections",
        json={
            "fields": {"guarantee_number": "BG-CORRECTED-001", "amount": 500000},
            "comment": "Fixed number",
            "submitted_by": "tester",
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["version"] == 1


@pytest.mark.asyncio
async def test_corrections_invalid_payload_422(client: AsyncClient):
    pdf_bytes = b"%PDF-1.4 fake"
    create_resp = await client.post(
        "/v1/jobs",
        files={"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={"pipeline": "ocr_only"},
    )
    job_id = create_resp.json()["job_id"]
    resp = await client.post(f"/v1/jobs/{job_id}/corrections", json={"bad": "data"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_corrections_job_not_found(client: AsyncClient):
    fake_id = uuid.uuid4()
    resp = await client.post(
        f"/v1/jobs/{fake_id}/corrections",
        json={"fields": {"guarantee_number": "X"}, "comment": None, "submitted_by": None},
    )
    assert resp.status_code == 404
