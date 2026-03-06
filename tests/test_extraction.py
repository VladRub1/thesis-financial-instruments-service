"""Tests for LLM extraction parsing + validation."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from app.llm.extract import _extract_json_block, _try_parse, extract_fields
from app.llm.schemas import ExtractionV1


def test_extraction_v1_valid_minimal():
    data = ExtractionV1(schema_version="v1")
    assert data.guarantee_number is None
    assert data.schema_version == "v1"


def test_extraction_v1_inn_validation():
    with pytest.raises(Exception):
        ExtractionV1(principal_inn="123", schema_version="v1")
    v = ExtractionV1(principal_inn="1234567890", schema_version="v1")
    assert v.principal_inn == "1234567890"


def test_extraction_v1_amount_negative():
    with pytest.raises(Exception):
        ExtractionV1(amount=-100, schema_version="v1")


def test_extract_json_block_plain():
    raw = '{"key": "value"}'
    assert _extract_json_block(raw) == raw


def test_extract_json_block_markdown_fenced():
    raw = '```json\n{"key": "val"}\n```'
    assert json.loads(_extract_json_block(raw)) == {"key": "val"}


def test_try_parse_valid():
    raw = json.dumps({"guarantee_number": "BG-001", "schema_version": "v1"})
    data, errors = _try_parse(raw)
    assert data is not None
    assert errors == []
    assert data["guarantee_number"] == "BG-001"


def test_try_parse_invalid_json():
    data, errors = _try_parse("not json at all")
    assert data is None
    assert len(errors) > 0


def test_extraction_success_json_only():
    """Mock the LLM engine and verify extract_fields returns validated JSON."""
    valid_json = json.dumps({
        "guarantee_number": "BG-123",
        "amount": 1000000.0,
        "currency": "RUB",
        "schema_version": "v1",
    })
    mock_engine = MagicMock()
    mock_engine.generate.return_value = valid_json

    result = extract_fields("some ocr text", engine=mock_engine)
    assert result.status == "succeeded"
    assert result.validated is not None
    assert result.validated.guarantee_number == "BG-123"
    assert result.validated.amount == 1000000.0


def test_extraction_invalid_json_retries_then_fails():
    """LLM returns garbage every time — should fail after retries."""
    mock_engine = MagicMock()
    mock_engine.generate.return_value = "I am not JSON sorry"

    result = extract_fields("some ocr text", engine=mock_engine)
    assert result.status == "failed_validation"
    assert result.validated is None
    assert len(result.errors) > 0
    assert mock_engine.generate.call_count == 3  # 1 initial + 2 retries
