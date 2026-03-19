"""Tests for LLM extraction parsing + validation."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from app.llm.extract import _extract_json_block, _try_parse, extract_fields
from app.llm.schemas import ExtractionV1, extraction_json_schema


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


# -- constrained-decoding & extra="forbid" tests ----------------------------


def test_extraction_v1_extra_forbid():
    """ExtractionV1 rejects unknown keys."""
    with pytest.raises(Exception):
        ExtractionV1(schema_version="v1", bogus_field="oops")


def test_extraction_json_schema_has_additional_properties_false():
    schema = extraction_json_schema()
    assert schema.get("additionalProperties") is False


def test_constrained_valid_json_succeeds():
    """Constrained decoding returns clean JSON — succeeds on first try."""
    valid = json.dumps({
        "guarantee_number": "БГ-123/2024",
        "issue_date": "2024-01-15",
        "amount": 500000.00,
        "currency": "RUB",
        "principal_inn": "7707083893",
        "schema_version": "v1",
    })
    engine = MagicMock()
    engine.generate.return_value = valid

    result = extract_fields("OCR text", engine=engine)
    assert result.status == "succeeded"
    assert result.validated.guarantee_number == "БГ-123/2024"
    assert result.raw == valid
    engine.generate.assert_called_once()
    _, kwargs = engine.generate.call_args
    assert kwargs["json_schema"] is not None


def test_constrained_malformed_json_preserves_raw():
    """Garbled output -> failed_validation with raw preserved."""
    garbled = '{"guarantee_number": "X", amount: BROKEN'
    engine = MagicMock()
    engine.generate.return_value = garbled

    result = extract_fields("OCR text", engine=engine)
    assert result.status == "failed_validation"
    assert result.validated is None
    assert result.raw == garbled
    assert any("JSON" in e or "parse" in e.lower() for e in result.errors)


def test_constrained_schema_invalid_inn_fails():
    """JSON is valid but INN has wrong length -> failed_validation."""
    bad_inn = json.dumps({
        "principal_inn": "123",
        "schema_version": "v1",
    })
    engine = MagicMock()
    engine.generate.return_value = bad_inn

    result = extract_fields("OCR text", engine=engine)
    assert result.status == "failed_validation"
    assert result.validated is None
    assert any("INN" in e or "inn" in e.lower() for e in result.errors)
