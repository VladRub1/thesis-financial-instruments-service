"""End-to-end extraction: OCR markdown → LLM → validated JSON."""
from __future__ import annotations

import json
import re

from pydantic import ValidationError

from app.core.config import settings
from app.core.logging import get_logger
from app.llm.engine import LLMEngine
from app.llm.postprocess import postprocess
from app.llm.prompts import SYSTEM_PROMPT, build_llm_input, build_retry_prompt, build_user_prompt
from app.llm.schemas import ExtractionResult, ExtractionV1, extraction_json_schema

log = get_logger(__name__)


def _extract_json_block(text: str) -> str:
    """Try to pull the first {...} block out of the LLM response."""
    # strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)

    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]
    return text.strip()


def _try_parse(raw: str) -> tuple[dict | None, list[str]]:
    json_str = _extract_json_block(raw)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return None, [f"JSON parse error: {exc}"]

    data = postprocess(data)
    try:
        validated = ExtractionV1.model_validate(data)
        return validated.model_dump(mode="json"), []
    except ValidationError as exc:
        return None, [str(e) for e in exc.errors()]


def extract_fields(
    ocr_md: str,
    layout_json: dict | None = None,
    schema_version: str = "v1",
    engine: LLMEngine | None = None,
) -> ExtractionResult:
    """Run LLM extraction with retry logic.  Returns ExtractionResult."""
    if engine is None:
        from app.llm.engine import get_engine
        engine = get_engine()

    llm_input = build_llm_input(ocr_md, layout_json)
    user_prompt = build_user_prompt(llm_input)
    schema = extraction_json_schema()

    max_retries = settings.LLM_MAX_RETRIES
    all_errors: list[str] = []
    raw_output = ""

    for attempt in range(1 + max_retries):
        try:
            if attempt == 0:
                raw_output = engine.generate(
                    SYSTEM_PROMPT, user_prompt, json_schema=schema,
                )
            else:
                retry_prompt = build_retry_prompt(llm_input, all_errors)
                raw_output = engine.generate(
                    SYSTEM_PROMPT, retry_prompt, json_schema=schema,
                )
        except Exception as exc:
            log.exception("LLM inference error on attempt %d", attempt + 1)
            return ExtractionResult(
                status="failed_runtime",
                raw=raw_output,
                errors=[str(exc)],
            )

        data, errors = _try_parse(raw_output)
        if data is not None:
            validated = ExtractionV1.model_validate(data)
            return ExtractionResult(
                status="succeeded",
                validated=validated,
                raw=raw_output,
                confidence=validated.extraction_confidence,
                warnings=validated.warnings,
            )
        all_errors = errors
        log.warning("Attempt %d/%d failed validation: %s", attempt + 1, 1 + max_retries, errors)

    return ExtractionResult(
        status="failed_validation",
        raw=raw_output,
        errors=all_errors,
    )
