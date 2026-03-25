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
from app.llm.schemas import ExtractionResult, extraction_json_schema, get_extraction_model

log = get_logger(__name__)

_DEFAULT_SCHEMA_VERSION = "v2"
_APPROX_CHARS_PER_TOKEN = 3
_PROMPT_OVERHEAD_TOKENS = 512
_MIN_INPUT_CHARS = 1200


def _effective_schema_version(schema_version: str) -> str:
    return schema_version if schema_version in ("v1", "v2") else _DEFAULT_SCHEMA_VERSION


def _budget_input_for_context(ocr_text: str, trace_id: str | None = None) -> str:
    """Trim oversized OCR input so completion has enough context/output room."""
    reserve_tokens = max(256, settings.LLM_MAX_TOKENS + _PROMPT_OVERHEAD_TOKENS)
    input_tokens_budget = max(512, settings.LLM_N_CTX - reserve_tokens)
    max_chars = max(_MIN_INPUT_CHARS, input_tokens_budget * _APPROX_CHARS_PER_TOKEN)

    if len(ocr_text) <= max_chars:
        return ocr_text

    head_chars = int(max_chars * 0.8)
    tail_chars = max(0, max_chars - head_chars)
    trimmed = ocr_text[:head_chars]
    if tail_chars > 0:
        trimmed += "\n\n...[TRUNCATED FOR CONTEXT BUDGET]...\n\n" + ocr_text[-tail_chars:]

    log_prefix = f"[{trace_id}] " if trace_id else ""
    log.warning(
        "%sOCR input truncated for context budget: %d -> %d chars (n_ctx=%d, max_tokens=%d)",
        log_prefix, len(ocr_text), len(trimmed), settings.LLM_N_CTX, settings.LLM_MAX_TOKENS,
    )
    return trimmed


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


def _json_candidates(raw: str) -> list[str]:
    json_str = _extract_json_block(raw)
    out = [json_str]

    # Common malformed-tail cleanup for almost-valid responses.
    cleaned = re.sub(r",\s*([}\]])", r"\1", json_str)
    if cleaned != json_str:
        out.append(cleaned)

    balance = cleaned.count("{") - cleaned.count("}")
    if balance > 0:
        out.append(cleaned + ("}" * balance))

    # Preserve order but remove duplicates.
    seen: set[str] = set()
    uniq: list[str] = []
    for c in out:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def _try_parse(raw: str, schema_version: str = _DEFAULT_SCHEMA_VERSION) -> tuple[dict | None, list[str]]:
    schema_version = _effective_schema_version(schema_version)
    model_cls = get_extraction_model(schema_version)
    data = None
    last_json_err: str | None = None

    for candidate in _json_candidates(raw):
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError as exc:
            last_json_err = f"JSON parse error: {exc}"

    if data is None:
        return None, [last_json_err or "JSON parse error"]

    if not isinstance(data, dict):
        return None, ["JSON root must be an object"]

    # Keep only schema keys to avoid failing on occasional extra fields.
    allowed_keys = set(model_cls.model_fields.keys())
    data = {k: v for k, v in data.items() if k in allowed_keys}
    if "schema_version" in allowed_keys and not data.get("schema_version"):
        data["schema_version"] = schema_version

    data = postprocess(data)
    try:
        validated = model_cls.model_validate(data)
        return validated.model_dump(mode="json"), []
    except ValidationError as exc:
        return None, [str(e) for e in exc.errors()]


def extract_fields(
    ocr_md: str,
    layout_json: dict | None = None,
    schema_version: str = _DEFAULT_SCHEMA_VERSION,
    engine: LLMEngine | None = None,
    trace_id: str | None = None,
) -> ExtractionResult:
    """Run LLM extraction with retry logic.  Returns ExtractionResult."""
    if engine is None:
        from app.llm.engine import get_engine
        engine = get_engine()

    schema_version = _effective_schema_version(schema_version)
    model_cls = get_extraction_model(schema_version)
    llm_input = build_llm_input(ocr_md, layout_json)
    llm_input = _budget_input_for_context(llm_input, trace_id=trace_id)
    user_prompt = build_user_prompt(llm_input, schema_version=schema_version)
    schema = extraction_json_schema(schema_version=schema_version)

    max_retries = settings.LLM_MAX_RETRIES
    all_errors: list[str] = []
    raw_output = ""
    log_prefix = f"[{trace_id}] " if trace_id else ""

    for attempt in range(1 + max_retries):
        log.info("%sLLM attempt %d/%d started", log_prefix, attempt + 1, 1 + max_retries)
        try:
            if attempt == 0:
                raw_output = engine.generate(
                    SYSTEM_PROMPT, user_prompt, json_schema=schema,
                )
            else:
                retry_prompt = build_retry_prompt(
                    llm_input,
                    all_errors,
                    schema_version=schema_version,
                )
                raw_output = engine.generate(
                    SYSTEM_PROMPT, retry_prompt, json_schema=schema,
                )
        except Exception as exc:
            log.exception("%sLLM inference error on attempt %d", log_prefix, attempt + 1)
            return ExtractionResult(
                status="failed_runtime",
                raw=raw_output,
                errors=[str(exc)],
            )

        data, errors = _try_parse(raw_output, schema_version=schema_version)
        if data is not None:
            validated = model_cls.model_validate(data)
            log.info("%sLLM attempt %d/%d succeeded", log_prefix, attempt + 1, 1 + max_retries)
            return ExtractionResult(
                status="succeeded",
                validated=validated,
                raw=raw_output,
                confidence=getattr(validated, "extraction_confidence", None),
                warnings=list(getattr(validated, "warnings", []) or []),
            )
        all_errors = errors
        log.warning("%sAttempt %d/%d failed validation: %s", log_prefix, attempt + 1, 1 + max_retries, errors)

    return ExtractionResult(
        status="failed_validation",
        raw=raw_output,
        errors=all_errors,
    )
