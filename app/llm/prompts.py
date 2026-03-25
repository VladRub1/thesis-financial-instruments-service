"""System and user prompt builders for LLM extraction."""
from __future__ import annotations

SYSTEM_PROMPT = """\
Extract fields from OCR text of Russian bank guarantees.
Return exactly ONE valid JSON object, no markdown, no prose, no extra keys.
Use YYYY-MM-DD dates, decimal dot for amount, and null when unknown.
Do not invent values. INN/IKZ must be digits only.
Currency should be a 3-letter code: RUB, USD, EUR, etc.\
"""

JSON_TEMPLATE_V2 = """\
{
  "guarantee_number": null,
  "issue_date": null,
  "start_date": null,
  "end_date": null,
  "amount": null,
  "currency": null,
  "principal_inn": null,
  "beneficiary_inn": null,
  "ikz": null,
  "bank_name": null,
  "bank_inn": null,
  "schema_version": "v2"
}\
"""


JSON_TEMPLATE_V1 = """\
{
  "guarantee_number": null,
  "issue_date": null,
  "start_date": null,
  "end_date": null,
  "amount": null,
  "currency": null,
  "principal_inn": null,
  "beneficiary_inn": null,
  "contract_number": null,
  "contract_date": null,
  "contract_name": null,
  "ikz": null,
  "bank_name": null,
  "bank_bic": null,
  "registry_number": null,
  "claim_period_days": null,
  "signatures_present": null,
  "schema_version": "v1",
  "extraction_confidence": null,
  "evidence": {},
  "warnings": []
}\
"""


def _template(schema_version: str) -> str:
    return JSON_TEMPLATE_V1 if schema_version == "v1" else JSON_TEMPLATE_V2


def build_user_prompt(ocr_text: str, *, schema_version: str = "v2") -> str:
    return (
        "Extract fields from OCR text.\n"
        "Output ONE valid JSON object only.\n\n"
        f"Schema template ({schema_version}):\n"
        f"```json\n{_template(schema_version)}\n```\n\n"
        "OCR text:\n"
        "---\n"
        f"{ocr_text}\n"
        "---"
    )


def build_retry_prompt(ocr_text: str, previous_errors: list[str], *, schema_version: str = "v2") -> str:
    errors_str = "\n".join(f"- {e}" for e in previous_errors)
    return (
        "Previous response is invalid.\n"
        "Fix all errors and return ONE valid JSON object only.\n\n"
        "Validation errors:\n"
        f"{errors_str}\n\n"
        f"Schema template ({schema_version}):\n"
        f"```json\n{_template(schema_version)}\n```\n\n"
        "OCR text:\n"
        "---\n"
        f"{ocr_text}\n"
        "---"
    )


def build_llm_input(ocr_md: str, layout_json: dict | None = None) -> str:
    """Build the text payload sent to the LLM.  Default: just the markdown."""
    return ocr_md
