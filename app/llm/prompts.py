"""System and user prompt builders for LLM extraction."""
from __future__ import annotations

SYSTEM_PROMPT = """\
You are a structured data extraction assistant.
You receive OCR text from Russian bank guarantee documents.
You MUST output a single JSON object and NOTHING ELSE — no markdown, no commentary, no extra keys.
Rules:
- Use ISO date format YYYY-MM-DD for all dates.
- Use decimal numbers with a dot (e.g. 1234567.89).
- If a field value is unknown or not present in the text, set it to null.
- Do NOT invent or hallucinate values.
- INN must be digits only (10 or 12 digits).
- IKZ must be digits only.
- Currency should be a 3-letter code: RUB, USD, EUR, etc. Map Russian names (e.g. "рублей" → "RUB").
- Return ONLY the JSON object.\
"""

JSON_TEMPLATE = """\
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


def build_user_prompt(ocr_text: str) -> str:
    return (
        "Extract structured fields from the following OCR text of a Russian bank guarantee.\n\n"
        "JSON template (fill in values or leave null):\n"
        f"```json\n{JSON_TEMPLATE}\n```\n\n"
        "OCR text:\n"
        "---\n"
        f"{ocr_text}\n"
        "---\n\n"
        "Return ONLY the JSON object."
    )


def build_retry_prompt(ocr_text: str, previous_errors: list[str]) -> str:
    errors_str = "\n".join(f"- {e}" for e in previous_errors)
    return (
        "Your previous response had validation errors:\n"
        f"{errors_str}\n\n"
        "Please fix these errors and try again.\n\n"
        "OCR text:\n"
        "---\n"
        f"{ocr_text}\n"
        "---\n\n"
        "Return ONLY the valid JSON object."
    )


def build_llm_input(ocr_md: str, layout_json: dict | None = None) -> str:
    """Build the text payload sent to the LLM.  Default: just the markdown."""
    return ocr_md
