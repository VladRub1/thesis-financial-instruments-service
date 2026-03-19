"""Pydantic models for LLM extraction output — schema v1."""
from __future__ import annotations

import re
from datetime import date

from pydantic import BaseModel, Field, field_validator, model_validator


class ExtractionV1(BaseModel):
    model_config = {"extra": "forbid"}

    guarantee_number: str | None = None
    issue_date: date | None = None
    start_date: date | None = None
    end_date: date | None = None
    amount: float | None = None
    currency: str | None = None
    principal_inn: str | None = None
    beneficiary_inn: str | None = None
    contract_number: str | None = None
    contract_date: date | None = None
    contract_name: str | None = None
    ikz: str | None = None

    bank_name: str | None = None
    bank_bic: str | None = None
    registry_number: str | None = None
    claim_period_days: int | None = None
    signatures_present: bool | None = None

    schema_version: str = "v1"
    extraction_confidence: float | None = Field(None, ge=0, le=1)
    evidence: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("principal_inn", "beneficiary_inn")
    @classmethod
    def validate_inn(cls, v: str | None) -> str | None:
        if v is None:
            return v
        digits = re.sub(r"\D", "", v)
        if len(digits) not in (10, 12):
            raise ValueError(f"INN must be 10 or 12 digits, got {len(digits)}: {v}")
        return digits

    @field_validator("ikz")
    @classmethod
    def validate_ikz(cls, v: str | None) -> str | None:
        if v is None:
            return v
        digits = re.sub(r"\D", "", v)
        if not digits:
            raise ValueError("IKZ must contain digits")
        return digits

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("Amount must be non-negative")
        return v

    @model_validator(mode="after")
    def check_date_order(self) -> ExtractionV1:
        dates = [
            ("issue_date", self.issue_date),
            ("start_date", self.start_date),
            ("end_date", self.end_date),
        ]
        present = [(n, d) for n, d in dates if d is not None]
        for i in range(len(present) - 1):
            n1, d1 = present[i]
            n2, d2 = present[i + 1]
            if d1 > d2:
                self.warnings.append(f"{n1} ({d1}) is after {n2} ({d2})")
        return self


class ExtractionResult(BaseModel):
    status: str  # succeeded | failed_validation | failed_runtime
    validated: ExtractionV1 | None = None
    raw: str = ""
    errors: list[str] = Field(default_factory=list)
    confidence: float | None = None
    warnings: list[str] = Field(default_factory=list)


def extraction_json_schema() -> dict:
    """Return a JSON Schema compatible with llama-cpp-python constrained decoding.

    Pydantic emits ``date`` fields with ``{"type": "string", "format": "date"}``.
    llama.cpp grammar understands ``"format": "date"`` natively, so no patching needed.
    """
    return ExtractionV1.model_json_schema()
