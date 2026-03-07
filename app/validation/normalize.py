"""Normalization rules for ground-truth and prediction fields.

Applied consistently to both gold and predicted values before any metric is computed.
"""
from __future__ import annotations

import re
from datetime import date, datetime

_CURRENCY_MAP: dict[str, str] = {
    "руб": "RUB", "руб.": "RUB", "рублей": "RUB", "рубль": "RUB", "рубли": "RUB",
    "российских рублей": "RUB", "рур": "RUB", "rur": "RUB", "rub": "RUB", "₽": "RUB",
    "usd": "USD", "долларов": "USD", "$": "USD", "доллар": "USD",
    "eur": "EUR", "евро": "EUR", "€": "EUR",
}

GT_TO_PRED_FIELD: dict[str, str] = {
    "pcpl_inn": "principal_inn",
    "bene_inn": "beneficiary_inn",
    "issue_date": "issue_date",
    "start_date": "start_date",
    "end_date": "end_date",
    "sum": "amount",
    "currency": "currency",
    "ikz": "ikz",
}

# bank_inn is GT-only metadata used for grouping analysis, not an extracted/evaluated field

EVALUATED_GT_FIELDS: list[str] = [
    "pcpl_inn", "bene_inn", "issue_date", "start_date", "end_date", "sum", "currency", "ikz",
]

EVALUATED_PRED_FIELDS: list[str] = [GT_TO_PRED_FIELD[f] for f in EVALUATED_GT_FIELDS]

STRING_ID_FIELDS = {"pcpl_inn", "bene_inn", "bank_inn", "principal_inn", "beneficiary_inn", "ikz"}
DATE_FIELDS = {"issue_date", "start_date", "end_date"}
AMOUNT_FIELDS = {"sum", "amount"}
CURRENCY_FIELDS = {"currency"}


def _to_str_or_none(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and v != v:  # NaN
        return None
    s = str(v).strip()
    return s if s else None


def normalize_inn(v) -> str | None:
    s = _to_str_or_none(v)
    if s is None:
        return None
    digits = re.sub(r"\D", "", s)
    return digits if digits else None


def normalize_ikz(v) -> str | None:
    s = _to_str_or_none(v)
    if s is None:
        return None
    digits = re.sub(r"\D", "", s)
    return digits if digits else None


def normalize_date(v) -> str | None:
    """Normalize to YYYY-MM-DD string."""
    if v is None:
        return None
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, datetime):
        return v.date().isoformat()
    s = str(v).strip()
    if not s or s.lower() in ("nat", "nan", "none", ""):
        return None
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S", "%d.%m.%y"):
        try:
            return datetime.strptime(s[:10], fmt).date().isoformat()
        except ValueError:
            continue
    return s[:10]


def normalize_amount(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        if v != v:  # NaN
            return None
        return round(float(v), 2)
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none"):
        return None
    s = re.sub(r"[^\d.,\-]", "", s)
    s = s.replace(" ", "")
    if "," in s and "." in s:
        s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def normalize_currency(v) -> str | None:
    s = _to_str_or_none(v)
    if s is None:
        return None
    key = s.lower().strip().rstrip(".")
    if key in _CURRENCY_MAP:
        return _CURRENCY_MAP[key]
    upper = s.upper().strip()
    return upper if upper else None


def normalize_field(field_name: str, value) -> str | float | None:
    """Normalize a single field value based on field name (works for both GT and pred names)."""
    if field_name in ("pcpl_inn", "bene_inn", "bank_inn", "principal_inn", "beneficiary_inn"):
        return normalize_inn(value)
    if field_name == "ikz":
        return normalize_ikz(value)
    if field_name in DATE_FIELDS:
        return normalize_date(value)
    if field_name in AMOUNT_FIELDS:
        return normalize_amount(value)
    if field_name in CURRENCY_FIELDS:
        return normalize_currency(value)
    return _to_str_or_none(value)


def normalize_gold_row(row: dict) -> dict:
    """Normalize a ground-truth row (GT field names)."""
    return {f: normalize_field(f, row.get(f)) for f in GT_TO_PRED_FIELD}


def normalize_pred_row(row: dict) -> dict:
    """Normalize a prediction row (pred field names)."""
    return {GT_TO_PRED_FIELD[f]: normalize_field(GT_TO_PRED_FIELD[f], row.get(GT_TO_PRED_FIELD[f]))
            for f in GT_TO_PRED_FIELD}
