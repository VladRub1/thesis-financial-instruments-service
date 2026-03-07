"""Regex-based baseline extractor — no ML/LLM, pure heuristics on OCR text."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_INN_RE = re.compile(r"\b(\d{10}|\d{12})\b")
_DATE_RE = re.compile(
    r"(?:«?\s*(\d{1,2})\s*»?\s*[./]\s*(\d{1,2})\s*[./]\s*(\d{4}))"
    r"|(?:(\d{1,2})\s*[./]\s*(\d{1,2})\s*[./]\s*(\d{4}))"
)
_AMOUNT_RE = re.compile(
    r"(\d[\d\s]*[\d](?:[.,]\d{1,2})?)\s*(?:руб|рублей|₽|RUB|руб\.|рубль)",
    re.IGNORECASE,
)
_AMOUNT_FALLBACK_RE = re.compile(
    r"(?:сумм[аыеой]|размер[еа]?|составляет)\s*[:.]?\s*(\d[\d\s]*[\d](?:[.,]\d{1,2})?)",
    re.IGNORECASE,
)
_IKZ_RE = re.compile(r"(?:ИКЗ|икз|I[KК]З)\s*[:.]?\s*(\d{20,40})", re.IGNORECASE)
_CURRENCY_RE = re.compile(r"(руб|рублей|рубль|руб\.|₽|RUB|USD|EUR|доллар|евро)", re.IGNORECASE)

_PRINCIPAL_KEYWORDS = re.compile(
    r"(принципал|заказчик|поставщик|подрядчик|исполнител)", re.IGNORECASE
)
_BENEFICIARY_KEYWORDS = re.compile(
    r"(бенефициар|заявител|получател|государственн.*заказчик)", re.IGNORECASE
)

_CURRENCY_MAP = {
    "руб": "RUB", "руб.": "RUB", "рублей": "RUB", "рубль": "RUB", "₽": "RUB", "rub": "RUB",
    "usd": "USD", "доллар": "USD", "eur": "EUR", "евро": "EUR",
}


def _clean_amount(s: str) -> float | None:
    s = re.sub(r"\s", "", s)
    s = s.replace(",", ".")
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def _format_date(d: int, m: int, y: int) -> str | None:
    if not (1 <= d <= 31 and 1 <= m <= 12 and 1900 <= y <= 2100):
        return None
    return f"{y:04d}-{m:02d}-{d:02d}"


def _find_dates(text: str) -> list[tuple[str, int]]:
    """Return (iso_date, char_position) pairs."""
    results = []
    for m in _DATE_RE.finditer(text):
        groups = m.groups()
        if groups[0] is not None:
            d, mo, y = int(groups[0]), int(groups[1]), int(groups[2])
        else:
            d, mo, y = int(groups[3]), int(groups[4]), int(groups[5])
        iso = _format_date(d, mo, y)
        if iso:
            results.append((iso, m.start()))
    return results


def _find_inns_with_context(text: str) -> list[tuple[str, int, str]]:
    """Return (inn_value, position, context_snippet) for each INN match."""
    results = []
    for m in _INN_RE.finditer(text):
        ctx_start = max(0, m.start() - 80)
        ctx_end = min(len(text), m.end() + 20)
        ctx = text[ctx_start:ctx_end].lower()
        results.append((m.group(1), m.start(), ctx))
    return results


def _classify_inn(
    inns: list[tuple[str, int, str]],
) -> tuple[str | None, str | None, str | None]:
    """Classify INNs into (principal, beneficiary, bank) using keyword proximity."""
    principal = None
    beneficiary = None
    bank = None
    for inn_val, _pos, ctx in inns:
        if "банк" in ctx or "гарант" in ctx.split("инн")[0][-40:] if "инн" in ctx else False:
            if bank is None:
                bank = inn_val
                continue
        if _BENEFICIARY_KEYWORDS.search(ctx):
            if beneficiary is None:
                beneficiary = inn_val
                continue
        if _PRINCIPAL_KEYWORDS.search(ctx):
            if principal is None:
                principal = inn_val
                continue
    unassigned = [inn for inn, _, _ in inns if inn not in (principal, beneficiary, bank)]
    if principal is None and unassigned:
        principal = unassigned.pop(0)
    if beneficiary is None and unassigned:
        beneficiary = unassigned.pop(0)
    return principal, beneficiary, bank


def extract_regex(ocr_text: str) -> dict:
    """Run regex baseline extraction on OCR text, returning the same schema as LLM extractor."""
    inns = _find_inns_with_context(ocr_text)
    principal_inn, beneficiary_inn, bank_inn = _classify_inn(inns)

    dates = _find_dates(ocr_text)

    amount = None
    for pattern in (_AMOUNT_RE, _AMOUNT_FALLBACK_RE):
        m = pattern.search(ocr_text)
        if m:
            amount = _clean_amount(m.group(1))
            if amount is not None:
                break

    currency = None
    cm = _CURRENCY_RE.search(ocr_text)
    if cm:
        currency = _CURRENCY_MAP.get(cm.group(1).lower().rstrip("."), cm.group(1).upper())

    ikz = None
    ikz_m = _IKZ_RE.search(ocr_text)
    if ikz_m:
        ikz = ikz_m.group(1)

    issue_date = dates[0][0] if len(dates) > 0 else None
    start_date = dates[1][0] if len(dates) > 1 else None
    end_date = dates[2][0] if len(dates) > 2 else None

    return {
        "guarantee_number": None,
        "issue_date": issue_date,
        "start_date": start_date,
        "end_date": end_date,
        "amount": amount,
        "currency": currency,
        "principal_inn": principal_inn,
        "beneficiary_inn": beneficiary_inn,
        "contract_number": None,
        "contract_date": None,
        "contract_name": None,
        "ikz": ikz,
        "bank_name": None,
        "bank_inn": bank_inn,
        "claim_period_days": None,
        "schema_version": "v1",
        "extraction_confidence": None,
        "evidence": {},
        "warnings": [],
    }
