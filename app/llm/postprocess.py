"""Lightweight post-processing rules applied after validated extraction."""
from __future__ import annotations

import re

_CURRENCY_MAP = {
    "руб": "RUB", "руб.": "RUB", "рублей": "RUB", "рубль": "RUB", "российских рублей": "RUB",
    "rub": "RUB", "₽": "RUB",
    "usd": "USD", "долларов": "USD", "$": "USD",
    "eur": "EUR", "евро": "EUR", "€": "EUR",
}


def postprocess(data: dict) -> dict:
    """Apply normalisation rules in-place and return the dict."""
    for key in ("principal_inn", "beneficiary_inn", "bank_inn", "ikz"):
        if data.get(key):
            data[key] = re.sub(r"\D", "", data[key])

    raw_cur = (data.get("currency") or "").strip().lower()
    if raw_cur in _CURRENCY_MAP:
        data["currency"] = _CURRENCY_MAP[raw_cur]
    elif raw_cur:
        data["currency"] = raw_cur.upper()

    for key in data:
        if isinstance(data[key], str):
            data[key] = " ".join(data[key].split())

    return data
