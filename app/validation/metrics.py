"""Metric computation for the validation subsystem.

All functions operate on normalised values (apply normalize_field before calling).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field

import numpy as np
from rapidfuzz.distance import Levenshtein

from app.validation.normalize import (
    AMOUNT_FIELDS,
    CURRENCY_FIELDS,
    DATE_FIELDS,
    EVALUATED_GT_FIELDS,
    GT_TO_PRED_FIELD,
    STRING_ID_FIELDS,
)

# ---------------------------------------------------------------------------
# Elementary helpers
# ---------------------------------------------------------------------------

def exact_match(pred, gold) -> bool:
    if pred is None and gold is None:
        return True
    if pred is None or gold is None:
        return False
    return str(pred) == str(gold)


def normalized_levenshtein_similarity(pred: str | None, gold: str | None) -> float:
    if pred is None and gold is None:
        return 1.0
    if pred is None or gold is None:
        return 0.0
    p, g = str(pred), str(gold)
    if not p and not g:
        return 1.0
    max_len = max(len(p), len(g))
    dist = Levenshtein.distance(p, g)
    return 1.0 - dist / max_len


def digit_accuracy(pred: str | None, gold: str | None) -> float:
    """Position-aligned digit accuracy for identifier fields."""
    import re
    p_digits = re.sub(r"\D", "", str(pred or ""))
    g_digits = re.sub(r"\D", "", str(gold or ""))
    if not g_digits and not p_digits:
        return 1.0
    if not g_digits or not p_digits:
        return 0.0
    max_len = max(len(p_digits), len(g_digits))
    matching = sum(a == b for a, b in zip(p_digits, g_digits))
    return matching / max_len


def amount_absolute_error(pred: float | None, gold: float | None) -> float | None:
    if pred is None or gold is None:
        return None
    return abs(pred - gold)


def amount_relative_error(pred: float | None, gold: float | None) -> float | None:
    if pred is None or gold is None:
        return None
    if gold == 0:
        return None
    return abs(pred - gold) / abs(gold)


def amount_within_tolerance(pred: float | None, gold: float | None, eps: float = 0.01) -> bool:
    if pred is None or gold is None:
        return pred is None and gold is None
    return abs(pred - gold) <= eps


# ---------------------------------------------------------------------------
# Per-document diagnostics
# ---------------------------------------------------------------------------

@dataclass
class FieldDiag:
    field: str
    pred_value: object = None
    gold_value: object = None
    is_match: bool = False
    edit_similarity: float | None = None
    digit_acc: float | None = None
    abs_error: float | None = None
    rel_error: float | None = None
    within_tolerance: bool | None = None
    error_type: str = "correct"  # correct | missing | wrong | extra


def compute_field_diagnostics(
    field_gt: str,
    pred_val,
    gold_val,
    tolerance: float = 0.01,
) -> FieldDiag:
    """Compute all diagnostics for a single (field, document) pair."""
    field_pred = GT_TO_PRED_FIELD.get(field_gt, field_gt)
    d = FieldDiag(field=field_gt, pred_value=pred_val, gold_value=gold_val)
    d.is_match = exact_match(pred_val, gold_val)

    if d.is_match:
        d.error_type = "correct"
    elif gold_val is None and pred_val is not None:
        d.error_type = "extra"
    elif gold_val is not None and pred_val is None:
        d.error_type = "missing"
    else:
        d.error_type = "wrong"

    if field_pred in STRING_ID_FIELDS or field_gt in STRING_ID_FIELDS:
        d.edit_similarity = normalized_levenshtein_similarity(
            str(pred_val) if pred_val is not None else None,
            str(gold_val) if gold_val is not None else None,
        )
        d.digit_acc = digit_accuracy(pred_val, gold_val)

    if field_gt in DATE_FIELDS or field_pred in DATE_FIELDS:
        d.edit_similarity = normalized_levenshtein_similarity(
            str(pred_val) if pred_val is not None else None,
            str(gold_val) if gold_val is not None else None,
        )

    if field_gt in AMOUNT_FIELDS or field_pred in AMOUNT_FIELDS:
        p = float(pred_val) if pred_val is not None else None
        g = float(gold_val) if gold_val is not None else None
        d.abs_error = amount_absolute_error(p, g)
        d.rel_error = amount_relative_error(p, g)
        d.within_tolerance = amount_within_tolerance(p, g, tolerance)

    return d


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

@dataclass
class SlotCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class AggregateMetrics:
    field_accuracy: dict[str, float] = dc_field(default_factory=dict)
    micro_accuracy: float = 0.0
    macro_accuracy: float = 0.0

    field_slot: dict[str, SlotCounts] = dc_field(default_factory=dict)
    micro_slot: SlotCounts = dc_field(default_factory=SlotCounts)
    micro_f1: float = 0.0
    macro_f1: float = 0.0

    doc_em: float = 0.0
    n_docs: int = 0

    avg_edit_sim: dict[str, float] = dc_field(default_factory=dict)
    avg_digit_acc: dict[str, float] = dc_field(default_factory=dict)
    amount_mae: float | None = None
    amount_tolerance_acc: float | None = None

    latency_ocr_ms: dict[str, float] = dc_field(default_factory=dict)
    latency_extract_ms: dict[str, float] = dc_field(default_factory=dict)
    latency_total_ms: dict[str, float] = dc_field(default_factory=dict)

    error_counts: dict[str, int] = dc_field(default_factory=dict)

    weighted_accuracy: float | None = None
    weighted_doc_score_mean: float | None = None


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"median": 0, "p95": 0, "p99": 0, "mean": 0}
    a = np.array(values, dtype=float)
    return {
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "mean": float(np.mean(a)),
    }


def compute_aggregate_metrics(
    doc_results: list[dict],
    evaluated_fields: list[str] | None = None,
    weights: dict[str, float] | None = None,
    wrong_counts_as_fn: bool = True,
    tolerance: float = 0.01,
) -> AggregateMetrics:
    """Compute all aggregate metrics from a list of per-document result dicts.

    Each dict in *doc_results* must contain:
      - "diags": list[FieldDiag] (or list of dicts with same keys)
      - "ocr_ms", "extract_ms", "total_ms" (floats)
    """
    if evaluated_fields is None:
        evaluated_fields = EVALUATED_GT_FIELDS

    m = AggregateMetrics()
    m.n_docs = len(doc_results)

    field_matches: dict[str, list[bool]] = {f: [] for f in evaluated_fields}
    field_slot: dict[str, SlotCounts] = {f: SlotCounts() for f in evaluated_fields}
    field_edit_sim: dict[str, list[float]] = {f: [] for f in evaluated_fields}
    field_digit_acc: dict[str, list[float]] = {f: [] for f in evaluated_fields}
    amount_errors: list[float] = []
    amount_tol: list[bool] = []
    doc_ems: list[bool] = []
    ocr_times: list[float] = []
    extract_times: list[float] = []
    total_times: list[float] = []
    error_counter: dict[str, int] = {"correct": 0, "missing": 0, "wrong": 0, "extra": 0}
    weighted_doc_scores: list[float] = []

    for doc in doc_results:
        diags = doc.get("diags", [])
        diag_map: dict[str, FieldDiag | dict] = {}
        for d in diags:
            key = d.field if isinstance(d, FieldDiag) else d["field"]
            diag_map[key] = d

        all_match = True
        doc_weighted_num = 0.0
        doc_weighted_den = 0.0

        for f in evaluated_fields:
            d = diag_map.get(f)
            if d is None:
                field_matches[f].append(False)
                all_match = False
                continue

            is_match = d.is_match if isinstance(d, FieldDiag) else d["is_match"]
            error_type = d.error_type if isinstance(d, FieldDiag) else d["error_type"]

            field_matches[f].append(is_match)
            if not is_match:
                all_match = False

            error_counter[error_type] = error_counter.get(error_type, 0) + 1

            sc = field_slot[f]
            if is_match:
                gold = d.gold_value if isinstance(d, FieldDiag) else d["gold_value"]
                if gold is not None:
                    sc.tp += 1
            elif error_type == "missing":
                sc.fn += 1
            elif error_type == "extra":
                sc.fp += 1
            elif error_type == "wrong":
                sc.fp += 1
                if wrong_counts_as_fn:
                    sc.fn += 1

            edit_sim = d.edit_similarity if isinstance(d, FieldDiag) else d.get("edit_similarity")
            if edit_sim is not None:
                field_edit_sim[f].append(edit_sim)

            digit_a = d.digit_acc if isinstance(d, FieldDiag) else d.get("digit_acc")
            if digit_a is not None:
                field_digit_acc[f].append(digit_a)

            abs_err = d.abs_error if isinstance(d, FieldDiag) else d.get("abs_error")
            if abs_err is not None:
                amount_errors.append(abs_err)
            within = d.within_tolerance if isinstance(d, FieldDiag) else d.get("within_tolerance")
            if within is not None:
                amount_tol.append(within)

            if weights and f in weights:
                doc_weighted_num += weights[f] * (1.0 if is_match else 0.0)
                doc_weighted_den += weights[f]

        doc_ems.append(all_match)
        if doc_weighted_den > 0:
            weighted_doc_scores.append(doc_weighted_num / doc_weighted_den)

        ocr_times.append(doc.get("ocr_ms", 0))
        extract_times.append(doc.get("extract_ms", 0))
        total_times.append(doc.get("total_ms", 0))

    for f in evaluated_fields:
        vals = field_matches[f]
        m.field_accuracy[f] = sum(vals) / len(vals) if vals else 0.0

    all_matches = [v for f in evaluated_fields for v in field_matches[f]]
    m.micro_accuracy = sum(all_matches) / len(all_matches) if all_matches else 0.0
    m.macro_accuracy = (
        sum(m.field_accuracy[f] for f in evaluated_fields) / len(evaluated_fields)
        if evaluated_fields else 0.0
    )

    m.field_slot = field_slot
    micro = SlotCounts()
    for sc in field_slot.values():
        micro.tp += sc.tp
        micro.fp += sc.fp
        micro.fn += sc.fn
    m.micro_slot = micro
    m.micro_f1 = micro.f1
    m.macro_f1 = (
        sum(sc.f1 for sc in field_slot.values()) / len(field_slot) if field_slot else 0.0
    )

    m.doc_em = sum(doc_ems) / len(doc_ems) if doc_ems else 0.0

    for f in evaluated_fields:
        if field_edit_sim[f]:
            m.avg_edit_sim[f] = sum(field_edit_sim[f]) / len(field_edit_sim[f])
        if field_digit_acc[f]:
            m.avg_digit_acc[f] = sum(field_digit_acc[f]) / len(field_digit_acc[f])

    if amount_errors:
        m.amount_mae = sum(amount_errors) / len(amount_errors)
    if amount_tol:
        m.amount_tolerance_acc = sum(amount_tol) / len(amount_tol)

    m.latency_ocr_ms = _percentiles(ocr_times)
    m.latency_extract_ms = _percentiles(extract_times)
    m.latency_total_ms = _percentiles(total_times)

    m.error_counts = error_counter

    if weights and weighted_doc_scores:
        m.weighted_doc_score_mean = sum(weighted_doc_scores) / len(weighted_doc_scores)
        w_total_num = sum(
            weights.get(f, 0) * sum(field_matches[f])
            for f in evaluated_fields if f in weights
        )
        w_total_den = sum(
            weights.get(f, 0) * len(field_matches[f])
            for f in evaluated_fields if f in weights
        )
        m.weighted_accuracy = w_total_num / w_total_den if w_total_den > 0 else 0.0

    return m
