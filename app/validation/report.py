"""Markdown report generator for validation metrics."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.validation.metrics import AggregateMetrics, SlotCounts, compute_aggregate_metrics
from app.validation.normalize import EVALUATED_GT_FIELDS, GT_TO_PRED_FIELD
from app.validation.storage import read_run_metadata


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _f3(v: float | None) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "—"
    return f"{v:.3f}"


def _f1(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.1f}"


def _slot_row(name: str, sc: SlotCounts) -> str:
    return f"| {name} | {sc.tp} | {sc.fp} | {sc.fn} | {_f3(sc.precision)} | {_f3(sc.recall)} | {_f3(sc.f1)} |"


def _latency_section(label: str, stats: dict[str, float]) -> str:
    return (
        f"| {label} | {_f1(stats.get('median', 0))} | "
        f"{_f1(stats.get('p95', 0))} | {_f1(stats.get('p99', 0))} | "
        f"{_f1(stats.get('mean', 0))} |"
    )


# ---------------------------------------------------------------------------
# Build doc-level dicts from parquet rows for the metrics engine
# ---------------------------------------------------------------------------

def _nan_to_none(v):
    """Convert NaN/inf (from parquet float columns) to None."""
    if v is None:
        return None
    try:
        if math.isnan(v) or math.isinf(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _rows_to_doc_dicts(df: pd.DataFrame) -> list[dict]:
    """Convert parquet rows to the list[dict] expected by compute_aggregate_metrics."""
    docs: list[dict] = []
    for _, row in df.iterrows():
        gold_json = row.get("gold_json", "{}")
        pred_json = row.get("pred_json", "{}")
        gold = json.loads(gold_json) if isinstance(gold_json, str) else (gold_json or {})
        pred = json.loads(pred_json) if isinstance(pred_json, str) else (pred_json or {})

        diags: list[dict] = []
        for f in EVALUATED_GT_FIELDS:
            abs_err = _nan_to_none(row.get(f"abs_error_{f}"))
            within_tol = _nan_to_none(row.get(f"within_tol_{f}"))
            if within_tol is None and f in ("sum",):
                from app.validation.metrics import amount_within_tolerance
                from app.validation.normalize import normalize_field
                gv = normalize_field(f, gold.get(f))
                pv = normalize_field("amount", pred.get(f))
                if gv is not None or pv is not None:
                    within_tol = amount_within_tolerance(pv, gv)
            diags.append({
                "field": f,
                "is_match": bool(row.get(f"match_{f}", False)),
                "error_type": row.get(f"error_type_{f}", "missing"),
                "edit_similarity": _nan_to_none(row.get(f"edit_sim_{f}")),
                "digit_acc": _nan_to_none(row.get(f"digit_acc_{f}")),
                "abs_error": abs_err,
                "within_tolerance": within_tol,
                "gold_value": gold.get(f),
                "pred_value": pred.get(f),
            })

        docs.append({
            "diags": diags,
            "ocr_ms": float(row.get("ocr_ms", 0) or 0),
            "extract_ms": float(row.get("extract_ms", 0) or 0),
            "total_ms": float(row.get("total_ms", 0) or 0),
        })
    return docs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_and_report(
    run_ids: list[str],
    *,
    weights: dict[str, float] | None = None,
    wrong_counts_as_fn: bool = True,
    tolerance: float = 0.01,
) -> tuple[str, dict]:
    """Compute metrics for given runs and return (markdown_str, metrics_dict).

    Metrics are grouped by (ocr_engine, extractor).
    """
    from app.validation.storage import load_results

    all_dfs: list[pd.DataFrame] = []
    meta_list: list[dict] = []
    for rid in run_ids:
        df = load_results(rid)
        if df.empty:
            continue
        df["run_id"] = rid
        all_dfs.append(df)
        try:
            meta_list.append(read_run_metadata(rid))
        except FileNotFoundError:
            meta_list.append({"run_id": rid})

    if not all_dfs:
        return "# No results found\n", {}

    combined = pd.concat(all_dfs, ignore_index=True)

    groups = combined.groupby(["ocr_engine", "extractor"])

    sections: list[str] = []
    all_metrics: dict[str, dict] = {}

    sections.append("# Validation Report\n")
    sections.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")
    sections.append(f"Run IDs: {', '.join(run_ids)}\n")
    if meta_list:
        m0 = meta_list[0]
        sections.append(f"- Sample size: {m0.get('sample_size', '?')}")
        sections.append(f"- Seed: {m0.get('seed_value', '?')}")
        sections.append(f"- LLM model: {m0.get('llm_model', '—')}")
        sections.append("")

    for (ocr_eng, extr), group_df in groups:
        succeeded = group_df[group_df["status"] == "succeeded"]
        total = len(group_df)
        n_ok = len(succeeded)
        n_fail = total - n_ok
        key = f"{ocr_eng}+{extr}"

        doc_dicts = _rows_to_doc_dicts(succeeded)
        m = compute_aggregate_metrics(
            doc_dicts,
            weights=weights,
            wrong_counts_as_fn=wrong_counts_as_fn,
            tolerance=tolerance,
        )

        sections.append(f"## {key}\n")
        sections.append(f"Documents: {total} total, {n_ok} succeeded, {n_fail} failed\n")

        # --- Field accuracy ---
        sections.append("### Field-Level Exact-Match Accuracy\n")
        sections.append("| Field | Accuracy |")
        sections.append("|-------|----------|")
        for f in EVALUATED_GT_FIELDS:
            sections.append(f"| {f} | {_pct(m.field_accuracy.get(f, 0))} |")
        sections.append(f"| **Micro** | **{_pct(m.micro_accuracy)}** |")
        sections.append(f"| **Macro** | **{_pct(m.macro_accuracy)}** |")
        sections.append("")

        # --- Slot P/R/F1 ---
        sections.append("### Slot-Filling Precision / Recall / F1\n")
        sections.append("| Field | TP | FP | FN | Precision | Recall | F1 |")
        sections.append("|-------|----|----|----|-----------|---------|----|")
        for f in EVALUATED_GT_FIELDS:
            sc = m.field_slot.get(f, SlotCounts())
            sections.append(_slot_row(f, sc))
        sections.append(_slot_row("**Micro**", m.micro_slot))
        sections.append(f"| **Macro F1** | | | | | | **{_f3(m.macro_f1)}** |")
        sections.append("")

        # --- Doc-EM ---
        sections.append(f"### Document-Level Exact Match (Doc-EM): **{_pct(m.doc_em)}**\n")

        # --- Diagnostics ---
        if m.avg_edit_sim:
            sections.append("### Normalised Edit Similarity (ANLS-style)\n")
            sections.append("| Field | Avg Similarity |")
            sections.append("|-------|---------------|")
            for f, v in m.avg_edit_sim.items():
                sections.append(f"| {f} | {_f3(v)} |")
            sections.append("")

        if m.avg_digit_acc:
            sections.append("### Digit-Level Accuracy\n")
            sections.append("| Field | Avg Digit Accuracy |")
            sections.append("|-------|--------------------|")
            for f, v in m.avg_digit_acc.items():
                sections.append(f"| {f} | {_f3(v)} |")
            sections.append("")

        if m.amount_mae is not None:
            sections.append("### Amount Metrics\n")
            sections.append(f"- MAE: {_f3(m.amount_mae)}")
            sections.append(f"- Tolerance accuracy (eps={tolerance}): {_pct(m.amount_tolerance_acc or 0)}")
            sections.append("")

        # --- Error breakdown ---
        sections.append("### Error Breakdown\n")
        sections.append("| Type | Count |")
        sections.append("|------|-------|")
        for etype in ("correct", "missing", "wrong", "extra"):
            sections.append(f"| {etype} | {m.error_counts.get(etype, 0)} |")
        sections.append("")

        # --- Latency ---
        sections.append("### Latency (ms)\n")
        sections.append("| Stage | Median | P95 | P99 | Mean |")
        sections.append("|-------|--------|-----|-----|------|")
        sections.append(_latency_section("OCR", m.latency_ocr_ms))
        sections.append(_latency_section("Extraction", m.latency_extract_ms))
        sections.append(_latency_section("Total", m.latency_total_ms))
        sections.append("")

        # --- Weighted ---
        if m.weighted_accuracy is not None:
            sections.append("### Weighted Metrics\n")
            sections.append(f"- Weighted field accuracy: {_pct(m.weighted_accuracy)}")
            sections.append(f"- Weighted document score (mean): {_f3(m.weighted_doc_score_mean)}")
            sections.append(f"- Strict Doc-EM (for comparison): {_pct(m.doc_em)}")
            sections.append("")

        all_metrics[key] = {
            "n_docs": total,
            "n_succeeded": n_ok,
            "field_accuracy": m.field_accuracy,
            "micro_accuracy": m.micro_accuracy,
            "macro_accuracy": m.macro_accuracy,
            "micro_f1": m.micro_f1,
            "macro_f1": m.macro_f1,
            "doc_em": m.doc_em,
            "avg_edit_sim": m.avg_edit_sim,
            "avg_digit_acc": m.avg_digit_acc,
            "amount_mae": m.amount_mae,
            "amount_tolerance_acc": m.amount_tolerance_acc,
            "error_counts": m.error_counts,
            "latency_ocr_ms": m.latency_ocr_ms,
            "latency_extract_ms": m.latency_extract_ms,
            "latency_total_ms": m.latency_total_ms,
            "weighted_accuracy": m.weighted_accuracy,
            "weighted_doc_score_mean": m.weighted_doc_score_mean,
        }

    # --- Comparison table ---
    if len(all_metrics) > 1:
        sections.append("## Comparison Summary\n")
        sections.append("| System | Docs | Micro Acc | Macro Acc | Micro F1 | Doc-EM |")
        sections.append("|--------|------|-----------|-----------|----------|--------|")
        for key, mv in all_metrics.items():
            sections.append(
                f"| {key} | {mv['n_succeeded']} | {_pct(mv['micro_accuracy'])} | "
                f"{_pct(mv['macro_accuracy'])} | {_f3(mv['micro_f1'])} | {_pct(mv['doc_em'])} |"
            )
        sections.append("")

    md = "\n".join(sections) + "\n"
    return md, all_metrics
