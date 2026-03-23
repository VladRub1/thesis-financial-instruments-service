"""Parquet-based storage for validation runs and per-document results."""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from app.validation.normalize import EVALUATED_GT_FIELDS, GT_TO_PRED_FIELD

_VAL_ROOT = Path("data/processed/validation")


def seeds_dir() -> Path:
    d = _VAL_ROOT / "seeds"
    d.mkdir(parents=True, exist_ok=True)
    return d


def runs_dir() -> Path:
    d = _VAL_ROOT / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_dir(run_id: str) -> Path:
    d = runs_dir() / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def artifacts_dir(run_id: str, doc_id: int) -> Path:
    d = run_dir(run_id) / "artifacts" / str(doc_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------

def generate_run_id(ocr_engine: str, extractor: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    short = uuid4().hex[:6]
    return f"{ts}_{ocr_engine}_{extractor}_{short}"


def write_run_metadata(
    run_id: str,
    *,
    seed_file: str,
    sample_size: int,
    seed_value: int | None,
    ocr_engine: str,
    extractor: str,
    llm_model: str | None = None,
    llm_device: str | None = None,
    llm_n_gpu_layers: int | None = None,
    regex_version: str = "v1",
    keep_artifacts: bool = False,
    extra: dict | None = None,
) -> Path:
    meta = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "machine": {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "seed_file": seed_file,
        "sample_size": sample_size,
        "seed_value": seed_value,
        "ocr_engine": ocr_engine,
        "extractor": extractor,
        "llm_model": llm_model,
        "llm_device": llm_device,
        "llm_n_gpu_layers": llm_n_gpu_layers,
        "regex_version": regex_version,
        "keep_artifacts": keep_artifacts,
    }
    if extra:
        meta.update(extra)
    p = run_dir(run_id) / "metadata.json"
    p.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return p


def read_run_metadata(run_id: str) -> dict:
    p = run_dir(run_id) / "metadata.json"
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Per-document results  (parquet)
# ---------------------------------------------------------------------------

_RESULTS_FILE = "results.parquet"


def _results_path(run_id: str) -> Path:
    return run_dir(run_id) / _RESULTS_FILE


def load_results(run_id: str) -> pd.DataFrame:
    p = _results_path(run_id)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def save_results(run_id: str, df: pd.DataFrame) -> Path:
    p = _results_path(run_id)
    df.to_parquet(p, index=False)
    return p


def append_results(run_id: str, new_rows: list[dict]) -> pd.DataFrame:
    """Append rows to the results parquet, avoiding duplicates by doc id."""
    existing = load_results(run_id)
    new_df = pd.DataFrame(new_rows)
    if existing.empty:
        combined = new_df
    else:
        existing_ids = set(existing["id"].tolist())
        new_df = new_df[~new_df["id"].isin(existing_ids)]
        combined = pd.concat([existing, new_df], ignore_index=True)
    save_results(run_id, combined)
    return combined


def build_result_row(
    doc_id: int,
    stored_path: str,
    ocr_engine: str,
    extractor: str,
    gold: dict,
    pred: dict,
    diags: list,
    timings: dict,
    status: str = "succeeded",
    error_msg: str | None = None,
) -> dict:
    """Build a flat dict for one document result row."""
    row: dict = {
        "id": doc_id,
        "stored_path": stored_path,
        "ocr_engine": ocr_engine,
        "extractor": extractor,
        "status": status,
        "error_msg": error_msg,
        "ocr_ms": timings.get("ocr_ms", 0),
        "extract_ms": timings.get("extract_ms", 0),
        "total_ms": timings.get("total_ms", 0),
        "gold_json": json.dumps(gold, ensure_ascii=False, default=str),
        "pred_json": json.dumps(pred, ensure_ascii=False, default=str),
    }
    for d in diags:
        fn = d["field"] if isinstance(d, dict) else d.field
        row[f"match_{fn}"] = d["is_match"] if isinstance(d, dict) else d.is_match
        row[f"error_type_{fn}"] = d["error_type"] if isinstance(d, dict) else d.error_type
        es = d.get("edit_similarity") if isinstance(d, dict) else d.edit_similarity
        if es is not None:
            row[f"edit_sim_{fn}"] = es
        da = d.get("digit_acc") if isinstance(d, dict) else d.digit_acc
        if da is not None:
            row[f"digit_acc_{fn}"] = da
        ae = d.get("abs_error") if isinstance(d, dict) else d.abs_error
        if ae is not None:
            row[f"abs_error_{fn}"] = ae
        wt = d.get("within_tolerance") if isinstance(d, dict) else d.within_tolerance
        if wt is not None:
            row[f"within_tol_{fn}"] = wt
    return row
