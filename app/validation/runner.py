"""Evaluation runner — parallel, resumable processing of sampled documents.

Runs OCR + extraction (LLM or regex) on each document, stores normalised
results in parquet, and checkpoints every batch for safe resume.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import pandas as pd
from tqdm import tqdm

from app.core.logging import get_logger
from app.validation import storage
from app.validation.metrics import FieldDiag, compute_field_diagnostics
from app.validation.normalize import (
    EVALUATED_GT_FIELDS,
    GT_TO_PRED_FIELD,
    normalize_field,
)

log = get_logger(__name__)


@contextmanager
def _suppress_native_stderr():
    """Redirect fd-level stderr to /dev/null (catches C-library output like ggml_metal_init)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_fd = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)


def _validation_n_gpu_layers(llm_device: str, llm_n_gpu_layers: int) -> int | None:
    """Map validation CLI device flags to llama-cpp n_gpu_layers."""
    if llm_device == "cuda":
        return llm_n_gpu_layers
    return None


# ---------------------------------------------------------------------------
# Single-document processing  (designed to run in a worker process)
# ---------------------------------------------------------------------------

def _file_to_images(file_path: Path) -> list[tuple]:
    """Convert PDF/TIF/TIFF to list[(PIL.Image, dpi)]."""
    import fitz
    from PIL import Image

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        doc = fitz.open(str(file_path))
        images = []
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append((img, 300))
        doc.close()
        return images

    if suffix in (".tif", ".tiff"):
        img = Image.open(str(file_path))
        pages = []
        try:
            for i in range(getattr(img, "n_frames", 1)):
                img.seek(i)
                frame = img.copy().convert("RGB")
                dpi_info = img.info.get("dpi", (300, 300))
                dpi = int(dpi_info[0]) if isinstance(dpi_info, (tuple, list)) else int(dpi_info)
                pages.append((frame, dpi))
        except EOFError:
            pass
        return pages

    if suffix in (".png", ".jpg", ".jpeg"):
        img = Image.open(str(file_path)).convert("RGB")
        return [(img, 300)]

    raise ValueError(f"Unsupported file type: {suffix}")


def _build_markdown(page_results) -> str:
    """Minimal markdown from OCR page results."""
    parts = []
    for pr in page_results:
        parts.append(f"# Page {pr.page_index + 1}\n")
        for block in sorted(pr.blocks, key=lambda b: (b.bbox[1], b.bbox[0])):
            parts.append(block.text)
            parts.append("")
    return "\n".join(parts).strip() + "\n"


def _compute_diags(gold_row: dict, pred: dict) -> tuple[dict, dict, list[dict]]:
    """Normalise gold/pred and compute per-field diagnostics."""
    norm_gold: dict = {}
    norm_pred: dict = {}
    for gt_f in EVALUATED_GT_FIELDS:
        pred_f = GT_TO_PRED_FIELD[gt_f]
        norm_gold[gt_f] = normalize_field(gt_f, gold_row.get(gt_f))
        norm_pred[gt_f] = normalize_field(pred_f, pred.get(pred_f))

    diags: list[dict] = []
    for gt_f in EVALUATED_GT_FIELDS:
        fd = compute_field_diagnostics(gt_f, norm_pred[gt_f], norm_gold[gt_f])
        diags.append({
            "field": fd.field, "is_match": fd.is_match, "error_type": fd.error_type,
            "edit_similarity": fd.edit_similarity, "digit_acc": fd.digit_acc,
            "abs_error": fd.abs_error, "within_tolerance": fd.within_tolerance,
        })
    return norm_gold, norm_pred, diags


def process_single_document(
    doc_id: int,
    stored_path: str,
    gold_row: dict,
    ocr_engine_name: str,
    extractor: str,
    lang: str = "rus+eng",
    llm_model_path: str | None = None,
    llm_device: str = "cpu",
    llm_n_gpu_layers: int = 0,
    keep_artifacts: bool = False,
    run_id: str | None = None,
) -> dict:
    """Process one document end-to-end. Safe to call from a worker process."""
    from app.services.pipeline import get_ocr_engine

    file_path = Path(stored_path)
    timings: dict[str, float] = {}
    pred: dict = {}
    status = "succeeded"
    error_msg = None

    t_total = time.perf_counter()

    try:
        t0 = time.perf_counter()
        engine = get_ocr_engine(ocr_engine_name, lang)
        images = _file_to_images(file_path)

        page_results = []
        for idx, (img, dpi) in enumerate(images):
            pr = engine.run_page(img, dpi=dpi)
            pr.page_index = idx
            page_results.append(pr)
        ocr_md = _build_markdown(page_results)
        timings["ocr_ms"] = (time.perf_counter() - t0) * 1000

        if keep_artifacts and run_id:
            art = storage.artifacts_dir(run_id, doc_id)
            (art / "ocr.md").write_text(ocr_md, encoding="utf-8")

        t1 = time.perf_counter()
        if extractor == "regex":
            from app.validation.regex_baseline import extract_regex
            pred = extract_regex(ocr_md)
        else:
            from app.llm.engine import LLMEngine
            from app.llm.extract import extract_fields

            llm = LLMEngine(
                n_gpu_layers=_validation_n_gpu_layers(llm_device, llm_n_gpu_layers),
            )
            if llm_model_path:
                from app.core.config import settings
                settings.LLM_MODEL_PATH = llm_model_path
            with _suppress_native_stderr():
                llm.load()
            result = extract_fields(ocr_md, None, engine=llm, trace_id=str(doc_id))
            if result.validated:
                pred = result.validated.model_dump(mode="json")
            else:
                status = "extraction_failed"
                error_msg = "; ".join(result.errors[:3])
            llm.unload()

        timings["extract_ms"] = (time.perf_counter() - t1) * 1000

        if keep_artifacts and run_id and pred:
            art = storage.artifacts_dir(run_id, doc_id)
            (art / "extraction.json").write_text(
                json.dumps(pred, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )

    except Exception:
        status = "failed"
        error_msg = traceback.format_exc()[-500:]

    timings["total_ms"] = (time.perf_counter() - t_total) * 1000

    norm_gold, norm_pred, diags_list = _compute_diags(gold_row, pred)

    return storage.build_result_row(
        doc_id=doc_id, stored_path=stored_path,
        ocr_engine=ocr_engine_name, extractor=extractor,
        gold=norm_gold, pred=norm_pred, diags=diags_list,
        timings=timings, status=status, error_msg=error_msg,
    )


# ---------------------------------------------------------------------------
# LLM-aware runner  (loads model once, processes docs sequentially)
# ---------------------------------------------------------------------------

def _process_items_with_llm(
    items: list[tuple[int, str, dict]],
    ocr_engine_name: str,
    lang: str,
    llm,
    keep_artifacts: bool,
    run_id: str,
    progress_cb: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Process items sequentially using a pre-loaded LLM instance."""
    from app.llm.extract import extract_fields
    from app.services.pipeline import get_ocr_engine

    engine = get_ocr_engine(ocr_engine_name, lang)
    results: list[dict] = []
    n_items = len(items)

    for idx, (doc_id, stored_path, gold_row) in enumerate(items, start=1):
        timings: dict[str, float] = {}
        pred: dict = {}
        status = "succeeded"
        error_msg = None
        t_total = time.perf_counter()

        log.info("[llm] Start doc_id=%s (%d/%d)", doc_id, idx, n_items)
        try:
            t0 = time.perf_counter()
            images = _file_to_images(Path(stored_path))
            page_results = []
            for idx, (img, dpi) in enumerate(images):
                pr = engine.run_page(img, dpi=dpi)
                pr.page_index = idx
                page_results.append(pr)
            ocr_md = _build_markdown(page_results)
            timings["ocr_ms"] = (time.perf_counter() - t0) * 1000

            if keep_artifacts:
                art = storage.artifacts_dir(run_id, doc_id)
                (art / "ocr.md").write_text(ocr_md, encoding="utf-8")

            t1 = time.perf_counter()
            result = extract_fields(ocr_md, None, engine=llm, trace_id=str(doc_id))
            if result.validated:
                pred = result.validated.model_dump(mode="json")
            else:
                status = "extraction_failed"
                error_msg = "; ".join(result.errors[:3])
            timings["extract_ms"] = (time.perf_counter() - t1) * 1000

            if keep_artifacts and pred:
                art = storage.artifacts_dir(run_id, doc_id)
                (art / "extraction.json").write_text(
                    json.dumps(pred, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
                )
        except Exception:
            status = "failed"
            error_msg = traceback.format_exc()[-500:]

        timings["total_ms"] = (time.perf_counter() - t_total) * 1000
        norm_gold, norm_pred, diags_list = _compute_diags(gold_row, pred)

        results.append(storage.build_result_row(
            doc_id=doc_id, stored_path=stored_path,
            ocr_engine=ocr_engine_name, extractor="llm",
            gold=norm_gold, pred=norm_pred, diags=diags_list,
            timings=timings, status=status, error_msg=error_msg,
        ))
        if progress_cb:
            progress_cb(results[-1])
    return results


def _run_batch_llm_subprocess(
    items: list[tuple[int, str, dict]],
    ocr_engine_name: str,
    lang: str,
    llm_model_path: str | None,
    llm_device: str,
    llm_n_gpu_layers: int,
    keep_artifacts: bool,
    run_id: str,
) -> list[dict]:
    """Subprocess entry: load model once, process items, unload."""
    from app.llm.engine import LLMEngine

    llm = LLMEngine(
        n_gpu_layers=_validation_n_gpu_layers(llm_device, llm_n_gpu_layers),
    )
    if llm_model_path:
        from app.core.config import settings
        settings.LLM_MODEL_PATH = llm_model_path
    with _suppress_native_stderr():
        llm.load()

    results = _process_items_with_llm(
        items, ocr_engine_name, lang, llm, keep_artifacts, run_id,
    )
    llm.unload()
    return results


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_evaluation(
    seed_df: pd.DataFrame,
    *,
    run_id: str,
    ocr_engine: str,
    extractor: str,
    lang: str = "rus+eng",
    llm_model_path: str | None = None,
    llm_device: str = "cpu",
    llm_n_gpu_layers: int = 0,
    keep_artifacts: bool = False,
    workers: int = 2,
    llm_workers: int = 1,
    batch_size: int = 10,
    resume: bool = True,
) -> pd.DataFrame:
    """Run evaluation pipeline over the seed set.

    Returns the full results DataFrame.
    """
    existing = storage.load_results(run_id)
    done_ids: set[int] = set()
    if resume and not existing.empty:
        done_ids = set(existing["id"].tolist())
        log.info("Resuming: %d docs already processed, skipping.", len(done_ids))

    pending = seed_df[~seed_df["id"].isin(done_ids)].to_dict("records")
    total = len(pending)
    if total == 0:
        log.info("All documents already processed.")
        return existing

    log.info("Processing %d documents (engine=%s extractor=%s)", total, ocr_engine, extractor)

    all_new_rows: list[dict] = []
    succeeded = sum(1 for _, r in existing.iterrows() if r.get("status") == "succeeded") if not existing.empty else 0
    failed = sum(1 for _, r in existing.iterrows() if r.get("status") != "succeeded") if not existing.empty else 0
    processed = 0
    t_run = time.perf_counter()

    pbar = tqdm(total=total, desc=f"eval [{ocr_engine}+{extractor}]", unit="doc")

    def _on_result(r: dict) -> None:
        nonlocal processed, succeeded, failed
        processed += 1
        if r["status"] == "succeeded":
            succeeded += 1
        else:
            failed += 1
        elapsed_ms = (time.perf_counter() - t_run) * 1000
        remaining = total - processed
        eta_ms = (elapsed_ms / processed) * remaining if processed > 0 else 0
        log.info(
            "[progress] doc_id=%s status=%s (%d/%d) elapsed=%s eta=%s",
            r.get("id"), r.get("status"),
            processed, total,
            _fmt_duration(elapsed_ms),
            _fmt_duration(eta_ms) if remaining > 0 else "—",
        )
        pbar.update(1)
        pbar.set_postfix(ok=succeeded, fail=failed)

    if extractor == "llm":
        log.info("LLM validation runtime: device=%s n_gpu_layers=%s", llm_device, llm_n_gpu_layers)
        if llm_workers <= 1:
            # Sequential: load model once, keep it across all batches
            from app.llm.engine import LLMEngine

            llm = LLMEngine(
                n_gpu_layers=_validation_n_gpu_layers(llm_device, llm_n_gpu_layers),
            )
            if llm_model_path:
                from app.core.config import settings
                settings.LLM_MODEL_PATH = llm_model_path
            log.info("Loading LLM model (noise suppressed) …")
            with _suppress_native_stderr():
                llm.load()
            log.info("LLM model ready (device=%s, n_gpu_layers=%s).", llm_device, llm_n_gpu_layers)

            for batch_start in range(0, total, batch_size):
                batch_items = [
                    (row["id"], row["stored_path"], row)
                    for row in pending[batch_start:batch_start + batch_size]
                ]
                batch_results = _process_items_with_llm(
                    batch_items, ocr_engine, lang, llm, keep_artifacts, run_id,
                    progress_cb=_on_result,
                )
                all_new_rows.extend(batch_results)
                storage.append_results(run_id, batch_results)

            llm.unload()
        else:
            # Parallel LLM workers: each subprocess loads its own model
            for batch_start in range(0, total, batch_size):
                batch_items = [
                    (row["id"], row["stored_path"], row)
                    for row in pending[batch_start:batch_start + batch_size]
                ]
                chunk_size = max(1, len(batch_items) // llm_workers)
                chunks = [
                    batch_items[i:i + chunk_size]
                    for i in range(0, len(batch_items), chunk_size)
                ]
                batch_results: list[dict] = []
                with ProcessPoolExecutor(max_workers=llm_workers) as pool:
                    futures = {
                        pool.submit(
                            _run_batch_llm_subprocess, chunk, ocr_engine, lang,
                            llm_model_path, llm_device, llm_n_gpu_layers, keep_artifacts, run_id,
                        ): chunk
                        for chunk in chunks
                    }
                    for fut in as_completed(futures):
                        batch_results.extend(fut.result())

                for r in batch_results:
                    _on_result(r)
                all_new_rows.extend(batch_results)
                storage.append_results(run_id, batch_results)

    else:
        # Regex: embarrassingly parallel with multiprocessing
        for batch_start in range(0, total, batch_size):
            batch = pending[batch_start:batch_start + batch_size]
            n_workers = min(workers, len(batch))

            batch_results: list[dict] = []
            if n_workers <= 1:
                for row in batch:
                    r = process_single_document(
                        row["id"], row["stored_path"], row,
                        ocr_engine, extractor, lang,
                        keep_artifacts=keep_artifacts, run_id=run_id,
                    )
                    batch_results.append(r)
            else:
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    futures = {
                        pool.submit(
                            process_single_document,
                            row["id"], row["stored_path"], row,
                            ocr_engine, extractor, lang,
                            keep_artifacts=keep_artifacts, run_id=run_id,
                        ): row["id"]
                        for row in batch
                    }
                    for fut in as_completed(futures):
                        try:
                            batch_results.append(fut.result())
                        except Exception as exc:
                            log.error("Worker failed: %s", exc)

            for r in batch_results:
                _on_result(r)
            all_new_rows.extend(batch_results)
            storage.append_results(run_id, batch_results)

    pbar.close()

    final = storage.load_results(run_id)
    log.info(
        "Run complete: %d succeeded, %d failed out of %d total",
        succeeded, failed, len(final),
    )
    return final
