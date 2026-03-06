"""CLI — evaluation scaffold: run OCR+LLM in-process, compare against ground truth.

Usage:
    uv run python -m app.cli.evaluate --input-root <attachments_dir> --gt-root <gt_dir>
                                       --out report.json --workers 2
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from app.core.logging import get_logger, setup_logging

log = get_logger(__name__)


def _process_and_compare(
    pdf_path: str, gt_path: str | None, engine_ocr: str, lang: str
) -> dict:
    """Runs in subprocess."""
    from app.services.pipeline import run_full_pipeline
    from app.llm.engine import get_engine

    llm = get_engine()
    try:
        llm.load()
    except Exception:
        llm = None

    t0 = time.perf_counter()
    result = run_full_pipeline(
        Path(pdf_path), engine_ocr_name=engine_ocr, lang=lang, pipeline="ocr+extract", llm_engine=llm,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    pred_json = None
    ext = result.get("extraction_result")
    if ext and ext.validated:
        pred_json = ext.validated.model_dump(mode="json")

    gt_json = None
    if gt_path and Path(gt_path).is_file():
        gt_json = json.loads(Path(gt_path).read_text(encoding="utf-8"))

    per_field: dict = {}
    if pred_json and gt_json:
        for key in pred_json:
            if key in ("schema_version", "extraction_confidence", "evidence", "warnings"):
                continue
            per_field[key] = {
                "pred": pred_json.get(key),
                "gt": gt_json.get(key),
                "match": pred_json.get(key) == gt_json.get(key),
            }

    return {
        "pdf": pdf_path,
        "gt_file": gt_path,
        "elapsed_ms": round(elapsed_ms),
        "pred": pred_json,
        "gt": gt_json,
        "per_field": per_field,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation scaffold")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True, help="Dir with GT JSON files (named <stem>.json)")
    parser.add_argument("--engine-ocr", default="tesseract")
    parser.add_argument("--lang", default="rus+eng")
    parser.add_argument("--out", type=Path, default=Path("eval_report.json"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    setup_logging(debug=False)

    pdfs: list[Path] = []
    for folder in sorted(args.input_root.iterdir()):
        if folder.is_dir():
            pdfs.extend(sorted(folder.glob("*.pdf")))
    log.info("Found %d PDFs for evaluation", len(pdfs))

    records: list[dict] = []
    t0 = time.perf_counter()

    def _gt_for(pdf: Path) -> str | None:
        gt = args.gt_root / f"{pdf.stem}.json"
        return str(gt) if gt.is_file() else None

    if args.workers <= 1:
        for pdf in pdfs:
            try:
                rec = _process_and_compare(str(pdf), _gt_for(pdf), args.engine_ocr, args.lang)
                records.append(rec)
            except Exception as e:
                log.error("Eval failed for %s: %s", pdf.name, e)
                records.append({"pdf": str(pdf), "error": str(e)})
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(_process_and_compare, str(pdf), _gt_for(pdf), args.engine_ocr, args.lang): pdf
                for pdf in pdfs
            }
            for fut in as_completed(futs):
                pdf = futs[fut]
                try:
                    records.append(fut.result())
                except Exception as e:
                    log.error("Eval failed for %s: %s", pdf.name, e)
                    records.append({"pdf": str(pdf), "error": str(e)})

    total_elapsed = time.perf_counter() - t0

    # TODO: compute F1 / precision / recall per field once GT format is finalised
    matched = sum(1 for r in records if r.get("per_field"))
    skipped = sum(1 for r in records if "error" in r)
    avg_latency = (
        sum(r.get("elapsed_ms", 0) for r in records if "elapsed_ms" in r) / max(len(records) - skipped, 1)
    )

    report = {
        "run_id": str(uuid.uuid4()),
        "total_files": len(pdfs),
        "evaluated_with_gt": matched,
        "skipped_or_failed": skipped,
        "avg_latency_ms": round(avg_latency),
        "total_elapsed_s": round(total_elapsed, 2),
        "records": records,
    }
    args.out.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False))
    log.info("Report written to %s", args.out)


if __name__ == "__main__":
    main()
