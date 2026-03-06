"""CLI — bulk OCR + extraction, runs in-process (no HTTP per file).

Usage:
    uv run python -m app.cli.ocr --input-root <dir> [--ids-file ids.csv]
                                  --engine-ocr tesseract --engine-llm llama_cpp --workers 4
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from app.core.logging import get_logger, setup_logging

log = get_logger(__name__)


def _collect_pdfs(input_root: Path, ids: list[str] | None) -> list[Path]:
    pdfs: list[Path] = []
    if ids:
        for gid in ids:
            folder = input_root / gid
            if folder.is_dir():
                pdfs.extend(sorted(folder.glob("*.pdf")))
    else:
        for folder in sorted(input_root.iterdir()):
            if folder.is_dir():
                pdfs.extend(sorted(folder.glob("*.pdf")))
    return pdfs


def _process_one(pdf_path: str, engine_ocr: str, engine_llm: str, lang: str, pipeline: str) -> dict:
    """Runs in a subprocess — must reimport everything."""
    from app.services.pipeline import run_full_pipeline
    from app.llm.engine import get_engine

    llm = None
    if pipeline == "ocr+extract":
        llm = get_engine()
        try:
            llm.load()
        except Exception:
            llm = None

    result = run_full_pipeline(
        Path(pdf_path),
        engine_ocr_name=engine_ocr,
        lang=lang,
        pipeline=pipeline,
        llm_engine=llm,
    )
    result.pop("extraction_result", None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk OCR + extraction CLI")
    parser.add_argument("--input-root", type=Path, required=True, help="Root attachments dir")
    parser.add_argument("--ids-file", type=Path, help="CSV with guarantee IDs (one per line)")
    parser.add_argument("--engine-ocr", default="tesseract")
    parser.add_argument("--engine-llm", default="llama_cpp")
    parser.add_argument("--lang", default="rus+eng")
    parser.add_argument("--pipeline", default="ocr+extract", choices=["ocr_only", "ocr+extract"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("bulk_manifest.json"))
    args = parser.parse_args()
    setup_logging(debug=False)

    ids = None
    if args.ids_file:
        with open(args.ids_file) as f:
            ids = [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]

    pdfs = _collect_pdfs(args.input_root, ids)
    log.info("Found %d PDFs to process", len(pdfs))

    run_id = str(uuid.uuid4())
    results: list[dict] = []
    failures: list[dict] = []
    t0 = time.perf_counter()

    if args.workers <= 1:
        for pdf in pdfs:
            try:
                r = _process_one(str(pdf), args.engine_ocr, args.engine_llm, args.lang, args.pipeline)
                results.append(r)
                log.info("Done: %s", pdf.name)
            except Exception as e:
                log.error("Failed: %s — %s", pdf.name, e)
                failures.append({"file": str(pdf), "error": str(e)})
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_process_one, str(pdf), args.engine_ocr, args.engine_llm, args.lang, args.pipeline): pdf
                for pdf in pdfs
            }
            for fut in as_completed(futures):
                pdf = futures[fut]
                try:
                    results.append(fut.result())
                    log.info("Done: %s", pdf.name)
                except Exception as e:
                    log.error("Failed: %s — %s", pdf.name, e)
                    failures.append({"file": str(pdf), "error": str(e)})

    elapsed = time.perf_counter() - t0
    manifest = {
        "run_id": run_id,
        "total_files": len(pdfs),
        "succeeded": len(results),
        "failed": len(failures),
        "elapsed_seconds": round(elapsed, 2),
        "files": results,
        "failures": failures,
    }
    args.out.write_text(json.dumps(manifest, indent=2, default=str, ensure_ascii=False))
    log.info("Manifest written to %s  (%d ok, %d failed, %.1fs)", args.out, len(results), len(failures), elapsed)


if __name__ == "__main__":
    main()
