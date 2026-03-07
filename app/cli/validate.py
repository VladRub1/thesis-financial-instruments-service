"""CLI for the validation subsystem: sample, run, metrics.

Usage examples:
    uv run python -m app.cli.validate sample --n 200 --seed 42
    uv run python -m app.cli.validate run --seed-file data/processed/validation/seeds/seed_n=200_seed=42.csv \
        --ocr-engine tesseract --extractor llm --llm-model models/qwen2.5-3b-instruct-q4_k_m.gguf
    uv run python -m app.cli.validate metrics --run-id <id> --out-md report.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from app.core.logging import get_logger, setup_logging

log = get_logger(__name__)

DEFAULT_DATASET = (
    "/Users/home/Work/10-edu/data-science/thesis/code/"
    "masters-thesis-dev/data/processed/final/dataset.csv"
)


# -----------------------------------------------------------------------
# sample
# -----------------------------------------------------------------------

def cmd_sample(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        log.error("Dataset not found: %s", dataset_path)
        sys.exit(1)

    df = pd.read_csv(dataset_path, dtype={"bank_inn": str, "pcpl_inn": str, "bene_inn": str, "ikz": str})
    log.info("Loaded dataset: %d rows", len(df))

    if args.n > len(df):
        log.warning("Requested %d but dataset has only %d — using all.", args.n, len(df))
        args.n = len(df)

    sampled = df.sample(n=args.n, random_state=args.seed)

    from app.validation.storage import seeds_dir
    name = f"seed_n={args.n}_seed={args.seed}.csv"
    out_path = seeds_dir() / name
    sampled.to_csv(out_path, index=False)
    log.info("Seed file written: %s  (%d rows)", out_path, len(sampled))
    print(f"Seed file: {out_path}")


# -----------------------------------------------------------------------
# run
# -----------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    seed_path = Path(args.seed_file)
    if not seed_path.exists():
        log.error("Seed file not found: %s", seed_path)
        sys.exit(1)

    seed_df = pd.read_csv(seed_path, dtype={"bank_inn": str, "pcpl_inn": str, "bene_inn": str, "ikz": str})
    log.info("Loaded seed file: %d documents", len(seed_df))

    missing = seed_df[~seed_df["stored_path"].apply(lambda p: Path(p).exists())]
    if len(missing) > 0:
        log.warning("%d documents have missing files — they will fail.", len(missing))

    from app.validation.storage import generate_run_id, write_run_metadata
    run_id = args.out_run_id or generate_run_id(args.ocr_engine, args.extractor)

    write_run_metadata(
        run_id,
        seed_file=str(seed_path),
        sample_size=len(seed_df),
        seed_value=_infer_seed(seed_path.name),
        ocr_engine=args.ocr_engine,
        extractor=args.extractor,
        llm_model=args.llm_model,
        keep_artifacts=args.keep_artifacts,
    )
    log.info("Run ID: %s", run_id)
    print(f"Run ID: {run_id}")

    from app.validation.runner import run_evaluation
    results = run_evaluation(
        seed_df,
        run_id=run_id,
        ocr_engine=args.ocr_engine,
        extractor=args.extractor,
        lang=args.lang,
        llm_model_path=args.llm_model,
        keep_artifacts=args.keep_artifacts,
        workers=args.workers,
        llm_workers=args.llm_workers,
        batch_size=args.batch_size,
        resume=args.resume,
    )
    n_ok = len(results[results["status"] == "succeeded"]) if not results.empty else 0
    n_fail = len(results) - n_ok
    print(f"\nDone: {n_ok} succeeded, {n_fail} failed")
    print(f"Results: data/processed/validation/runs/{run_id}/results.parquet")


def _infer_seed(filename: str) -> int | None:
    import re
    m = re.search(r"seed=(\d+)", filename)
    return int(m.group(1)) if m else None


# -----------------------------------------------------------------------
# metrics
# -----------------------------------------------------------------------

def cmd_metrics(args: argparse.Namespace) -> None:
    run_ids = [r.strip() for r in args.run_id.split(",")]

    weights = None
    if args.weights:
        wp = Path(args.weights)
        if wp.exists():
            weights = json.loads(wp.read_text(encoding="utf-8"))
        else:
            try:
                weights = json.loads(args.weights)
            except json.JSONDecodeError:
                log.error("Cannot parse weights: %s", args.weights)
                sys.exit(1)

    from app.validation.report import compute_and_report
    md, metrics_dict = compute_and_report(
        run_ids,
        weights=weights,
        wrong_counts_as_fn=args.wrong_counts_as_fn,
        tolerance=args.tolerance,
    )

    out_md = Path(args.out_md)
    out_md.write_text(md, encoding="utf-8")
    log.info("Report written: %s", out_md)
    print(f"Report: {out_md}")

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.write_text(json.dumps(metrics_dict, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"JSON metrics: {out_json}")

    print("\n" + md)


# -----------------------------------------------------------------------
# main
# -----------------------------------------------------------------------

def main() -> None:
    setup_logging(debug=os.environ.get("DEBUG", "").lower() in ("1", "true"))

    parser = argparse.ArgumentParser(
        prog="validate",
        description="Validation subsystem: sample, run, metrics",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- sample --
    sp = sub.add_parser("sample", help="Create a reproducible sample from the dataset")
    sp.add_argument("--n", type=int, required=True, help="Number of documents to sample")
    sp.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    sp.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to dataset.csv")

    # -- run --
    rp = sub.add_parser("run", help="Run evaluation pipeline on a seed file")
    rp.add_argument("--seed-file", required=True, help="Path to seed CSV")
    rp.add_argument("--ocr-engine", required=True, choices=["tesseract", "paddleocr"])
    rp.add_argument("--extractor", required=True, choices=["llm", "regex"])
    rp.add_argument("--llm-model", default=None, help="Path to GGUF model for LLM extractor")
    rp.add_argument("--lang", default="rus+eng", help="OCR language (default: rus+eng)")
    rp.add_argument("--keep-artifacts", action="store_true", default=False,
                     help="Keep OCR/extraction artifacts on disk")
    rp.add_argument("--no-keep-artifacts", dest="keep_artifacts", action="store_false")
    rp.add_argument("--resume", action="store_true", default=True,
                     help="Resume from checkpoint (default: true)")
    rp.add_argument("--no-resume", dest="resume", action="store_false")
    rp.add_argument("--batch-size", type=int, default=10, help="Checkpoint every N docs (default: 10)")
    rp.add_argument("--workers", type=int, default=2, help="OCR parallelism (default: 2)")
    rp.add_argument("--llm-workers", type=int, default=1,
                     help="LLM concurrency — careful with memory (default: 1)")
    rp.add_argument("--out-run-id", default=None, help="Override auto-generated run ID")

    # -- metrics --
    mp = sub.add_parser("metrics", help="Compute metrics and generate report")
    mp.add_argument("--run-id", required=True,
                     help="Run ID(s) — comma-separated for multi-run comparison")
    mp.add_argument("--out-md", default="data/processed/validation/report.md",
                     help="Output markdown report path (default: data/processed/validation/report.md)")
    mp.add_argument("--out-json", default=None, help="Optional JSON metrics output path")
    mp.add_argument("--weights", default=None,
                     help="Field weights: JSON string or path to weights.json")
    mp.add_argument("--wrong-counts-as-fn", action="store_true", default=True)
    mp.add_argument("--no-wrong-counts-as-fn", dest="wrong_counts_as_fn", action="store_false")
    mp.add_argument("--tolerance", type=float, default=0.01,
                     help="Amount tolerance for equality (default: 0.01)")

    args = parser.parse_args()

    if args.command == "sample":
        cmd_sample(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "metrics":
        cmd_metrics(args)


if __name__ == "__main__":
    main()
