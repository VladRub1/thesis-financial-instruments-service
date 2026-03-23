#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Copy files referenced by a validation seed CSV into a flat docs/ folder "
            "and write a rewritten CSV with portable stored_path values."
        )
    )
    p.add_argument(
        "--seed-file",
        required=True,
        help="Path to the sampled seed CSV (must contain stored_path).",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output bundle directory, e.g. data/processed/validation/bundles/n1000_seed42",
    )
    p.add_argument(
        "--docs-dir-name",
        default="docs",
        help="Name of the flat documents folder inside the bundle (default: docs).",
    )
    p.add_argument(
        "--out-csv-name",
        default="seed_bundled.csv",
        help="Name of the rewritten CSV inside the bundle (default: seed_bundled.csv).",
    )
    p.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink", "symlink"],
        default="copy",
        help="How to place files into the bundle (default: copy).",
    )
    p.add_argument(
        "--archive",
        default=None,
        help=(
            "Optional archive output path. Supported extensions: .tar.gz, .tgz, .zip "
            "(example: validation_bundle_n1000_seed42.tar.gz)"
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory if it exists.",
    )
    return p.parse_args()


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {path}. Use --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def unique_name(candidate: str, used: Dict[str, int]) -> str:
    if candidate not in used:
        used[candidate] = 1
        return candidate

    stem = Path(candidate).stem
    suffix = Path(candidate).suffix
    idx = used[candidate]
    while True:
        new_name = f"{stem}__{idx}{suffix}"
        if new_name not in used:
            used[candidate] += 1
            used[new_name] = 1
            return new_name
        idx += 1


def place_file(src: Path, dst: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def make_archive(src_dir: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    if str(archive_path).endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "w:gz") as tf:
            tf.add(src_dir, arcname=src_dir.name)
    elif str(archive_path).endswith(".zip"):
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in src_dir.rglob("*"):
                zf.write(p, arcname=p.relative_to(src_dir.parent))
    else:
        raise ValueError("Archive must end with .tar.gz, .tgz, or .zip")


def main() -> int:
    args = parse_args()

    seed_file = Path(args.seed_file).expanduser().resolve()
    if not seed_file.exists():
        print(f"ERROR: seed file does not exist: {seed_file}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir).expanduser().resolve()
    docs_dir = out_dir / args.docs_dir_name
    out_csv = out_dir / args.out_csv_name

    ensure_clean_dir(out_dir, overwrite=args.overwrite)
    docs_dir.mkdir(parents=True, exist_ok=True)

    used_names: Dict[str, int] = {}
    total = 0
    copied = 0
    missing = 0

    with seed_file.open("r", encoding="utf-8-sig", newline="") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = list(reader.fieldnames or [])
        if "stored_path" not in fieldnames:
            print("ERROR: seed CSV must contain a 'stored_path' column.", file=sys.stderr)
            return 1

        extra_cols = ["original_stored_path", "original_stored_filename", "bundle_filename"]
        for col in extra_cols:
            if col not in fieldnames:
                fieldnames.append(col)

        rows_out = []

        for row in reader:
            total += 1
            src = Path(row["stored_path"]).expanduser()

            if not src.exists():
                missing += 1
                print(f"[MISSING] {src}", file=sys.stderr)
                continue

            base_name = row.get("stored_filename") or src.name
            safe_name = unique_name(base_name, used_names)
            dst = docs_dir / safe_name

            place_file(src, dst, args.copy_mode)
            copied += 1

            row["original_stored_path"] = row["stored_path"]
            row["original_stored_filename"] = row.get("stored_filename", "")
            row["stored_filename"] = safe_name
            row["stored_path"] = f"{args.docs_dir_name}/{safe_name}"  # portable relative path
            row["bundle_filename"] = safe_name
            rows_out.append(row)

    with out_csv.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Seed file:     {seed_file}")
    print(f"Bundle dir:    {out_dir}")
    print(f"Docs dir:      {docs_dir}")
    print(f"Rewritten CSV: {out_csv}")
    print(f"Total rows:    {total}")
    print(f"Copied:        {copied}")
    print(f"Missing:       {missing}")
    print(f"Mode:          {args.copy_mode}")

    if args.archive:
        archive_path = Path(args.archive).expanduser().resolve()
        make_archive(out_dir, archive_path)
        print(f"Archive:       {archive_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
