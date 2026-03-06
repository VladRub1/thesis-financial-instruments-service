"""Deterministic output path generation."""
from __future__ import annotations

from pathlib import Path

from app.core.config import settings


def output_dir(guarantee_id: str, attachment_stem: str) -> Path:
    d = settings.processed_path / guarantee_id / attachment_stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def ocr_json_path(guarantee_id: str, attachment_stem: str) -> Path:
    return output_dir(guarantee_id, attachment_stem) / "ocr.json"


def ocr_md_path(guarantee_id: str, attachment_stem: str) -> Path:
    return output_dir(guarantee_id, attachment_stem) / "ocr.md"


def extraction_json_path(guarantee_id: str, attachment_stem: str) -> Path:
    return output_dir(guarantee_id, attachment_stem) / "extraction.json"


def meta_json_path(guarantee_id: str, attachment_stem: str) -> Path:
    return output_dir(guarantee_id, attachment_stem) / "meta.json"


def source_pdf_path(guarantee_id: str, attachment_stem: str) -> Path:
    return output_dir(guarantee_id, attachment_stem) / "source.pdf"


def parse_identifiers(filename: str) -> tuple[str, str]:
    """Derive guarantee_id and attachment_stem from a filename like '1589112_1.pdf'."""
    stem = Path(filename).stem
    parts = stem.rsplit("_", maxsplit=1)
    guarantee_id = parts[0] if len(parts) == 2 else stem
    return guarantee_id, stem
