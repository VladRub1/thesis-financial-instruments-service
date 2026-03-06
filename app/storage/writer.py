"""Artifact writer — serializes OCR and extraction outputs to disk."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from app.core.config import settings
from app.ocr.base import OCRDocument, PageResult
from app.storage import paths


def write_ocr_artifacts(
    doc: OCRDocument, guarantee_id: str, attachment_stem: str
) -> tuple[Path, Path]:
    """Write ocr.json and ocr.md, return their paths."""
    json_path = paths.ocr_json_path(guarantee_id, attachment_stem)
    md_path = paths.ocr_md_path(guarantee_id, attachment_stem)

    json_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(doc.pages), encoding="utf-8")
    return json_path, md_path


def write_extraction(data: dict, guarantee_id: str, attachment_stem: str) -> Path:
    p = paths.extraction_json_path(guarantee_id, attachment_stem)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return p


def write_meta(meta: dict, guarantee_id: str, attachment_stem: str) -> Path:
    p = paths.meta_json_path(guarantee_id, attachment_stem)
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return p


def copy_source_pdf(src: Path, guarantee_id: str, attachment_stem: str) -> Path | None:
    if not settings.COPY_SOURCE_PDF:
        return None
    dst = paths.source_pdf_path(guarantee_id, attachment_stem)
    shutil.copy2(src, dst)
    return dst


def _render_markdown(pages: list[PageResult]) -> str:
    parts: list[str] = []
    for page in pages:
        parts.append(f"# Page {page.page_index + 1}\n")
        sorted_blocks = sorted(page.blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
        for block in sorted_blocks:
            parts.append(block.text)
            parts.append("")
        parts.append("")
    return "\n".join(parts).strip() + "\n"
