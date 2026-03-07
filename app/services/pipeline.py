"""OCR + LLM orchestration pipeline — runs inside the worker, never in API handlers."""
from __future__ import annotations

import json
import time
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from app.core.logging import get_logger
from app.llm.engine import LLMEngine
from app.llm.extract import extract_fields
from app.llm.schemas import ExtractionResult
from app.ocr.base import OCRDocument, OCREngine, PageResult
from app.ocr.tesseract import TesseractEngine
from app.storage import paths as sp
from app.storage.writer import copy_source_pdf, write_extraction, write_meta, write_ocr_artifacts

log = get_logger(__name__)

_RENDER_DPI = 300


def _fmt_duration(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


def get_ocr_engine(name: str, lang: str = "rus+eng") -> OCREngine:
    if name == "tesseract":
        return TesseractEngine(lang=lang)
    if name == "paddleocr":
        from app.ocr.paddle import PaddleEngine
        return PaddleEngine(lang=lang)
    raise ValueError(f"Unknown OCR engine: {name}")


def pdf_to_images(pdf_path: Path) -> list[tuple[Image.Image, int]]:
    suffix = pdf_path.suffix.lower()

    if suffix in (".tif", ".tiff"):
        img = Image.open(str(pdf_path))
        pages: list[tuple[Image.Image, int]] = []
        for i in range(getattr(img, "n_frames", 1)):
            img.seek(i)
            frame = img.copy().convert("RGB")
            dpi_info = img.info.get("dpi", (_RENDER_DPI, _RENDER_DPI))
            dpi = int(dpi_info[0]) if isinstance(dpi_info, (tuple, list)) else int(dpi_info)
            pages.append((frame, dpi))
        return pages

    if suffix in (".png", ".jpg", ".jpeg"):
        img = Image.open(str(pdf_path)).convert("RGB")
        return [(img, _RENDER_DPI)]

    doc = fitz.open(str(pdf_path))
    images: list[tuple[Image.Image, int]] = []
    for page in doc:
        pix = page.get_pixmap(dpi=_RENDER_DPI)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append((img, _RENDER_DPI))
    doc.close()
    return images


def run_ocr(
    pdf_path: Path,
    engine: OCREngine,
    guarantee_id: str,
    attachment_stem: str,
    *,
    on_page_done: callable | None = None,
) -> tuple[OCRDocument, Path, Path, float]:
    """Run OCR on all pages.  Returns (doc, json_path, md_path, duration_ms)."""
    log.info("[%s] Rendering PDF to images …", attachment_stem)
    t_render = time.perf_counter()
    images = pdf_to_images(pdf_path)
    render_ms = (time.perf_counter() - t_render) * 1000
    log.info("[%s] Rendered %d page(s) in %s", attachment_stem, len(images), _fmt_duration(render_ms))

    t0 = time.perf_counter()
    page_results: list[PageResult] = []
    for idx, (img, dpi) in enumerate(images):
        t_page = time.perf_counter()
        pr = engine.run_page(img, dpi=dpi)
        pr.page_index = idx
        page_results.append(pr)
        page_ms = (time.perf_counter() - t_page) * 1000
        elapsed_total = (time.perf_counter() - t0) * 1000
        remaining = len(images) - (idx + 1)
        avg_per_page = elapsed_total / (idx + 1)
        eta_ms = avg_per_page * remaining
        log.info(
            "[%s] OCR page %d/%d done in %s  |  elapsed: %s  |  ETA: %s",
            attachment_stem,
            idx + 1,
            len(images),
            _fmt_duration(page_ms),
            _fmt_duration(elapsed_total),
            _fmt_duration(eta_ms) if remaining > 0 else "—",
        )
        if on_page_done:
            on_page_done(idx + 1, len(images))

    ocr_doc = engine.build_document(page_results, doc_id=guarantee_id, attachment_id=attachment_stem)
    duration_ms = (time.perf_counter() - t0) * 1000
    log.info("[%s] OCR complete: %d pages in %s", attachment_stem, len(images), _fmt_duration(duration_ms))

    json_path, md_path = write_ocr_artifacts(ocr_doc, guarantee_id, attachment_stem)
    copy_source_pdf(pdf_path, guarantee_id, attachment_stem)
    return ocr_doc, json_path, md_path, duration_ms


def run_extraction(
    ocr_md: str,
    layout_json: dict | None,
    guarantee_id: str,
    attachment_stem: str,
    llm_engine: LLMEngine,
) -> tuple[ExtractionResult, Path | None, float]:
    """Run LLM extraction.  Returns (result, extraction_path_or_none, duration_ms)."""
    log.info("[%s] Starting LLM extraction …", attachment_stem)
    t0 = time.perf_counter()
    result = extract_fields(ocr_md, layout_json, engine=llm_engine)
    duration_ms = (time.perf_counter() - t0) * 1000
    log.info("[%s] LLM extraction finished in %s — status: %s", attachment_stem, _fmt_duration(duration_ms), result.status)

    ext_path: Path | None = None
    if result.validated:
        ext_path = write_extraction(result.validated.model_dump(mode="json"), guarantee_id, attachment_stem)

    return result, ext_path, duration_ms


def run_full_pipeline(
    pdf_path: Path,
    *,
    engine_ocr_name: str = "tesseract",
    lang: str = "rus+eng",
    pipeline: str = "ocr+extract",
    llm_engine: LLMEngine | None = None,
    on_page_done: callable | None = None,
) -> dict:
    """Complete pipeline for one PDF.  Called by the worker or CLI."""
    guarantee_id, attachment_stem = sp.parse_identifiers(pdf_path.name)
    t_total = time.perf_counter()
    log.info(
        "━━━ Pipeline start: %s  |  engine=%s  lang=%s  pipeline=%s ━━━",
        pdf_path.name, engine_ocr_name, lang, pipeline,
    )

    log.info("[%s] Initialising OCR engine: %s …", attachment_stem, engine_ocr_name)
    t_init = time.perf_counter()
    ocr_engine = get_ocr_engine(engine_ocr_name, lang)
    init_ms = (time.perf_counter() - t_init) * 1000
    log.info("[%s] OCR engine ready in %s", attachment_stem, _fmt_duration(init_ms))

    ocr_doc, json_path, md_path, ocr_ms = run_ocr(
        pdf_path, ocr_engine, guarantee_id, attachment_stem, on_page_done=on_page_done,
    )
    ocr_md = md_path.read_text(encoding="utf-8")

    result_dict: dict = {
        "guarantee_id": guarantee_id,
        "attachment_stem": attachment_stem,
        "page_count": len(ocr_doc.pages),
        "ocr_json_path": str(json_path),
        "ocr_md_path": str(md_path),
        "duration_ms_ocr": round(ocr_ms),
        "extraction_json_path": None,
        "extraction_result": None,
        "duration_ms_llm": None,
    }

    if pipeline == "ocr+extract" and llm_engine is not None:
        layout = json.loads(json_path.read_text(encoding="utf-8"))
        ext_result, ext_path, llm_ms = run_extraction(
            ocr_md, layout, guarantee_id, attachment_stem, llm_engine,
        )
        result_dict["extraction_json_path"] = str(ext_path) if ext_path else None
        result_dict["extraction_result"] = ext_result
        result_dict["duration_ms_llm"] = round(llm_ms)

    meta = {
        "engine_ocr": ocr_engine.name(),
        "engine_ocr_version": ocr_engine.version(),
        "lang": lang,
        "pipeline": pipeline,
        "duration_ms_ocr": result_dict["duration_ms_ocr"],
        "duration_ms_llm": result_dict.get("duration_ms_llm"),
    }
    meta_path = write_meta(meta, guarantee_id, attachment_stem)
    result_dict["meta_path"] = str(meta_path)

    total_ms = (time.perf_counter() - t_total) * 1000
    log.info(
        "━━━ Pipeline done: %s  |  OCR: %s  |  LLM: %s  |  Total: %s ━━━",
        pdf_path.name,
        _fmt_duration(result_dict["duration_ms_ocr"]),
        _fmt_duration(result_dict["duration_ms_llm"]) if result_dict["duration_ms_llm"] else "skipped",
        _fmt_duration(total_ms),
    )
    return result_dict
