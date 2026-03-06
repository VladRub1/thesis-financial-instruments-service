"""Tesseract OCR engine implementation."""
from __future__ import annotations

import pytesseract
from PIL import Image

from app.core.config import settings
from app.ocr.base import (
    BlockResult,
    LineResult,
    OCREngine,
    PageResult,
    WordResult,
)
from app.ocr.preprocess import preprocess

pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD


class TesseractEngine(OCREngine):
    def __init__(self, lang: str = "rus+eng"):
        self._lang = lang

    def name(self) -> str:
        return "tesseract"

    def version(self) -> str:
        try:
            return pytesseract.get_tesseract_version().vstring  # type: ignore[union-attr]
        except Exception:
            return "unknown"

    def run_page(self, image: Image.Image, dpi: int = 300) -> PageResult:
        processed = preprocess(image, current_dpi=dpi)
        data = pytesseract.image_to_data(
            processed, lang=self._lang, output_type=pytesseract.Output.DICT
        )
        blocks = self._assemble_blocks(data)
        return PageResult(
            page_index=0,
            width=processed.width,
            height=processed.height,
            dpi=dpi,
            blocks=blocks,
        )

    @staticmethod
    def _assemble_blocks(data: dict) -> list[BlockResult]:
        blocks_map: dict[int, list[dict]] = {}
        n = len(data["text"])
        for i in range(n):
            block_num = data["block_num"][i]
            blocks_map.setdefault(block_num, []).append(
                {
                    "text": data["text"][i],
                    "left": data["left"][i],
                    "top": data["top"][i],
                    "width": data["width"][i],
                    "height": data["height"][i],
                    "conf": float(data["conf"][i]),
                    "line_num": data["line_num"][i],
                    "word_num": data["word_num"][i],
                }
            )

        result: list[BlockResult] = []
        for _block_num, items in sorted(blocks_map.items()):
            lines_map: dict[int, list[dict]] = {}
            for it in items:
                lines_map.setdefault(it["line_num"], []).append(it)

            lines: list[LineResult] = []
            block_bbox = [999999, 999999, 0, 0]
            block_confs: list[float] = []

            for _line_num, words_raw in sorted(lines_map.items()):
                word_objs: list[WordResult] = []
                line_bbox = [999999, 999999, 0, 0]
                line_confs: list[float] = []
                line_texts: list[str] = []

                for w in words_raw:
                    txt = (w["text"] or "").strip()
                    if not txt:
                        continue
                    x0, y0 = w["left"], w["top"]
                    x1, y1 = x0 + w["width"], y0 + w["height"]
                    conf = max(w["conf"], 0.0) / 100.0
                    word_objs.append(WordResult(text=txt, bbox=[x0, y0, x1, y1], confidence=conf))
                    line_bbox = [
                        min(line_bbox[0], x0), min(line_bbox[1], y0),
                        max(line_bbox[2], x1), max(line_bbox[3], y1),
                    ]
                    line_confs.append(conf)
                    line_texts.append(txt)

                if not word_objs:
                    continue
                avg_conf = sum(line_confs) / len(line_confs) if line_confs else 0.0
                lines.append(
                    LineResult(
                        text=" ".join(line_texts),
                        bbox=line_bbox,
                        confidence=round(avg_conf, 4),
                        words=word_objs,
                    )
                )
                block_bbox = [
                    min(block_bbox[0], line_bbox[0]), min(block_bbox[1], line_bbox[1]),
                    max(block_bbox[2], line_bbox[2]), max(block_bbox[3], line_bbox[3]),
                ]
                block_confs.extend(line_confs)

            if not lines:
                continue
            avg_block_conf = sum(block_confs) / len(block_confs) if block_confs else 0.0
            block_text = "\n".join(ln.text for ln in lines)
            result.append(
                BlockResult(
                    text=block_text,
                    bbox=block_bbox,
                    confidence=round(avg_block_conf, 4),
                    lines=lines,
                )
            )
        return result
