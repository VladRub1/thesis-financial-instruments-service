"""PaddleOCR engine — optional, behind feature flag / extra deps.

Requires PaddleOCR >= 3.0 (PP-OCRv5).
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")

from PIL import Image

from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.ocr.base import (
    BlockResult,
    LineResult,
    OCREngine,
    PageResult,
    WordResult,
)

log = get_logger(__name__)

_LANG_TO_REC_MODEL: dict[str, str] = {
    "rus":     "eslav_PP-OCRv5_mobile_rec",
    "rus+eng": "eslav_PP-OCRv5_mobile_rec",
    "eng":     "en_PP-OCRv5_mobile_rec",
    "chi_sim": "PP-OCRv5_mobile_rec",
    "deu":     "latin_PP-OCRv5_mobile_rec",
    "fra":     "latin_PP-OCRv5_mobile_rec",
}


def _rec_model(tesseract_lang: str) -> str:
    key = tesseract_lang.strip().lower()
    return _LANG_TO_REC_MODEL.get(key, "eslav_PP-OCRv5_mobile_rec")


def _assert_paddle_runtime_compatibility() -> None:
    """Fail fast on incompatible Paddle runtime variants (common in Colab)."""
    try:
        import paddle  # type: ignore[import-untyped]
        from paddle import inference as paddle_infer  # type: ignore[import-untyped]
    except Exception as exc:
        raise RuntimeError(
            "Paddle runtime is unavailable. Install compatible versions: "
            "paddleocr==3.3.3 and paddlepaddle==3.2.0."
        ) from exc

    config_cls = getattr(paddle_infer, "Config", None)
    has_opt_level = bool(config_cls) and hasattr(config_cls, "set_optimization_level")
    if not has_opt_level:
        paddle_version = getattr(paddle, "__version__", "unknown")
        raise RuntimeError(
            "Incompatible paddlepaddle runtime detected "
            f"(found: {paddle_version}). PaddleOCR 3.3.3 expects "
            "`paddle.inference.Config.set_optimization_level`. "
            "Do not use `paddlepaddle-gpu` 2.x here; install `paddlepaddle==3.2.0`."
        )


class PaddleEngine(OCREngine):
    def __init__(self, lang: str = "rus+eng"):
        cache_home = Path(settings.PADDLE_PDX_CACHE_HOME)
        cache_home.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_home.resolve()))

        try:
            from paddleocr import PaddleOCR  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR is not installed. Install with: uv pip install paddleocr paddlepaddle"
            ) from exc

        _assert_paddle_runtime_compatibility()
        rec_model = _rec_model(lang)
        self._ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name=rec_model,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
            enable_mkldnn=True,
            cpu_threads=8,
        )
        log.info("PaddleOCR ready (device=cpu)")
        # Paddle may reconfigure root logging; restore project logging format/level.
        setup_logging(settings.DEBUG)

    def name(self) -> str:
        return "paddleocr"

    def version(self) -> str:
        try:
            import paddleocr  # type: ignore[import-untyped]
            return getattr(paddleocr, "__version__", "unknown")
        except Exception:
            return "unknown"

    def run_page(self, image: Image.Image, dpi: int = 300) -> PageResult:
        """PaddleOCR has its own preprocessing — skip ours, pass RGB directly."""
        import numpy as np

        img_arr = np.array(image.convert("RGB"))

        blocks: list[BlockResult] = []
        try:
            results = list(self._ocr.predict(img_arr))
            for res in results:
                texts, scores, polys, boxes = self._extract_fields(res, log)
                for i, text in enumerate(texts):
                    if not text or not text.strip():
                        continue
                    conf = float(scores[i]) if i < len(scores) else 0.0
                    bbox = self._get_bbox(polys, boxes, i)
                    word = WordResult(text=text, bbox=bbox, confidence=round(conf, 4))
                    line = LineResult(text=text, bbox=bbox, confidence=round(conf, 4), words=[word])
                    blocks.append(
                        BlockResult(text=text, bbox=bbox, confidence=round(conf, 4), lines=[line])
                    )
        except Exception as exc:
            raise RuntimeError(f"PaddleOCR prediction failed: {exc}") from exc

        log.info("PaddleOCR extracted %d text blocks from page", len(blocks))
        return PageResult(
            page_index=0,
            width=image.width,
            height=image.height,
            dpi=dpi,
            blocks=blocks,
        )

    @staticmethod
    def _extract_fields(res, log) -> tuple[list, list, object, object]:
        """Handle multiple PaddleOCR v3 result formats."""
        # Try dict-style access (res might be a dict or have a dict 'res' attr)
        d = None
        if isinstance(res, dict):
            d = res.get("res", res)
        elif hasattr(res, "res") and isinstance(res.res, dict):
            d = res.res
        elif hasattr(res, "__dict__"):
            d = res.__dict__

        if d and isinstance(d, dict):
            texts = d.get("rec_texts", [])
            scores = list(d.get("rec_scores", []))
            polys = d.get("dt_polys", d.get("rec_polys", None))
            boxes = d.get("rec_boxes", None)
            if texts:
                return list(texts), scores, polys, boxes

        # Try direct attribute access
        for attr_texts in ("rec_texts", "texts"):
            texts = getattr(res, attr_texts, None)
            if texts is not None and len(texts) > 0:
                scores = list(getattr(res, "rec_scores", getattr(res, "scores", [])))
                polys = getattr(res, "dt_polys", getattr(res, "rec_polys", None))
                boxes = getattr(res, "rec_boxes", None)
                return list(texts), scores, polys, boxes

        # Log the result structure for debugging
        log.warning(
            "PaddleOCR result has unexpected structure: type=%s, dir=%s",
            type(res).__name__,
            [a for a in dir(res) if not a.startswith("_")],
        )
        return [], [], None, None

    @staticmethod
    def _get_bbox(polys, boxes, i: int) -> list[int]:
        if polys is not None and i < len(polys):
            pts = polys[i]
            x_coords = [int(p[0]) for p in pts]
            y_coords = [int(p[1]) for p in pts]
            return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        if boxes is not None and i < len(boxes):
            b = boxes[i]
            return [int(b[0]), int(b[1]), int(b[2]), int(b[3])]
        return [0, 0, 0, 0]
