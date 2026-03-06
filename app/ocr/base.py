"""Abstract OCR engine and shared result data structures."""
from __future__ import annotations

import abc
from datetime import datetime

from pydantic import BaseModel


class WordResult(BaseModel):
    text: str
    bbox: list[int]  # [x0, y0, x1, y1]
    confidence: float


class LineResult(BaseModel):
    text: str
    bbox: list[int]
    confidence: float
    words: list[WordResult] = []


class BlockResult(BaseModel):
    text: str
    bbox: list[int]
    confidence: float
    lines: list[LineResult] = []


class PageResult(BaseModel):
    page_index: int
    width: int
    height: int
    dpi: int
    blocks: list[BlockResult] = []


class OCRDocument(BaseModel):
    doc_id: str
    attachment_id: str
    engine: str
    engine_version: str
    created_at: str
    pages: list[PageResult] = []


class OCREngine(abc.ABC):
    """Interface every OCR backend must implement."""

    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def version(self) -> str: ...

    @abc.abstractmethod
    def run_page(self, image, dpi: int = 300) -> PageResult:
        """Run OCR on a single PIL Image and return structured result."""
        ...

    def build_document(
        self, pages: list[PageResult], doc_id: str, attachment_id: str
    ) -> OCRDocument:
        return OCRDocument(
            doc_id=doc_id,
            attachment_id=attachment_id,
            engine=self.name(),
            engine_version=self.version(),
            created_at=datetime.utcnow().isoformat(),
            pages=pages,
        )
