"""llama-cpp-python wrapper — singleton model loader."""
from __future__ import annotations

import threading

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

_lock = threading.Lock()
_instance: LLMEngine | None = None


class LLMEngine:
    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        from llama_cpp import Llama  # type: ignore[import-untyped]

        log.info("Loading LLM model from %s …", settings.LLM_MODEL_PATH)
        self._model = Llama(
            model_path=settings.LLM_MODEL_PATH,
            n_ctx=settings.LLM_N_CTX,
            n_threads=settings.LLM_THREADS,
            verbose=settings.DEBUG,
        )
        log.info("LLM model loaded.")

    def generate(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict | None = None,
    ) -> str:
        """Run chat completion. If *json_schema* is given, enable constrained decoding."""
        if self._model is None:
            raise RuntimeError("LLM model not loaded — call .load() first")

        kwargs: dict = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": settings.LLM_MAX_TOKENS,
            "temperature": settings.LLM_TEMPERATURE,
            "top_p": settings.LLM_TOP_P,
        }
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_object",
                "schema": json_schema,
            }

        resp = self._model.create_chat_completion(**kwargs)
        return resp["choices"][0]["message"]["content"]  # type: ignore[index]

    def unload(self) -> None:
        self._model = None


def get_engine() -> LLMEngine:
    """Return the process-wide singleton engine (thread-safe)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = LLMEngine()
    return _instance
