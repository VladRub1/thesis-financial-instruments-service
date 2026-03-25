"""llama-cpp-python wrapper — singleton model loader."""
from __future__ import annotations

import threading

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

_lock = threading.Lock()
_instance: LLMEngine | None = None


class LLMEngine:
    def __init__(
        self,
        *,
        n_gpu_layers: int | None = None,
        require_gpu_offload: bool = False,
    ) -> None:
        self._model = None
        self._n_gpu_layers = n_gpu_layers
        self._require_gpu_offload = require_gpu_offload

    def load(self) -> None:
        if self._model is not None:
            return
        from llama_cpp import Llama  # type: ignore[import-untyped]

        log.info("Loading LLM model from %s …", settings.LLM_MODEL_PATH)
        gpu_offload_supported: bool | None = None
        try:
            import llama_cpp  # type: ignore[import-untyped]

            log.info(
                "llama_cpp module: %s (version=%s)",
                getattr(llama_cpp, "__file__", "unknown"),
                getattr(llama_cpp, "__version__", "unknown"),
            )
            support_fn = getattr(llama_cpp, "llama_supports_gpu_offload", None)
            if callable(support_fn):
                gpu_offload_supported = bool(support_fn())
        except Exception:
            gpu_offload_supported = None

        kwargs: dict = {
            "model_path": settings.LLM_MODEL_PATH,
            "n_ctx": settings.LLM_N_CTX,
            "n_threads": settings.LLM_THREADS,
            "verbose": settings.DEBUG,
        }
        if self._n_gpu_layers is not None:
            kwargs["n_gpu_layers"] = self._n_gpu_layers
            log.info("LLM offload request: n_gpu_layers=%s", self._n_gpu_layers)
        offload_requested = self._n_gpu_layers is not None and self._n_gpu_layers != 0
        if gpu_offload_supported is not None:
            log.info("llama.cpp GPU offload support: %s", gpu_offload_supported)
            if (
                offload_requested
                and not gpu_offload_supported
            ):
                log.warning(
                    "GPU offload requested, but llama.cpp reports no GPU offload support. "
                    "This build will run on CPU."
                )
            elif offload_requested and gpu_offload_supported:
                log.info("GPU offload requested and supported by this llama.cpp build.")
        if self._require_gpu_offload and offload_requested and gpu_offload_supported is False:
            raise RuntimeError(
                "GPU offload was requested, but llama.cpp reports no GPU support. "
                "Install CUDA-enabled llama-cpp-python in the same environment used by `uv run`. "
                "In Colab, `uv run` may re-sync from lock and replace custom builds; use "
                "`uv run --no-sync ...` (or `.venv/bin/python`) after installing the CUDA build."
            )
        self._model = Llama(
            **kwargs,
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
            "stop": ["```"],
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
