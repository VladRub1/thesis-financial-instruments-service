from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_NAME: str = "thesis-financial-instruments-service"
    DEBUG: bool = False
    ADMIN_API_KEY: str = ""

    # Postgres (async)
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/docai"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # OCR
    DEFAULT_OCR_ENGINE: str = "tesseract"
    TESSERACT_CMD: str = "tesseract"

    # LLM
    LLM_MODEL_PATH: str = "/models/qwen2.5-3b-instruct-q4_k_m.gguf"
    LLM_N_CTX: int = 4096
    LLM_MAX_TOKENS: int = 2048
    LLM_TEMPERATURE: float = 0.0
    LLM_TOP_P: float = 1.0
    LLM_THREADS: int = 4
    LLM_MAX_RETRIES: int = 2

    # Storage
    PROCESSED_DIR: str = "data/processed"
    COPY_SOURCE_PDF: bool = False

    # Security — semicolon-separated absolute paths
    ALLOWED_INPUT_ROOTS: str = ""

    # Worker
    WORKER_MAX_JOBS: int = 2

    # --- derived helpers (not env vars) ---

    @property
    def processed_path(self) -> Path:
        return Path(self.PROCESSED_DIR)

    @property
    def allowed_roots(self) -> list[Path]:
        if not self.ALLOWED_INPUT_ROOTS:
            return []
        return [Path(p.strip()) for p in self.ALLOWED_INPUT_ROOTS.split(";") if p.strip()]


settings = Settings()
