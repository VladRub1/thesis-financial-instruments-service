# Document AI PoC — Bank Guarantee Extraction Service

On-prem backend for processing scanned Russian bank guarantee documents.  
Runs **OCR** (Tesseract / PaddleOCR) and **LLM extraction** (llama-cpp-python with GGUF models) locally — no cloud calls.

## Architecture

```
┌───────────┐      ┌──────────┐      ┌───────┐      ┌────────────┐
│ Streamlit │─────▶│ FastAPI  │─────▶│ Redis │─────▶│   Worker   │
│    UI     │◀─────│   API    │      │ Queue │      │ OCR + LLM  │
└───────────┘      └──────────┘      └───────┘      └────────────┘
                        │                                 │
                        ▼                                 ▼
                   ┌──────────┐                    ┌────────────┐
                   │ Postgres │                    │ Filesystem │
                   │  (audit) │                    │ (artifacts)│
                   └──────────┘                    └────────────┘
```

**Flow:** Upload PDF → API creates job → Redis enqueues → Worker runs OCR page-by-page → Worker runs LLM extraction → Results stored in Postgres + filesystem → Client polls for status/results.

## Quick Start (local development)

### Prerequisites

- Python 3.12+, [uv](https://docs.astral.sh/uv/)
- Tesseract OCR with Russian language pack (`brew install tesseract tesseract-lang` on macOS)
- PostgreSQL 14+ and Redis 7+ (or use Docker Compose)
- A GGUF model file (e.g. `Qwen2.5-3B-Instruct-Q4_K_M.gguf`)

### 1. Install dependencies

```bash
uv sync
# For development tools (pytest, ruff, etc.):
uv sync --extra dev
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set DATABASE_URL, REDIS_URL, LLM_MODEL_PATH, ALLOWED_INPUT_ROOTS
```

### 3. Start infrastructure

```bash
# Option A: Docker Compose (recommended)
docker compose up -d postgres redis

# Option B: use local Postgres + Redis
```

### 4. Run database migrations

```bash
uv run alembic upgrade head
```

### 5. Start API server

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Start worker

```bash
uv run arq app.workers.tasks.WorkerSettings
```

### 7. Start Streamlit UI

```bash
uv run streamlit run streamlit_app.py
```

Open http://localhost:8501 in your browser.

## Docker Compose (full stack)

```bash
# Place your GGUF model in ./models/ directory
docker compose up --build
```

Services:
| Service   | Port | Description              |
|-----------|------|--------------------------|
| api       | 8000 | FastAPI backend          |
| worker    | —    | arq background worker    |
| streamlit | 8501 | Streamlit frontend       |
| postgres  | 5432 | PostgreSQL database      |
| redis     | 6379 | Redis job queue          |

## Using the Streamlit UI

Open http://localhost:8501 after starting all services. The sidebar lets you configure processing options before submitting a document.

### Sidebar settings

- **API URL** — address of the FastAPI backend (default `http://localhost:8000`; change if running remotely).
- **Pipeline** — `ocr+extract` runs OCR followed by LLM field extraction; `ocr_only` runs OCR and skips the LLM step (useful for inspecting raw OCR quality).
- **OCR engine** — `tesseract` (default, lightweight) or `paddleocr` (alternative engine, requires extra dependencies).
- **Language** — Tesseract language codes to use during recognition. `rus+eng` applies both Russian and English models simultaneously, which is important for bank guarantees that mix Cyrillic body text with Latin abbreviations, BIC codes, and legal references. Other common values: `rus` (Russian only), `eng` (English only). You can use any language code installed via `tesseract-lang`.

### Workflow

1. **Upload a PDF** — go to the "Upload PDF" tab, select a scanned bank guarantee PDF, and click "Process uploaded file". A progress bar shows page-by-page OCR progress.
2. **Or submit by server path** — go to the "By file path" tab, enter the absolute path to a PDF on the server (must be under an allowed root), and click "Process by path".
3. **View OCR output** — once processing completes, the left column shows the raw OCR text in Markdown format. Review it to check recognition quality.
4. **View extracted fields** — the right column shows the structured JSON extracted by the LLM: guarantee number, dates, amounts, INN, contract details, etc.
5. **Submit corrections** — if any extracted fields are wrong, edit them in the correction form below the extraction results, add an optional comment, and click "Submit corrections". Each correction is versioned and stored for audit.
6. **Look up past jobs** — go to the "Job results" tab and paste a job ID to fetch results from a previous run.

## API Endpoints

### Create job (file upload)

```bash
curl -X POST http://localhost:8000/v1/jobs \
  -F "file=@guarantee.pdf" \
  -F "pipeline=ocr+extract" \
  -F "engine_ocr=tesseract"
```

### Create job (server-side path)

```bash
curl -X POST http://localhost:8000/v1/jobs/by-path \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/data/raw/attachments/1589112/1589112_1.pdf"}'
```

### Poll job status

```bash
curl http://localhost:8000/v1/jobs/{job_id}
```

### Get results

```bash
curl http://localhost:8000/v1/jobs/{job_id}/result
```

### Submit corrections

```bash
curl -X POST http://localhost:8000/v1/jobs/{job_id}/corrections \
  -H "Content-Type: application/json" \
  -d '{"fields": {"guarantee_number": "BG-001-CORRECTED"}, "comment": "Fixed typo"}'
```

### Admin endpoints (require API key)

Admin endpoints are protected by the `ADMIN_API_KEY` environment variable. Set it in your `.env` file:

```
ADMIN_API_KEY=my-secret-key-123
```

No code changes needed — the API reads the key from `.env` at startup. Restart the API server after changing it.

Pass the key via the `X-API-Key` header:

```bash
# Service health — checks DB, Redis, and LLM model availability
curl http://localhost:8000/v1/admin/health \
  -H "X-API-Key: my-secret-key-123"

# List all jobs (with optional filters)
curl "http://localhost:8000/v1/admin/jobs?status=succeeded&limit=10" \
  -H "X-API-Key: my-secret-key-123"

# List failed jobs
curl "http://localhost:8000/v1/admin/jobs?status=failed" \
  -H "X-API-Key: my-secret-key-123"
```

A missing or wrong key returns `403 Forbidden`. If `ADMIN_API_KEY` is empty in `.env`, all admin endpoints are locked.

## CLI Tools

### Bulk OCR + extraction

```bash
uv run python -m app.cli.ocr \
  --input-root data/raw/attachments \
  --engine-ocr tesseract \
  --engine-llm llama_cpp \
  --pipeline ocr+extract \
  --workers 4 \
  --out bulk_manifest.json
```

### Evaluation scaffold

```bash
uv run python -m app.cli.evaluate \
  --input-root data/raw/attachments \
  --gt-root data/ground_truth \
  --out eval_report.json \
  --workers 2
```

## Running Tests

```bash
uv run pytest -q
```

Tests use an in-memory SQLite database and mock the Redis queue + LLM engine — no real infrastructure needed.

## Environment Variables

| Variable              | Default                        | Description                              |
|-----------------------|--------------------------------|------------------------------------------|
| `DATABASE_URL`        | `postgresql+asyncpg://…`       | Async Postgres connection string         |
| `REDIS_URL`           | `redis://localhost:6379/0`     | Redis for arq job queue                  |
| `ADMIN_API_KEY`       | (empty)                        | API key for admin endpoints              |
| `LLM_MODEL_PATH`     | `/models/qwen2.5-…`           | Path to GGUF model file                  |
| `LLM_N_CTX`          | `4096`                         | LLM context window                       |
| `LLM_MAX_TOKENS`     | `2048`                         | Max tokens for LLM response              |
| `LLM_TEMPERATURE`    | `0.0`                          | LLM temperature (0 = deterministic)      |
| `LLM_THREADS`        | `4`                            | CPU threads for LLM inference            |
| `LLM_MAX_RETRIES`    | `2`                            | Retry count for failed JSON validation   |
| `DEFAULT_OCR_ENGINE`  | `tesseract`                   | Default OCR engine                       |
| `TESSERACT_CMD`       | `tesseract`                   | Path to tesseract binary                 |
| `PROCESSED_DIR`       | `data/processed`              | Root for output artifacts                |
| `COPY_SOURCE_PDF`     | `false`                       | Copy source PDF into output directory    |
| `ALLOWED_INPUT_ROOTS` | (empty)                        | Semicolon-separated allowed path roots   |
| `WORKER_MAX_JOBS`     | `2`                            | Max concurrent jobs per worker           |

## Project Structure

```
app/
├── main.py                  # FastAPI application entry point
├── api/v1/
│   ├── jobs.py              # Job CRUD endpoints
│   └── admin.py             # Admin endpoints (API-key protected)
├── core/
│   ├── config.py            # Pydantic settings
│   ├── logging.py           # Logging setup
│   └── security.py          # API key + file path validation
├── db/
│   ├── models.py            # SQLAlchemy ORM models
│   └── session.py           # Async session factory
├── ocr/
│   ├── base.py              # OCR engine interface + result models
│   ├── preprocess.py        # Image preprocessing pipeline
│   ├── tesseract.py         # Tesseract engine
│   └── paddle.py            # PaddleOCR engine (optional)
├── llm/
│   ├── engine.py            # llama-cpp-python wrapper (singleton)
│   ├── schemas.py           # ExtractionV1 Pydantic model + validators
│   ├── prompts.py           # System/user prompt templates
│   ├── extract.py           # Extraction with retry logic
│   └── postprocess.py       # Post-extraction normalisation
├── services/
│   ├── jobs.py              # Job lifecycle (DB operations)
│   └── pipeline.py          # OCR + LLM orchestration
├── storage/
│   ├── paths.py             # Deterministic output paths
│   └── writer.py            # Artifact serialisation
├── workers/
│   └── tasks.py             # arq worker tasks
├── cli/
│   ├── ocr.py               # Bulk OCR CLI
│   └── evaluate.py          # Evaluation scaffold CLI
├── schemas/
│   └── jobs.py              # API request/response models
streamlit_app.py             # Streamlit frontend
tests/
├── conftest.py              # Fixtures (in-memory DB, mocked queue)
├── test_api_jobs.py         # API endpoint tests
├── test_extraction.py       # LLM extraction + validation tests
└── test_corrections.py      # Correction submission tests
alembic/                     # Database migrations
docker/Dockerfile
docker-compose.yml
```

## License

MIT
