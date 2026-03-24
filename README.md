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
- A GGUF model file — see **Model download** below

### 1. Install dependencies

```bash
uv sync
# For development tools (pytest, ruff, etc.):
uv sync --extra dev
# Add `--extra paddle` if PaddleOCR is needed locally
```

#### llama-cpp-python install notes

**macOS (Apple Silicon / Metal acceleration):**

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

The `uv sync` step installs the default wheel which already enables Metal on Apple Silicon.
If you experience issues, reinstall with the flag above.

**Linux VM (CPU only):**

```bash
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python
```

Or simply `uv sync` — the default build uses CPU. Install `gcc`, `g++`, `cmake`, `make` if building from source (already handled in Docker).

### Model download

Download the default GGUF model into `./models/`:

```bash
uv run python -m app.cli.download_model
```

This downloads **Qwen3-4B-Instruct-2507-Q5_K_M.gguf** (~2.9 GB) from [unsloth/Qwen3-4B-Instruct-2507-GGUF](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF) and saves it as `models/qwen3-4b-instruct-2507-q5_k_m.gguf` (lowercase).

For faster downloads, authenticate with Hugging Face (one of):

```bash
huggingface-cli login          # interactive, cached for future runs
export HF_TOKEN=hf_...         # or pass --token hf_...
```

To download a different quant or verify integrity:

```bash
uv run python -m app.cli.download_model \
  --file Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --local-name qwen3-4b-instruct-2507-q4_k_m.gguf \
  --sha256 <expected-hash>
```

If you already have a model file, place it in `./models/` and set `LLM_MODEL_PATH` in `.env`.

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set DATABASE_URL, REDIS_URL, LLM_MODEL_PATH, ALLOWED_INPUT_ROOTS
# For public demo: set DEMO_PASSWORD to gate the Streamlit UI
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

Run the entire application with a single command — no local Python, Tesseract, or Postgres install needed.

### Prerequisites

1. Docker and Docker Compose installed
2. A GGUF model file placed in `./models/` (run `uv run python -m app.cli.download_model` or place manually)
3. A `.env` file (`cp .env.example .env` — defaults work out of the box for Docker)
4. Optional: set `DOCKER_ENABLE_PADDLE=true` in `.env` to include PaddleOCR in the image build

### Start

```bash
docker compose up --build
```

This will:
- Build the application image (installs Python, Tesseract, all dependencies)
- Start PostgreSQL and Redis
- Run database migrations automatically (Alembic)
- Start the API server, background worker, and Streamlit UI

### PaddleOCR in Docker (optional)

PaddleOCR is controlled by a compose build flag, so you can enable/disable it without editing `docker/Dockerfile`:

```bash
# .env
DOCKER_ENABLE_PADDLE=true

# rebuild image with Paddle dependencies
docker compose up --build -d
```

Notes:
- `DOCKER_ENABLE_PADDLE=false` (default) keeps the image smaller and avoids platform issues.
- Paddle wheels are typically available on Linux `x86_64`.
- On Linux `aarch64` (common on Apple Silicon Docker hosts), Paddle builds may fail due to missing wheels.
- For server stability, this project pins `paddlepaddle==3.2.0` and `paddleocr==3.3.3` to avoid OneDNN runtime crashes (`ConvertPirAttribute2RuntimeAttribute`).
- Paddle model cache is persisted at `PADDLE_PDX_CACHE_HOME` (compose default: `/app/data/paddle-cache`) so models are not re-downloaded on each container restart.

### Access

| Service   | URL / Port | Description |
|-----------|------------|-------------|
| Streamlit | http://localhost:8501 | Web UI for document upload and results |
| API docs  | http://localhost:8000/docs | Swagger / OpenAPI interface |
| API       | http://localhost:8000 | REST API |
| PostgreSQL | localhost:5432 | Database (user: postgres, pass: postgres, db: docai) |
| Redis     | localhost:6379 | Job queue |

For public deployment behind Caddy/reverse proxy, keep FastAPI internal and expose only Streamlit.

### Shared storage

All containers share a Docker volume (`app-data`) for:
- **Uploads** (`/app/data/uploads`) — files uploaded via the API; accessible by the worker for processing
- **Processed** (`/app/data/processed`) — OCR and extraction artifacts; accessible by Streamlit for display

The model directory (`./models/`) is bind-mounted read-only into all app containers.

### Stop / reset

```bash
# Stop all services (data persists)
docker compose down

# Stop and delete all data (clean slate)
docker compose down -v
```

### Migrating to a remote server

1. Copy the project to the server
2. Place the GGUF model in `./models/`
3. Copy or create `.env` (defaults work; set `ADMIN_API_KEY` and `DEMO_PASSWORD` for public demo; set `DOCKER_ENABLE_PADDLE=true` only if you need PaddleOCR and host is compatible)
4. Run `docker compose up --build -d`
5. Expose Streamlit (e.g. via Caddy) at `https://thesis-guarantee.ru/`; keep API internal

## Using the Streamlit UI

Open http://localhost:8501 after starting all services. The sidebar lets you configure processing options before submitting a document.

### Public demo login gate

- Set `DEMO_PASSWORD` in `.env`.
- Until the correct password is entered, users only see a macOS-style login window with a random ASCII art and haiku.
- Auth state is stored in Streamlit session state for the current browser session.
- Login UI lives in `app/ui/login_gate.py` and `app/ui/demo_content.py`.

### Sidebar settings

- **Pipeline** — `ocr+extract` runs OCR followed by LLM field extraction; `ocr_only` runs OCR and skips the LLM step (useful for inspecting raw OCR quality).
- **OCR engine** — `tesseract` (default, lightweight) or `paddleocr` (requires Docker build with `DOCKER_ENABLE_PADDLE=true` or local install with `uv sync --extra paddle`; server mode uses pinned CPU-only Paddle settings).
- **Language** — Tesseract language codes to use during recognition. `rus+eng` applies both Russian and English models simultaneously, which is important for bank guarantees that mix Cyrillic body text with Latin abbreviations, BIC codes, and legal references. Other common values: `rus` (Russian only), `eng` (English only). You can use any language code installed via `tesseract-lang`.

### Workflow

Public demo protections:
- Processing typically takes **2–3 minutes** per document.
- If the worker is busy, jobs may remain in `queued` state before processing starts.
- Streamlit limits uploads to **10 MB** (`.streamlit/config.toml`).
- New submissions are disabled while one job is actively tracked in the same browser session.

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

## Validation & Metrics

A standalone CLI subsystem for systematic evaluation of OCR + extraction quality against a ground-truth dataset.

### What it does

1. **Samples** N documents reproducibly from `dataset.csv` (the ground-truth table of ~33k bank guarantee records).
2. **Processes** each document through a configurable pipeline: OCR engine (tesseract or paddleocr) + extractor (LLM or regex baseline).
3. **Normalises** both predictions and gold values (digits-only for INNs/IKZ, ISO dates, ISO currency codes, robust float parsing).
4. **Computes metrics** — field-level exact match, slot-filling P/R/F1, Doc-EM, ANLS edit similarity, digit accuracy, amount MAE, latency percentiles, weighted scores.
5. **Generates** a Markdown report comparing all system configurations side-by-side.

Processing is **parallel** (multiprocessing for OCR, controlled concurrency for LLM) and **resumable** (checkpointed every batch to parquet).

### Ground-truth dataset

The file `dataset.csv` (external, not in this repo) contains one row per bank guarantee scan with columns: `id`, `bank_inn`, `bank_name`, `pcpl_inn`, `bene_inn`, `issue_date`, `start_date`, `end_date`, `sum`, `currency`, `ikz`, `stored_filename`, `stored_path`.

The document ID can be obtained from the file name (e.g. `.../1962714/1962714_1.pdf` → ID `1962714`).

`bank_inn` is metadata for grouping analysis; it is **not** an evaluated extraction field.

**Evaluated fields:** `pcpl_inn`, `bene_inn`, `issue_date`, `start_date`, `end_date`, `sum`, `currency`, `ikz`.

### Metrics included

| Metric | Description |
|--------|-------------|
| **Field accuracy** (micro/macro) | Fraction of exact matches per field after normalisation |
| **Slot P/R/F1** (micro/macro) | TP if match, FP+FN if wrong, FN if missing |
| **Doc-EM** | 1 if all required fields match for a document |
| **Normalised edit similarity** | Levenshtein-based, ANLS-style (string/identifier fields) |
| **Digit accuracy** | Position-aligned digit match ratio (INN/IKZ) |
| **Amount MAE** | Mean absolute error on `sum` |
| **Amount tolerance accuracy** | Fraction within configurable epsilon (default 0.01) |
| **Latency** | Median, P95, P99 for OCR / extraction / total |
| **Weighted field accuracy** | User-supplied per-field weights |

### CLI commands

All commands are run via:

```bash
uv run python -m app.cli.validate <command> [options]
```

#### 1.1 `sample` — create a reproducible document subset

```bash
uv run python -m app.cli.validate sample --n 200 --seed 42
# Output: data/processed/validation/seeds/seed_n=200_seed=42.csv
```

Options:
- `--n <int>` — number of documents to sample (required)
- `--seed <int>` — random seed (default: 42)
- `--dataset <path>` — path to dataset.csv (has a sensible default)

The seed file is a CSV containing all GT columns for the sampled rows. Re-running with the same `--n` and `--seed` produces the identical sample.

#### 1.2 Bundle a sampled subset (helper)

```bash
uv run python app/validation/bundle_validation_sample.py \
  --seed-file data/processed/validation/seeds/seed_n=10_seed=42.csv \
  --out-dir data/processed/validation/bundles/n10_seed42 \
  --archive data/processed/validation/bundles/n10_seed42.tar.gz
```

#### 1.3 Fetch a validation bundle from Dropbox (helper)

```bash
bash scripts/fetch_validation_bundle.sh \
  "https://www.dropbox.com/scl/fi/<id>/validation_bundle.tar.gz?dl=0" \
  data/processed/validation/bundles/from_dropbox \
  validation_bundle.tar.gz
```

### Colab GPU setup (validation-only)

Validation CLI can run LLM extraction with CUDA in Colab while leaving the web service CPU-oriented.

```bash
# System deps
apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng

# One-command setup: sync (colab extra) + CUDA rebuild + verification
bash scripts/colab_gpu_setup.sh

# One-command setup with PaddleOCR CPU runtime + compatibility checks
COLAB_WITH_PADDLE=1 bash scripts/colab_gpu_setup.sh

# Optional manual PaddleOCR CPU install (Colab, validation-only)
# uv pip install --python .venv/bin/python --force-reinstall \
#   paddleocr==3.3.3 paddlepaddle==3.2.0
# Do not install paddlepaddle-gpu 2.x here; it is incompatible with PaddleOCR 3.3.3.

# If you prefer manual setup, run:
uv sync --extra colab --frozen
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
  uv pip install --python .venv/bin/python --force-reinstall --no-cache-dir llama-cpp-python==0.3.16

# Verify GPU offload support from that environment
uv run --no-sync python -c "import llama_cpp; support_fn=getattr(llama_cpp, 'llama_supports_gpu_offload', None); print('llama_cpp module:', getattr(llama_cpp, '__file__', 'unknown')); print('llama_cpp version:', getattr(llama_cpp, '__version__', 'unknown')); print('gpu_offload_support:', bool(support_fn()) if callable(support_fn) else 'unknown')"
```

If `uv pip` prints `Using Python ... at: /usr`, you are installing into the wrong environment; pass `--python .venv/bin/python` as shown above. In Colab, run sync first, then CUDA reinstall, and avoid running `uv sync` again afterwards in the same session. Use `uv run --no-sync ...` for verification and validation so lockfile sync does not replace your CUDA build. In notebooks, use `%env UV_NO_SYNC=1` (not `!export UV_NO_SYNC=1`) if you want the setting to persist across cells. `--ocr-engine paddleocr` runs on CPU in this setup.

Run validation with CUDA offload:

```bash
uv run --no-sync python -m app.cli.validate run \
  --seed-file data/processed/validation/seeds/seed_n=200_seed=42.csv \
  --ocr-engine tesseract --extractor llm \
  --llm-model models/qwen3-4b-instruct-2507-q5_k_m.gguf \
  --llm-device cuda --llm-n-gpu-layers -1
```


#### 2. `run` — process documents through OCR + extraction

```bash
# Tesseract + LLM
uv run python -m app.cli.validate run \
  --seed-file data/processed/validation/seeds/seed_n=200_seed=42.csv \
  --ocr-engine tesseract --extractor llm \
  --llm-model models/qwen3-4b-instruct-2507-q5_k_m.gguf \
  --workers 4 --batch-size 10

# PaddleOCR + LLM
uv run python -m app.cli.validate run \
  --seed-file data/processed/validation/seeds/seed_n=200_seed=42.csv \
  --ocr-engine paddleocr --extractor llm \
  --llm-model models/qwen3-4b-instruct-2507-q5_k_m.gguf \
  --workers 2

# Tesseract + regex baseline
uv run python -m app.cli.validate run \
  --seed-file data/processed/validation/seeds/seed_n=200_seed=42.csv \
  --ocr-engine tesseract --extractor regex --workers 4

# PaddleOCR + regex baseline
uv run python -m app.cli.validate run \
  --seed-file data/processed/validation/seeds/seed_n=200_seed=42.csv \
  --ocr-engine paddleocr --extractor regex --workers 2
```

Options:
- `--ocr-engine {tesseract,paddleocr}` — OCR backend (required)
- `--extractor {llm,regex}` — extraction method (required)
- `--llm-model <path>` — GGUF model file (required for `--extractor llm`)
- `--llm-device {cpu,cuda}` — validation LLM device (default: `cpu`)
- `--llm-n-gpu-layers <int>` — llama-cpp `n_gpu_layers` (`0` default CPU behavior, `-1` full offload)
- `--workers <int>` — OCR/regex parallelism (default: 2)
- `--llm-workers <int>` — LLM concurrency; careful with memory (default: 1)
- `--batch-size <int>` — checkpoint every N docs (default: 10)
- `--resume / --no-resume` — resume from last checkpoint (default: resume)
- `--keep-artifacts / --no-keep-artifacts` — retain OCR markdown and extraction JSON per doc (default: false)
- `--lang <str>` — OCR language code (default: `rus+eng`)
- `--out-run-id <str>` — override auto-generated run ID

**Resuming:** progress is saved to `data/processed/validation/runs/<run_id>/results.parquet` every `--batch-size` documents. If interrupted, re-run the same command and already-processed documents will be skipped.

**Artifact retention:** with `--no-keep-artifacts` (default), only the minimal results parquet is written — no OCR/extraction files hit disk. With `--keep-artifacts`, per-document OCR markdown and extraction JSON are saved under `runs/<run_id>/artifacts/<doc_id>/`. For large samples this can use significant disk space.

#### 3. `metrics` — compute metrics and generate report

```bash
# Single run
uv run python -m app.cli.validate metrics \
  --run-id <run_id> --out-md report.md

# Compare multiple runs
uv run python -m app.cli.validate metrics \
  --run-id "<id1>,<id2>,<id3>,<id4>" --out-md report.md

# With custom field weights
uv run python -m app.cli.validate metrics \
  --run-id <run_id> --out-md report.md \
  --weights weights.json --out-json metrics.json
```

Options:
- `--run-id <str>` — comma-separated run IDs (required)
- `--out-md <path>` — output Markdown report (required)
- `--out-json <path>` — optional JSON metrics output
- `--weights <json-or-path>` — field weights for weighted accuracy
- `--tolerance <float>` — amount equality tolerance (default: 0.01)
- `--wrong-counts-as-fn / --no-wrong-counts-as-fn` — slot-filling convention (default: yes)

### Field weights format

Create a JSON file (e.g. `weights.json`):

```json
{
  "pcpl_inn": 1.0,
  "bene_inn": 1.0,
  "issue_date": 0.8,
  "start_date": 0.5,
  "end_date": 0.5,
  "sum": 1.0,
  "currency": 0.3,
  "ikz": 0.7
}
```

Keys are ground-truth field names. Fields not listed are excluded from the weighted score.

### Output structure

```
data/processed/validation/
├── seeds/
│   └── seed_n=200_seed=42.csv
└── runs/
    └── <run_id>/
        ├── metadata.json       # run config, machine info, timestamps
        ├── results.parquet     # per-doc predictions, gold, diagnostics, timings
        └── artifacts/          # (only with --keep-artifacts)
            └── <doc_id>/
                ├── ocr.md
                └── extraction.json
```

### Normalization rules

Applied to both gold and predicted values before comparison:

- **INN / IKZ**: strip all non-digits; empty → null. Leading zeros preserved.
- **Dates**: parse to `YYYY-MM-DD`. Accepts `DD.MM.YYYY`, `DD/MM/YYYY`, ISO, and 2-digit year formats.
- **Currency**: map Russian variants (`руб.`, `рублей`, `₽`, `RUR`) → `RUB`; `долларов` → `USD`; `евро` → `EUR`.
- **Amount**: strip non-numeric characters, accept comma as decimal separator, round to 2 decimal places for equality, keep raw numeric for MAE.

### Supported file formats

PDF, TIF, TIFF, PNG, JPG/JPEG — multi-page TIFFs are handled page-by-page.

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
| `DEMO_PASSWORD`       | (empty)                        | Streamlit public demo login password     |
| `LLM_MODEL_PATH`     | `models/qwen3-4b-…q5_k_m.gguf`| Path to GGUF model file                  |
| `LLM_N_CTX`          | `4096`                         | LLM context window                       |
| `LLM_MAX_TOKENS`     | `2048`                         | Max tokens for LLM response              |
| `LLM_TEMPERATURE`    | `0.0`                          | LLM temperature (0 = deterministic)      |
| `LLM_THREADS`        | `4`                            | CPU threads for LLM inference            |
| `LLM_MAX_RETRIES`    | `2`                            | Retry count for failed JSON validation   |
| `DEFAULT_OCR_ENGINE`  | `tesseract`                   | Default OCR engine                       |
| `TESSERACT_CMD`       | `tesseract`                   | Path to tesseract binary                 |
| `PADDLE_PDX_CACHE_HOME` | `data/paddle-cache`        | Persistent Paddle model cache directory  |
| `PROCESSED_DIR`       | `data/processed`              | Root for output artifacts                |
| `UPLOAD_DIR`          | `data/uploads`                | Directory for uploaded files             |
| `COPY_SOURCE_PDF`     | `false`                       | Copy source PDF into output directory    |
| `ALLOWED_INPUT_ROOTS` | (empty)                        | Semicolon-separated allowed path roots   |
| `WORKER_MAX_JOBS`     | `1` (demo) / `2` (dev)         | Max concurrent jobs per worker; use 1 for stable public demo |
| `DOCKER_ENABLE_PADDLE`| `false`                        | Docker Compose build flag: include PaddleOCR deps in image |

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
│   ├── base.py              # SQLAlchemy DeclarativeBase
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
├── validation/
│   ├── normalize.py         # Field normalisation rules
│   ├── metrics.py           # Metric computation engine
│   ├── regex_baseline.py    # Regex-based baseline extractor
│   ├── runner.py            # Parallel, resumable evaluation runner
│   ├── report.py            # Markdown report generator
│   └── storage.py           # Parquet-based result storage
├── cli/
│   ├── ocr.py               # Bulk OCR CLI
│   ├── validate.py          # Validation CLI (sample, run, metrics)
│   ├── evaluate.py          # Evaluation scaffold CLI (legacy)
│   └── download_model.py    # GGUF model downloader (HuggingFace)
├── schemas/
│   └── jobs.py              # API request/response models
├── ui/
│   ├── demo_content.py      # ASCII art + haiku pools for login gate
│   └── login_gate.py        # Streamlit password gate (macOS-style window)
streamlit_app.py             # Streamlit frontend (imports app.ui.login_gate)
.streamlit/config.toml       # Streamlit config (maxUploadSize=10)
tests/
├── conftest.py              # Fixtures (in-memory DB, mocked queue)
├── test_api_jobs.py         # API endpoint tests
├── test_extraction.py       # LLM extraction + validation tests
└── test_corrections.py      # Correction submission tests
alembic/                     # Database migrations
docker/Dockerfile            # App image with optional Paddle deps via ENABLE_PADDLE build arg
docker-compose.yml
```

## License

MIT
