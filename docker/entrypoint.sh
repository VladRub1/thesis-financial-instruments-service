#!/bin/bash
set -e

case "$1" in
  api)
    echo "Running database migrations..."
    uv run alembic upgrade head
    echo "Starting API server..."
    exec uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
    ;;
  worker)
    echo "Starting worker..."
    exec uv run arq app.workers.tasks.WorkerSettings
    ;;
  streamlit)
    echo "Starting Streamlit UI..."
    exec uv run streamlit run streamlit_app.py \
      --server.port 8501 \
      --server.address 0.0.0.0 \
      --server.headless true
    ;;
  migrate)
    echo "Running database migrations..."
    exec uv run alembic upgrade head
    ;;
  *)
    exec "$@"
    ;;
esac
