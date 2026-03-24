#!/usr/bin/env bash
set -euo pipefail

LLAMA_VERSION="${LLAMA_VERSION:-0.3.16}"

if [[ ! -f "pyproject.toml" ]]; then
  echo "Run this script from the repository root (where pyproject.toml exists)."
  exit 1
fi

echo "[1/3] Syncing project deps for Colab..."
uv sync --extra colab --frozen

echo "[2/3] Rebuilding llama-cpp-python with CUDA..."
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
  uv pip install --python .venv/bin/python --force-reinstall --no-cache-dir "llama-cpp-python==${LLAMA_VERSION}"

echo "[3/3] Verifying GPU offload support..."
uv run --no-sync python -c "import sys, llama_cpp; support_fn=getattr(llama_cpp, 'llama_supports_gpu_offload', None); supported=bool(support_fn()) if callable(support_fn) else None; print('llama_cpp module:', getattr(llama_cpp, '__file__', 'unknown')); print('llama_cpp version:', getattr(llama_cpp, '__version__', 'unknown')); print('gpu_offload_support:', supported if supported is not None else 'unknown'); sys.exit(0 if supported else 2)"

echo
echo "Done. Keep using 'uv run --no-sync ...' (or export UV_NO_SYNC=1) for validation commands in this session."
