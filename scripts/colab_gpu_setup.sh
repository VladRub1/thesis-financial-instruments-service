#!/usr/bin/env bash
set -euo pipefail

LLAMA_VERSION="${LLAMA_VERSION:-0.3.16}"
COLAB_WITH_PADDLE="${COLAB_WITH_PADDLE:-0}"

if [[ ! -f "pyproject.toml" ]]; then
  echo "Run this script from the repository root (where pyproject.toml exists)."
  exit 1
fi

TOTAL_STEPS=3
if [[ "$COLAB_WITH_PADDLE" == "1" ]]; then
  TOTAL_STEPS=4
fi

echo "[1/${TOTAL_STEPS}] Syncing project deps for Colab..."
SYNC_ARGS=(--extra colab --frozen)
if [[ "$COLAB_WITH_PADDLE" == "1" ]]; then
  SYNC_ARGS+=(--extra paddle)
fi
uv sync "${SYNC_ARGS[@]}"

echo "[2/${TOTAL_STEPS}] Rebuilding llama-cpp-python with CUDA..."
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
  uv pip install --python .venv/bin/python --force-reinstall --no-cache-dir "llama-cpp-python==${LLAMA_VERSION}"

echo "[3/${TOTAL_STEPS}] Verifying GPU offload support..."
uv run --no-sync python -c "import sys, llama_cpp; support_fn=getattr(llama_cpp, 'llama_supports_gpu_offload', None); supported=bool(support_fn()) if callable(support_fn) else None; print('llama_cpp module:', getattr(llama_cpp, '__file__', 'unknown')); print('llama_cpp version:', getattr(llama_cpp, '__version__', 'unknown')); print('gpu_offload_support:', supported if supported is not None else 'unknown'); sys.exit(0 if supported else 2)"

if [[ "$COLAB_WITH_PADDLE" == "1" ]]; then
  echo "[4/${TOTAL_STEPS}] Verifying Paddle runtime compatibility..."
  uv run --no-sync python -c "import sys, paddle, paddleocr; from paddle import inference as paddle_infer; ok=hasattr(paddle_infer.Config, 'set_optimization_level'); print('paddle version:', getattr(paddle, '__version__', 'unknown')); print('paddleocr version:', getattr(paddleocr, '__version__', 'unknown')); print('has_set_optimization_level:', ok); sys.exit(0 if ok else 2)"
fi

echo
echo "Done. Keep using 'uv run --no-sync ...' for validation commands in this session."
echo "In notebooks, persist no-sync with: %env UV_NO_SYNC=1"
