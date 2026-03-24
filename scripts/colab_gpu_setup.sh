#!/usr/bin/env bash
set -euo pipefail

LLAMA_VERSION="${LLAMA_VERSION:-0.3.16}"
COLAB_WITH_PADDLE="${COLAB_WITH_PADDLE:-0}"
PADDLE_GPU_VERSION="${PADDLE_GPU_VERSION:-3.0.0}"
PADDLE_GPU_INDEX="${PADDLE_GPU_INDEX:-https://www.paddlepaddle.org.cn/packages/stable/cu118/}"

if [[ ! -f "pyproject.toml" ]]; then
  echo "Run this script from the repository root (where pyproject.toml exists)."
  exit 1
fi

TOTAL_STEPS=3
if [[ "$COLAB_WITH_PADDLE" == "1" ]]; then
  TOTAL_STEPS=5
fi

echo "[1/${TOTAL_STEPS}] Syncing project deps for Colab..."
SYNC_ARGS=(--extra colab --frozen)
if [[ "$COLAB_WITH_PADDLE" == "1" ]]; then
  SYNC_ARGS+=(--extra paddle)
fi
uv sync "${SYNC_ARGS[@]}"

if [[ "$COLAB_WITH_PADDLE" == "1" ]]; then
  echo "[2/${TOTAL_STEPS}] Installing Paddle GPU runtime for Colab..."
  uv pip uninstall --python .venv/bin/python paddlepaddle || true
  uv pip install --python .venv/bin/python --force-reinstall \
    --index-url "${PADDLE_GPU_INDEX}" \
    --extra-index-url https://pypi.org/simple \
    "paddlepaddle-gpu==${PADDLE_GPU_VERSION}"
fi

echo "[3/${TOTAL_STEPS}] Rebuilding llama-cpp-python with CUDA..."
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
  uv pip install --python .venv/bin/python --force-reinstall --no-cache-dir "llama-cpp-python==${LLAMA_VERSION}"

echo "[4/${TOTAL_STEPS}] Verifying GPU offload support..."
uv run --no-sync python -c "import sys, llama_cpp; support_fn=getattr(llama_cpp, 'llama_supports_gpu_offload', None); supported=bool(support_fn()) if callable(support_fn) else None; print('llama_cpp module:', getattr(llama_cpp, '__file__', 'unknown')); print('llama_cpp version:', getattr(llama_cpp, '__version__', 'unknown')); print('gpu_offload_support:', supported if supported is not None else 'unknown'); sys.exit(0 if supported else 2)"

if [[ "$COLAB_WITH_PADDLE" == "1" ]]; then
  echo "[5/${TOTAL_STEPS}] Verifying Paddle runtime compatibility..."
  uv run --no-sync python -c "import sys, paddle, paddleocr; from paddle import inference as paddle_infer; has_opt=hasattr(paddle_infer.Config, 'set_optimization_level'); has_cuda=bool(paddle.is_compiled_with_cuda()); print('paddle version:', getattr(paddle, '__version__', 'unknown')); print('paddleocr version:', getattr(paddleocr, '__version__', 'unknown')); print('has_set_optimization_level:', has_opt); print('paddle_cuda_enabled:', has_cuda); sys.exit(0 if (has_opt and has_cuda) else 2)"
fi

echo
echo "Done. Keep using 'uv run --no-sync ...' for validation commands in this session."
echo "In notebooks, persist no-sync with: %env UV_NO_SYNC=1"
if [[ "$COLAB_WITH_PADDLE" == "1" ]]; then
  echo "Paddle validation in Colab will auto-use device gpu:0."
fi
