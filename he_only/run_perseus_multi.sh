#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${QWEN_FHE_MODEL_GPU:?Select the arithmetic GPU}"
: "${QWEN_FHE_BOOTSTRAP_GPU:?Select a different bootstrap GPU}"
test "$QWEN_FHE_MODEL_GPU" != "$QWEN_FHE_BOOTSTRAP_GPU"
export CUDA_VISIBLE_DEVICES="$QWEN_FHE_MODEL_GPU"
export PYTHONPATH="$root/perseus-core/python:$root${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$root/.cuda/lib64:$root/.rust-native/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
exec "${QWEN_HE_GPU_PYTHON:-$root/.venv/bin/python}" "$root/run_demo.py" --backend perseus-multigpu "$@"
