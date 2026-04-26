#!/bin/bash
set -euo pipefail

MODEL_ALIAS="${1:?usage: run_current_model_matrix_parallel.sh MODEL_ALIAS [MAX_JOBS] [runner args...]}"
MAX_JOBS="${2:-${MAX_JOBS:-4}}"
if [[ $# -ge 2 ]]; then
  shift 2
else
  shift 1
fi
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-experiments/vllm_matrix_20260426}"

cd /fsx/users/haoli/structured-cot-vllm-20260426
source .venv/bin/activate

NVIDIA_LIBS=$(find .venv/lib/python*/site-packages/nvidia -type d -path "*/lib" | paste -sd: -)
export LD_LIBRARY_PATH="$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}"
export HF_HOME=/fsx/users/haoli/.cache/huggingface

python scripts/run_vllm_matrix_parallel.py \
  --model-alias "$MODEL_ALIAS" \
  --experiment-root "$EXPERIMENT_ROOT" \
  --max-jobs "$MAX_JOBS" \
  --request-timeout "${REQUEST_TIMEOUT:-900}" \
  "$@"
