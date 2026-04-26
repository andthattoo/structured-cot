#!/bin/bash
set -euo pipefail

MODEL_ALIAS="${1:?usage: run_current_model_matrix.sh MODEL_ALIAS}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-experiments/vllm_matrix_20260426}"

cd /fsx/users/haoli/structured-cot-vllm-20260426
source .venv/bin/activate

NVIDIA_LIBS=$(find .venv/lib/python*/site-packages/nvidia -type d -path "*/lib" | paste -sd: -)
export LD_LIBRARY_PATH="$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}"
export HF_HOME=/fsx/users/haoli/.cache/huggingface

python scripts/run_vllm_matrix.py \
  --model-alias "$MODEL_ALIAS" \
  --experiment-root "$EXPERIMENT_ROOT" \
  --key-repeats "${KEY_REPEATS:-3}" \
  --exploratory-repeats "${EXPLORATORY_REPEATS:-1}" \
  --request-timeout "${REQUEST_TIMEOUT:-900}"
