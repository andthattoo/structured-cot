#!/bin/bash
#SBATCH --job-name=structured-cot-vllm-1gpu
#SBATCH --partition=main
#SBATCH --nodelist=ip-10-1-75-55
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=20
#SBATCH --mem-per-gpu=160G
#SBATCH --time=48:00:00
#SBATCH --output=/fsx/users/haoli/structured-cot-vllm-20260426/slurm_logs/%x-%j.out
#SBATCH --error=/fsx/users/haoli/structured-cot-vllm-20260426/slurm_logs/%x-%j.err

set -euo pipefail

cd /fsx/users/haoli/structured-cot-vllm-20260426
mkdir -p slurm_logs sagemaker-empty

source .venv/bin/activate

MODEL_ID="${MODEL_ID:?set MODEL_ID}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(echo "$MODEL_ID" | tr '/[:upper:]' '-[:lower:]')}"
PORT="${PORT:-18080}"
EXTRA_VLLM_ARGS=()
if [[ -n "${MAX_MODEL_LEN:-}" ]]; then
  EXTRA_VLLM_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
fi
if [[ -n "${MAX_NUM_SEQS:-}" ]]; then
  EXTRA_VLLM_ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")
fi
if [[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]]; then
  EXTRA_VLLM_ARGS+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi
if [[ -n "${GPU_MEMORY_UTILIZATION:-}" ]]; then
  EXTRA_VLLM_ARGS+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
fi

NVIDIA_LIBS=$(find .venv/lib/python*/site-packages/nvidia -type d -path "*/lib" | paste -sd: -)
export LD_LIBRARY_PATH="$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}"
export HF_HOME=/fsx/users/haoli/.cache/huggingface
export SAGEMAKER_MODEL_PATH=/fsx/users/haoli/structured-cot-vllm-20260426/sagemaker-empty
export PYTHONUNBUFFERED=1

echo "job_id=${SLURM_JOB_ID}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "model_id=${MODEL_ID}"
echo "served_model_name=${SERVED_MODEL_NAME}"
echo "port=${PORT}"
echo "max_model_len=${MAX_MODEL_LEN:-vllm_default}"
echo "max_num_seqs=${MAX_NUM_SEQS:-vllm_default}"
echo "max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-vllm_default}"
echo "gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-vllm_default}"
echo "slurm_job_gpus=${SLURM_JOB_GPUS:-unset}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "cpus_on_node=${SLURM_CPUS_ON_NODE:-unset}"
echo "mem_per_gpu=160G"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits

exec vllm serve "$MODEL_ID" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  "${EXTRA_VLLM_ARGS[@]}"
