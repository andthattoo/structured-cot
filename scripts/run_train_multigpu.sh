#!/usr/bin/env bash
# Multi-GPU SFT training launcher for etpi-phase1.
#
# Designed to run inside the lmsysorg/sglang:latest docker container so we
# inherit its torch/CUDA/flash-attn-3 stack. Auto-detects the GPU count and
# launches train_sft.py under accelerate with FSDP (full shard).
#
# Usage (on the target box):
#
#   sudo docker run -d --name sft_train --gpus all --ipc=host --shm-size=32g \
#       -v $PWD/scripts/train_sft.py:/workspace/train_sft.py:ro \
#       -v $PWD/scripts/run_train_multigpu.sh:/workspace/run.sh:ro \
#       -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#       -v $HOME/sft_run:/workspace/sft_run \
#       -w /workspace \
#       lmsysorg/sglang:latest \
#       bash /workspace/run.sh
#
# Override hyperparams via env vars:
#   NUM_EPOCHS=5 MAX_LENGTH=16384 LR=3e-4 bash run_train_multigpu.sh
#
# Memory math for 27B + LoRA at seq 32k, FSDP across N GPUs:
#   - 1×80GB: OOM (~30GB short)
#   - 2×80GB: tight; works at seq 24k
#   - 4×80GB: comfortable at seq 32k
#   - 8×80GB: lots of headroom, ~5x faster wall-clock

set -euo pipefail

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.6-27B}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
# This is a short format-warmup SFT pass, not long-context finetuning.
# 32k makes rare long traces painfully slow with SDPA/torch fallback on A100s.
MAX_LENGTH="${MAX_LENGTH:-8192}"
LR="${LR:-3e-4}"
SAVE_STEPS="${SAVE_STEPS:-50}"
LOGGING_STEPS="${LOGGING_STEPS:-2}"
DATASET="${DATASET:-andthattoo/etpi-sft}"

echo "[launcher] detected $NUM_GPUS GPUs"
echo "[launcher] config: epochs=$NUM_EPOCHS max_length=$MAX_LENGTH lr=$LR"
echo "[launcher] base model: $BASE_MODEL"
echo "[launcher] dataset: $DATASET"

# Install training deps that aren't already in the sglang image.
pip install --quiet \
    "trl>=1.0" \
    "peft>=0.10" \
    "bitsandbytes>=0.43" \
    "liger-kernel" \
    "datasets" \
    "huggingface_hub" \
    "accelerate>=1.0" 2>&1 | tail -3

# Resolve the model snapshot once before spawning distributed workers. If the
# shared HF cache has an incomplete snapshot, this repairs or fails it in one
# process instead of letting eight ranks race and killing the job with a vague
# torch.distributed ChildFailedError.
python - <<PY
from huggingface_hub import snapshot_download

model_id = "$BASE_MODEL"
path = snapshot_download(
    repo_id=model_id,
    allow_patterns=[
        "config.json",
        "configuration*.py",
        "generation_config.json",
        "model*.safetensors",
        "model.safetensors.index.json",
        "tokenizer*",
        "*.model",
        "*.tiktoken",
        "*.json",
        "*.py",
    ],
)
print(f"[launcher] cached model snapshot: {path}")
PY

# Pull the SFT dataset locally as JSONL.
python -c "
from datasets import load_dataset
ds = load_dataset('$DATASET', split='train')
ds.to_json('/workspace/sft.jsonl')
print(f'saved {len(ds)} examples to /workspace/sft.jsonl')
"

# Write a minimal FSDP accelerate config. TRANSFORMER_BASED_WRAP auto-wraps
# decoder layers; FULL_SHARD shards weights+grads+optim across GPUs;
# use_orig_params=true is required for PEFT/LoRA to work under FSDP.
cat > /tmp/fsdp_config.yaml <<YAML
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: ${NUM_GPUS}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
YAML

# Launch.
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

accelerate launch \
    --config_file /tmp/fsdp_config.yaml \
    --num_processes "$NUM_GPUS" \
    /workspace/train_sft.py \
    --sft-jsonl /workspace/sft.jsonl \
    --base-model "$BASE_MODEL" \
    --out-dir /workspace/sft_run \
    --num-epochs "$NUM_EPOCHS" \
    --max-length "$MAX_LENGTH" \
    --lr "$LR" \
    --logging-steps "$LOGGING_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --attn-impl sdpa
