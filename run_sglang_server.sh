#!/usr/bin/env bash
# Launch SGLang's server in Docker for grammar-constrained rollouts.
# Tuned for 2x A100 80GB; tp=2 by default, xgrammar grammar backend.
#
# Env vars (all optional):
#   MODEL          HF model id or local path
#                  (default: Qwen/Qwen3.6-27B — the RL target model)
#   PORT           host port to expose         (default: 30000)
#   TP             tensor-parallel degree      (default: 2)
#   IMAGE          docker image                (default: lmsysorg/sglang:latest)
#   HF_CACHE       host path for HF cache      (default: ~/.cache/huggingface)
#   MAX_LEN        max context length          (default: 32768)
#   BIND_HOST      host address to bind to     (default: 127.0.0.1)
#                  Use 0.0.0.0 only if you've put SGLang behind a proper auth
#                  layer; otherwise the open port becomes a free-LLM proxy
#                  for internet scanners.
#   EXTRA_ARGS     anything else to forward to launch_server
#
# Pin IMAGE to a specific tag once you've found a build you like —
# `latest` can break the grammar backend across versions.
#
# Quick smoke test once up (in another shell):
#   curl http://localhost:30000/generate -H 'Content-Type: application/json' -d '{
#     "text":"hi ",
#     "sampling_params":{"max_new_tokens":8,
#       "ebnf":"root ::= \"omer\" | \"friend\""}}'

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
PORT="${PORT:-30000}"
TP="${TP:-2}"
IMAGE="${IMAGE:-lmsysorg/sglang:latest}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
MAX_LEN="${MAX_LEN:-32768}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$HF_CACHE"

echo "Pulling $IMAGE (5-15 GB; one-time per host)..."
docker pull "$IMAGE"

echo
echo "Starting SGLang server"
echo "  model = $MODEL"
echo "  tp    = $TP"
echo "  bind  = $BIND_HOST:$PORT  (host -> container 30000)"
echo "  cache = $HF_CACHE -> /root/.cache/huggingface"
echo "  ctx   = $MAX_LEN"
echo "  image = $IMAGE"
echo
echo "Parsers: --tool-call-parser qwen3_coder"
echo "(reasoning parser intentionally OFF: with it on, SGLang carves"
echo " the <think> region out of xgrammar enforcement and our IR grammar"
echo " ends up applied to content instead of reasoning. Off = grammar"
echo " constrains the whole generation including the thinking block.)"
echo

# --ipc=host and --shm-size are required for multi-GPU NCCL; without them
# tp=2 deadlocks or OOMs at startup with cryptic errors.
#
# Port mapping binds to $BIND_HOST (default 127.0.0.1) so the server isn't
# reachable from the public internet. Override with BIND_HOST=0.0.0.0 only
# behind auth.
exec docker run --gpus all --rm -it \
  --ipc=host --shm-size=32g \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -p "$BIND_HOST:$PORT:30000" \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp "$TP" \
    --host 0.0.0.0 --port 30000 \
    --grammar-backend xgrammar \
    --context-length "$MAX_LEN" \
    --tool-call-parser qwen3_coder \
    $EXTRA_ARGS
