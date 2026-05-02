# structured-cot

**Grammar-constrained chain-of-thought for reasoning LMs. Zero training. 22× compression on HumanEval+; +14 pp on a recent LiveCodeBench public-test slice.**

## The finding

On HumanEval+ (164 problems) with `unsloth/Qwen3.6-35B-A3B-GGUF` at Q4_K_M, running on one H100 via `llama-cpp-python`:

| Mode | pass@1 | mean thinking tokens | mean total tokens |
|---|---|---|---|
| Free-form `<think>...</think>` | **151 / 164 = 92.1%** | 3087 | 3410 |
| Grammar-constrained `<think>GOAL/APPROACH/EDGE</think>` | **152 / 164 = 92.7%** | **138** | **408** |
| Prompt-only terse `GOAL/APPROACH/EDGE` | **153 / 164 = 93.3%** | 2298 | 2764 |
| **FSM vs FREE Δ** | **+0.6 pp** | **22.4× shorter** | **8.4× shorter** |

No distillation. No fine-tuning. No reward-model. A ~20-line GBNF grammar applied to the `<think>` block at inference time matches full-thinking accuracy with an order-of-magnitude fewer tokens. The prompt-only terse control slightly improves pass@1 on this run, but still uses 2298 thinking tokens on average; the grammar is what reliably enforces the compact token regime.

On a harder recent LiveCodeBench v6 LeetCode slice (50 problems, `contest_date >= 2025-01-01`, public functional tests), the richer [`grammars/fsm_grammar_lcb_plan.gbnf`](grammars/fsm_grammar_lcb_plan.gbnf) grammar improves both reliability and token use:

| Mode | pass@1 | mean thinking tokens | mean total tokens | extraction issues |
|---|---:|---:|---:|---|
| Free-form `<think>...</think>` | 25 / 50 = 50.0% | 11553 | 13632 | empty_code=18 |
| FSM_PLAN `GOAL/STATE/ALGO/EDGE/VERIFY` | **32 / 50 = 64.0%** | **267** | **2743** | **none** |
| **FSM_PLAN vs FREE Δ** | **+14.0 pp** | **43.3× shorter** | **5.0× shorter** | - |

The LiveCodeBench result is a public-test result, not an official private leaderboard score. It also reveals a useful caveat: on harder tasks, the model can move some deliberation into comments or post-think answer text, so total tokens and comment bloat need to be tracked alongside `<think>` tokens.

See [RESULTS.md](RESULTS.md) for the full experimental writeup, per-problem breakdown, and discussion of limitations.

## Why this matters

Reasoning models like Qwen3, DeepSeek-R1, QwQ spend thousands of tokens in verbose prose thinking — exploring alternatives, restating, hedging. This work shows, on one benchmark, that for a large chunk of that reasoning, the verbose scaffolding isn't doing real work. The model already has the reasoning capability internally; grammar constraint just extracts it in a denser form.

The engineering implication is direct: **inference-time thinking compute can be cut dramatically via a grammar file alone**, with no training pipeline or serving changes beyond a GBNF argument. HumanEval+ shows the clean compression case; LiveCodeBench shows the harder-task behavior, where the grammar also prevents many no-code failures but can displace some reasoning into the answer channel.

## How it works

Single GBNF grammar (see [`grammars/fsm_grammar.gbnf`](grammars/fsm_grammar.gbnf)):

```gbnf
root  ::= think code
think ::= "<think>\n" "GOAL: " line "APPROACH: " line "EDGE: " line "</think>\n\n"
line  ::= [^\n]+ "\n"
code  ::= [\x09\x0A\x0D\x20-\x7E]+
```

Three lines of structured plan. Code after is unconstrained.

The grammar is applied via llama-cpp-python's server which accepts `grammar` in the request body. At each generation step inside `<think>`, logits for tokens that violate the grammar are masked to -∞; the model samples from the constrained distribution.

## Setup

Tested on an H100 with CUDA 12.4.

```bash
uv sync
# llama-cpp-python is pulled from the prebuilt cu124 index via tool.uv.sources.
# CUDA runtime libs pinned so it coexists with torch's cu13 default.

# Before running, expose the bundled CUDA libs to the dynamic loader:
export LD_LIBRARY_PATH=$(uv run python -c "
import site, os
sp = site.getsitepackages()[0]
nv = os.path.join(sp, 'nvidia')
print(':'.join(os.path.join(nv, d, 'lib') for d in os.listdir(nv)
               if os.path.isdir(os.path.join(nv, d, 'lib'))))
")
```

### Native llama-server (recommended for speed)

For long generations, native `llama-server` is faster than `llama-cpp-python`, and can use llama.cpp speculative decoding:

```bash
sudo apt-get update
sudo apt-get install -y git cmake build-essential libcurl4-openssl-dev

git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build \
    -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build ~/llama.cpp/build --config Release -j

export PATH="$HOME/llama.cpp/build/bin:$PATH"
llama-server --help | head
```

The repo includes [`run_llama_server.sh`](run_llama_server.sh), which defaults to:

```bash
llama-server -hf ggml-org/Qwen3.6-27B-GGUF --spec-default \
    --host 127.0.0.1 --port 8000 -c 32768 -ngl 999 \
    --flash-attn on --reasoning-format none
```

The older [`run_server.sh`](run_server.sh) still starts `llama-cpp-python` and is kept for reproducing the original run.

### Model download

```bash
export HF_TOKEN=hf_...
huggingface-cli download unsloth/Qwen3.6-35B-A3B-GGUF \
    --include "*Q4_K_M*" --local-dir ~/models/qwen3.6-gguf
```

## Run

The server can run in the foreground in one pane, or in the background from a single terminal.

### Start the server

```bash
BACKGROUND=1 ./run_llama_server.sh
tail -f server.log
curl http://127.0.0.1:8000/v1/models
```

By default this downloads/serves `ggml-org/Qwen3.6-27B-GGUF` through native `llama-server` with `--spec-default` and `--reasoning-format none`, so `<think>...</think>` remains visible for token accounting. Override with env vars:

```bash
HF_REPO=ggml-org/Qwen3.6-27B-GGUF N_CTX=32768 BACKGROUND=1 ./run_llama_server.sh
MODEL_PATH=/path/to/model.gguf BACKGROUND=1 ./run_llama_server.sh
KV_TYPE=q8_0 BACKGROUND=1 ./run_llama_server.sh
REASONING_FORMAT=deepseek-legacy BACKGROUND=1 ./run_llama_server.sh
```

Background mode writes `server.log` and `server.pid`. Stop it with:

```bash
kill "$(cat server.pid)"
```

If you prefer the original Python server, use `./run_server.sh`. It auto-discovers the GGUF from `~/.cache/huggingface/hub/` or `~/models/`, uses 8-bit KV cache (`q8_0`), flash attention, and `n_ctx=65536`.

### Run the comparison

```bash
# Smoke test
uv run python fsm_vs_free_eval.py --n-problems 10 --max-tokens 4096

# Full HumanEval+ with FREE + FSM + PROMPT_TERSE controls
uv run python fsm_vs_free_eval.py --n-problems 164 --max-tokens 8192 --only all

# MBPP+
uv run python fsm_vs_free_eval.py --n-problems 100 --dataset mbpp --max-tokens 8192

# LiveCodeBench v6 recent subset (public functional tests)
uv run python fsm_vs_free_eval.py --dataset livecodebench \
    --lcb-version release_v6 --date-cutoff 2025-01-01 --platform leetcode \
    --n-problems 50 --max-tokens 16384 --only all \
    --out-dir lcb_v6_2025_01_01_n50_all

# LiveCodeBench FSM-only baseline grammar
uv run python fsm_vs_free_eval.py --dataset livecodebench \
    --lcb-version release_v6 --date-cutoff 2025-01-01 --platform leetcode \
    --n-problems 50 --max-tokens 16384 --only fsm \
    --grammar-file grammars/fsm_grammar.gbnf \
    --out-dir lcb_v6_2025_01_01_fsm_base_grammar

# LiveCodeBench FSM-only plan grammar
uv run python fsm_vs_free_eval.py --dataset livecodebench \
    --lcb-version release_v6 --date-cutoff 2025-01-01 --platform leetcode \
    --n-problems 50 --max-tokens 16384 --only fsm \
    --grammar-file grammars/fsm_grammar_lcb_plan.gbnf \
    --out-dir lcb_v6_2025_01_01_fsm_lcb_plan
```

### Terminal-Bench smoke test

Terminal-Bench is the first multi-turn agent probe in this repo. It uses a
custom [`BaseAgent`](terminal_bench_structured_cot_agent.py) that talks to an
OpenAI-compatible local endpoint and exposes the terminal as `run_shell` /
`finish` tools. This is meant for the patched llama.cpp reasoning-grammar path,
where the user grammar applies to `message.reasoning_content` and llama.cpp's
normal tool-call grammar still handles tool calls.

Terminal-Bench requires Docker and its CLI:

```bash
uv tool install terminal-bench
```

Start patched native llama.cpp with reasoning extraction:

```bash
REASONING_FORMAT=deepseek BACKGROUND=1 ./run_llama_server.sh
```

Then compare the same task with and without reasoning grammar:

```bash
GRAMMAR_MODE=none ./scripts/run_terminal_bench_smoke.sh
GRAMMAR_MODE=reasoning ./scripts/run_terminal_bench_smoke.sh  # STEP only
GRAMMAR_MODE=step_status ./scripts/run_terminal_bench_smoke.sh # STEP/STATUS
GRAMMAR_MODE=phase ./scripts/run_terminal_bench_smoke.sh      # PHASE/CHECK/NEXT
GRAMMAR_MODE=dsl ./scripts/run_terminal_bench_smoke.sh        # PLAN/STATE/RISK/NEXT
```

The `dsl` mode is a tiny existing-token symbolic trace, meant as a first
step toward "abstract CoT without new tokens":

```text
PLAN: seq(inspect,edit,verify,finish)
STATE: need_verify
RISK: premature_finish
NEXT: run_shell
```

It is deliberately closer to behavior-tree / HTN control flow than natural
language prose. The current grammar keeps the shape flat because that has been
more reliable with the pre-trigger grammar patch; the system prompt supplies
the semantics for when `STATE: ready` / `RISK: none` / `NEXT: finish` should be
used. DSL runs also write `grammar_violations.jsonl` if `reasoning_content`
escapes the expected shape. This is still a hand-written inference-time
grammar; the longer-term experiment is to mine these labels from successful
agent traces, then train or dynamically select compact reasoning-state
grammars per turn.

For LoRA adapters trained on the QwenXML compact-DSL data, test the learned
format without server-side grammar by using the prompt-only XML mode. This
uses the same generic compact-DSL + QwenXML contract as the SFT data first,
then appends Terminal-Bench-specific `run_shell` / `finish` instructions:

```bash
GRAMMAR_MODE=none TOOL_MODE=qwen_xml MODEL=qwen-leo-pi-warm-1 \
    ./scripts/run_terminal_bench_smoke.sh
```

This asks the model to emit the trained shape directly:

```text
<think>
PLAN: ...
STATE: ...
RISK: ...
NEXT: tool_call
</think>
<tool_call>
<function=run_shell>
<parameter=command>
...
</parameter>
</function>
</tool_call>
```

The script defaults to `terminal-bench-core==0.1.1` and `hello-world`. Override
with `TASK_ID=...`, `DATASET=...`, `MODEL=...`, or `BASE_URL=...`. Use
`TASK_ID=all` to omit `--task-id` and run the full dataset:

```bash
TASK_ID=all GRAMMAR_MODE=none ./scripts/run_terminal_bench_smoke.sh
```

The runner defaults to `N_CONCURRENT=1` because Terminal-Bench defaults to
four concurrent trials, which can overwhelm a single local `llama-server`.
For a smaller serial slice:

```bash
TASK_ID=all N_TASKS=10 GRAMMAR_MODE=none ./scripts/run_terminal_bench_smoke.sh
```

### Compact-DSL training data

The inference-only grammars above are useful probes, but the stronger branch is
to train the model to use a compact symbolic reasoning trace before tool calls.
For a first pass, convert the filtered Hermes agent traces into SFT JSONL with
verbose `<think>` blocks rewritten as a tiny existing-token DSL:

```bash
uv run python scripts/prepare_hermes_dsl_sft.py \
    --dataset DJLougen/hermes-agent-traces-filtered \
    --limit 200 \
    --out hermes_dsl_sft_sample.jsonl
```

For the full filtered set:

```bash
uv run python scripts/prepare_hermes_dsl_sft.py \
    --dataset DJLougen/hermes-agent-traces-filtered \
    --out hermes_dsl_sft_full.jsonl
```

The output keeps the original tool calls and tool responses, but replaces
assistant reasoning blocks with traces like:

```text
<think>
PLAN: seq(observe,act,verify,finish)
STATE: need_verify
RISK: bad_tool_args
NEXT: tool_call
</think>
<tool_call>
{"name": "search_files", "arguments": {"pattern": "*.yaml"}}
</tool_call>
```

For Qwen3.6, prefer the model-native XML tool-call target instead of the
Hermes JSON target:

```bash
uv run python scripts/prepare_hermes_dsl_sft.py \
    --dataset DJLougen/hermes-agent-traces-filtered \
    --limit 200 \
    --labeler local_grammar \
    --labeler-base-url http://127.0.0.1:8002/v1 \
    --labeler-model ggml-org/Qwen3.6-27B-GGUF \
    --store-original-think \
    --tool-format qwen_xml \
    --out hermes_dsl_sft_qwenxml_200_bottleneck.jsonl
```

This rewrites assistant tool calls to:

```text
<tool_call>
<function=run_shell>
<parameter=command>
printf 'Hello, world!\n' > hello.txt
</parameter>
</function>
</tool_call>
```

By default, natural-language prose between `</think>` and `<tool_call>` is
stripped so the SFT target teaches abstract reasoning followed directly by tool
calls. Pass `--keep-pre-tool-prose` if you want to preserve the original
assistant prose. This script is still a heuristic bootstrapper: it uses the
teacher trace and previous tool response to choose labels such as
`RISK: tool_failure`, but an LLM relabeling pass is the cleaner next step before
serious training.

This treats the larger-model rollouts as noisy action demonstrations, not as
ground-truth prose reasoning. The compact DSL is meant for LoRA/SFT experiments
where the model learns the symbolic state language before emitting normal tool
calls.

The default labels are heuristic. For cleaner labels, use a model-constrained
relabeling pass. With a local OpenAI-compatible server that accepts `grammar`:

```bash
uv run python scripts/prepare_hermes_dsl_sft.py \
    --dataset DJLougen/hermes-agent-traces-filtered \
    --limit 200 \
    --labeler local_grammar \
    --labeler-base-url http://127.0.0.1:8000/v1 \
    --labeler-model ggml-org/Qwen3.6-27B-GGUF \
    --out hermes_dsl_sft_local_labeled.jsonl
```

If you want the Abstract-CoT-style bottleneck experiment, keep the original
verbose thinking blocks as metadata during relabeling:

```bash
uv run python scripts/prepare_hermes_dsl_sft.py \
    --dataset DJLougen/hermes-agent-traces-filtered \
    --limit 200 \
    --labeler local_grammar \
    --labeler-base-url http://127.0.0.1:8002/v1 \
    --labeler-model ggml-org/Qwen3.6-27B-GGUF \
    --labeler-fallback heuristic \
    --store-original-think \
    --out hermes_dsl_sft_local_labeled_200_bottleneck.jsonl
```

That file can train a factorized bottleneck objective: one example compresses
`context + teacher_think -> DSL`, and a second example trains
`context + DSL -> tool_call/final`. This is not the exact Abstract-CoT paper
attention-mask bottleneck, but it is the closest simple SFT approximation
without custom training internals.

With OpenRouter, set `OPENROUTER_API_KEY` and use a model that supports
structured outputs. OpenRouter uses JSON Schema structured outputs here rather
than GBNF:

```bash
export OPENROUTER_API_KEY=...
uv run python scripts/prepare_hermes_dsl_sft.py \
    --dataset DJLougen/hermes-agent-traces-filtered \
    --limit 200 \
    --labeler openrouter \
    --labeler-model openai/gpt-4o-mini \
    --out hermes_dsl_sft_openrouter_labeled.jsonl
```

For a first pure-SFT LoRA pilot on a labeled JSONL file:

```bash
uv run --with peft --with accelerate --with bitsandbytes \
  python scripts/train_dsl_sft_lora.py \
    --train-jsonl hermes_dsl_sft_local_labeled_200_rerun.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output-dir runs/dsl-sft-qwen2p5-7b-lora \
    --max-seq-len 4096 \
    --context-messages 12 \
    --epochs 1 \
    --batch-size 1 \
    --grad-accum 16 \
    --learning-rate 2e-4
```

This is plain behavior-cloning SFT: no KL, no preference tuning, and no RL. The
script creates one example per assistant turn and masks all context tokens, so
the loss is only on the current assistant DSL/tool-call/final-answer message.
It trains a LoRA adapter from a Hugging Face Transformers checkpoint; GGUF files
served by llama.cpp cannot be fine-tuned directly.

For the factorized bottleneck objective, use a JSONL produced with
`--store-original-think`:

```bash
uv run --with peft --with accelerate --with bitsandbytes \
  python scripts/train_dsl_sft_lora.py \
    --train-jsonl hermes_dsl_sft_local_labeled_200_bottleneck.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output-dir runs/dsl-bottleneck-qwen2p5-7b-lora \
    --objective factorized_bottleneck \
    --max-seq-len 4096 \
    --context-messages 12 \
    --epochs 1 \
    --batch-size 1 \
    --grad-accum 16 \
    --learning-rate 2e-4
```

For the paper-style masked bottleneck objective, use the same JSONL but train a
single sequence shaped as `context + teacher_think + DSL + action`. The loss is
on `DSL + action`, while a block attention mask prevents action/final tokens
from attending to the teacher-think span:

```bash
uv run --with peft --with accelerate --with bitsandbytes \
  python scripts/train_dsl_sft_lora.py \
    --train-jsonl hermes_dsl_sft_local_labeled_200_bottleneck.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output-dir runs/dsl-bottleneck-masked-qwen2p5-7b-lora \
    --objective bottleneck_masked \
    --max-seq-len 4096 \
    --context-messages 12 \
    --epochs 1 \
    --batch-size 1 \
    --grad-accum 16 \
    --learning-rate 2e-4
```

This path defaults to `attn_implementation=eager`, because the custom 4D mask
is more reliable there than through FlashAttention/SDPA during early tests.

### AgentTrove MQE critic pilot

AgentTrove Terminus-2 traces can also train a small directed-distance critic for
offline self-distillation/ranking. First sample or stream AgentTrove rows, then
turn each assistant command turn into `(state, action, next_state, goal,
distance_to_goal)` tuples:

```bash
uv run --with datasets python scripts/prepare_agenttrove_mqe.py \
  --dataset open-thoughts/AgentTrove \
  --limit-transitions 50000 \
  --scan-limit 1500000 \
  --out-dir data/mqe/agenttrove_50k_features
```

The prepared rows include `action_features`: compact verb, flag, path, mutation,
test, network, install, and hashed argument features. They also include
transition fields (`action_text`, `action_signature`, and deterministic hard
negative indices) for training a goal-free transition encoder.

To train the transition encoder with Qwen3-Embedding-0.6B LoRA:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run --with torch --with transformers --with peft --with accelerate python scripts/train_transition_encoder.py \
  --data-dir data/mqe/agenttrove_50k_features \
  --model Qwen/Qwen3-Embedding-0.6B \
  --output-dir outputs/transition_encoder/qwen06b_agenttrove_50k_v0 \
  --max-state-tokens 2048 \
  --max-action-tokens 512 \
  --lora-r 16 \
  --lora-alpha 32 \
  --batch-size 8 \
  --grad-accum 4 \
  --epochs 1 \
  --bf16
```

Cache the trained transition embeddings:

```bash
uv run --with torch --with transformers --with peft python scripts/cache_transition_embeddings.py \
  --data-dir data/mqe/agenttrove_50k_features \
  --encoder-dir outputs/transition_encoder/qwen06b_agenttrove_50k_v0 \
  --out data/mqe/cache/agenttrove_50k_qwen06b_transition.pt \
  --dtype bfloat16
```

Then train MQE against the cached transition tensors:

```bash
uv run --with torch python scripts/train_mqe_critic.py \
  --data-dir data/mqe/agenttrove_50k_features \
  --encoder-backend transition-cache \
  --cache-path data/mqe/cache/agenttrove_50k_qwen06b_transition.pt \
  --output-dir outputs/mqe/agenttrove_50k_transition_qwen06b \
  --epochs 10
```

For a fast hashing-encoder smoke train:

```bash
uv run --with torch python scripts/train_mqe_critic.py \
  --data-dir data/mqe/agenttrove_50k_features \
  --output-dir outputs/mqe/agenttrove_hash_50k_features \
  --encoder-backend hashing \
  --embedding-dim 2048 \
  --epochs 10
```

Once the data path and ranking metrics look sane, switch to a frozen embedding
model:

```bash
uv run --with torch --with sentence-transformers python scripts/train_mqe_critic.py \
  --data-dir data/mqe/agenttrove_50k_features \
  --output-dir outputs/mqe/agenttrove_qwen_embed_50k_features \
  --encoder-backend sentence-transformers \
  --encoder-model Qwen/Qwen3-Embedding-4B \
  --encoder-device cuda \
  --encoder-dtype bfloat16 \
  --encoder-max-length 2048 \
  --embed-batch-size 8 \
  --cache-path data/mqe/cache/agenttrove_50k_features_qwen3_embed_len2048_b8.pt \
  --epochs 10
```

Use `--no-use-action-features` as an ablation if you want to compare against the
text-embedding-only action head.

This critic is not a verifier. It learns a progress heuristic from offline
rollout order: real actions should reduce directed distance to the final goal
state. Use it later to rank/filter self-distilled candidate actions alongside
hard validity checks for DSL/action JSON. The most relevant action-ranking
metrics are `val_action_choice_acc` and `val_action_choice_margin`, which
compare the real next action against shuffled negative actions for the same
state/goal. `val_spearman_state` is useful for checking whether the critic
understands rollout progress, but it is not enough on its own for action
selection.

To publish the transition encoder and MQE critic to Hugging Face Hub:

```bash
export HF_TOKEN=...

uv run --with huggingface-hub --with torch --with transformers --with peft \
  python scripts/upload_hf_artifacts.py \
  --transition-dir outputs/transition_encoder/qwen06b_agenttrove_50k_v0_steps2000_neg4 \
  --transition-repo driaforall/code-state-embedding \
  --critic-dir outputs/mqe/agenttrove_50k_transition_qwen06b_steps2000_neg4 \
  --critic-repo driaforall/code-mqe-critic \
  --merge-transition-encoder \
  --dry-run
```

Remove `--dry-run` to create/update the model repos. The uploader stages only
the reusable artifacts. With `--merge-transition-encoder`, the Qwen
0.6B embedding model is saved with the LoRA already merged, alongside
`transition_head.pt`, `best.pt`, `metrics.json`, and generated model cards.

Each run produces in `fsm_vs_free/`:
- `results.jsonl` — per-problem raw generations, extracted think/code, pass/fail, errors, extraction metadata
- `summary.json` — aggregate stats, pass-set overlap, and failure accounting
- `per_problem.md` — human-readable report with outcome tags (🔺 / 🔻 / 🟰 / ❌)

The summary also reports `post_think_tokens_mean`, `answer_channel_bloat`,
`code_comment_tokens_mean`, and `comment_bloat`. Those are useful on
LiveCodeBench because a model can obey a short `<think>` grammar while moving
some of the missing scratchpad into comments or other post-think answer text.

## Architecture notes

Qwen3.6-35B-A3B is a **MoE hybrid**:
- 40 layers total, pattern `10 × (3 × GatedDeltaNet + 1 × GatedAttention)`
- 256 experts, 9 active per token (3B active params)
- Native 262K context, extensible to 1M via YaRN

For the FSM experiment, the base-LM architecture doesn't matter much — we're constraining what tokens are emitted, not how they're processed internally. The same GBNF approach should work with any model served through llama.cpp, vLLM, SGLang, TGI, etc., as long as the server accepts a grammar parameter.

## Limitations and open questions

1. **Contamination.** HumanEval has been in training corpora for years. All modes may be recalling solutions, not reasoning. The "FSM matches FREE" result could mean "grammar extracts the same memorized solution in fewer tokens" rather than "grammar preserves reasoning capability." LiveCodeBench release v6 currently reaches April 2025, so it is a useful recent/lower-contamination check but not a strict post-Qwen3.6-release cutoff.

2. **Grammar specificity.** The GOAL/APPROACH/EDGE format was tuned for short coding tasks. The active LiveCodeBench grammar, [`grammars/fsm_grammar_lcb_plan.gbnf`](grammars/fsm_grammar_lcb_plan.gbnf), adds `GOAL/STATE/ALGO/EDGE/VERIFY` while leaving the answer permissive. Math / logic / planning domains may need different symbolic formats. Unclear whether a single "universal" compressed-thinking grammar exists or whether each domain needs its own.

3. **Reasoning depth.** The current result is on problems solvable in one forward pass. For multi-step problems (SWE-Bench, long-horizon planning, agentic tasks), whether grammar compression preserves capability is untested.

4. **Model dependency.** Run with one model (Qwen3.6-35B-A3B). We don't know yet whether smaller models (1B-7B) can be grammar-constrained the same way or whether the capability requires scale.

## Status

- ✅ HumanEval+ full (164 problems): FSM 152/164 vs FREE 151/164, 22× think-token compression
- ✅ LiveCodeBench v6 recent LeetCode subset, public tests (50 problems): FREE 25/50 vs `grammars/fsm_grammar_lcb_plan.gbnf` 32/50; FSM_PLAN improves pass@1 while cutting mean total tokens from 13632 to 2743
- ⏳ MBPP+ (planned)
- 🔲 Other domains (math, logic, planning)
- 🔲 Cross-model transfer (smaller models)

## References

The direct technical antecedent is [Coconut: Training Large Language Models to Reason in a Continuous Latent Space (Hao et al., 2024)](https://arxiv.org/abs/2412.06769), which compresses CoT into continuous latents via fine-tuning. This work tests whether a much cheaper intervention — grammar constraint at inference, no training — captures a meaningful fraction of the same benefit.
