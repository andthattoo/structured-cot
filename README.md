# structured-cot

**Grammar-constrained chain-of-thought for reasoning LMs. Zero training. 22× compression with no accuracy loss on HumanEval+.**

## The finding

On HumanEval+ (164 problems) with `unsloth/Qwen3.6-35B-A3B-GGUF` at Q4_K_M, running on one H100 via `llama-cpp-python`:

| Mode | pass@1 | mean thinking tokens | mean total tokens |
|---|---|---|---|
| Free-form `<think>...</think>` | **151 / 164 = 92.1%** | 3087 | 3410 |
| Grammar-constrained `<think>GOAL/APPROACH/EDGE</think>` | **152 / 164 = 92.7%** | **138** | **408** |
| Prompt-only terse `GOAL/APPROACH/EDGE` | **153 / 164 = 93.3%** | 2298 | 2764 |
| **FSM vs FREE Δ** | **+0.6 pp** | **22.4× shorter** | **8.4× shorter** |

No distillation. No fine-tuning. No reward-model. A ~20-line GBNF grammar applied to the `<think>` block at inference time matches full-thinking accuracy with an order-of-magnitude fewer tokens. The prompt-only terse control slightly improves pass@1 on this run, but still uses 2298 thinking tokens on average; the grammar is what reliably enforces the compact token regime.

See [RESULTS.md](RESULTS.md) for the full experimental writeup, per-problem breakdown, and discussion of limitations (including contamination).

## Why this matters

Reasoning models like Qwen3, DeepSeek-R1, QwQ spend thousands of tokens in verbose prose thinking — exploring alternatives, restating, hedging. This work shows, on one benchmark, that for a large chunk of that reasoning, the verbose scaffolding isn't doing real work. The model already has the reasoning capability internally; grammar constraint just extracts it in a denser form.

If this holds on recent or post-release benchmarks (next: LiveCodeBench), the engineering implication is direct: **inference-time thinking compute can be cut ~10× via a grammar file alone**, with no training pipeline or serving changes beyond a GBNF argument.

## How it works

Single GBNF grammar (see [`fsm_grammar.gbnf`](fsm_grammar.gbnf)):

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
    --host 127.0.0.1 --port 8000 -c 32768 -ngl 999 --flash-attn
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

By default this downloads/serves `ggml-org/Qwen3.6-27B-GGUF` through native `llama-server` with `--spec-default`. Override with env vars:

```bash
HF_REPO=ggml-org/Qwen3.6-27B-GGUF N_CTX=32768 BACKGROUND=1 ./run_llama_server.sh
MODEL_PATH=/path/to/model.gguf BACKGROUND=1 ./run_llama_server.sh
KV_TYPE=q8_0 BACKGROUND=1 ./run_llama_server.sh
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
    --n-problems 50 --max-tokens 8192 --only all \
    --out-dir lcb_v6_2025_01_01_n50_all
```

Each run produces in `fsm_vs_free/`:
- `results.jsonl` — per-problem raw generations, extracted think/code, pass/fail, errors, extraction metadata
- `summary.json` — aggregate stats, pass-set overlap, and failure accounting
- `per_problem.md` — human-readable report with outcome tags (🔺 / 🔻 / 🟰 / ❌)

## Architecture notes

Qwen3.6-35B-A3B is a **MoE hybrid**:
- 40 layers total, pattern `10 × (3 × GatedDeltaNet + 1 × GatedAttention)`
- 256 experts, 9 active per token (3B active params)
- Native 262K context, extensible to 1M via YaRN

For the FSM experiment, the base-LM architecture doesn't matter much — we're constraining what tokens are emitted, not how they're processed internally. The same GBNF approach should work with any model served through llama.cpp, vLLM, SGLang, TGI, etc., as long as the server accepts a grammar parameter.

## Limitations and open questions

1. **Contamination.** HumanEval has been in training corpora for years. All modes may be recalling solutions, not reasoning. The "FSM matches FREE" result could mean "grammar extracts the same memorized solution in fewer tokens" rather than "grammar preserves reasoning capability." LiveCodeBench release v6 currently reaches April 2025, so it is a useful recent/lower-contamination check but not a strict post-Qwen3.6-release cutoff.

2. **Grammar specificity.** The GOAL/APPROACH/EDGE format was tuned for coding. Math / logic / planning domains may need different symbolic formats. Unclear whether a single "universal" compressed-thinking grammar exists or whether each domain needs its own.

3. **Reasoning depth.** The current result is on problems solvable in one forward pass. For multi-step problems (SWE-Bench, long-horizon planning, agentic tasks), whether grammar compression preserves capability is untested.

4. **Model dependency.** Run with one model (Qwen3.6-35B-A3B). We don't know yet whether smaller models (1B-7B) can be grammar-constrained the same way or whether the capability requires scale.

## Status

- ✅ HumanEval+ full (164 problems): FSM 152/164 vs FREE 151/164, 22× think-token compression
- 🧪 LiveCodeBench v6 recent subset (script ready; run pending) — public functional-test contamination pressure
- ⏳ MBPP+ (planned)
- 🔲 Other domains (math, logic, planning)
- 🔲 Cross-model transfer (smaller models)

## References

The direct technical antecedent is [Coconut: Training Large Language Models to Reason in a Continuous Latent Space (Hao et al., 2024)](https://arxiv.org/abs/2412.06769), which compresses CoT into continuous latents via fine-tuning. This work tests whether a much cheaper intervention — grammar constraint at inference, no training — captures a meaningful fraction of the same benefit.
