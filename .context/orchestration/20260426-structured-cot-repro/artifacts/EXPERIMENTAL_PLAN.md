# structured-cot reproduction and generalization plan

Date: 2026-04-26
Repo: https://github.com/andthattoo/structured-cot
Local clone: `/Users/haoli/Documents/Codex/2026-04-26/help-me-make-a-plan-to/structured-cot`
Snapshot: `83b16be0226d10627b27cc3af2b87209fab007b7`

## 0. Current read of the claims

The current repo supports a narrow but interesting claim:

- On HumanEval+ full, `FSM` is reported as 152/164 versus `FREE` 151/164, with mean think tokens 138 versus 3087.
- On a LiveCodeBench release_v6 LeetCode public-test slice, `FSM_PLAN` is reported as 32/50 versus `FREE` 25/50, with mean think tokens 267 versus 11553.
- The stronger claim is not "the grammar improves reasoning" yet. The cleaner claim is that grammar-constrained decoding reliably forces visible `<think>` compression, and on the reported coding slices it does not obviously hurt pass@1.

Important caveat: the repo does not commit raw `results.jsonl`, `summary.json`, `per_problem.md`, server logs, exact model checksums, dataset task IDs, or a lockfile. Exact reproduction therefore requires rerunning the study, not recomputing the headline tables from committed artifacts.

This plan is intentionally constrained to vLLM only. It is a vLLM replication and generalization study, not a bit-for-bit reproduction of the original llama-cpp-python serving stack.

## 1. vLLM-only reproduction target

Treat this as the reference configuration for the vLLM replication of the headline HumanEval+ result:

- Hardware: Linux CUDA host with enough aggregate VRAM for vLLM defaults.
- Model: HF-format `Qwen/Qwen3.6-35B-A3B`, not GGUF.
- Server: vLLM OpenAI-compatible server only.
- Server settings: vLLM defaults unless required for correctness, plus `--reasoning-parser qwen3`.
- Generation: `temperature=0.0`, `max_tokens=8192` for HumanEval+, `max_tokens=16384` for LiveCodeBench.
- Evaluator: `fsm_vs_free_eval.py` at repo snapshot `83b16be`.

Because the original report used a GGUF model through llama-cpp-python, vLLM cannot be treated as an exact serving-stack reproduction. The main comparison is whether the same qualitative effect survives under vLLM: visible `<think>` compression with no material pass@1 loss.

## 1.1 VRAM budget

For the constrained vLLM-only plan, the peak VRAM cell is `Qwen/Qwen3.6-35B-A3B` in vLLM default dtype/weight format.

Recommended allocation:

- Safe default/full-context target: 8 x 80GB GPUs, matching the model card's vLLM example for `--max-model-len 262144`.
- Likely practical lower bound for this benchmark only: 2 x 80GB GPUs, assuming tensor parallelism and low concurrency. This should be treated as a validation target, not a guaranteed default-context setup.
- Single 80GB GPU: not recommended for the non-quantized 35B default-weight run. BF16 weights alone are roughly 70 GB decimal before vLLM overhead, CUDA graphs, activations, and KV cache.
- Single H200 141GB-class GPU: likely enough for low-concurrency benchmark runs, but still needs a vLLM smoke test because default max context and multimodal profiling can affect startup memory.

If compute cost matters, the cleanest first probe is to start vLLM on 2 x 80GB with the intended default settings and run the grammar sentinel plus a 3-problem smoke test. If vLLM rejects the default max context, do not silently lower context in the main experiment; record that as a failed default-setting cell, then decide whether to create a separate capped-context pilot.

## 2. Reproduction setup checklist

Run on the GPU host:

```bash
git clone https://github.com/andthattoo/structured-cot.git
cd structured-cot
git checkout 83b16be0226d10627b27cc3af2b87209fab007b7

uv sync

huggingface-cli download Qwen/Qwen3.6-35B-A3B \
  --local-dir ~/models/qwen3.6-35b-a3b

find ~/models/qwen3.6-35b-a3b -type f -name '*.safetensors' -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  | tee model.sha256.txt

vllm serve Qwen/Qwen3.6-35B-A3B \
  --port 8000 \
  --tensor-parallel-size 8 \
  --reasoning-parser qwen3
```

In a second pane:

```bash
curl http://127.0.0.1:8000/v1/models
nvidia-smi | tee nvidia-smi.txt
git rev-parse HEAD | tee repo-head.txt
uv pip freeze | tee uv-freeze.txt
vllm --version | tee vllm-version.txt
```

Smoke:

```bash
uv run python fsm_vs_free_eval.py \
  --n-problems 10 \
  --max-tokens 4096 \
  --only all \
  --out-dir repro_smoke_humaneval_n10
```

HumanEval+ headline rerun:

```bash
uv run python fsm_vs_free_eval.py \
  --n-problems 164 \
  --max-tokens 8192 \
  --only all \
  --out-dir repro_humaneval_full_all
```

LiveCodeBench reported table rerun:

```bash
uv run python fsm_vs_free_eval.py \
  --dataset livecodebench \
  --lcb-version release_v6 \
  --date-cutoff 2025-01-01 \
  --platform leetcode \
  --n-problems 50 \
  --max-tokens 16384 \
  --only free \
  --out-dir repro_lcb_v6_2025_01_01_free_n50

uv run python fsm_vs_free_eval.py \
  --dataset livecodebench \
  --lcb-version release_v6 \
  --date-cutoff 2025-01-01 \
  --platform leetcode \
  --n-problems 50 \
  --max-tokens 16384 \
  --only fsm \
  --grammar-file grammars/fsm_grammar_lcb_plan.gbnf \
  --out-dir repro_lcb_v6_2025_01_01_fsm_plan_n50
```

Also run the base grammar on the same LCB slice to separate "grammar at all" from "richer plan grammar":

```bash
uv run python fsm_vs_free_eval.py \
  --dataset livecodebench \
  --lcb-version release_v6 \
  --date-cutoff 2025-01-01 \
  --platform leetcode \
  --n-problems 50 \
  --max-tokens 16384 \
  --only fsm \
  --grammar-file grammars/fsm_grammar.gbnf \
  --out-dir repro_lcb_v6_2025_01_01_fsm_base_n50
```

Archive every output directory plus vLLM logs, `model.sha256.txt`, `repo-head.txt`, `uv-freeze.txt`, `vllm-version.txt`, `nvidia-smi.txt`, and the full server startup command.

## 3. Internal validity audit

Before believing the headline numbers, check these failure modes.

### 3.1 Raw artifact gap

The repo ignores eval outputs (`fsm_vs_free/`, `*.jsonl`) and does not commit the raw generations. Require raw artifacts from the rerun for every claim:

- `results.jsonl`
- `summary.json`
- `per_problem.md`
- exact `task_id` list
- raw vLLM server log
- model checksum
- dependency and server version dump

### 3.2 Dataset pinning

`load_dataset(...)` is used without a `revision`. For an auditable rerun, record:

- Hugging Face dataset repo and revision for `evalplus/humanevalplus`.
- Hugging Face dataset repo, `version_tag`, and revision for `livecodebench/code_generation_lite`.
- The exact ordered task IDs after filtering.

Add a small manifest next to each result directory:

```json
{
  "repo_head": "83b16be0226d10627b27cc3af2b87209fab007b7",
  "dataset": "livecodebench/code_generation_lite",
  "version_tag": "release_v6",
  "date_cutoff": "2025-01-01",
  "platform": "leetcode",
  "task_ids": ["..."],
  "model_path_sha256": "...",
  "server": "vLLM",
  "server_flags": "..."
}
```

### 3.3 Scoring audit

HumanEval+/MBPP+:

- Compare the custom temp-file harness against the official EvalPlus evaluation path on a sample of generated outputs.
- Confirm missing entry points, syntax errors, timeouts, and runtime errors bucket identically.

LiveCodeBench:

- Treat current LCB numbers as public-test smoke metrics only.
- Compare a sample against the official LCB evaluator where possible.
- Flag tasks involving in-place mutation, custom structures, stateful classes, stdin-style input, or timeout-sensitive behavior. The current harness is intentionally simple and may not match official semantics.

### 3.4 Token accounting audit

The evaluator counts think tokens with the HF tokenizer but usually takes total tokens from server `usage.completion_tokens`. For auditability:

- Store server usage tokens and HF-retokenized completion tokens separately.
- Fail loudly if tokenizer loading falls back to `len(text)//4`.
- Compute `post_think_tokens` from a single tokenizer as a cross-check.
- Report wall-clock latency, tokens/sec, and total completion tokens. Visible think-token compression alone is not enough if reasoning moves into comments or answer text.

### 3.5 Grammar enforcement audit

For every FSM output:

- Parse/assert the expected `<think>` shape.
- Count line lengths inside each required section.
- Record whether the answer channel contains large comments, prose, or extra scratchpad.
- Patch the evaluator's FSM request for vLLM structured outputs, for example `extra_body={"structured_outputs": {"grammar": grammar}}`, instead of llama.cpp's `extra_body={"grammar": grammar}`.
- Include a small sentinel test where an impossible grammar is rejected or a toy grammar forces an obvious prefix, proving vLLM honors the structured-output grammar.

### 3.6 Determinism and uncertainty

Temperature 0 does not guarantee bit-for-bit determinism across kernels or vLLM versions. For final claims:

- Run at least 3 repeats for each key cell.
- Use paired comparisons by task ID.
- Report confidence intervals for pass@1 and bootstrap intervals for token means.
- Use McNemar-style paired tests for FREE vs FSM pass-set differences.

## 4. Generalization experiment matrix

Run this after exact reproduction passes.

| Axis | Cells | Purpose |
| --- | --- | --- |
| Model architecture | Qwen3.6 MoE, dense Qwen/Llama-style model, DeepSeek/QwQ-style reasoning model | Test whether the effect is architecture-specific. |
| Model scale | 7B-ish, 14B-ish, 30B+ | Test whether smaller models lose accuracy when visible reasoning is compressed. |
| Grammar | `GOAL/APPROACH/EDGE`, `GOAL/STATE/ALGO/EDGE/VERIFY`, length-capped free-form, JSON/XML plan, random/permuted labels | Separate label semantics, length control, and structure. |
| Prompt control | FREE, PROMPT_TERSE, FSM, no-think/direct-code | Separate "asked to be terse" from "forced to be terse". |
| Dataset | HumanEval+, MBPP+, LCB recent public tests, LCB post-release if enough tasks, APPS/CodeContests subset | Separate contamination, difficulty, and benchmark-family effects. |
| Non-code domain | GSM8K, MATH/AIME-style, BBH logic/planning | Test whether this is coding-specific. Requires new scorer/extractor. |
| Serving implementation | vLLM only | Keep serving infra fixed so the study isolates model, grammar, and benchmark effects. |

Primary dependent variables:

- pass@1 or task-native score
- mean think tokens
- mean total tokens
- post-think tokens
- code/comment tokens or answer-channel scratchpad tokens
- extraction issue rate
- timeout/no-code rate
- pass-set overlap and paired deltas
- latency, tokens/sec, and GPU-hours

## 5. Minimum viable generalization run

If compute is limited, do this first:

- Models: Qwen/Qwen3.6-35B-A3B, one smaller Qwen reasoning model, one non-Qwen reasoning model, all in their vLLM default dtype/weight format.
- Benchmarks: HumanEval+ full, MBPP+ 100, LCB release_v6 first 50, LCB post-2026-04-23 if enough tasks exist.
- Modes: FREE, PROMPT_TERSE, FSM base, FSM_PLAN.
- Repeats: 3 per cell for the headline cells, 1 for exploratory cells.
- Metrics: all current evaluator metrics plus wall-clock, backend usage tokens, HF-retokenized total tokens, and task IDs.

Decision rule for a credible generalized claim:

- Compression: FSM think tokens at least 5x lower than FREE, and total tokens materially lower after counting answer-channel displacement.
- Accuracy: paired pass@1 delta within a predeclared tolerance, for example no worse than -2 pp on easy coding and no worse than -5 pp on hard coding, unless a gain is statistically stable across repeats.
- Robustness: no major increase in extraction failures, no vLLM structured-output failures, and no large comment/prose displacement that erases the total-token gain.
- Scope statement: claims must name the model family, benchmark family, grammar, and vLLM version. Do not call it architecture-general until at least two materially different architectures pass.

## 6. Concrete issues to resolve before publication-quality claims

1. Commit or publish raw run artifacts and manifests.
2. Pin dependency and dataset revisions.
3. State clearly that the new study is vLLM-only and is not an exact reproduction of the original llama-cpp-python/GGUF run.
4. Split the LCB claim into "public-test answer-forcing reliability" and "reasoning-quality improvement"; the current +14 pp includes many FREE empty-code failures.
5. Replace or cross-check the custom LCB harness with official evaluation where possible.
6. Make token accounting internally consistent and prevent silent tokenizer fallback.
7. Add repeat runs and paired uncertainty reporting.
