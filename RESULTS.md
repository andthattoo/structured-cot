# Results — HumanEval+ full run, cleaned evaluator

**Date:** 2026-04-24
**Hardware:** 1× H100 (Lambda Labs)
**Model:** `unsloth/Qwen3.6-35B-A3B-GGUF`, quant `UD-Q4_K_M` (~22 GB on disk)
**Inference:** `llama-cpp-python` 0.3+ server, flash-attn on, KV cache `q8_0`, `n_ctx=65536`
**Benchmark:** [`evalplus/humanevalplus`](https://huggingface.co/datasets/evalplus/humanevalplus) — 164 problems (HumanEval with augmented tests)
**Budget:** `max_new_tokens=8192`, greedy (`temperature=0`)
**Run:** `FREE`, `FSM`, and `PROMPT_TERSE` for 3 × 164 = 492 rollouts

This supersedes the earlier two-mode report. The evaluator now calls `check(entry_point)` directly; it no longer catches `NameError`, so missing functions and undefined-name implementations fail normally.

## Headline

| Mode | pass@1 | mean think tokens | mean total tokens |
|---|---:|---:|---:|
| **FREE** (natural `<think>...</think>`) | **151 / 164 = 92.1%** | 3087 | 3410 |
| **FSM** (grammar-constrained `GOAL/APPROACH/EDGE`) | **152 / 164 = 92.7%** | **138** | **408** |
| **PROMPT_TERSE** (prompt asks for `GOAL/APPROACH/EDGE`, no grammar) | **153 / 164 = 93.3%** | 2298 | 2764 |

- **Accuracy Δ (FSM − FREE):** **+0.6 pp**
- **Think-token compression (FREE / FSM):** **22.45×**
- **Total-token compression (FREE / FSM):** **8.36×**
- **Prompt-only think compression (FREE / PROMPT_TERSE):** **1.34×**
- **Grammar gain over prompt-only compression (PROMPT_TERSE / FSM):** **16.65×**

The clean takeaway is not that terse prompting is enough. `PROMPT_TERSE` slightly improves pass@1 on this run, but it still spends 2298 thinking tokens on average. The grammar-constrained run preserves the same accuracy band while forcing a radically smaller thinking format.

## Pass-set overlap

| Outcome | Count | Problems |
|---|---:|---|
| Both pass | 146 | (most of the benchmark) |
| Both fail | 7 | 32, 76, 91, 132, 145, 151, 163 |
| **FSM-only pass** (FREE fail → FSM pass) | 6 | 9, 26, 93, 129, 141, 154 |
| **FREE-only pass** (FSM fail → FREE pass) | 5 | 86, 97, 124, 125, 134 |

The pairwise disagreement remains small: 11 / 164 problems differ between `FREE` and `FSM`, with a slight edge to `FSM`.

## Prompt-terse control

| Outcome | Count | Problems |
|---|---:|---|
| All pass | 144 | (most of the benchmark) |
| All fail | 6 | 32, 76, 91, 132, 145, 151 |
| **PROMPT_TERSE-only pass** | 1 | 163 |
| **PROMPT_TERSE-only fail** | 2 | 12, 95 |
| PROMPT_TERSE matches FREE, not FSM | 6 | 97, 124, 125, 134, 141, 154 |
| PROMPT_TERSE matches FSM, not FREE | 5 | 9, 26, 86, 93, 129 |

This control separates "asked to be terse" from "forced to be terse." Prompt-only sometimes changes outcomes, but it does not reliably control reasoning length. In several logged cases, `PROMPT_TERSE` used thousands of thinking tokens despite the instruction to use a three-line plan.

## Failure accounting

| Mode | Failures | Failure types | Extraction issues |
|---|---:|---|---|
| FREE | 13 | missing_entry_point=1, runtime_error=9, runtime_name_error=2, timeout=1 | none |
| FSM | 12 | runtime_error=12 | none |
| PROMPT_TERSE | 11 | runtime_error=8, runtime_name_error=1, syntax_error=1, timeout=1 | none |

The absence of extraction issues is important. The cleaned run is not being decided by fenced-code parsing failures; the remaining failures are execution failures, timeouts, or generated code that does not satisfy the tests.

## Representative examples

### HumanEval/9 — FSM and PROMPT_TERSE rescue a FREE failure

Logged result:

| Mode | Outcome | Think tokens |
|---|---:|---:|
| FREE | fail | 2835 |
| FSM | pass | 102 |
| PROMPT_TERSE | pass | 4161 |

This is not clean evidence that the grammar improved reasoning over prompt-only. `PROMPT_TERSE` also passed. But it is strong evidence that grammar constraint enforces the compact token regime: prompt-only passed while spending about 41× more thinking tokens than `FSM`.

### HumanEval/86 — compressed formats regress

Logged result:

| Mode | Outcome | Think tokens |
|---|---:|---:|
| FREE | pass | 3721 |
| FSM | fail | 56 |
| PROMPT_TERSE | fail | 1062 |

This is the expected failure mode for aggressive compression: a compact plan can skip an underspecified edge case that verbose reasoning happens to explore. In this run, `PROMPT_TERSE` also failed and produced a syntax/indentation failure, so this problem should be inspected in `per_problem.md` before attributing the failure entirely to reasoning compression.

### HumanEval/83 — large compression with no accuracy loss

Logged result:

| Mode | Outcome | Think tokens |
|---|---:|---:|
| FREE | pass | 4054 |
| FSM | pass | 48 |
| PROMPT_TERSE | pass | 2625 |

This is the central pattern: `FREE` and `PROMPT_TERSE` spend thousands of reasoning tokens, while `FSM` emits a tiny structured plan and still passes.

## Token-budget sensitivity

`FREE` averages 3410 total completion tokens, well below the 8192-token budget. The compression result is therefore not mainly explained by `FREE` being cut off. `FSM` averages 408 total tokens, making its budget pressure practically irrelevant on HumanEval+.

`PROMPT_TERSE` averages 2764 total tokens. It reduces verbosity somewhat, but not enough to behave like a controlled compression method.

## LiveCodeBench v6 recent public-test run

To stress the claim beyond HumanEval+, we ran a recent LiveCodeBench slice:

- Dataset: `livecodebench/code_generation_lite`, `release_v6`
- Filter: `contest_date >= 2025-01-01`, `platform == leetcode`, functional tests only
- Size: first 50 filtered problems
- Scoring: public functional tests from the dataset, not the official private leaderboard
- Budget: `max_new_tokens=16384`

The active LiveCodeBench grammar is [`grammars/fsm_grammar_lcb_plan.gbnf`](grammars/fsm_grammar_lcb_plan.gbnf). It uses a richer five-field plan:

```text
GOAL / STATE / ALGO / EDGE / VERIFY
```

and leaves the answer/code region permissive.

| Metric | FREE | FSM_PLAN |
|---|---:|---:|
| pass@1 | 25 / 50 = 50.0% | **32 / 50 = 64.0%** |
| mean think tokens | 11553 | **267** |
| mean total tokens | 13632 | **2743** |
| mean post-think tokens | 2079 | 2476 |
| mean comment tokens | 52 | 1753 |
| answer-channel bloat | 11 / 50 | 16 / 50 |
| comment bloat | 0 / 50 | 19 / 50 |
| extraction issues | empty_code=18 | none |

Failure accounting:

| Mode | Failures | Failure types |
|---|---:|---|
| FREE | 25 / 50 | extraction_empty_code=18, syntax_error=4, missing_entry_point=1, timeout=1, wrong_answer=1 |
| FSM_PLAN | 18 / 50 | wrong_answer=13, syntax_error=3, runtime_error=1, timeout=1 |

The clean takeaway: on this public-test slice, `FSM_PLAN` improves pass@1 by **+14 percentage points**, cuts mean explicit think tokens by about **43x**, and cuts mean total completion tokens by about **5x**. The improvement is partly mechanical: FREE often spends the full 16k-token budget in `<think>` and never emits runnable code, while the grammar forces an early compact plan and gets to code reliably.

The caveat is equally important. LiveCodeBench exposes **reasoning displacement**: constraining `<think>` does not fully constrain total deliberation. On harder tasks, `FSM_PLAN` often moves some reasoning into comments or other post-think answer-channel text. That is why the evaluator reports `post_think_tokens`, `answer_channel_bloat`, `code_comment_tokens`, and `comment_bloat` in addition to think tokens.

Representative examples:

| Task | FREE | FSM_PLAN |
|---|---|---|
| 3702 | fail, 16384 think, empty code | pass, 114 think / 454 total |
| 3715 | fail, 16384 think, empty code | pass, 232 think / 1536 total |
| 3781 | fail, 16384 think, empty code | pass, 260 think / 7914 total |
| 3750 | pass, 15241 think / 15717 total | pass, 182 think / 724 total |
| 3708 | pass, 4370 think / 4492 total | pass, 191 think / 361 total |

The repo now keeps only two active grammars:

- [`grammars/fsm_grammar.gbnf`](grammars/fsm_grammar.gbnf): compact `GOAL/APPROACH/EDGE` plan for short coding tasks.
- [`grammars/fsm_grammar_lcb_plan.gbnf`](grammars/fsm_grammar_lcb_plan.gbnf): richer `GOAL/STATE/ALGO/EDGE/VERIFY` plan for LiveCodeBench.

Recommended LiveCodeBench A/B:

```bash
uv run python fsm_vs_free_eval.py --dataset livecodebench \
  --lcb-version release_v6 --date-cutoff 2025-01-01 --platform leetcode \
  --n-problems 50 --max-tokens 16384 --only free \
  --out-dir lcb_v6_2025_01_01_free_n50

uv run python fsm_vs_free_eval.py --dataset livecodebench \
  --lcb-version release_v6 --date-cutoff 2025-01-01 --platform leetcode \
  --n-problems 50 --max-tokens 16384 --only fsm \
  --grammar-file grammars/fsm_grammar_lcb_plan.gbnf \
  --out-dir lcb_v6_2025_01_01_fsm_lcb_plan_n50
```

## Methodology notes

**Modes**

- `FREE`: standard thinking-mode generation with the shared system prompt.
- `FSM`: same user prompt, but the server receives a GBNF grammar. The default `grammars/fsm_grammar.gbnf` forces `<think>` to contain exactly `GOAL`, `APPROACH`, and `EDGE` lines before unconstrained code. The LiveCodeBench plan grammar uses `GOAL/STATE/ALGO/EDGE/VERIFY`.
- `PROMPT_TERSE`: no grammar; the system prompt merely asks the model to use the same `GOAL/APPROACH/EDGE` thinking format.

**Test execution** — each generated code block is combined with the benchmark's `test` body and `entry_point`, written to a temp file, and executed via `python $FILE` in a subprocess with a 30s timeout. The harness calls `check(entry_point)` directly. Missing functions and undefined names now fail normally.

**Token counts** — using the `Qwen/Qwen3.6-35B-A3B` tokenizer loaded via `transformers`. Think tokens are measured on the extracted portion of the response between, or before, `<think>/</think>` tags. Total tokens come from the server's `usage.completion_tokens`.

**Code extraction** — prefers the last fenced `python` block after `</think>`, then the last fenced block anywhere, then a `def ...` block, then a fallback. The cleaned run reports no extraction issues for any mode.

## Limitations and caveats

1. **HumanEval contamination.** HumanEval has been public since 2021. Qwen3.6 may have seen many solutions during training. This result shows that grammar-constrained thinking preserves performance on this benchmark; it does not prove post-cutoff reasoning preservation.

2. **Single model, single quant.** All results are on Qwen3.6-35B-A3B with `Q4_K_M` quantization. Other models, quantizations, and reasoning recipes may have different compression ceilings.

3. **One domain.** HumanEval+ is short-form Python function synthesis. Math, logic, planning, and long-horizon agentic tasks may need different compressed formats or may lose more accuracy.

4. **Grammar specificity.** The `GOAL/APPROACH/EDGE` format was tuned for short coding tasks. It is a minimum viable grammar, not evidence of an optimal structure. The LiveCodeBench run suggests that harder tasks benefit from a richer grammar that gives the model structured space for state, algorithm, edge cases, and verification.

5. **Public benchmark scoring.** HumanEval+ uses the dataset's available augmented tests. LiveCodeBench numbers here are public functional-test pass rates, not official/private leaderboard scores.

## Next runs

1. **LiveCodeBench post-release subset.** For Qwen3.6, use `contest_date >= 2026-04-23` as the strict cutoff. If that yields too few problems, also run `contest_date >= 2026-03-01` and label it as recent-but-risky rather than clean.
2. **Measure reasoning displacement.** Continue tracking `post_think_tokens`, `answer_channel_bloat`, `code_comment_tokens`, and `comment_bloat` on harder coding subsets.
3. **Richer coding grammar.** Continue testing `grammars/fsm_grammar_lcb_plan.gbnf` against the current `GOAL/APPROACH/EDGE` grammar and new datasets.
4. **MBPP+.** Bigger and messier than HumanEval+, useful as a second coding benchmark.
5. **Smaller models.** Test whether grammar compression still works when the model has less latent capability.
6. **Math / logic.** Try the same grammar and domain-specific grammars on GSM8K, MATH/AIME-style tasks, and logic benchmarks.

## Raw data

All per-problem data — raw responses, extracted think/code, token counts, errors, extraction metadata, and failure buckets — is written to `fsm_vs_free/results.jsonl` by the evaluator. The aggregate report is `fsm_vs_free/summary.json`, and the readable per-problem report is `fsm_vs_free/per_problem.md`.
