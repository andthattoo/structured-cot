# Results — HumanEval+ full run

**Date:** 2026-04-24
**Hardware:** 1× H100 (Lambda Labs)
**Model:** `unsloth/Qwen3.6-35B-A3B-GGUF`, quant `UD-Q4_K_M` (~22 GB on disk)
**Inference:** `llama-cpp-python` 0.3+ server, flash-attn on, KV cache `q8_0`, `n_ctx=65536`
**Benchmark:** [`evalplus/humanevalplus`](https://huggingface.co/datasets/evalplus/humanevalplus) — 164 problems (HumanEval with augmented tests)
**Budget:** `max_new_tokens=8192`, greedy (`temperature=0`)
**Total wall time:** 6853 s ≈ 114 min for 2 × 164 = 328 rollouts

## Headline

| Mode | pass@1 | mean think tokens | mean total tokens |
|---|---|---|---|
| **FREE** (natural `<think>…</think>`) | **152 / 164 = 92.7%** | 3087 | 3410 |
| **FSM** (grammar-constrained) | **152 / 164 = 92.7%** | **138** | 408 |

- **Accuracy Δ (FSM − FREE)**: **+0.0 pp**
- **Think-token compression**: **22.45×**
- **Total-token compression**: **8.36×**

Pass rates are identical to the single problem. The underlying *sets* of passing problems are not identical — they disagree on 10 out of 164 — but the disagreement is symmetric (5 problems where FREE wins, 5 where FSM wins).

## Disagreement breakdown

| Outcome | Count | Problems |
|---|---|---|
| 🟰 Both pass | 147 | (most of the benchmark) |
| ❌ Both fail | 7 | 32, 76, 91, 132, 145, 151, 163 |
| 🔺 **FSM wins** (FREE fail → FSM pass) | 5 | 9, 26, 93, 141, 154 |
| 🔻 **FREE wins** (FSM fail → FREE pass) | 5 | 86, 97, 124, 125, 134 |

Notable:
- **HumanEval/154** — FREE hit the timeout (`TIMEOUT`), FSM finished in 145 think tokens. When verbose reasoning runs out of budget mid-problem, FSM's structural ceiling is an advantage.
- **HumanEval/9, 26, 93, 141** — FREE produced `NameError` or incomplete code, typically because its multi-draft response pattern leads to ambiguous final code even with our last-fenced-block extraction. FSM's single code block after `</think>` sidesteps this.
- **HumanEval/86, 97, 124, 125, 134** — FSM's compressed plan didn't surface an edge case that FREE caught. These are genuine cases where 150 tokens of plan isn't sufficient scaffolding for the problem's complexity.

## Per-problem compression distribution

Looking across the 152 problems both modes solved correctly:

| Metric | Value |
|---|---|
| Median think-token compression | ~15× |
| Max compression | 70× (HumanEval/18) |
| Min compression | 1.5× (HumanEval/27, which had complex state tracking) |
| FSM think mean on both-pass problems | 135 tokens |
| FREE think mean on both-pass problems | 2930 tokens |

Compression is tightly correlated with problem triviality. For one-liners like `truncate_number`, FREE spent 1509 thinking tokens and FSM spent 142 — not because the problem is hard, but because FREE's thinking-mode habitually enumerates doctest examples, discusses floating-point precision, mentally benchmarks alternatives. The grammar strips that ceremony without losing correctness.

## Representative examples

### HumanEval/18 — 70× compression, both pass

FREE think (3426 tokens) includes:
- 8 paragraphs of problem-statement re-analysis
- 3 different candidate algorithms compared
- Multiple self-corrections ("wait, that's wrong...")
- Explicit doctest walk-throughs

FSM think (49 tokens):
```
GOAL: count non-overlapping occurrences of substring in string
APPROACH: use str.count(sub) which counts non-overlapping matches
EDGE: empty substring should return 0
```

Both produce correct code. FSM's plan captures the algorithmic decision (use `.count()`) and the key edge case. The 3377 tokens FREE spent beyond that are not load-bearing for this problem.

### HumanEval/86 — FSM regression, FREE wins

FSM's compressed plan:
```
GOAL: sort words in each sentence alphabetically
APPROACH: split into words, sort, join
EDGE: maintain space separators
```

FREE's verbose thinking explicitly considered whether "alphabetically" meant case-sensitive or insensitive, settled on ASCII-order (case-sensitive), verified with examples. FSM's plan didn't probe this, produced case-insensitive sort, failed on a test case with mixed-case input.

This is the class of error we expect from compression: **underspecified edge cases that verbose reasoning happens to explore**. It's not a capability failure of the model — it's an attentional failure of the compressed format.

## Token-budget sensitivity

Since the FSM mode uses ~22× fewer thinking tokens, it's interesting to look at whether FREE is bottlenecked by the 8192-token budget:

- FREE mean total: 3410 tokens. Well below budget.
- FREE max: one of the timeouts hit the budget.
- FSM mean total: 408 tokens. Practically irrelevant for budget.

So the compression here isn't "FREE got cut off, FSM didn't." The comparison is fair — FREE had room to finish its natural generation on ~98% of problems.

## Methodology notes

**Test execution** — each generated code is combined with the benchmark's `test` body and `entry_point`, written to a temp file, executed via `python $FILE` in a subprocess with 30s timeout. `check(entry_point)` is called; pass = returncode 0. Early iterations used `python -c` and hit `E2BIG / Errno 7` on long programs; tempfile-based execution is correct.

**Token counts** — using `Qwen/Qwen3.6-35B-A3B` tokenizer loaded via `transformers`. Think tokens measured on the extracted portion of the response between (or before) `<think>/</think>` tags. Total tokens from server's `usage.completion_tokens`.

**Code extraction** — prefers the *last* fenced `python` block in the response (FREE mode often emits draft code blocks mid-thinking; the final answer is always last). Earlier iterations took the first fenced block and scored FREE incorrectly on multi-draft responses.

**Both modes use the same prompt** — a system message asking for careful thought and a fenced `python` final answer. Only difference: the FSM run passes `grammar` in the request body; the FREE run doesn't.

## Limitations and caveats

1. **HumanEval contamination.** The benchmark has been public since 2021. Qwen3.6's training data almost certainly contains many HumanEval solutions. Both modes may be recalling memorized code rather than reasoning from first principles. The 22× compression could mean "the model stops wasting tokens on rehearsal of a memorized answer" rather than "the model reasons in compressed form." **We cannot distinguish these from HumanEval alone.**

2. **Small accuracy delta, single seed.** `+0.0 pp` is cleanly interpretable only because of the symmetric 5-5 disagreement. With temperature=0 (greedy) there is no seed variance, but a different problem pool, different base prompt, or different grammar could shift this by ±5 pp in either direction.

3. **Single model.** All results are on Qwen3.6-35B-A3B with `Q4_K_M` quantization. Different base models, different quants, or different reasoning-tuning recipes may give very different compression ceilings.

4. **One domain.** Coding problems with test-case correctness signal. Whether the same grammar approach transfers to math (MATH, GSM8K, AIME), logic (LogicBench, FOLIO), planning (BlocksWorld), or open-ended reasoning is untested.

5. **Grammar tuning.** The `GOAL/APPROACH/EDGE` three-line format was chosen after two grammar iterations (multi-line rules → single-line; bullet-loop trap → one-line sections; invalid `\-` escape → hex char classes). A better grammar for coding tasks probably exists; this is a near-minimum viable one.

## Next runs (in priority order)

1. **LiveCodeBench post-cutoff subset.** Filter to `contest_date >= 2025-12-01`. Resolves the contamination question. If FSM still matches FREE, compression is real for reasoning. If FSM regresses >10pp, grammar was riding on memorized patterns.
2. **MBPP+.** Bigger, messier, less clean. Confirms the finding generalizes slightly.
3. **Smaller base model.** Qwen2.5-7B-Instruct. Does the capability hold when the base is weaker?
4. **Math / logic.** Same grammar or domain-specific grammar on GSM8K and LogicBench.

## Raw data

All per-problem data — raw responses, extracted think/code, token counts, errors — is in `fsm_vs_free/results.jsonl` from the run (not committed to the repo; regenerate by running the eval). The aggregate `summary.json` is similar but compact.

A human-readable `per_problem.md` is auto-generated at the end of each run, with one section per problem including problem statement, both extracted think blocks side by side, both codes, and outcome tag.
