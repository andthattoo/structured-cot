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

## LiveCodeBench early observation: reasoning displacement

Early LiveCodeBench v6 runs on recent LeetCode-style public functional tests show a different failure pattern from HumanEval+. The current grammar reliably caps the explicit `<think>` block, but on harder tasks the model can move reasoning into the unconstrained answer/code region after `</think>`.

Example logged pattern:

| Task | Mode | Outcome | Think tokens | Total tokens | Notes |
|---|---|---:|---:|---:|---|
| 3715 | FSM | fail | 134 | 7609 | Short plan, very long answer/code region |
| 3562 | FSM | fail | 190 | 7047 | Short plan, syntax failure deep in generated code |
| 3634 | FSM | pass | 130 | (small enough to pass cleanly) | Compression behaves as intended |

This is **reasoning displacement**: constraining the thought channel does not necessarily constrain total deliberation. If the problem is hard enough, the model may spend the missing reasoning budget in comments, pseudo-code, multiple drafts, or overly long code after the grammar becomes permissive.

That makes total completion tokens a required guardrail metric. On HumanEval+, `FSM` compresses both think tokens and total tokens. On harder LiveCodeBench tasks, `FSM` may still compress think tokens while losing much of the total-token advantage.

This suggests a next grammar iteration should constrain more of the response shape, not just the `<think>` prefix. A candidate coding grammar:

    <think>
    GOAL: ...
    STATE: ...
    ALGO: ...
    EDGE: ...
    VERIFY: ...
    </think>

    ```python
    # final code only
    ...
    ```

The goal is not to make the plan verbose. It is to give the model enough structured slots for harder tasks while preventing hidden scratchpad migration into the answer channel. Early strict-grammar logs show the tradeoff clearly: `fsm_grammar_lcb.gbnf` collapses huge 7k-token answer regions into roughly 1k-token answers on hard failures, but can also regress passing tasks and still allows plain English inside the Python fence if the model chooses invalid code-like text.

The repo now keeps three LiveCodeBench grammar variants:

- `fsm_grammar_lcb_plan.gbnf`: richer `GOAL/STATE/ALGO/EDGE/VERIFY` plan, permissive answer region.
- `fsm_grammar_lcb_fenced.gbnf`: richer plan plus exactly one fenced Python block; comments allowed, backticks disallowed.
- `fsm_grammar_lcb_no_comments.gbnf`: richer plan, permissive answer region, but no `#` comments or backticks.
- `fsm_grammar_lcb_bounded_no_comments.gbnf`: richer plan, no `#` comments or backticks. Despite the historical filename, this is now parse-safe and does not use `{m,n}` line bounds.
- `fsm_grammar_lcb_code_start_bounded.gbnf`: no-comments grammar whose answer must start with `from`, `import`, `class`, or `def`. Despite the historical filename, this is now parse-safe and does not use `{m,n}` line bounds.
- `fsm_grammar_lcb.gbnf`: strictest version; one fenced Python block, no `#` comments, no backticks.

Early 10-problem `fsm_grammar_lcb_fenced.gbnf` run:

| Metric | Value |
|---|---:|
| pass@1 | 2 / 10 = 20.0% |
| mean think tokens | 411 |
| mean total tokens | 2944 |
| mean post-think tokens | 2534 |
| answer-channel bloat | 2 / 10 |

This is not a good accuracy result, but it is a useful failure probe. The fenced grammar reduces some of the worst multi-draft markdown behavior, yet comments become an escape hatch: the model can continue its scratchpad as `# ...` lines inside the Python block. That is why the evaluator now tracks comment-token bloat separately. Re-run these LCB numbers before treating them as final because the local harness now injects `from typing import *`, matching common LeetCode-style ambient type hints and removing false `List`/`Optional` failures.

Early 10-problem `fsm_grammar_lcb_plan.gbnf` run after the harness/comment-metric fix:

| Metric | Value |
|---|---:|
| pass@1 | 6 / 10 = 60.0% |
| mean think tokens | 411 |
| mean total tokens | 4349 |
| mean post-think tokens | 3938 |
| mean comment tokens | 3134 |
| answer-channel bloat | 4 / 10 |
| comment bloat | 4 / 10 |

This is a much better accuracy signal than the fenced/strict variants, but it is not a compression win. The model still moves most of its deliberation into the answer channel, mostly as Python comments. The next useful ablation is therefore `fsm_grammar_lcb_no_comments.gbnf`: keep the richer five-field plan, do not force a fence, but remove the `#` comment escape hatch.

Full 50-problem `fsm_grammar_lcb_plan.gbnf` run:

| Metric | Value |
|---|---:|
| pass@1 | 32 / 50 = 64.0% |
| mean think tokens | 267 |
| mean total tokens | 2743 |
| mean post-think tokens | 2476 |
| mean comment tokens | 1753 |
| answer-channel bloat | 16 / 50 |
| comment bloat | 19 / 50 |
| failures | 18 / 50 |

Failure accounting: `wrong_answer=13`, `syntax_error=3`, `runtime_error=1`, `timeout=1`. Extraction issues: none.

This is the best LiveCodeBench grammar result so far. The richer plan preserves substantially more capability than the no-comments/fenced/code-start variants, but the token accounting shows why think-token compression alone is not enough on harder tasks. The useful interpretation is demand-adaptive reasoning: easy problems stay compact; harder problems often spend extra tokens in the answer channel, especially comments.

Shared rollouts reveal two separate displacement channels:

- Comment displacement: `3562` produced 217 comment lines and 4050 comment tokens, then failed with an indentation error. This is the pure `#`-scratchpad pathology.
- Field stuffing: `3684` passed, but spent 2175 think tokens because the grammar allowed arbitrarily long `GOAL/STATE/ALGO/EDGE/VERIFY` lines.

The no-comments grammar removes comments/backticks in the answer. An attempted bounded-line version used GBNF `{m,n}` repetition ranges, but the deployed llama.cpp parser rejected that syntax, so the current checked-in grammars use parse-safe unbounded lines.

Initial 10-problem `fsm_grammar_lcb_bounded_no_comments.gbnf` run:

| Metric | Value |
|---|---:|
| pass@1 | 1 / 10 = 10.0% |
| mean think tokens | 143 |
| mean total tokens | 3602 |
| mean post-think tokens | 3459 |
| mean comment tokens | 0 |
| answer-channel bloat | 4 / 10 |
| comment bloat | 0 / 10 |
| prose-before-code extraction | 8 / 10 |
| generation errors | 2 / 10 |

This showed that removing comments does remove the comment scratchpad, but the model then moved reasoning into raw answer prose. It also exposed a prompt confound: this grammar forbids backticks, while the shared FSM prompt still asked the model to return a fenced code block. The evaluator now switches to a direct-code prompt for grammars that cannot emit markdown fences. Treat the first bounded-no-comments result as diagnostic only and rerun after that prompt fix.

The shared rollouts also showed a harmless extraction artifact: no-fence grammars can still produce a bare `python` language label before code. The extractor now reports that separately as `language_label_before_code` instead of mixing it with true `prose_before_code`. The next stricter variant, `fsm_grammar_lcb_code_start_bounded.gbnf`, blocks both the language label and raw prose by forcing the answer to begin with a Python code start token.

Implementation note: llama.cpp's GBNF parser in this setup rejects `{m,n}` repetition ranges and underscores in rule names, so the checked-in experimental grammars avoid both.

Early 10-problem `fsm_grammar_lcb_code_start_bounded.gbnf` run:

| Metric | Value |
|---|---:|
| pass@1 | 2 / 10 = 20.0% |
| mean think tokens | 221 |
| mean total tokens | 1346 |
| mean post-think tokens | 1125 |
| mean comment tokens | 0 |
| answer-channel bloat | 1 / 10 |
| comment bloat | 0 / 10 |
| generation errors | 2 / 10 |

This confirms the token side of the hypothesis: code-start/no-comments collapses comments and most answer-channel bloat. It also confirms the capability cost: pass rate drops well below the permissive `plan` grammar. The original summary overcounted `prose_before_code` because the extractor treated the normal blank line after `</think>` as prose; the extractor now ignores leading whitespace, treats bare `python` labels separately, and prefers code after the last `</think>` if the model emits a stray second close tag.

For LiveCodeBench, useful reporting should include:

- `think_tokens`
- `total_tokens`
- `post_think_tokens = total_tokens - think_tokens`
- `code_comment_tokens`
- extraction failures such as `empty_code`
- syntax/runtime failures caused by answer-channel bloat

The evaluator now records `post_think_tokens = total_tokens - think_tokens` and an `answer_channel_bloat` flag using `--bloat-threshold` (default: 2048). It also records `code_comment_tokens` and `comment_bloat` using `--comment-bloat-threshold` (default: 1024). That makes this failure mode visible in `summary.json` and `per_problem.md`.

Recommended A/B:

```bash
uv run python fsm_vs_free_eval.py --dataset livecodebench \
  --lcb-version release_v6 --date-cutoff 2025-01-01 --platform leetcode \
  --n-problems 50 --max-tokens 8192 --only fsm \
  --grammar-file fsm_grammar.gbnf \
  --out-dir lcb_v6_2025_01_01_fsm_base_grammar

uv run python fsm_vs_free_eval.py --dataset livecodebench \
  --lcb-version release_v6 --date-cutoff 2025-01-01 --platform leetcode \
  --n-problems 50 --max-tokens 8192 --only fsm \
  --grammar-file fsm_grammar_lcb.gbnf \
  --out-dir lcb_v6_2025_01_01_fsm_lcb_grammar
```

If the strict grammar loses too much pass rate, run the two middle variants next:

```bash
uv run python fsm_vs_free_eval.py --dataset livecodebench \
  --lcb-version release_v6 --date-cutoff 2025-01-01 --platform leetcode \
  --n-problems 50 --max-tokens 8192 --only fsm \
  --grammar-file fsm_grammar_lcb_plan.gbnf \
  --out-dir lcb_v6_2025_01_01_fsm_lcb_plan

uv run python fsm_vs_free_eval.py --dataset livecodebench \
  --lcb-version release_v6 --date-cutoff 2025-01-01 --platform leetcode \
  --n-problems 50 --max-tokens 8192 --only fsm \
  --grammar-file fsm_grammar_lcb_fenced.gbnf \
  --out-dir lcb_v6_2025_01_01_fsm_lcb_fenced

uv run python fsm_vs_free_eval.py --dataset livecodebench \
  --lcb-version release_v6 --date-cutoff 2025-01-01 --platform leetcode \
  --n-problems 50 --max-tokens 8192 --only fsm \
  --grammar-file fsm_grammar_lcb_no_comments.gbnf \
  --out-dir lcb_v6_2025_01_01_fsm_lcb_no_comments

uv run python fsm_vs_free_eval.py --dataset livecodebench \
  --lcb-version release_v6 --date-cutoff 2025-01-01 --platform leetcode \
  --n-problems 50 --max-tokens 8192 --only fsm \
  --grammar-file fsm_grammar_lcb_bounded_no_comments.gbnf \
  --out-dir lcb_v6_2025_01_01_fsm_lcb_bounded_no_comments

uv run python fsm_vs_free_eval.py --dataset livecodebench \
  --lcb-version release_v6 --date-cutoff 2025-01-01 --platform leetcode \
  --n-problems 50 --max-tokens 8192 --only fsm \
  --grammar-file fsm_grammar_lcb_code_start_bounded.gbnf \
  --out-dir lcb_v6_2025_01_01_fsm_lcb_code_start_bounded
```

## Methodology notes

**Modes**

- `FREE`: standard thinking-mode generation with the shared system prompt.
- `FSM`: same user prompt, but the server receives a GBNF grammar. The default `fsm_grammar.gbnf` forces `<think>` to contain exactly `GOAL`, `APPROACH`, and `EDGE` lines before unconstrained code. The LiveCodeBench variants use `GOAL/STATE/ALGO/EDGE/VERIFY`, with separate permissive, no-comments, code-start, fenced, and strict answer-channel constraints.
- `PROMPT_TERSE`: no grammar; the system prompt merely asks the model to use the same `GOAL/APPROACH/EDGE` thinking format.

**Test execution** — each generated code block is combined with the benchmark's `test` body and `entry_point`, written to a temp file, and executed via `python $FILE` in a subprocess with a 30s timeout. The harness calls `check(entry_point)` directly. Missing functions and undefined names now fail normally.

**Token counts** — using the `Qwen/Qwen3.6-35B-A3B` tokenizer loaded via `transformers`. Think tokens are measured on the extracted portion of the response between, or before, `<think>/</think>` tags. Total tokens come from the server's `usage.completion_tokens`.

**Code extraction** — prefers the last fenced `python` block after `</think>`, then the last fenced block anywhere, then a `def ...` block, then a fallback. The cleaned run reports no extraction issues for any mode.

## Limitations and caveats

1. **HumanEval contamination.** HumanEval has been public since 2021. Qwen3.6 may have seen many solutions during training. This result shows that grammar-constrained thinking preserves performance on this benchmark; it does not prove post-cutoff reasoning preservation.

2. **Single model, single quant.** All results are on Qwen3.6-35B-A3B with `Q4_K_M` quantization. Other models, quantizations, and reasoning recipes may have different compression ceilings.

3. **One domain.** HumanEval+ is short-form Python function synthesis. Math, logic, planning, and long-horizon agentic tasks may need different compressed formats or may lose more accuracy.

4. **Grammar specificity.** The `GOAL/APPROACH/EDGE` format was tuned for short coding tasks. It is a minimum viable grammar, not evidence of an optimal structure. Early LiveCodeBench runs suggest that harder tasks may need a richer grammar that gives the model structured space for state, algorithm, edge cases, and verification.

5. **Public benchmark scoring.** The run uses the dataset's available augmented tests. LiveCodeBench public functional tests are the next step, but should be labeled as public-test pass rate unless the official/private evaluator is wired in.

## Next runs

1. **LiveCodeBench post-release subset.** For Qwen3.6, use `contest_date >= 2026-04-23` as the strict cutoff. If that yields too few problems, also run `contest_date >= 2026-03-01` and label it as recent-but-risky rather than clean.
2. **Measure reasoning displacement.** Use the new `post_think_tokens` and `answer_channel_bloat` metrics to compare LiveCodeBench grammar variants.
3. **Richer coding grammar.** Test `fsm_grammar_lcb_plan.gbnf`, `fsm_grammar_lcb_no_comments.gbnf`, `fsm_grammar_lcb_bounded_no_comments.gbnf`, `fsm_grammar_lcb_code_start_bounded.gbnf`, `fsm_grammar_lcb_fenced.gbnf`, and `fsm_grammar_lcb.gbnf` against the current `GOAL/APPROACH/EDGE` grammar.
4. **MBPP+.** Bigger and messier than HumanEval+, useful as a second coding benchmark.
5. **Smaller models.** Test whether grammar compression still works when the model has less latent capability.
6. **Math / logic.** Try the same grammar and domain-specific grammars on GSM8K, MATH/AIME-style tasks, and logic benchmarks.

## Raw data

All per-problem data — raw responses, extracted think/code, token counts, errors, extraction metadata, and failure buckets — is written to `fsm_vs_free/results.jsonl` by the evaluator. The aggregate report is `fsm_vs_free/summary.json`, and the readable per-problem report is `fsm_vs_free/per_problem.md`.
