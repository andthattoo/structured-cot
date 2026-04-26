# structured-cot reproduction and generalization design

## Core question

Does grammar-constrained thinking preserve or improve coding-task pass@1 while substantially reducing thinking tokens, and is the effect robust across models, grammars, benchmark families, and serving stacks?

## Study phases

1. Exact reproduction: rerun the committed HumanEval+, MBPP+, and LiveCodeBench commands with the same model family, grammar files, max-token budgets, server settings, and evaluator.
2. Internal validity audit: verify that pass/fail scoring, code extraction, token accounting, grammar injection, and dataset filtering match the claims.
3. Generalization tests: vary one axis at a time: model architecture/scale, grammar format, benchmark domain, and inference stack.
4. Stress tests: track whether compressed thinking moves into comments or post-think answer text, especially on harder problems.

## Non-goals

- Do not treat public-test LiveCodeBench as official leaderboard evidence.
- Do not infer cross-domain reasoning generality from coding-only results.
- Do not conflate shorter visible `<think>` text with lower total reasoning compute unless total tokens and runtime are also measured.

## Primary dependent variables

- pass@1
- mean thinking tokens
- mean total tokens
- post-think tokens
- code comment tokens
- extraction issue rate
- wall-clock latency and tokens/sec
- pass-set overlap between modes

## Required controls

- FREE baseline: unconstrained `<think>...</think>`.
- PROMPT_TERSE baseline: same structure requested in prompt but no grammar.
- FSM base grammar: `GOAL/APPROACH/EDGE`.
- FSM_PLAN grammar: `GOAL/STATE/ALGO/EDGE/VERIFY`.
- No-think or answer-only control where supported.

