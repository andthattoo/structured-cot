# structured-cot reproduction planning state

Objective: build a concrete plan to reproduce the structured-cot claims, audit their reproducibility, and test whether the result generalizes beyond the reported Qwen3.6 + GBNF + coding-benchmark setting.

Constraints:
- Do not modify upstream experiment code during planning.
- Treat README/RESULTS numbers as claims until independently rerun or backed by committed raw artifacts.
- Keep exact reproduction separate from extension experiments.
- Prefer public-test/eval commands that can be rerun from the cloned repo.

Primary evidence:
- Upstream repo: https://github.com/andthattoo/structured-cot
- Local clone: /Users/haoli/Documents/Codex/2026-04-26/help-me-make-a-plan-to/structured-cot
- Snapshot HEAD: 83b16be0226d10627b27cc3af2b87209fab007b7

Acceptance criteria:
- Enumerate exact commands needed for first reproduction.
- Identify claims that are not exactly reproducible from committed artifacts.
- Define validation checks for scoring, extraction, token accounting, and grammar enforcement.
- Propose an experimental matrix that isolates model, grammar, serving stack, dataset, and benchmark effects.

Stop conditions:
- Exact numerical reproduction requires an H100-class CUDA host and model download credentials.
- LiveCodeBench/HumanEval+ dataset revisions must be pinned before final numeric comparison.

