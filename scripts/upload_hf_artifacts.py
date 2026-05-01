#!/usr/bin/env python3
"""Package and upload transition-encoder / MQE critic artifacts to HF Hub."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_TRANSITION_DIR = Path("outputs/transition_encoder/qwen06b_agenttrove_50k_v0_steps2000_neg4")
DEFAULT_CRITIC_DIR = Path("outputs/mqe/agenttrove_50k_transition_qwen06b_steps2000_neg4")
DEFAULT_TRANSITION_REPO = "driaforall/code-state-embedding"
DEFAULT_CRITIC_REPO = "driaforall/code-mqe-critic"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def final_metrics(metrics_blob: dict[str, Any]) -> dict[str, Any]:
    metrics = metrics_blob.get("metrics")
    if isinstance(metrics, list) and metrics:
        last = metrics[-1]
        return last if isinstance(last, dict) else {}
    final = metrics_blob.get("final_metrics")
    return final if isinstance(final, dict) else {}


def fmt_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, int):
        return str(value)
    if value is None:
        return "n/a"
    return str(value)


def metric_table(metrics: dict[str, Any], keys: list[str]) -> str:
    rows = ["| Metric | Value |", "|---|---:|"]
    for key in keys:
        if key in metrics:
            rows.append(f"| `{key}` | {fmt_metric(metrics[key])} |")
    return "\n".join(rows)


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def copy_transition_artifacts(src: Path, dst: Path) -> None:
    for name in [
        "transition_config.json",
        "transition_head.pt",
        "metrics.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "README.md",
    ]:
        copy_if_exists(src / name, dst / name)
    copy_if_exists(src / "encoder_adapter", dst / "encoder_adapter")


def copy_critic_artifacts(src: Path, dst: Path) -> None:
    for name in ["best.pt", "metrics.json", "README.md"]:
        copy_if_exists(src / name, dst / name)


def transition_readme(src: Path, repo_id: str) -> str:
    config = read_json(src / "transition_config.json")
    metrics_blob = read_json(src / "metrics.json")
    metrics = final_metrics(metrics_blob)
    base_model = config.get("base_model", "Qwen/Qwen3-Embedding-0.6B")
    embedding_dim = config.get("embedding_dim", metrics_blob.get("embedding_dim", "unknown"))
    action_feature_dim = config.get("action_feature_dim", metrics_blob.get("action_feature_dim", "unknown"))
    train_rows = metrics_blob.get("train_rows", "unknown")
    val_rows = metrics_blob.get("val_rows", "unknown")

    table = metric_table(
        metrics,
        [
            "step",
            "train_next_retrieval_acc@1",
            "val_next_retrieval_acc@1",
            "val_next_retrieval_acc@5",
            "val_same_rollout_hard_negative_acc",
            "val_same_signature_hard_negative_acc",
            "val_action_type_acc",
            "val_mean_positive_margin",
        ],
    )
    return f"""---
library_name: peft
base_model: {base_model}
tags:
- code
- embeddings
- agents
- transition-model
- mqe
- qwen
pipeline_tag: feature-extraction
---

# Code State Transition Encoder

This repository contains a goal-free transition encoder for code-agent states.
It was trained for MQE-style progress critics on AgentTrove-derived transitions.

The encoder learns the transition relation:

```text
E_state(s), E_action(a), action_features -> T(s, a) ~= E_state(s_next)
```

It is intended to produce cached tensors for downstream goal-conditioned MQE
critics, not to be used as a standalone verifier.

## Contents

- `encoder_adapter/`: LoRA adapter for `{base_model}`
- `transition_head.pt`: transition MLP and action-type auxiliary head
- `transition_config.json`: training/config metadata
- tokenizer files copied from the training output
- `metrics.json`: training and validation metrics

## Training Data

- Source: `open-thoughts/AgentTrove`
- Prepared schema: canonical `(state_text, action_text, next_state_text, action_features)`
- Train rows: `{train_rows}`
- Validation rows: `{val_rows}`
- Embedding dim: `{embedding_dim}`
- Action feature dim: `{action_feature_dim}`
- State/next/goal context budget: 1024 tokens in this run
- Action context budget: 512 tokens
- Negatives: deterministic hard negatives plus in-batch negatives

## Metrics

{table}

## Usage

This artifact is easiest to use from the `structured-cot` repository:

```bash
uv run --with torch --with transformers --with peft python scripts/cache_transition_embeddings.py \\
  --data-dir data/mqe/agenttrove_50k_features_v2 \\
  --encoder-dir <downloaded {repo_id} folder> \\
  --out data/mqe/cache/agenttrove_50k_qwen06b_transition.pt \\
  --dtype bfloat16
```

The cache contains `z_state`, `z_action`, `z_next`, `z_goal`,
`z_state_action`, and `action_features` tensors for MQE critic training.

## Limitations

This is a research artifact. It was trained on offline agent rollouts and
optimizes transition retrieval, not task success directly. Use it as a
candidate-action reranking feature or as input to an MQE critic, then validate
against real task outcomes.
"""


def critic_readme(src: Path, repo_id: str, transition_repo: str) -> str:
    metrics_blob = read_json(src / "metrics.json")
    metrics = final_metrics(metrics_blob)
    config = metrics_blob.get("config") if isinstance(metrics_blob.get("config"), dict) else {}
    train_rows = metrics_blob.get("train_rows", "unknown")
    val_rows = metrics_blob.get("val_rows", "unknown")
    input_dim = metrics_blob.get("input_dim", "unknown")
    precomputed = metrics_blob.get("precomputed_state_action", "unknown")

    table = metric_table(
        metrics,
        [
            "val_action_choice_acc",
            "val_action_choice_margin",
            "val_spearman_action",
            "val_spearman_state",
            "val_mae_steps",
            "val_action_mae_steps",
            "train_action_contrastive_acc",
            "train_loss",
        ],
    )
    return f"""---
library_name: pytorch
tags:
- code
- agents
- critic
- reward-model
- mqe
- reranker
---

# Code MQE Critic

This repository contains a lightweight directed quasimetric critic trained to
estimate code-agent progress toward a goal. It is designed for action reranking
and self-distillation experiments.

The critic scores:

```text
d(E_state(s), E_state(goal))
d(T(s, a), E_state(goal))
improvement = d(state, goal) - d(action, goal)
```

Lower action distance, or higher predicted improvement, indicates a candidate
action that is expected to move the agent closer to the goal.

## Contents

- `best.pt`: PyTorch checkpoint for `DirectedQuasimetricCritic`
- `metrics.json`: training config, metrics history, and summary

This critic expects transition-cache tensors produced by:

- `{transition_repo}`

## Training Data

- Source: `open-thoughts/AgentTrove`
- Train rows: `{train_rows}`
- Validation rows: `{val_rows}`
- Input dim: `{input_dim}`
- Uses precomputed state-action transition embeddings: `{precomputed}`
- Distance scale mode: `{config.get("distance_scale_mode", "unknown")}`

## Metrics

{table}

The most important metric for reranking is `val_action_choice_acc`, which tests
whether the critic ranks the true next action above sampled negatives for the
same state/goal.

## Usage

From the `structured-cot` repository:

```python
from scripts.train_mqe_critic import load_checkpoint_model

model, config, checkpoint = load_checkpoint_model("best.pt", device="cuda")
model.eval()
```

For training/evaluation with cached transition embeddings:

```bash
uv run --with torch python scripts/train_mqe_critic.py \\
  --data-dir data/mqe/agenttrove_50k_features_v2 \\
  --encoder-backend transition-cache \\
  --cache-path data/mqe/cache/agenttrove_50k_qwen06b_transition.pt \\
  --output-dir outputs/mqe/repro \\
  --epochs 10
```

## Limitations

This critic is not a verifier and should not be treated as a task-success
oracle. It is a dense progress heuristic learned from offline rollouts. The
recommended use is inference-time candidate reranking followed by execution in a
real environment, then self-distillation from successful traces.
"""


def prepare_transition(src: Path, dst: Path, repo_id: str) -> None:
    if not (src / "transition_config.json").exists():
        raise FileNotFoundError(f"transition_config.json not found in {src}")
    if not (src / "transition_head.pt").exists():
        raise FileNotFoundError(f"transition_head.pt not found in {src}")
    copy_transition_artifacts(src, dst)
    (dst / "README.md").write_text(transition_readme(src, repo_id) + "\n")


def prepare_critic(src: Path, dst: Path, repo_id: str, transition_repo: str) -> None:
    if not (src / "best.pt").exists():
        raise FileNotFoundError(f"best.pt not found in {src}")
    copy_critic_artifacts(src, dst)
    (dst / "README.md").write_text(critic_readme(src, repo_id, transition_repo) + "\n")


def upload_folder(repo_id: str, folder: Path, *, private: bool, token_env: str, commit_message: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Run with:\n"
            "  uv run --with huggingface-hub python scripts/upload_hf_artifacts.py ..."
        ) from exc

    token = os.environ.get(token_env)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder),
        commit_message=commit_message,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package/upload HF artifacts for transition encoder and MQE critic.")
    parser.add_argument("--transition-dir", type=Path, default=DEFAULT_TRANSITION_DIR)
    parser.add_argument("--critic-dir", type=Path, default=DEFAULT_CRITIC_DIR)
    parser.add_argument("--transition-repo", default=DEFAULT_TRANSITION_REPO)
    parser.add_argument("--critic-repo", default=DEFAULT_CRITIC_REPO)
    parser.add_argument("--only", choices=["both", "transition", "critic"], default="both")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--staging-dir", type=Path, default=None)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--commit-message", default="Upload structured-cot research artifact")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    temp_ctx = None
    if args.staging_dir is not None:
        root = args.staging_dir
    elif args.dry_run:
        root = Path(tempfile.mkdtemp(prefix="structured_cot_hf_artifacts_"))
    else:
        temp_ctx = tempfile.TemporaryDirectory()
        root = Path(temp_ctx.name)
    assert root is not None
    root.mkdir(parents=True, exist_ok=True)

    staged: list[tuple[str, str, Path]] = []
    try:
        if args.only in {"both", "transition"}:
            dst = root / "code-state-embedding"
            prepare_transition(args.transition_dir, dst, args.transition_repo)
            staged.append(("transition", args.transition_repo, dst))
        if args.only in {"both", "critic"}:
            dst = root / "code-mqe-critic"
            prepare_critic(args.critic_dir, dst, args.critic_repo, args.transition_repo)
            staged.append(("critic", args.critic_repo, dst))

        print(
            json.dumps(
                {
                    "dry_run": args.dry_run,
                    "staging_root": str(root),
                    "artifacts": [
                        {
                            "kind": kind,
                            "repo": repo,
                            "folder": str(folder),
                            "files": sorted(str(path.relative_to(folder)) for path in folder.rglob("*") if path.is_file()),
                        }
                        for kind, repo, folder in staged
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        if not args.dry_run:
            for kind, repo, folder in staged:
                print(f"Uploading {kind} artifact to {repo} ...", flush=True)
                upload_folder(
                    repo,
                    folder,
                    private=args.private,
                    token_env=args.hf_token_env,
                    commit_message=args.commit_message,
                )
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
