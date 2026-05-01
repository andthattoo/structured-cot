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


def torch_dtype_from_name(dtype_name: str):
    import torch

    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError("merge dtype must be one of float32, bfloat16, float16")


def merge_transition_encoder(src: Path, dst: Path, *, dtype_name: str, device: str, token: str | None = None) -> None:
    try:
        from peft import PeftModel
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing merge dependencies. Run with:\n"
            "  uv run --with huggingface-hub --with torch --with transformers --with peft "
            "python scripts/upload_hf_artifacts.py ..."
        ) from exc

    config = read_json(src / "transition_config.json")
    base_model = config.get("base_model")
    if not base_model:
        raise ValueError(f"Missing base_model in {src / 'transition_config.json'}")
    adapter_dir = src / "encoder_adapter"
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing transition encoder adapter: {adapter_dir}")

    dtype = torch_dtype_from_name(dtype_name)
    print(
        json.dumps(
            {
                "phase": "merge_transition_encoder",
                "base_model": base_model,
                "adapter_dir": str(adapter_dir),
                "dtype": dtype_name,
                "device": device,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True, token=token)
    base = AutoModel.from_pretrained(base_model, torch_dtype=dtype, trust_remote_code=True, token=token).to(device)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(dst, safe_serialization=True)
    tokenizer.save_pretrained(dst)


def transition_readme(src: Path, repo_id: str, *, merged_encoder: bool) -> str:
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
    library_name = "transformers" if merged_encoder else "peft"
    contents = (
        f"- merged `{base_model}` encoder weights with the transition LoRA applied\n"
        if merged_encoder
        else f"- `encoder_adapter/`: LoRA adapter for `{base_model}`\n"
    )
    loader_note = (
        "The encoder weights in this repository are already merged with the LoRA adapter."
        if merged_encoder
        else "Load the base model and apply `encoder_adapter/` with PEFT before encoding."
    )
    encoder_setup = (
        "encoder = AutoModel.from_pretrained(repo_dir, trust_remote_code=True).to(device).eval()"
        if merged_encoder
        else "\n".join(
            [
                "from peft import PeftModel",
                f'base = AutoModel.from_pretrained("{base_model}", trust_remote_code=True).to(device)',
                'encoder = PeftModel.from_pretrained(base, Path(repo_dir) / "encoder_adapter").to(device).eval()',
            ]
        )
    )

    return f"""---
library_name: {library_name}
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

{contents.rstrip()}
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

{loader_note}

Generic single-transition example:

```python
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer


class TransitionHead(nn.Module):
    def __init__(self, embedding_dim, action_feature_dim, hidden_dim, n_action_types):
        super().__init__()
        self.action_feature_dim = int(action_feature_dim)
        self.transition = nn.Sequential(
            nn.Linear(embedding_dim * 2 + action_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.action_type_head = nn.Linear(embedding_dim, n_action_types)

    def forward(self, state_z, action_z, action_features):
        if self.action_feature_dim:
            action_features = action_features.to(state_z.dtype)
            inputs = torch.cat([state_z, action_z, action_features], dim=-1)
        else:
            inputs = torch.cat([state_z, action_z], dim=-1)
        pred_next_z = F.normalize(self.transition(inputs), p=2, dim=-1)
        return pred_next_z, self.action_type_head(pred_next_z)


def encode_text(encoder, tokenizer, text, *, max_length, truncation_side, device):
    old_side = tokenizer.truncation_side
    tokenizer.truncation_side = truncation_side
    try:
        batch = tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    finally:
        tokenizer.truncation_side = old_side
    batch = {{key: value.to(device) for key, value in batch.items()}}
    with torch.no_grad():
        output = encoder(**batch)
    positions = torch.arange(batch["attention_mask"].shape[1], device=device).unsqueeze(0)
    last_index = (positions * batch["attention_mask"]).argmax(dim=1)
    pooled = output.last_hidden_state[torch.arange(1, device=device), last_index]
    return F.normalize(pooled.float(), p=2, dim=-1)


device = "cuda" if torch.cuda.is_available() else "cpu"
repo_dir = snapshot_download("{repo_id}")

tokenizer = AutoTokenizer.from_pretrained(repo_dir, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
{encoder_setup}

head_blob = torch.load(Path(repo_dir) / "transition_head.pt", map_location="cpu")
head = TransitionHead(
    int(head_blob["embedding_dim"]),
    int(head_blob["action_feature_dim"]),
    int(head_blob["config"]["transition_hidden_dim"]),
    len(head_blob["action_types"]),
).to(device)
head.load_state_dict(head_blob["transition_head_state_dict"])
head.eval()

state_text = "Current workspace state, recent logs, and relevant files."
action_text = "Run the unit tests for the changed module."
action_features = torch.zeros((1, int(head_blob["action_feature_dim"])), device=device)

z_state = encode_text(encoder, tokenizer, state_text, max_length=1024, truncation_side="left", device=device)
z_action = encode_text(encoder, tokenizer, action_text, max_length=512, truncation_side="right", device=device)
with torch.no_grad():
    z_pred_next, action_type_logits = head(z_state, z_action, action_features)

print(z_pred_next.shape)
```

The zero `action_features` vector is acceptable for a minimal example. For
ranking actions in a real harness, pass the same structured action features used
during cache generation.

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


def prepare_transition(
    src: Path,
    dst: Path,
    repo_id: str,
    *,
    merge_encoder: bool,
    merge_dtype: str,
    merge_device: str,
    token: str | None = None,
) -> None:
    if not (src / "transition_config.json").exists():
        raise FileNotFoundError(f"transition_config.json not found in {src}")
    if not (src / "transition_head.pt").exists():
        raise FileNotFoundError(f"transition_head.pt not found in {src}")
    if merge_encoder:
        merge_transition_encoder(src, dst, dtype_name=merge_dtype, device=merge_device, token=token)
        for name in ["transition_config.json", "transition_head.pt", "metrics.json"]:
            copy_if_exists(src / name, dst / name)
    else:
        copy_transition_artifacts(src, dst)
    (dst / "README.md").write_text(
        transition_readme(src, repo_id, merged_encoder=merge_encoder) + "\n"
    )


def prepare_critic(src: Path, dst: Path, repo_id: str, transition_repo: str) -> None:
    if not (src / "best.pt").exists():
        raise FileNotFoundError(f"best.pt not found in {src}")
    copy_critic_artifacts(src, dst)
    (dst / "README.md").write_text(critic_readme(src, repo_id, transition_repo) + "\n")


def resolve_hf_token(token_env: str, *, required: bool) -> tuple[str | None, str | None]:
    try:
        from huggingface_hub import get_token
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Run with:\n"
            "  uv run --with huggingface-hub python scripts/upload_hf_artifacts.py ..."
        ) from exc

    token = os.environ.get(token_env)
    if token:
        return token, token_env
    token = get_token()
    if token:
        return token, "huggingface-cli cache"
    if required:
        raise SystemExit(
            f"No Hugging Face token found. Set {token_env} or run `hf auth login` "
            "with a token that has write access to the target repos."
        )
    return None, None


def validate_hf_token(token: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        whoami = api.whoami()
    except Exception as exc:
        raise SystemExit(
            "A Hugging Face token was found, but the Hub rejected it. "
            "Run `hf auth whoami` and confirm the token has write access to the target org."
        ) from exc
    print(
        json.dumps(
            {
                "hf_user": whoami.get("name") or whoami.get("fullname") or "unknown",
                "phase": "hf_auth",
            },
            sort_keys=True,
        ),
        flush=True,
    )


def upload_folder(repo_id: str, folder: Path, *, private: bool, token: str, commit_message: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder),
        commit_message=commit_message,
        token=token,
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
    parser.add_argument(
        "--merge-transition-encoder",
        action="store_true",
        help="Merge the transition LoRA into the Qwen embedding base model before upload.",
    )
    parser.add_argument(
        "--merge-dtype",
        choices=["float32", "bfloat16", "float16"],
        default="bfloat16",
        help="Dtype used when saving a merged transition encoder.",
    )
    parser.add_argument(
        "--merge-device",
        default="cpu",
        help="Device used for merging the transition encoder, e.g. cpu or cuda.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hf_token, token_source = resolve_hf_token(args.hf_token_env, required=not args.dry_run)
    if token_source:
        print(json.dumps({"hf_token_source": token_source, "phase": "hf_auth"}, sort_keys=True), flush=True)
    if hf_token and not args.dry_run:
        validate_hf_token(hf_token)

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
            prepare_transition(
                args.transition_dir,
                dst,
                args.transition_repo,
                merge_encoder=args.merge_transition_encoder,
                merge_dtype=args.merge_dtype,
                merge_device=args.merge_device,
                token=hf_token,
            )
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
                    token=hf_token,
                    commit_message=args.commit_message,
                )
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
