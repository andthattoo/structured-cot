#!/usr/bin/env python3
"""Visualize a small transition-encoder embedding sample with t-SNE."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from train_transition_encoder import (
    ACTION_TYPES,
    build_transition_head_class,
    encode_texts,
    resolve_device,
)


DEFAULT_ENCODER = "driaforall/code-state-embedding"
DEFAULT_OUT = Path("outputs/figures/code_state_embedding_tsne.png")


@dataclass(frozen=True)
class VizExample:
    example_id: str
    group: str
    domain: str
    state_text: str
    action_text: str


def builtin_examples() -> list[VizExample]:
    return [
        VizExample(
            "parser_all_tests",
            "same_task_parser",
            "python_tests",
            """Task: Fix the CSV parser.

Current state:
- parser.py was edited recently.
- tests/test_parser.py fails around quoted commas.
- The likely bug is token splitting inside quoted fields.
""",
            "python -m pytest tests/test_parser.py -q",
        ),
        VizExample(
            "parser_one_case",
            "same_task_parser",
            "python_tests",
            """Task: Fix the CSV parser.

Current state:
- parser.py was edited recently.
- tests/test_parser.py fails around quoted commas.
- The likely bug is token splitting inside quoted fields.
""",
            "python -m pytest tests/test_parser.py::test_quoted_commas -q",
        ),
        VizExample(
            "parser_maxfail_verbose",
            "same_task_parser",
            "python_tests",
            """Task: Fix the CSV parser.

Current state:
- parser.py was edited recently.
- tests/test_parser.py fails around quoted commas.
- The likely bug is token splitting inside quoted fields.
""",
            "python -m pytest tests/test_parser.py --maxfail=1 -vv",
        ),
        VizExample(
            "api_put_a1",
            "same_domain_api",
            "rest_api",
            """Task: Populate a spreadsheet through a local REST API.

Current state:
- API docs are available at http://api:8000/docs/json.
- Spreadsheet id 1 and sheet id 1 already exist.
- Need to write financial report cells.
""",
            """curl -s -X PUT http://api:8000/sheets/1/cells/A1 -H 'Content-Type: application/json' -d '{"value":"Month"}'""",
        ),
        VizExample(
            "api_put_b2",
            "same_domain_api",
            "rest_api",
            """Task: Populate a spreadsheet through a local REST API.

Current state:
- API docs are available at http://api:8000/docs/json.
- Spreadsheet id 1 and sheet id 1 already exist.
- Need to write financial report cells.
""",
            """curl -s -X PUT http://api:8000/sheets/1/cells/B2 -H 'Content-Type: application/json' -d '{"value":10000}'""",
        ),
        VizExample(
            "api_get_schema",
            "same_domain_api",
            "rest_api",
            """Task: Populate a spreadsheet through a local REST API.

Current state:
- API docs are available at http://api:8000/docs/json.
- Spreadsheet id 1 and sheet id 1 already exist.
- Need to inspect request schemas before updating cells.
""",
            "curl -s http://api:8000/docs/json | python3 -m json.tool",
        ),
        VizExample(
            "sqlite_gcov_build",
            "different_domains",
            "c_build",
            """Task: Compile SQLite with gcov instrumentation.

Current state:
- /app/sqlite source tree is present.
- sqlite3 must be placed on PATH.
- Build flags need coverage instrumentation.
""",
            "cd /app/sqlite && CFLAGS='-O0 -g --coverage' LDFLAGS='--coverage' ./configure && make -j2",
        ),
        VizExample(
            "frontend_npm_test",
            "different_domains",
            "frontend",
            """Task: Fix a React component regression.

Current state:
- package.json exists.
- src/SearchBox.tsx was changed.
- The failing behavior is keyboard navigation in a dropdown.
""",
            "npm test -- --runInBand SearchBox",
        ),
        VizExample(
            "sql_migration_check",
            "different_domains",
            "database",
            """Task: Validate a database migration.

Current state:
- migrations/004_add_indexes.sql was added.
- Query plans should use the new customer_id index.
- Need a quick smoke check against a local sqlite database.
""",
            "sqlite3 app.db 'EXPLAIN QUERY PLAN SELECT * FROM orders WHERE customer_id = 42;'",
        ),
        VizExample(
            "docs_markdown_lint",
            "different_domains",
            "docs",
            """Task: Update developer documentation.

Current state:
- README.md and docs/setup.md were edited.
- Broken anchors or malformed markdown would block publishing.
- Need a lightweight docs check.
""",
            "npx markdownlint README.md docs/setup.md",
        ),
    ]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def load_examples(path: Path | None) -> list[VizExample]:
    if path is None:
        return builtin_examples()
    examples: list[VizExample] = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [
                key
                for key in ["example_id", "group", "domain", "state_text", "action_text"]
                if not row.get(key)
            ]
            if missing:
                raise ValueError(f"{path}:{line_no} missing fields: {missing}")
            examples.append(VizExample(**{key: str(row[key]) for key in VizExample.__annotations__}))
    return examples


def resolve_encoder_dir(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Run with:\n"
            "  uv run --with huggingface-hub --with torch --with transformers "
            "--with scikit-learn --with matplotlib python scripts/visualize_transition_embeddings.py ..."
        ) from exc
    return Path(snapshot_download(value))


def torch_dtype(dtype_name: str, device: str):
    import torch

    if dtype_name == "auto":
        return torch.bfloat16 if device.startswith("cuda") else torch.float32
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    raise ValueError("dtype must be auto, float32, bfloat16, or float16")


def load_encoder_bundle(encoder_dir: Path, device: str, dtype_name: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    config = read_json(encoder_dir / "transition_config.json")
    base_model = str(config.get("base_model") or "Qwen/Qwen3-Embedding-0.6B")
    dtype = torch_dtype(dtype_name, device)

    tokenizer = AutoTokenizer.from_pretrained(encoder_dir, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    adapter_dir = encoder_dir / "encoder_adapter"
    if (adapter_dir / "adapter_config.json").exists():
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise SystemExit(
                "This encoder directory contains a PEFT adapter. Run with `--with peft`."
            ) from exc
        encoder = AutoModel.from_pretrained(base_model, torch_dtype=dtype, trust_remote_code=True).to(device)
        encoder = PeftModel.from_pretrained(encoder, adapter_dir).to(device)
    else:
        encoder = AutoModel.from_pretrained(encoder_dir, torch_dtype=dtype, trust_remote_code=True).to(device)
    encoder.eval()

    head_blob = torch.load(encoder_dir / "transition_head.pt", map_location="cpu")
    TransitionHead = build_transition_head_class()
    head = TransitionHead(
        int(head_blob["embedding_dim"]),
        int(head_blob["action_feature_dim"]),
        int(head_blob["config"]["transition_hidden_dim"]),
        len(head_blob.get("action_types") or ACTION_TYPES),
    ).to(device)
    head.load_state_dict(head_blob["transition_head_state_dict"])
    head.eval()
    return encoder, tokenizer, head, head_blob


def embed_examples(examples: list[VizExample], encoder, tokenizer, head, head_blob, args: argparse.Namespace):
    import torch

    state_texts = [example.state_text for example in examples]
    action_texts = [example.action_text for example in examples]
    feature_dim = int(head_blob["action_feature_dim"])
    features = torch.zeros((len(examples), feature_dim), dtype=torch.float32, device=args.device)
    with torch.no_grad():
        z_state = encode_texts(
            encoder,
            tokenizer,
            state_texts,
            max_length=args.max_state_tokens,
            device=args.device,
            truncation_side="left",
        )
        z_action = encode_texts(
            encoder,
            tokenizer,
            action_texts,
            max_length=args.max_action_tokens,
            device=args.device,
            truncation_side="right",
        )
        z_state_action, _ = head(z_state, z_action, features)
    return {
        "state": z_state.cpu(),
        "action": z_action.cpu(),
        "state_action": z_state_action.cpu(),
    }


def selected_points(examples: list[VizExample], embeddings: dict[str, Any], which: str):
    import torch

    if which != "all":
        return embeddings[which], [
            {
                **asdict(example),
                "kind": which,
                "plot_label": example.example_id,
            }
            for example in examples
        ]

    vectors = []
    rows = []
    for kind in ["state", "action", "state_action"]:
        vectors.append(embeddings[kind])
        for example in examples:
            rows.append(
                {
                    **asdict(example),
                    "kind": kind,
                    "plot_label": f"{example.example_id}:{kind}",
                }
            )
    return torch.cat(vectors, dim=0), rows


def compute_tsne(vectors, *, perplexity: float | None, seed: int):
    from sklearn.manifold import TSNE

    n_samples = int(vectors.shape[0])
    if n_samples < 4:
        raise ValueError("t-SNE needs at least 4 points for this visualization")
    if perplexity is None:
        perplexity = min(5.0, max(2.0, (n_samples - 1) / 3.0))
    if perplexity >= n_samples:
        raise ValueError(f"perplexity must be less than n_samples ({n_samples})")
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(vectors.numpy())


def plot_points(rows: list[dict[str, Any]], coords, out: Path, *, title: str) -> None:
    import matplotlib.pyplot as plt

    groups = sorted({row["group"] for row in rows})
    cmap = plt.get_cmap("tab10")
    colors = {group: cmap(index % 10) for index, group in enumerate(groups)}
    markers = {"state": "o", "action": "^", "state_action": "s"}

    fig, ax = plt.subplots(figsize=(12, 8), dpi=160)
    for group in groups:
        indices = [index for index, row in enumerate(rows) if row["group"] == group]
        ax.scatter(
            coords[indices, 0],
            coords[indices, 1],
            s=72,
            color=colors[group],
            label=group,
            alpha=0.86,
            edgecolors="white",
            linewidths=0.8,
            marker="o",
        )
    for index, row in enumerate(rows):
        marker = markers.get(row["kind"], "o")
        ax.scatter(
            [coords[index, 0]],
            [coords[index, 1]],
            s=84,
            facecolors="none",
            edgecolors="black",
            linewidths=0.8,
            marker=marker,
        )
        ax.annotate(
            row["plot_label"],
            (coords[index, 0], coords[index, 1]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize transition encoder embeddings with t-SNE.")
    parser.add_argument("--encoder-dir", default=DEFAULT_ENCODER, help="Local encoder dir or HF repo id.")
    parser.add_argument("--examples-jsonl", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--coords-out", type=Path, default=None)
    parser.add_argument("--which", choices=["state", "action", "state_action", "all"], default="state_action")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-state-tokens", type=int, default=1024)
    parser.add_argument("--max-action-tokens", type=int, default=512)
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--perplexity", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.device = resolve_device(args.device)
    examples = load_examples(args.examples_jsonl)
    if args.limit > 0:
        examples = examples[: args.limit]
    if len(examples) < 4:
        raise ValueError("Need at least 4 examples to build a t-SNE plot")

    encoder_dir = resolve_encoder_dir(args.encoder_dir)
    encoder, tokenizer, head, head_blob = load_encoder_bundle(encoder_dir, args.device, args.dtype)
    embeddings = embed_examples(examples, encoder, tokenizer, head, head_blob, args)
    vectors, rows = selected_points(examples, embeddings, args.which)
    coords = compute_tsne(vectors, perplexity=args.perplexity, seed=args.seed)
    title = f"Code transition encoder t-SNE ({args.which}, n={len(rows)})"
    plot_points(rows, coords, args.out, title=title)

    coords_out = args.coords_out or args.out.with_suffix(".json")
    coords_out.parent.mkdir(parents=True, exist_ok=True)
    coords_rows = [
        {
            **row,
            "x": float(coords[index, 0]),
            "y": float(coords[index, 1]),
        }
        for index, row in enumerate(rows)
    ]
    coords_out.write_text(json.dumps(coords_rows, indent=2) + "\n")
    print(
        json.dumps(
            {
                "encoder_dir": str(encoder_dir),
                "examples": len(examples),
                "points": len(rows),
                "which": args.which,
                "out": str(args.out),
                "coords_out": str(coords_out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
