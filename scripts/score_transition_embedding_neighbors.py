#!/usr/bin/env python3
"""Score transition-encoder nearest neighbors for a tiny qualitative set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from train_transition_encoder import encode_texts, resolve_device
from visualize_transition_embeddings import (
    DEFAULT_ENCODER,
    builtin_examples,
    embed_examples,
    load_encoder_bundle,
    load_examples,
    resolve_encoder_dir,
)


DEFAULT_OUT = Path("outputs/figures/code_state_embedding_neighbors.json")


BUILTIN_NEXT_STATE_TEXTS = {
    "parser_all_tests": """Task: Fix the CSV parser.

Terminal output after running the full parser test file:
- Most parser tests passed.
- test_quoted_commas still fails.
- AssertionError: expected ['a,b', 'c'], got ['a', 'b', 'c'].
""",
    "parser_one_case": """Task: Fix the CSV parser.

Terminal output after running the focused quoted-comma test:
- tests/test_parser.py::test_quoted_commas failed.
- The parser is still splitting commas inside quoted fields.
- Only the focused failing case was executed.
""",
    "parser_maxfail_verbose": """Task: Fix the CSV parser.

Verbose pytest output:
- Collection succeeded.
- test_quoted_commas failed with a detailed assertion diff.
- Pytest stopped after the first failure because maxfail=1 was set.
""",
    "api_put_a1": """Task: Populate a spreadsheet through a local REST API.

API response:
- Cell A1 was updated.
- The value is now "Month".
- The sheet timestamp changed.
""",
    "api_put_b2": """Task: Populate a spreadsheet through a local REST API.

API response:
- Cell B2 was updated.
- The value is now 10000.
- The sheet timestamp changed.
""",
    "api_get_schema": """Task: Populate a spreadsheet through a local REST API.

Terminal output:
- OpenAPI JSON was printed.
- Paths and schemas for spreadsheets, sheets, and cells are visible.
- No spreadsheet data was modified.
""",
    "sqlite_gcov_build": """Task: Compile SQLite with gcov instrumentation.

Build output:
- configure completed in /app/sqlite.
- make compiled sqlite3 with coverage flags.
- sqlite3 can now be copied or symlinked into PATH.
""",
    "frontend_npm_test": """Task: Fix a React component regression.

Test output:
- Jest ran the SearchBox test suite.
- Keyboard navigation assertions are shown.
- The result identifies the component behavior that still needs attention.
""",
    "sql_migration_check": """Task: Validate a database migration.

SQLite output:
- EXPLAIN QUERY PLAN was printed.
- The query plan shows whether the customer_id index is used.
- No database rows were modified.
""",
    "docs_markdown_lint": """Task: Update developer documentation.

Markdown lint output:
- README.md and docs/setup.md were checked.
- Any malformed headings, bad lists, or broken markdown style issues are listed.
- Documentation content was not modified.
""",
}


def next_state_text(example: Any) -> str:
    value = getattr(example, "next_state_text", None)
    if value:
        return str(value)
    builtin = BUILTIN_NEXT_STATE_TEXTS.get(example.example_id)
    if builtin:
        return builtin
    return "\n".join(
        [
            example.state_text,
            "",
            "After action:",
            example.action_text,
        ]
    )


def topk_rows(
    *,
    query_index: int,
    scores,
    examples: list[Any],
    k: int,
    exclude_self: bool,
) -> list[dict[str, Any]]:
    import torch

    row = scores[query_index].clone()
    if exclude_self:
        row[query_index] = -float("inf")
    k = min(k, len(examples) - (1 if exclude_self else 0))
    values, indices = torch.topk(row, k=max(0, k))
    return [
        {
            "rank": rank + 1,
            "example_id": examples[int(index)].example_id,
            "group": examples[int(index)].group,
            "domain": examples[int(index)].domain,
            "score": float(value),
            "same_group": examples[int(index)].group == examples[query_index].group,
            "is_self": int(index) == query_index,
        }
        for rank, (value, index) in enumerate(zip(values, indices))
    ]


def rank_of_self(scores, index: int) -> int:
    import torch

    order = torch.argsort(scores[index], descending=True)
    matches = (order == index).nonzero(as_tuple=False)
    return int(matches[0].item()) + 1


def score_examples(examples: list[Any], encoder, tokenizer, head, head_blob, args: argparse.Namespace) -> dict[str, Any]:
    import torch

    embeddings = embed_examples(examples, encoder, tokenizer, head, head_blob, args)
    next_texts = [next_state_text(example) for example in examples]
    with torch.no_grad():
        z_next = encode_texts(
            encoder,
            tokenizer,
            next_texts,
            max_length=args.max_state_tokens,
            device=args.device,
            truncation_side="left",
        ).cpu()

    transition_to_next = embeddings["state_action"] @ z_next.T
    transition_to_transition = embeddings["state_action"] @ embeddings["state_action"].T
    state_to_state = embeddings["state"] @ embeddings["state"].T
    action_to_action = embeddings["action"] @ embeddings["action"].T

    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        rows.append(
            {
                "example_id": example.example_id,
                "group": example.group,
                "domain": example.domain,
                "self_next_rank": rank_of_self(transition_to_next, index),
                "self_next_score": float(transition_to_next[index, index]),
                "same_group_next_topk": sum(
                    1 for row in topk_rows(query_index=index, scores=transition_to_next, examples=examples, k=args.top_k, exclude_self=False)
                    if row["same_group"]
                ),
                "nearest_next": topk_rows(
                    query_index=index,
                    scores=transition_to_next,
                    examples=examples,
                    k=args.top_k,
                    exclude_self=False,
                ),
                "nearest_state_action": topk_rows(
                    query_index=index,
                    scores=transition_to_transition,
                    examples=examples,
                    k=args.top_k,
                    exclude_self=True,
                ),
                "nearest_state": topk_rows(
                    query_index=index,
                    scores=state_to_state,
                    examples=examples,
                    k=args.top_k,
                    exclude_self=True,
                ),
                "nearest_action": topk_rows(
                    query_index=index,
                    scores=action_to_action,
                    examples=examples,
                    k=args.top_k,
                    exclude_self=True,
                ),
            }
        )

    exact_at_1 = sum(1 for row in rows if row["self_next_rank"] == 1) / max(1, len(rows))
    exact_at_3 = sum(1 for row in rows if row["self_next_rank"] <= 3) / max(1, len(rows))
    mean_self_score = sum(row["self_next_score"] for row in rows) / max(1, len(rows))
    return {
        "summary": {
            "examples": len(examples),
            "top_k": args.top_k,
            "transition_to_next_exact_at_1": exact_at_1,
            "transition_to_next_exact_at_3": exact_at_3,
            "transition_to_next_mean_self_score": mean_self_score,
        },
        "rows": rows,
    }


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Transition Embedding Neighbor Report",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in payload["summary"].items():
        if isinstance(value, float):
            rendered = f"{value:.4f}"
        else:
            rendered = str(value)
        lines.append(f"| `{key}` | {rendered} |")
    lines.extend(["", "## Per-Example Neighbors", ""])
    for row in payload["rows"]:
        lines.extend(
            [
                f"### `{row['example_id']}`",
                "",
                f"- group: `{row['group']}`",
                f"- domain: `{row['domain']}`",
                f"- true next-state rank from `state_action`: `{row['self_next_rank']}`",
                f"- true next-state cosine: `{row['self_next_score']:.4f}`",
                "",
                "| Rank | Nearest next-state | Group | Score |",
                "|---:|---|---|---:|",
            ]
        )
        for item in row["nearest_next"]:
            marker = " <- true" if item["is_self"] else ""
            lines.append(
                f"| {item['rank']} | `{item['example_id']}`{marker} | `{item['group']}` | {item['score']:.4f} |"
            )
        lines.extend(["", "| Rank | Nearest state_action | Group | Score |", "|---:|---|---|---:|"])
        for item in row["nearest_state_action"]:
            lines.append(f"| {item['rank']} | `{item['example_id']}` | `{item['group']}` | {item['score']:.4f} |")
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score nearest neighbors for transition encoder embeddings.")
    parser.add_argument("--encoder-dir", default=DEFAULT_ENCODER, help="Local encoder dir or HF repo id.")
    parser.add_argument("--examples-jsonl", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-state-tokens", type=int, default=1024)
    parser.add_argument("--max-action-tokens", type=int, default=512)
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.device = resolve_device(args.device)
    examples = load_examples(args.examples_jsonl)
    if args.limit > 0:
        examples = examples[: args.limit]
    if len(examples) < 2:
        raise ValueError("Need at least 2 examples for neighbor scoring")

    encoder_dir = resolve_encoder_dir(args.encoder_dir)
    encoder, tokenizer, head, head_blob = load_encoder_bundle(encoder_dir, args.device, args.dtype)
    payload = score_examples(examples, encoder, tokenizer, head, head_blob, args)
    payload["summary"]["encoder_dir"] = str(encoder_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    md_out = args.md_out or args.out.with_suffix(".md")
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(markdown_report(payload) + "\n")

    print(
        json.dumps(
            {
                **payload["summary"],
                "out": str(args.out),
                "md_out": str(md_out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
