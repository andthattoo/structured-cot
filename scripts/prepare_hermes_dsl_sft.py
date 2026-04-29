#!/usr/bin/env python3
"""Prepare Hermes agent traces for compact-DSL SFT.

The source dataset contains verbose <think> blocks before tool calls. This
script rewrites those blocks into a small existing-token DSL while preserving
the original tool calls and tool responses:

    <think>
    PLAN: seq(observe,act,verify,finish)
    STATE: need_action
    RISK: bad_tool_args
    NEXT: tool_call
    </think>
    <tool_call>...</tool_call>

The goal is not to treat the teacher's prose as ground truth. The teacher trace
provides the action/observation trajectory; the DSL is a compact supervised
state label for the next assistant action.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET = "DJLougen/hermes-agent-traces-filtered"

DSL_SYSTEM_PROMPT = (
    "When thinking before tool use, use this compact DSL inside <think> tags:\n"
    "PLAN: one symbolic control-flow plan\n"
    "STATE: current state\n"
    "RISK: main risk to avoid\n"
    "NEXT: tool_call or final\n"
    "Use the DSL as a decision record for the next action, then emit the "
    "tool call or final answer normally."
)

THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE
)

ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "tool": "tool",
    "function": "tool",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite Hermes agent reasoning traces into compact DSL SFT data."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Hugging Face dataset id to load. Default: {DEFAULT_DATASET}",
    )
    source.add_argument(
        "--input",
        type=Path,
        help="Local JSON/JSONL file with Hermes-style rows.",
    )
    parser.add_argument("--split", default="train", help="Dataset split to load.")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSONL path. The repo ignores *.jsonl by default.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit.")
    parser.add_argument(
        "--min-rewrites",
        type=int,
        default=1,
        help="Skip rows with fewer rewritten assistant think blocks.",
    )
    parser.add_argument(
        "--no-system-dsl-prompt",
        action="store_true",
        help="Do not append the DSL instruction to the first system message.",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Do not include a rendered ChatML-ish text field.",
    )
    return parser.parse_args()


def iter_local_rows(path: Path) -> Iterable[dict[str, Any]]:
    text = path.read_text()
    stripped = text.lstrip()
    if stripped.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path} JSON root must be a list or JSONL rows")
        yield from data
        return

    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: row must be an object")
        yield row


def iter_hf_rows(dataset_id: str, split: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install with `uv sync` or "
            "`uv pip install datasets>=3,<4`."
        ) from exc

    yield from load_dataset(dataset_id, split=split)


def normalize_conversations(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("conversations must be a list or JSON string list")

    conversations: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        speaker = str(item.get("from") or item.get("role") or "").strip()
        content = item.get("value", item.get("content", ""))
        conversations.append({"from": speaker, "value": str(content)})
    return conversations


def compact_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def has_any(text: str, words: Iterable[str]) -> bool:
    return any(word in text for word in words)


def has_tool_call(assistant_text: str) -> bool:
    return TOOL_CALL_RE.search(assistant_text) is not None


def infer_dsl(think: str, assistant_text: str) -> dict[str, str]:
    text = compact_ws(think)
    next_action = "tool_call" if has_tool_call(assistant_text) else "final"

    if next_action == "final":
        return {
            "PLAN": "seq(observe,verify,finish)",
            "STATE": "ready",
            "RISK": "none",
            "NEXT": "final",
        }

    repair_markers = [
        "error",
        "failed",
        "failure",
        "exception",
        "bug",
        "fix",
        "debug",
        "retry",
        "wrong",
    ]
    verify_markers = [
        "verify",
        "validate",
        "confirm",
        "check",
        "test",
        "ensure",
    ]
    context_markers = [
        "inspect",
        "look",
        "read",
        "search",
        "find",
        "explore",
        "list",
        "schema",
        "docs",
    ]
    arg_markers = [
        "argument",
        "parameter",
        "json",
        "schema",
        "format",
        "quote",
        "malformed",
    ]

    if has_any(text, repair_markers):
        plan = "seq(observe,act,verify,repair,finish)"
        state = "need_fix"
    elif has_any(text, verify_markers):
        plan = "seq(observe,act,verify,finish)"
        state = "need_verify"
    elif has_any(text, context_markers):
        plan = "seq(observe,act,verify,finish)"
        state = "need_context"
    else:
        plan = "seq(observe,act,verify,finish)"
        state = "need_action"

    if has_any(text, arg_markers):
        risk = "bad_tool_args"
    elif has_any(text, ["ambiguous", "unclear", "missing", "not enough", "unknown"]):
        risk = "missing_context"
    elif has_any(text, ["repeat", "again", "same command", "loop"]):
        risk = "repeat_loop"
    elif has_any(text, repair_markers):
        risk = "tool_failure"
    elif "finish" in text or "complete" in text or "done" in text:
        risk = "premature_final"
    else:
        risk = "none"

    return {"PLAN": plan, "STATE": state, "RISK": risk, "NEXT": next_action}


def render_dsl(dsl: dict[str, str]) -> str:
    return (
        f"PLAN: {dsl['PLAN']}\n"
        f"STATE: {dsl['STATE']}\n"
        f"RISK: {dsl['RISK']}\n"
        f"NEXT: {dsl['NEXT']}"
    )


def rewrite_assistant_value(value: str) -> tuple[str, int, list[dict[str, str]]]:
    rewrites: list[dict[str, str]] = []

    def replace(match: re.Match[str]) -> str:
        think = match.group(1)
        dsl = infer_dsl(think, value)
        rewrites.append(dsl)
        return "<think>\n" + render_dsl(dsl) + "\n</think>"

    rewritten, count = THINK_RE.subn(replace, value, count=1)
    return rewritten, count, rewrites


def append_dsl_system_prompt(conversations: list[dict[str, str]]) -> None:
    for item in conversations:
        if ROLE_MAP.get(item["from"], item["from"]) == "system":
            item["value"] = item["value"].rstrip() + "\n\n" + DSL_SYSTEM_PROMPT
            return
    conversations.insert(0, {"from": "system", "value": DSL_SYSTEM_PROMPT})


def to_messages(conversations: list[dict[str, str]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in conversations:
        role = ROLE_MAP.get(item["from"], item["from"])
        messages.append({"role": role, "content": item["value"]})
    return messages


def render_chatml(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = message["role"]
        content = message["content"].rstrip()
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts) + "\n"


def convert_row(
    row: dict[str, Any],
    *,
    add_system_prompt: bool,
    include_text: bool,
) -> dict[str, Any]:
    conversations = normalize_conversations(row.get("conversations", []))
    rewritten_conversations: list[dict[str, str]] = []
    rewritten_blocks = 0
    tool_turns = 0
    labels: list[dict[str, str]] = []

    for item in conversations:
        new_item = dict(item)
        role = ROLE_MAP.get(new_item["from"], new_item["from"])
        if role == "assistant":
            new_value, count, row_labels = rewrite_assistant_value(new_item["value"])
            new_item["value"] = new_value
            rewritten_blocks += count
            labels.extend(row_labels)
            if has_tool_call(new_value):
                tool_turns += 1
        rewritten_conversations.append(new_item)

    if add_system_prompt:
        append_dsl_system_prompt(rewritten_conversations)

    messages = to_messages(rewritten_conversations)
    out = {
        "id": row.get("id"),
        "category": row.get("category"),
        "subcategory": row.get("subcategory"),
        "task": row.get("task"),
        "tools": row.get("tools"),
        "conversations": rewritten_conversations,
        "messages": messages,
        "dsl_stats": {
            "rewritten_think_blocks": rewritten_blocks,
            "assistant_tool_turns": tool_turns,
            "labels": labels,
        },
    }
    if include_text:
        out["text"] = render_chatml(messages)
    return out


def main() -> None:
    args = parse_args()
    rows: Iterable[dict[str, Any]]
    if args.input:
        rows = iter_local_rows(args.input)
    else:
        rows = iter_hf_rows(args.dataset, args.split)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    seen = 0
    written = 0
    rewritten_total = 0
    with args.out.open("w") as f:
        for row in rows:
            if args.limit is not None and seen >= args.limit:
                break
            seen += 1
            converted = convert_row(
                row,
                add_system_prompt=not args.no_system_dsl_prompt,
                include_text=not args.no_text,
            )
            rewrites = int(converted["dsl_stats"]["rewritten_think_blocks"])
            if rewrites < args.min_rewrites:
                continue
            rewritten_total += rewrites
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")
            written += 1

    print(
        json.dumps(
            {
                "input_rows_seen": seen,
                "output_rows_written": written,
                "rewritten_think_blocks": rewritten_total,
                "out": str(args.out),
            },
            indent=2,
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
