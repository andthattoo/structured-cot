#!/usr/bin/env python3
"""Summarize Pi session JSONL traces.

This is intentionally lightweight: it reads Pi's native session export and
prints enough structure to compare ETPI data-generation runs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def content_text_length(content: list[dict[str, Any]], kind: str, key: str) -> int:
    return sum(len(str(part.get(key) or "")) for part in content if part.get("type") == kind)


def assistant_row(row: dict[str, Any]) -> dict[str, Any] | None:
    message = row.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return None
    content = message.get("content") or []
    if not isinstance(content, list):
        content = []
    content_types = [str(part.get("type")) for part in content if isinstance(part, dict)]
    tool_calls = [part for part in content if isinstance(part, dict) and part.get("type") == "toolCall"]
    usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
    return {
        "id": row.get("id"),
        "stop_reason": message.get("stopReason"),
        "error": message.get("errorMessage"),
        "content_types": content_types,
        "thinking_chars": content_text_length(content, "thinking", "thinking"),
        "text_chars": content_text_length(content, "text", "text"),
        "tool_calls": [
            {
                "name": call.get("name"),
                "arguments": call.get("arguments") or {},
            }
            for call in tool_calls
        ],
        "input_tokens": usage.get("input", 0),
        "output_tokens": usage.get("output", 0),
        "total_tokens": usage.get("totalTokens", 0),
        "cost": (usage.get("cost") or {}).get("total", 0) if isinstance(usage.get("cost"), dict) else 0,
    }


def summarize(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    type_counts = Counter(str(row.get("type")) for row in rows)
    role_counts = Counter(
        str((row.get("message") or {}).get("role"))
        for row in rows
        if isinstance(row.get("message"), dict)
    )
    assistants = [summary for row in rows if (summary := assistant_row(row)) is not None]
    return {
        "path": str(path),
        "records": len(rows),
        "type_counts": dict(sorted(type_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "assistant_turns": assistants,
        "assistant_count": len(assistants),
        "tool_call_count": sum(len(turn["tool_calls"]) for turn in assistants),
        "thinking_chars": sum(int(turn["thinking_chars"]) for turn in assistants),
        "text_chars": sum(int(turn["text_chars"]) for turn in assistants),
        "input_tokens": sum(int(turn["input_tokens"] or 0) for turn in assistants),
        "output_tokens": sum(int(turn["output_tokens"] or 0) for turn in assistants),
        "cost": sum(float(turn["cost"] or 0) for turn in assistants),
        "errors": [turn for turn in assistants if turn.get("error") or turn.get("stop_reason") == "error"],
    }


def render_text(payload: dict[str, Any], *, show_tools: bool) -> str:
    lines = [
        f"Trace: {payload['path']}",
        f"records={payload['records']} assistants={payload['assistant_count']} tools={payload['tool_call_count']}",
        f"tokens input={payload['input_tokens']} output={payload['output_tokens']} cost={payload['cost']:.6f}",
        f"types={payload['type_counts']}",
        f"roles={payload['role_counts']}",
        "",
    ]
    for index, turn in enumerate(payload["assistant_turns"], 1):
        lines.append(
            f"{index:02d} id={turn['id']} stop={turn['stop_reason']} "
            f"content={','.join(turn['content_types'])} "
            f"think_chars={turn['thinking_chars']} text_chars={turn['text_chars']} "
            f"tools={len(turn['tool_calls'])} in={turn['input_tokens']} out={turn['output_tokens']}"
        )
        if turn.get("error"):
            lines.append(f"   error={turn['error']}")
        if show_tools:
            for call in turn["tool_calls"]:
                lines.append(f"   tool={call['name']} args={json.dumps(call['arguments'], ensure_ascii=False)}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Pi session JSONL traces.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit JSON summaries instead of text.")
    parser.add_argument("--show-tools", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = [summarize(path) for path in args.paths]
    if args.json:
        print(json.dumps(summaries if len(summaries) > 1 else summaries[0], indent=2, sort_keys=True))
    else:
        for index, summary in enumerate(summaries):
            if index:
                print()
            print(render_text(summary, show_tools=args.show_tools))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
