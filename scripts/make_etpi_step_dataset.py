#!/usr/bin/env python3
"""Convert ETPI trajectory rows into assistant-decision step rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


TEXT_KEYS = ("text", "thinking", "content")
TOOL_RESULT_ROLES = {"tool", "toolResult"}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield row


def input_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.input_index:
        yield from read_jsonl(args.input_index)
        return

    if not args.hf_dataset:
        raise ValueError("pass --input-index or --hf-dataset")

    from datasets import load_dataset

    yield from load_dataset(args.hf_dataset, split=args.split, streaming=True)


def content_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part)
            elif part is not None:
                parts.append({"type": "text", "text": str(part)})
        return parts
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if content is None:
        return []
    return [{"type": "text", "text": str(content)}]


def part_text(part: dict[str, Any]) -> str:
    for key in TEXT_KEYS:
        value = part.get(key)
        if value is not None:
            return str(value)
    return ""


def parts_text(parts: list[dict[str, Any]], *, include_thinking: bool = True) -> str:
    texts: list[str] = []
    for part in parts:
        if not include_thinking and part.get("type") == "thinking":
            continue
        text = part_text(part)
        if text:
            texts.append(text)
    return "\n".join(texts)


def thinking_text(parts: list[dict[str, Any]]) -> str:
    return "\n".join(part_text(part) for part in parts if part.get("type") == "thinking" and part_text(part))


def tool_calls(parts: list[dict[str, Any]], message: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for part in parts:
        if part.get("type") != "toolCall":
            continue
        calls.append(
            {
                "id": "" if part.get("id") is None else str(part.get("id")),
                "name": "" if part.get("name") is None else str(part.get("name")),
                "arguments": part.get("arguments") or {},
            }
        )

    raw_calls = message.get("tool_calls")
    if isinstance(raw_calls, list):
        for call in raw_calls:
            if not isinstance(call, dict):
                continue
            calls.append(
                {
                    "id": "" if call.get("id") is None else str(call.get("id")),
                    "name": "" if call.get("name") is None else str(call.get("name")),
                    "arguments": call.get("arguments") or {},
                }
            )
    return calls


def message_record(record: dict[str, Any]) -> dict[str, Any] | None:
    if record.get("type") != "message" and not ("role" in record and "content" in record):
        return None

    message = record.get("message")
    if not isinstance(message, dict):
        message = record

    role = message.get("role")
    if not isinstance(role, str):
        return None

    normalized_role = "tool" if role in TOOL_RESULT_ROLES else role
    parts = content_parts(message.get("content"))
    calls = tool_calls(parts, message)
    usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
    return {
        "id": "" if record.get("id") is None else str(record.get("id")),
        "role": normalized_role,
        "raw_role": role,
        "content": parts,
        "text": parts_text(parts),
        "text_without_thinking": parts_text(parts, include_thinking=False),
        "tool_calls": calls,
        "stop_reason": "" if message.get("stopReason") is None else str(message.get("stopReason")),
        "error": "" if message.get("errorMessage") is None else str(message.get("errorMessage")),
        "usage": usage,
    }


def trajectory_messages(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [message for record in records if (message := message_record(record)) is not None]


def trajectory_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("trajectory_json")
    if not isinstance(value, str) or not value.strip():
        return []
    records = json.loads(value)
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def metadata_from_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "run_id",
        "task_id",
        "source",
        "thinking_level",
        "base_task_id",
        "repo_id",
        "repo_name",
        "domain",
        "persona_id",
        "intent",
        "language",
        "needs_workspace",
        "task_kind",
        "task_profile",
    ]
    return {key: row.get(key) for key in keys if key in row}


def step_rows_for_trace(row: dict[str, Any], *, include_empty_assistant: bool = False) -> list[dict[str, Any]]:
    records = trajectory_from_row(row)
    if not records:
        return []

    messages = trajectory_messages(records)
    steps: list[dict[str, Any]] = []
    state: list[dict[str, Any]] = []
    assistant_index = 0
    metadata = metadata_from_row(row)
    run_id = str(row.get("run_id") or "")
    task_id = str(row.get("task_id") or "")

    for message in messages:
        if message["role"] != "assistant":
            state.append(message)
            continue

        if not include_empty_assistant and not message["content"] and not message["tool_calls"]:
            state.append(message)
            continue

        raw_thinking = thinking_text(message["content"])
        calls = message["tool_calls"]
        step = {
            "id": f"{run_id}/{task_id}/assistant_{assistant_index:04d}",
            "run_id": run_id,
            "task_id": task_id,
            "source": row.get("source") or "",
            "thinking_level": row.get("thinking_level") or "",
            "state_messages": list(state),
            "target_assistant": message,
            "raw_thinking": raw_thinking,
            "compressed_thinking": None,
            "loss_mask": {
                "state": False,
                "tool_results": False,
                "assistant_action": True,
                "raw_verbose_thinking": False,
                "compressed_thinking": False,
            },
            "reward_features": {
                "trace_success": str(row.get("status") or "ok") == "ok",
                "step_index": len(steps),
                "assistant_index": assistant_index,
                "state_messages": len(state),
                "has_tool_call": bool(calls),
                "tool_names": [call.get("name") or "" for call in calls],
                "stop_reason": message.get("stop_reason") or "",
                "raw_thinking_chars": len(raw_thinking),
                "target_text_chars": len(message.get("text_without_thinking") or ""),
            },
            "metadata": metadata,
        }
        steps.append(step)
        assistant_index += 1
        state.append(message)

    return steps


def update_stats(stats: dict[str, Any], trace_row: dict[str, Any], steps: list[dict[str, Any]]) -> None:
    stats["trajectories"] += 1
    if not trace_row.get("trajectory_json"):
        stats["blank_trajectories"] += 1
    if not steps:
        stats["trajectories_without_steps"] += 1

    stats["steps"] += len(steps)
    stats["runs"][str(trace_row.get("run_id") or "")] += len(steps)
    stats["sources"][str(trace_row.get("source") or "")] += len(steps)

    for step in steps:
        target = step["target_assistant"]
        stats["assistant_stop_reasons"][str(target.get("stop_reason") or "")] += 1
        stats["role_counts"].update(message["role"] for message in step["state_messages"])
        if step["reward_features"]["has_tool_call"]:
            stats["tool_call_steps"] += 1
            stats["tool_names"].update(step["reward_features"]["tool_names"])
        else:
            stats["non_tool_steps"] += 1
        stats["raw_thinking_chars"] += int(step["reward_features"]["raw_thinking_chars"])
        stats["target_text_chars"] += int(step["reward_features"]["target_text_chars"])
        stats["max_state_messages"] = max(stats["max_state_messages"], int(step["reward_features"]["state_messages"]))


def serializable_stats(stats: dict[str, Any]) -> dict[str, Any]:
    steps = stats["steps"] or 1
    return {
        "trajectories": stats["trajectories"],
        "blank_trajectories": stats["blank_trajectories"],
        "trajectories_without_steps": stats["trajectories_without_steps"],
        "steps": stats["steps"],
        "tool_call_steps": stats["tool_call_steps"],
        "non_tool_steps": stats["non_tool_steps"],
        "avg_raw_thinking_chars": stats["raw_thinking_chars"] / steps,
        "avg_target_text_chars": stats["target_text_chars"] / steps,
        "max_state_messages": stats["max_state_messages"],
        "runs": dict(sorted(stats["runs"].items())),
        "sources": dict(sorted(stats["sources"].items())),
        "assistant_stop_reasons": dict(sorted(stats["assistant_stop_reasons"].items())),
        "role_counts": dict(sorted(stats["role_counts"].items())),
        "tool_names": dict(sorted(stats["tool_names"].items())),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an ETPI assistant-step dataset from trajectory rows.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-index", type=Path, help="Local index JSONL with trajectory_json rows.")
    group.add_argument("--hf-dataset", help="Hugging Face dataset repo id, e.g. user/etpi-pi-traces.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stats-out", type=Path, default=None)
    parser.add_argument("--canary-out", type=Path, default=None)
    parser.add_argument("--canary-size", type=int, default=32)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--include-empty-assistant", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "trajectories": 0,
        "blank_trajectories": 0,
        "trajectories_without_steps": 0,
        "steps": 0,
        "tool_call_steps": 0,
        "non_tool_steps": 0,
        "raw_thinking_chars": 0,
        "target_text_chars": 0,
        "max_state_messages": 0,
        "runs": Counter(),
        "sources": Counter(),
        "assistant_stop_reasons": Counter(),
        "role_counts": Counter(),
        "tool_names": Counter(),
    }
    canary_rows: list[dict[str, Any]] = []

    with args.out.open("w") as out_file:
        for index, row in enumerate(input_rows(args), 1):
            if args.max_trajectories is not None and index > args.max_trajectories:
                break
            steps = step_rows_for_trace(row, include_empty_assistant=args.include_empty_assistant)
            update_stats(stats, row, steps)
            for step in steps:
                out_file.write(json.dumps(step, ensure_ascii=False) + "\n")
                if len(canary_rows) < args.canary_size:
                    canary_rows.append(step)

    summary = serializable_stats(stats)
    if args.stats_out:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        args.stats_out.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    if args.canary_out:
        args.canary_out.parent.mkdir(parents=True, exist_ok=True)
        with args.canary_out.open("w") as f:
            for row in canary_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"out": str(args.out), **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
