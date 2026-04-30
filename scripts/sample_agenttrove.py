#!/usr/bin/env python3
"""Stream a small AgentTrove sample for format inspection.

AgentTrove is large, so this script uses Hugging Face streaming by default and
writes only a small JSONL subset plus a human-readable preview of conversation
turn shapes. It is intentionally an inspection tool, not a converter.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET = "open-thoughts/AgentTrove"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample AgentTrove rows for inspection.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, default=Path("agenttrove_sample.jsonl"))
    parser.add_argument("--preview-out", type=Path, default=Path("agenttrove_sample_preview.txt"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--scan-limit", type=int, default=50_000)
    parser.add_argument(
        "--reward",
        type=float,
        default=None,
        help="Optional exact reward filter, e.g. 1.0 for successful traces.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Keep only rows whose original_source/task matches this value. Can repeat.",
    )
    parser.add_argument(
        "--teacher",
        action="append",
        default=[],
        help="Keep only rows whose original_teacher/model contains this case-insensitive substring. Can repeat.",
    )
    parser.add_argument("--max-turns-preview", type=int, default=8)
    parser.add_argument("--max-content-chars", type=int, default=700)
    parser.add_argument("--no-streaming", action="store_true")
    return parser.parse_args()


def compact(text: Any, *, max_chars: int) -> str:
    value = "" if text is None else str(text)
    value = value.replace("\r", "\\r").replace("\n", "\\n")
    if len(value) > max_chars:
        return value[:max_chars] + "..."
    return value


def row_value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def row_matches(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.reward is not None:
        try:
            reward = float(row.get("reward"))
        except (TypeError, ValueError):
            return False
        if reward != args.reward:
            return False

    if args.source:
        source = str(row_value(row, "original_source", "task", "source") or "")
        if source not in set(args.source):
            return False

    if args.teacher:
        teacher = str(row_value(row, "original_teacher", "model", "teacher") or "").lower()
        if not any(item.lower() in teacher for item in args.teacher):
            return False

    return True


def iter_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Run with:\n"
            "  uv run --with datasets python scripts/sample_agenttrove.py ..."
        ) from exc

    dataset = load_dataset(
        args.dataset,
        split=args.split,
        streaming=not args.no_streaming,
    )
    yield from dataset


def write_preview(rows: list[dict[str, Any]], args: argparse.Namespace, scanned: int) -> None:
    source_counts = Counter(str(row_value(row, "original_source", "task", "source")) for row in rows)
    teacher_counts = Counter(str(row_value(row, "original_teacher", "model", "teacher")) for row in rows)
    reward_counts = Counter(str(row.get("reward")) for row in rows)
    result_counts = Counter(str(row.get("result")) for row in rows)

    parts: list[str] = []
    parts.append(f"dataset: {args.dataset}")
    parts.append(f"split: {args.split}")
    parts.append(f"scanned: {scanned}")
    parts.append(f"written: {len(rows)}")
    parts.append(f"sources: {source_counts.most_common(20)}")
    parts.append(f"teachers: {teacher_counts.most_common(20)}")
    parts.append(f"rewards: {reward_counts.most_common(20)}")
    parts.append(f"results: {result_counts.most_common(20)}")

    for index, row in enumerate(rows[: min(10, len(rows))]):
        parts.append("\n" + "=" * 100)
        parts.append(f"ROW {index}")
        for key in [
            "original_source",
            "original_teacher",
            "reward",
            "result",
            "task",
            "task_id",
            "trial_name",
            "model",
            "model_provider",
            "trace_source",
        ]:
            if key in row:
                parts.append(f"{key}: {compact(row.get(key), max_chars=180)}")

        messages = row.get("messages") or row.get("conversations") or []
        parts.append(f"turns: {len(messages)}")
        for turn_index, message in enumerate(messages[: args.max_turns_preview]):
            if not isinstance(message, dict):
                parts.append(f"turn {turn_index}: {compact(message, max_chars=args.max_content_chars)}")
                continue
            role = message.get("role") or message.get("from")
            content = message.get("content") if "content" in message else message.get("value")
            parts.append(
                f"turn {turn_index:02d} role={role!r}: "
                f"{compact(content, max_chars=args.max_content_chars)}"
            )

    args.preview_out.write_text("\n".join(parts) + "\n")


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    scanned = 0

    with args.out.open("w") as out:
        for row in iter_rows(args):
            scanned += 1
            if row_matches(row, args):
                rows.append(row)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                if len(rows) >= args.limit:
                    break
            if scanned >= args.scan_limit:
                break

    write_preview(rows, args, scanned)
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "split": args.split,
                "scanned": scanned,
                "written": len(rows),
                "out": str(args.out),
                "preview_out": str(args.preview_out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
