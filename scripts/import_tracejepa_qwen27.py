#!/usr/bin/env python3
"""Import Qwen-27B Pi rows from TraceJEPA into the ETPI HF index schema."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

import upload_pi_trace_dataset as etpi_upload


DEFAULT_SOURCE_DATASET = "driaforall/tracejepa-pi-2500-v1"
DEFAULT_TARGET_REPO = "andthattoo/etpi-pi-traces"
DEFAULT_RUN_ID = "tracejepa_pi_2500_v1_qwen27"
DEFAULT_MODEL_PATTERN = r"qwen[^,\n\r\t\"']*(?:27\s*b|27b)"
DEFAULT_CONFIG = "transitions"


def iter_hf_rows(
    dataset: str,
    *,
    config: str | None,
    split: str,
    streaming: bool,
) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split, streaming=streaming)
    for row in ds:
        if isinstance(row, dict):
            yield row


def iter_jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield row


def nested_get(row: dict[str, Any], *paths: str) -> Any:
    for path in paths:
        value: Any = row
        ok = True
        for part in path.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                ok = False
                break
        if ok and value not in (None, ""):
            return value
    return None


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def row_search_text(row: dict[str, Any]) -> str:
    fields = [
        nested_get(row, "model", "model_id", "model_name", "provider_model", "metadata.model"),
        nested_get(row, "pi_command", "manifest.pi_command"),
        nested_get(row, "trajectory_json", "session_json", "trajectory", "session", "events", "messages"),
    ]
    return "\n".join(compact_json(field) if not isinstance(field, str) else field for field in fields if field is not None)


def row_matches_model(row: dict[str, Any], pattern: re.Pattern[str]) -> bool:
    return pattern.search(row_search_text(row)) is not None


def trajectory_value(row: dict[str, Any]) -> str:
    value = nested_get(
        row,
        "trajectory_json",
        "session_json",
        "session",
        "trajectory",
        "events",
        "messages",
        "goal_events",
        "next_prefix_events",
        "prefix_events",
    )
    if value is None:
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        try:
            json.loads(stripped)
        except json.JSONDecodeError:
            return compact_json([{"type": "text", "text": stripped}])
        return stripped
    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        event_types = {str(item.get("type") or "") for item in value}
        if event_types & {"instruction", "action", "observation"}:
            return compact_json(tracejepa_events_to_pi_records(value, task_id=str(row.get("task_id") or "")))
    return compact_json(value)


def tracejepa_events_to_pi_records(events: list[dict[str, Any]], *, task_id: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = [
        {
            "type": "session",
            "id": f"tracejepa-{task_id or 'task'}",
            "cwd": "/home/user",
        }
    ]
    last_tool_call_id = ""
    assistant_index = 0
    tool_index = 0
    for index, event in enumerate(events):
        event_type = str(event.get("type") or "")
        text = "" if event.get("text") is None else str(event.get("text"))
        metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
        event_id = str(event.get("event_id") or f"{task_id}-event-{index:05d}")

        if event_type == "instruction":
            records.append(
                {
                    "type": "message",
                    "id": event_id,
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": text}],
                    },
                }
            )
            continue

        if event_type == "action":
            tool_name = str(metadata.get("tool_name") or "bash")
            last_tool_call_id = f"tracejepa-call-{assistant_index:04d}"
            records.append(
                {
                    "type": "message",
                    "id": event_id,
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "toolCall",
                                "id": last_tool_call_id,
                                "name": tool_name,
                                "arguments": {"command": text},
                            }
                        ],
                        "stopReason": "toolUse",
                    },
                }
            )
            assistant_index += 1
            continue

        if event_type == "observation":
            tool_name = str(metadata.get("tool_name") or "bash")
            records.append(
                {
                    "type": "message",
                    "id": event_id,
                    "message": {
                        "role": "toolResult",
                        "toolCallId": metadata.get("tool_call_id") or last_tool_call_id or f"tracejepa-tool-{tool_index:04d}",
                        "toolName": tool_name,
                        "content": [{"type": "text", "text": text}],
                    },
                }
            )
            tool_index += 1
            continue

        if text:
            records.append(
                {
                    "type": "message",
                    "id": event_id,
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": text}],
                    },
                }
            )
    return records


def command_value(command: Any, flag: str) -> str | None:
    if not isinstance(command, list):
        return None
    try:
        index = command.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    value = command[index + 1]
    return str(value) if value is not None else None


def normalize_tracejepa_row(
    row: dict[str, Any],
    *,
    run_id: str,
    row_index: int,
    source_dataset: str,
) -> dict[str, str | float | bool]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    task = row.get("task") if isinstance(row.get("task"), dict) else {}
    pi_command = nested_get(row, "pi_command", "manifest.pi_command")
    task_id = nested_get(row, "task_id", "id", "session_id", "trace_id", "base_task_id")
    prompt = nested_get(row, "prompt", "instruction", "message", "task.prompt", "metadata.prompt")
    if prompt is None and isinstance(row.get("goal_events"), list):
        for event in row["goal_events"]:
            if isinstance(event, dict) and event.get("type") == "instruction" and event.get("text"):
                prompt = event.get("text")
                break
    cwd = nested_get(row, "cwd", "repo", "task.cwd", "metadata.cwd")
    model = nested_get(row, "model", "model_id", "model_name", "provider_model", "metadata.model")
    provider = nested_get(row, "provider", "metadata.provider")
    status = nested_get(row, "status", "manifest.status")
    if status is None and "success" in row:
        status = "ok" if row.get("success") else "error"

    return etpi_upload.normalize_index_row(
        {
            "run_id": run_id,
            "task_id": task_id or f"{run_id}_{row_index:06d}",
            "status": status or "ok",
            "prompt": prompt,
            "cwd": cwd,
            "session_path": nested_get(row, "session_path"),
            "rpc_events_path": nested_get(row, "rpc_events_path"),
            "stderr_path": nested_get(row, "stderr_path"),
            "manifest_path": f"runs/{run_id}/manifest.jsonl",
            "provider": provider or command_value(pi_command, "--provider"),
            "model": model or command_value(pi_command, "--model"),
            "elapsed_sec": nested_get(row, "elapsed_sec"),
            "started_at": nested_get(row, "started_at"),
            "ended_at": nested_get(row, "ended_at"),
            "error": nested_get(row, "error"),
            "thinking_level": nested_get(row, "thinking_level", "task.thinking_level", "metadata.thinking_level"),
            "base_task_id": nested_get(row, "base_task_id", "metadata.base_task_id"),
            "repo_id": nested_get(row, "repo_id", "metadata.repo_id"),
            "repo_name": nested_get(row, "repo_name", "metadata.repo_name"),
            "domain": nested_get(row, "domain", "metadata.domain"),
            "source": nested_get(row, "source", "metadata.source") or source_dataset,
            "verifiable": nested_get(row, "verifiable", "metadata.verifiable"),
            "persona_id": nested_get(row, "persona_id", "metadata.persona_id"),
            "intent": nested_get(row, "intent", "metadata.intent"),
            "language": nested_get(row, "language", "metadata.language"),
            "needs_workspace": nested_get(row, "needs_workspace", "metadata.needs_workspace"),
            "difficulty": nested_get(row, "difficulty", "metadata.difficulty"),
            "generator_model": nested_get(row, "generator_model", "metadata.generator_model"),
            "task_generation_fallback": nested_get(row, "task_generation_fallback", "metadata.task_generation_fallback"),
            "trajectory_json": trajectory_value(row),
        }
    )


def select_rows(
    rows: Iterable[dict[str, Any]],
    *,
    run_id: str,
    source_dataset: str,
    model_pattern: str,
    max_rows: int | None = None,
) -> tuple[list[dict[str, str | float | bool]], dict[str, int]]:
    pattern = re.compile(model_pattern, re.IGNORECASE)
    selected: list[dict[str, str | float | bool]] = []
    seen_trajectories: set[str] = set()
    scanned = 0
    skipped = 0
    duplicate_transitions = 0
    for row in rows:
        scanned += 1
        if not row_matches_model(row, pattern):
            skipped += 1
            continue
        trajectory_id = row.get("trajectory_id")
        if isinstance(trajectory_id, str) and trajectory_id:
            if trajectory_id in seen_trajectories:
                duplicate_transitions += 1
                continue
            seen_trajectories.add(trajectory_id)
        selected.append(
            normalize_tracejepa_row(
                row,
                run_id=run_id,
                row_index=scanned,
                source_dataset=source_dataset,
            )
        )
        if max_rows is not None and len(selected) >= max_rows:
            break
    return selected, {
        "scanned": scanned,
        "selected": len(selected),
        "skipped": skipped,
        "duplicate_transitions": duplicate_transitions,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def upload_index_rows(
    rows: list[dict[str, str | float | bool]],
    *,
    repo_id: str,
    run_id: str,
    token: str | None,
    private: bool,
    commit_message: str | None,
) -> None:
    if not token:
        raise SystemExit("HF_TOKEN is not set")
    with tempfile.TemporaryDirectory(prefix="tracejepa-qwen27-hf-") as tmp:
        staging = Path(tmp)
        index_path = staging / "index" / f"{run_id}.jsonl"
        write_jsonl(index_path, rows)
        run_root = staging / "runs" / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        write_jsonl(run_root / "manifest.jsonl", [{"run_id": run_id, "imported_rows": len(rows)}])
        (staging / "README.md").write_text(etpi_upload.dataset_card(repo_id))
        etpi_upload.upload_folder(
            staging,
            repo_id=repo_id,
            private=private,
            commit_message=commit_message or f"Import TraceJEPA Qwen-27B rows as {run_id}",
            token=token,
        )


def resolve_hf_token(token_env: str) -> str | None:
    token = os.environ.get(token_env)
    if token:
        return token
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    return get_token()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract TraceJEPA Qwen-27B Pi rows into ETPI's HF index schema.")
    parser.add_argument("--source-dataset", default=DEFAULT_SOURCE_DATASET)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="HF dataset config. TraceJEPA currently exposes trajectories and transitions.",
    )
    parser.add_argument("--target-repo", default=DEFAULT_TARGET_REPO)
    parser.add_argument("--split", default="train")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--model-pattern", default=DEFAULT_MODEL_PATTERN)
    parser.add_argument("--source-jsonl", type=Path, help="Read rows from local JSONL instead of HF.")
    parser.add_argument("--out-index", type=Path, help="Write selected normalized rows to a local index JSONL.")
    parser.add_argument("--max-rows", type=int, help="Limit selected rows for a smoke run.")
    parser.add_argument("--no-streaming", action="store_true", help="Disable datasets streaming.")
    parser.add_argument("--upload", action="store_true", help="Append the selected rows to --target-repo.")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--commit-message")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_rows: Iterable[dict[str, Any]]
    if args.source_jsonl:
        source_rows = iter_jsonl_rows(args.source_jsonl)
    else:
        source_rows = iter_hf_rows(
            args.source_dataset,
            config=args.config,
            split=args.split,
            streaming=not args.no_streaming,
        )

    rows, stats = select_rows(
        source_rows,
        run_id=args.run_id,
        source_dataset=args.source_dataset,
        model_pattern=args.model_pattern,
        max_rows=args.max_rows,
    )
    if args.out_index:
        write_jsonl(args.out_index, rows)

    if args.upload:
        upload_index_rows(
            rows,
            repo_id=args.target_repo,
            run_id=args.run_id,
            token=resolve_hf_token(args.token_env),
            private=args.private,
            commit_message=args.commit_message,
        )

    stats.update(
        {
            "run_id": args.run_id,
            "source_dataset": args.source_dataset,
            "target_repo": args.target_repo,
            "out_index": str(args.out_index) if args.out_index else "",
            "uploaded": bool(args.upload),
        }
    )
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
