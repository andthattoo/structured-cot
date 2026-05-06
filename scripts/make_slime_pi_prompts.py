#!/usr/bin/env python3
"""Convert ETPI task JSONL into a slime prompt dataset.

The output keeps the raw task prompt plus an OpenAI-style ``messages`` field so
slime can either use a plain prompt or apply a chat template. Task metadata is
packed under ``metadata`` for custom slime generation/reward hooks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SYSTEM_PROMPT = """You are an autonomous coding agent in a local workspace.
Use terminal actions when inspection or verification is useful.

Action protocol:
- Run a shell command by replying exactly with <bash>COMMAND</bash>.
- Finish by replying with <final>SUMMARY</final>.
- Keep commands focused and avoid destructive changes unless the task asks for them.
"""


KNOWN_TOP_LEVEL = {
    "task_id",
    "id",
    "prompt",
    "message",
    "instruction",
    "cwd",
    "repo",
    "verify_commands",
}


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield line_no, row


def task_prompt(row: dict[str, Any], *, line_no: int) -> str:
    prompt = row.get("prompt") or row.get("message") or row.get("instruction")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"task row {line_no}: missing prompt/message/instruction")
    return prompt.strip()


def normalize_verify_commands(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        commands: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                commands.append(item)
        return commands
    return []


def slime_row(row: dict[str, Any], *, line_no: int, system_prompt: str) -> dict[str, Any]:
    prompt = task_prompt(row, line_no=line_no)
    task_id = str(row.get("task_id") or row.get("id") or f"task_{line_no:05d}")
    cwd = row.get("cwd") or row.get("repo")
    metadata = {key: value for key, value in row.items() if key not in KNOWN_TOP_LEVEL}
    metadata.update(
        {
            "task_id": task_id,
            "cwd": str(cwd) if cwd else "",
            "verify_commands": normalize_verify_commands(row.get("verify_commands")),
        }
    )
    return {
        "prompt": f"{system_prompt.strip()}\n\nTask:\n{prompt}",
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": prompt},
        ],
        "label": "",
        "metadata": metadata,
    }


def convert_tasks(
    tasks_path: Path,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_tasks: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, row in read_jsonl(tasks_path):
        rows.append(slime_row(row, line_no=line_no, system_prompt=system_prompt))
        if max_tasks is not None and len(rows) >= max_tasks:
            break
    if not rows:
        raise ValueError(f"{tasks_path}: no task rows found")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare slime prompt JSONL from ETPI task JSONL.")
    parser.add_argument("--tasks", type=Path, required=True, help="Input ETPI/Pi task JSONL.")
    parser.add_argument("--out", type=Path, required=True, help="Output slime prompt JSONL.")
    parser.add_argument("--system-prompt-file", type=Path, help="Optional replacement system prompt text.")
    parser.add_argument("--max-tasks", type=int, help="Limit rows for a smoke dataset.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    system_prompt = (
        args.system_prompt_file.read_text()
        if args.system_prompt_file
        else DEFAULT_SYSTEM_PROMPT
    )
    rows = convert_tasks(args.tasks, system_prompt=system_prompt, max_tasks=args.max_tasks)
    write_jsonl(args.out, rows)
    print(json.dumps({"out": str(args.out), "rows": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
