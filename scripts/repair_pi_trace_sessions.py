#!/usr/bin/env python3
"""Backfill missing Pi session JSONL files from captured RPC events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import generate_pi_rpc_traces as pi_traces


def command_value(command: list[str], flag: str, default: str) -> str:
    try:
        index = command.index(flag)
    except ValueError:
        return default
    if index + 1 >= len(command):
        return default
    return str(command[index + 1])


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def repair_rows(rows: list[dict[str, Any]], *, trace_dir: Path, dry_run: bool) -> int:
    repaired = 0
    session_dir = trace_dir / "sessions"
    for row in rows:
        if row.get("status") != "ok" or row.get("session_file"):
            continue
        rpc_events_file = Path(str(row.get("rpc_events_file") or ""))
        if not rpc_events_file.exists():
            continue
        task = row.get("task") if isinstance(row.get("task"), dict) else {}
        command = row.get("pi_command") if isinstance(row.get("pi_command"), list) else []
        task_id = str(row.get("task_id") or task.get("task_id") or rpc_events_file.stem)
        task_slug = pi_traces.slug(task_id)
        provider = command_value(command, "--provider", "openrouter")
        model = command_value(command, "--model", "qwen/qwen3.5-27b")
        session_name = str(task.get("session_name") or f"etpi_{task_slug}_reconstructed")
        session_file = pi_traces.reconstruct_session_file(
            rpc_events_file,
            session_dir,
            task_slug=task_slug,
            cwd=str(row.get("cwd") or task.get("cwd") or "."),
            provider=provider,
            model=model,
            thinking_level=task.get("thinking_level"),
            session_name=session_name,
            started_at=str(row.get("started_at") or pi_traces.utc_now()),
        )
        if session_file is None:
            continue
        repaired += 1
        if not dry_run:
            row["session_file"] = str(session_file)
    return repaired


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair Pi trace manifests with reconstructed session files.")
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trace_dir = args.trace_dir
    manifest = args.manifest or trace_dir / "manifest.jsonl"
    rows = read_manifest(manifest)
    repaired = repair_rows(rows, trace_dir=trace_dir, dry_run=args.dry_run)
    if repaired and not args.dry_run:
        write_manifest(manifest, rows)
    print(json.dumps({"manifest": str(manifest), "repaired": repaired, "rows": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
