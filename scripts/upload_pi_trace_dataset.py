#!/usr/bin/env python3
"""Upload Pi trace runs to an append-only Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any


RUN_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def slug(value: str, fallback: str = "run") -> str:
    cleaned = RUN_ID_RE.sub("_", value.strip()).strip("._-")
    return cleaned[:160] or fallback


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def relative_to_trace(path_value: str | None, trace_dir: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        return path.resolve().relative_to(trace_dir.resolve())
    except ValueError:
        return None


def copy_trace_file(trace_dir: Path, staging_dir: Path, run_id: str, rel_path: Path) -> str:
    src = trace_dir / rel_path
    dst_rel = Path("runs") / run_id / rel_path
    dst = staging_dir / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst_rel.as_posix()


def dataset_card(repo_id: str) -> str:
    return f"""---
license: apache-2.0
task_categories:
- text-generation
language:
- en
tags:
- pi
- agent-traces
- efficient-thinking
- code-agent
---

# ETPI Pi Traces

Append-only Pi RPC trace dataset for efficient-thinker experiments.

Each upload adds:

- `runs/<run_id>/manifest.jsonl`: original run manifest.
- `runs/<run_id>/sessions/...`: Pi session JSONL, usually reconstructed from RPC events.
- `index/<run_id>.jsonl`: compact index pointing to usable session files.

Dataset repo: `{repo_id}`.
"""


def build_upload_tree(
    trace_dir: Path,
    staging_dir: Path,
    *,
    run_id: str,
    repo_id: str,
    include_errors: bool = False,
    include_rpc: bool = False,
    include_stderr: bool = False,
) -> dict[str, Any]:
    trace_dir = trace_dir.resolve()
    manifest_path = trace_dir / "manifest.jsonl"
    rows = read_jsonl(manifest_path)
    run_root = staging_dir / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, run_root / "manifest.jsonl")

    index_rows: list[dict[str, Any]] = []
    copied_sessions = 0
    copied_rpc = 0
    copied_stderr = 0

    for row in rows:
        status = row.get("status")
        session_rel = relative_to_trace(row.get("session_file"), trace_dir)
        if status != "ok" and not include_errors:
            continue
        if status == "ok" and session_rel is None:
            continue

        session_path = None
        if session_rel is not None and (trace_dir / session_rel).exists():
            session_path = copy_trace_file(trace_dir, staging_dir, run_id, session_rel)
            copied_sessions += 1

        rpc_path = None
        rpc_rel = relative_to_trace(row.get("rpc_events_file"), trace_dir)
        if include_rpc and rpc_rel is not None and (trace_dir / rpc_rel).exists():
            rpc_path = copy_trace_file(trace_dir, staging_dir, run_id, rpc_rel)
            copied_rpc += 1

        stderr_path = None
        stderr_rel = relative_to_trace(row.get("stderr_file"), trace_dir)
        if include_stderr and stderr_rel is not None and (trace_dir / stderr_rel).exists():
            stderr_path = copy_trace_file(trace_dir, staging_dir, run_id, stderr_rel)
            copied_stderr += 1

        task = row.get("task") if isinstance(row.get("task"), dict) else {}
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        index_rows.append(
            {
                "run_id": run_id,
                "task_id": row.get("task_id"),
                "status": status,
                "session_path": session_path,
                "rpc_events_path": rpc_path,
                "stderr_path": stderr_path,
                "manifest_path": f"runs/{run_id}/manifest.jsonl",
                "elapsed_sec": row.get("elapsed_sec"),
                "error": row.get("error"),
                "thinking_level": task.get("thinking_level"),
                "base_task_id": metadata.get("base_task_id"),
                "repo_id": metadata.get("repo_id"),
                "repo_name": metadata.get("repo_name"),
                "domain": metadata.get("domain"),
                "source": metadata.get("source"),
                "verifiable": metadata.get("verifiable"),
            }
        )

    index_path = staging_dir / "index" / f"{run_id}.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w") as f:
        for row in index_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    readme_path = staging_dir / "README.md"
    readme_path.write_text(dataset_card(repo_id))

    return {
        "run_id": run_id,
        "rows": len(rows),
        "indexed_rows": len(index_rows),
        "copied_sessions": copied_sessions,
        "copied_rpc": copied_rpc,
        "copied_stderr": copied_stderr,
    }


def upload_folder(
    staging_dir: Path,
    *,
    repo_id: str,
    private: bool,
    commit_message: str,
    token: str | None,
) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=staging_dir,
        commit_message=commit_message,
        token=token,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a Pi trace directory to a HF dataset repo.")
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True, help="HF dataset repo, e.g. user/etpi-pi-traces")
    parser.add_argument("--run-id", default=None, help="Defaults to trace directory name.")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--include-errors", action="store_true")
    parser.add_argument("--include-rpc", action="store_true")
    parser.add_argument("--include-stderr", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--commit-message", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trace_dir = args.trace_dir.resolve()
    run_id = slug(args.run_id or trace_dir.name)
    token = os.environ.get(args.token_env)
    if not args.dry_run and not token:
        raise SystemExit(f"{args.token_env} is not set")

    with tempfile.TemporaryDirectory(prefix="pi-trace-hf-") as tmp:
        staging_dir = Path(tmp)
        stats = build_upload_tree(
            trace_dir,
            staging_dir,
            run_id=run_id,
            repo_id=args.repo_id,
            include_errors=args.include_errors,
            include_rpc=args.include_rpc,
            include_stderr=args.include_stderr,
        )
        if args.dry_run:
            stats["staging_dir"] = str(staging_dir)
            stats["files"] = sorted(path.relative_to(staging_dir).as_posix() for path in staging_dir.rglob("*") if path.is_file())
            print(json.dumps(stats, indent=2, sort_keys=True))
            return 0

        upload_folder(
            staging_dir,
            repo_id=args.repo_id,
            private=args.private,
            commit_message=args.commit_message or f"Add Pi trace run {run_id}",
            token=token,
        )
        stats["url"] = f"https://huggingface.co/datasets/{args.repo_id}"
        print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
