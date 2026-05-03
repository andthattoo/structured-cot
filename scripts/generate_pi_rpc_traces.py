#!/usr/bin/env python3
"""Generate Pi-native RPC traces from a JSONL task list.

The goal of this script is to collect native Pi session JSONL for efficient
thinker training. It does not reimplement Pi; it wraps ``pi --mode rpc`` and
records both the RPC event stream and Pi's own persisted session file.

Task JSONL rows accept these fields:

    {"task_id": "repo_read_001", "prompt": "Read the repo.", "cwd": "/repo"}

``message`` and ``instruction`` are accepted as aliases for ``prompt``.
Per-task ``thinking_level`` and ``session_name`` override the CLI defaults.
All unknown fields are preserved in the output manifest under ``metadata``.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MODEL = "qwen/qwen3.6-27b"
DEFAULT_PROVIDER = "openrouter"
DEFAULT_TIMEOUT_SEC = 900.0
TASK_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    prompt: str
    cwd: str | None = None
    thinking_level: str | None = None
    session_name: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class TaskResult:
    task_id: str
    status: str
    started_at: str
    ended_at: str
    elapsed_sec: float
    cwd: str
    session_file: str | None
    rpc_events_file: str
    stderr_file: str
    stdout_events: int
    stderr_lines: int
    agent_end_seen: bool
    exit_code: int | None
    error: str | None
    pi_command: list[str]
    task: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(value: str, fallback: str = "task") -> str:
    cleaned = TASK_ID_RE.sub("_", value.strip()).strip("._-")
    return cleaned[:120] or fallback


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


def task_from_row(row: dict[str, Any], *, line_no: int) -> TaskSpec:
    prompt = row.get("prompt") or row.get("message") or row.get("instruction")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"task row {line_no}: missing prompt/message/instruction")
    raw_task_id = row.get("task_id") or row.get("id") or f"task_{line_no:05d}"
    task_id = slug(str(raw_task_id), fallback=f"task_{line_no:05d}")
    known = {"task_id", "id", "prompt", "message", "instruction", "cwd", "repo", "thinking_level", "session_name"}
    metadata = {key: value for key, value in row.items() if key not in known}
    cwd = row.get("cwd") or row.get("repo")
    return TaskSpec(
        task_id=task_id,
        prompt=prompt.strip(),
        cwd=str(cwd) if cwd else None,
        thinking_level=str(row["thinking_level"]) if row.get("thinking_level") else None,
        session_name=str(row["session_name"]) if row.get("session_name") else None,
        metadata=metadata,
    )


def load_tasks(path: Path, *, max_tasks: int | None = None) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    for line_no, row in read_jsonl(path):
        tasks.append(task_from_row(row, line_no=line_no))
        if max_tasks is not None and len(tasks) >= max_tasks:
            break
    if not tasks:
        raise ValueError(f"{path}: no tasks found")
    return tasks


def build_pi_command(args: argparse.Namespace, session_dir: Path) -> list[str]:
    command = [
        args.pi_bin,
        "--mode",
        "rpc",
        "--provider",
        args.provider,
        "--model",
        args.model,
        "--session-dir",
        str(session_dir),
    ]
    command.extend(args.pi_arg or [])
    return command


def list_session_files(session_dir: Path) -> set[Path]:
    if not session_dir.exists():
        return set()
    return {path.resolve() for path in session_dir.rglob("*.jsonl") if path.is_file()}


def newest_created_session_file(session_dir: Path, before: set[Path], started_monotonic: float) -> Path | None:
    candidates = [path for path in list_session_files(session_dir) if path not in before]
    if not candidates:
        # Fall back to mtime in case Pi reused a known file after set_session_name.
        wall_started = time.time() - max(0.0, time.monotonic() - started_monotonic)
        candidates = [
            path
            for path in list_session_files(session_dir)
            if path.stat().st_mtime >= wall_started - 1.0
        ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def write_rpc(stdin, payload: dict[str, Any]) -> None:
    stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
    stdin.flush()


def stream_reader(stream, out_queue: "queue.Queue[tuple[str, str]]", label: str) -> None:
    try:
        for line in stream:
            out_queue.put((label, line.rstrip("\n")))
    finally:
        out_queue.put((label, None))  # type: ignore[arg-type]


def terminate_process(proc: subprocess.Popen[str], *, grace_sec: float = 5.0) -> int | None:
    if proc.poll() is not None:
        return proc.returncode
    try:
        if proc.stdin:
            proc.stdin.close()
    except Exception:
        pass
    try:
        return proc.wait(timeout=grace_sec)
    except subprocess.TimeoutExpired:
        proc.terminate()
    try:
        return proc.wait(timeout=grace_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.wait(timeout=grace_sec)


def env_check(provider: str, *, skip: bool) -> None:
    if skip:
        return
    if provider == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "OPENROUTER_API_KEY is not set. Export it or pass --skip-env-check "
            "if your Pi config supplies credentials another way."
        )


def run_one_task(task: TaskSpec, args: argparse.Namespace, out_dir: Path) -> TaskResult:
    task_slug = slug(task.task_id)
    session_dir = out_dir / "sessions"
    events_dir = out_dir / "rpc-events"
    stderr_dir = out_dir / "stderr"
    for path in [session_dir, events_dir, stderr_dir]:
        path.mkdir(parents=True, exist_ok=True)

    cwd = Path(task.cwd or args.cwd).expanduser().resolve()
    if not cwd.exists():
        raise FileNotFoundError(f"Task {task.task_id} cwd does not exist: {cwd}")

    command = build_pi_command(args, session_dir)
    rpc_events_file = events_dir / f"{task_slug}.jsonl"
    stderr_file = stderr_dir / f"{task_slug}.log"
    before_sessions = list_session_files(session_dir)
    started_at = utc_now()
    started_monotonic = time.monotonic()
    out_queue: "queue.Queue[tuple[str, str | None]]" = queue.Queue()
    stdout_done = False
    stderr_done = False
    stdout_events = 0
    stderr_lines = 0
    agent_end_seen = False
    error: str | None = None

    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None
    threading.Thread(target=stream_reader, args=(proc.stdout, out_queue, "stdout"), daemon=True).start()
    threading.Thread(target=stream_reader, args=(proc.stderr, out_queue, "stderr"), daemon=True).start()

    session_name = task.session_name or f"etpi_{task_slug}_{int(started_monotonic)}"
    thinking_level = task.thinking_level or args.thinking_level
    write_rpc(proc.stdin, {"id": f"{task_slug}-session-name", "type": "set_session_name", "name": session_name})
    if thinking_level:
        write_rpc(proc.stdin, {"id": f"{task_slug}-thinking", "type": "set_thinking_level", "level": thinking_level})
    write_rpc(proc.stdin, {"id": f"{task_slug}-prompt", "type": "prompt", "message": task.prompt})

    deadline = time.monotonic() + args.timeout_sec
    with rpc_events_file.open("w") as events_out, stderr_file.open("w") as stderr_out:
        while time.monotonic() < deadline:
            if proc.poll() is not None and stdout_done and stderr_done:
                break
            try:
                label, line = out_queue.get(timeout=0.2)
            except queue.Empty:
                if agent_end_seen:
                    break
                continue
            if line is None:
                if label == "stdout":
                    stdout_done = True
                else:
                    stderr_done = True
                continue
            if label == "stderr":
                stderr_lines += 1
                stderr_out.write(line + "\n")
                stderr_out.flush()
                continue

            stdout_events += 1
            events_out.write(line + "\n")
            events_out.flush()
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("type") == "agent_end":
                agent_end_seen = True
                break
            if isinstance(event, dict) and event.get("type") in {"error", "agent_error"}:
                error = json.dumps(event, ensure_ascii=False)

        if not agent_end_seen and proc.poll() is None and time.monotonic() >= deadline:
            error = f"timed out after {args.timeout_sec:.1f}s"

    exit_code = terminate_process(proc)
    session_file = newest_created_session_file(session_dir, before_sessions, started_monotonic)
    ended_at = utc_now()
    elapsed = time.monotonic() - started_monotonic
    status = "ok" if agent_end_seen and error is None else "error"
    if not agent_end_seen and error is None:
        error = f"pi exited before agent_end (exit_code={exit_code})"

    return TaskResult(
        task_id=task.task_id,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        elapsed_sec=round(elapsed, 3),
        cwd=str(cwd),
        session_file=str(session_file) if session_file else None,
        rpc_events_file=str(rpc_events_file),
        stderr_file=str(stderr_file),
        stdout_events=stdout_events,
        stderr_lines=stderr_lines,
        agent_end_seen=agent_end_seen,
        exit_code=exit_code,
        error=error,
        pi_command=command,
        task=asdict(task),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Pi RPC traces from JSONL tasks.")
    parser.add_argument("--tasks", type=Path, required=True, help="JSONL tasks with prompt/message/instruction fields.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory. Defaults to data/pi_traces/<timestamp>.")
    parser.add_argument("--pi-bin", default="pi", help="Pi executable path.")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--thinking-level", default="medium", help="Default Pi thinking level; empty string disables command.")
    parser.add_argument("--cwd", default=".", help="Default cwd for tasks without cwd/repo.")
    parser.add_argument("--timeout-sec", type=float, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--skip-env-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pi-arg", action="append", default=[], help="Extra argument passed through to pi; repeatable.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.thinking_level == "":
        args.thinking_level = None
    env_check(args.provider, skip=args.skip_env_check or args.dry_run)
    tasks = load_tasks(args.tasks, max_tasks=args.max_tasks)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path("data/pi_traces") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    if args.dry_run:
        session_dir = out_dir / "sessions"
        plan = {
            "tasks": [asdict(task) for task in tasks],
            "pi_command": build_pi_command(args, session_dir),
            "out_dir": str(out_dir),
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    ok = 0
    with manifest_path.open("a") as manifest:
        for task in tasks:
            try:
                result = run_one_task(task, args, out_dir)
            except Exception as exc:
                now = utc_now()
                result = TaskResult(
                    task_id=task.task_id,
                    status="error",
                    started_at=now,
                    ended_at=now,
                    elapsed_sec=0.0,
                    cwd=str(Path(task.cwd or args.cwd).expanduser()),
                    session_file=None,
                    rpc_events_file="",
                    stderr_file="",
                    stdout_events=0,
                    stderr_lines=0,
                    agent_end_seen=False,
                    exit_code=None,
                    error=repr(exc),
                    pi_command=[],
                    task=asdict(task),
                )
            if result.status == "ok":
                ok += 1
            manifest.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            manifest.flush()
            print(
                json.dumps(
                    {
                        "task_id": result.task_id,
                        "status": result.status,
                        "elapsed_sec": result.elapsed_sec,
                        "session_file": result.session_file,
                        "error": result.error,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    print(json.dumps({"out_dir": str(out_dir), "manifest": str(manifest_path), "ok": ok, "total": len(tasks)}, sort_keys=True))
    return 0 if ok == len(tasks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
