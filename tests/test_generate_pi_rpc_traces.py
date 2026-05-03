from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "generate_pi_rpc_traces.py"
SPEC = importlib.util.spec_from_file_location("generate_pi_rpc_traces", SCRIPT_PATH)
assert SPEC is not None
generate_pi_rpc_traces = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = generate_pi_rpc_traces
SPEC.loader.exec_module(generate_pi_rpc_traces)

REPAIR_SCRIPT_PATH = ROOT / "scripts" / "repair_pi_trace_sessions.py"
REPAIR_SPEC = importlib.util.spec_from_file_location("repair_pi_trace_sessions", REPAIR_SCRIPT_PATH)
assert REPAIR_SPEC is not None
repair_pi_trace_sessions = importlib.util.module_from_spec(REPAIR_SPEC)
assert REPAIR_SPEC.loader is not None
sys.modules[REPAIR_SPEC.name] = repair_pi_trace_sessions
REPAIR_SPEC.loader.exec_module(repair_pi_trace_sessions)


def test_load_tasks_accepts_prompt_aliases_and_preserves_metadata(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "repo/read 1", "prompt": "Read repo", "cwd": "/tmp/repo", "source": "manual"}),
                json.dumps({"id": "msg", "message": "Explain script", "thinking_level": "high"}),
                json.dumps({"instruction": "Find tests", "repo": "/tmp/other"}),
            ]
        )
        + "\n"
    )

    tasks = generate_pi_rpc_traces.load_tasks(tasks_path)

    assert [task.task_id for task in tasks] == ["repo_read_1", "msg", "task_00003"]
    assert tasks[0].prompt == "Read repo"
    assert tasks[0].cwd == "/tmp/repo"
    assert tasks[0].metadata == {"source": "manual"}
    assert tasks[1].thinking_level == "high"
    assert tasks[2].cwd == "/tmp/other"


def test_build_pi_command_uses_rpc_openrouter_model_and_session_dir(tmp_path: Path) -> None:
    args = argparse.Namespace(
        pi_bin="pi",
        provider="openrouter",
        model="qwen/qwen3.5-27b",
        pi_arg=["--some-flag", "value"],
    )

    command = generate_pi_rpc_traces.build_pi_command(args, tmp_path / "sessions")

    assert command[:8] == [
        "pi",
        "--mode",
        "rpc",
        "--provider",
        "openrouter",
        "--model",
        "qwen/qwen3.5-27b",
        "--session-dir",
    ]
    assert command[-2:] == ["--some-flag", "value"]


def test_dry_run_outputs_plan_without_openrouter_key(tmp_path: Path, capsys) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    out_dir = tmp_path / "out"
    tasks_path.write_text(json.dumps({"task_id": "one", "prompt": "Read the repo"}) + "\n")

    rc = generate_pi_rpc_traces.main(
        [
            "--tasks",
            str(tasks_path),
            "--out-dir",
            str(out_dir),
            "--dry-run",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tasks"][0]["task_id"] == "one"
    assert payload["pi_command"][:3] == ["pi", "--mode", "rpc"]
    assert payload["out_dir"] == str(out_dir)


def test_newest_created_session_file_prefers_new_file(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    old_file = session_dir / "old.jsonl"
    old_file.write_text("{}\n")
    before = generate_pi_rpc_traces.list_session_files(session_dir)
    started = time.monotonic()
    new_file = session_dir / "new.jsonl"
    new_file.write_text("{}\n")

    selected = generate_pi_rpc_traces.newest_created_session_file(session_dir, before, started)

    assert selected == new_file.resolve()


def test_session_discovery_ignores_reconstructed_files(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions"
    reconstructed_dir = session_dir / "reconstructed"
    reconstructed_dir.mkdir(parents=True)
    reconstructed_file = reconstructed_dir / "old.jsonl"
    reconstructed_file.write_text("{}\n")

    assert generate_pi_rpc_traces.list_session_files(session_dir) == set()


def test_thinking_level_sweep_expands_tasks() -> None:
    tasks = [
        generate_pi_rpc_traces.TaskSpec(
            task_id="read_repo",
            prompt="Read repo",
            cwd=".",
            thinking_level="medium",
            metadata={"source": "manual"},
        )
    ]

    expanded = generate_pi_rpc_traces.expand_tasks_for_thinking_levels(tasks, ["off", "high"])

    assert [task.task_id for task in expanded] == ["read_repo__think_off", "read_repo__think_high"]
    assert [task.thinking_level for task in expanded] == ["off", "high"]
    assert expanded[0].metadata["base_task_id"] == "read_repo"
    assert expanded[0].metadata["thinking_level_sweep"] is True


def test_event_error_message_catches_assistant_api_error() -> None:
    event = {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "stopReason": "error",
            "errorMessage": "401 User not found.",
        },
    }

    assert generate_pi_rpc_traces.event_error_message(event) == "401 User not found."


def test_event_error_message_catches_failed_response() -> None:
    event = {"type": "response", "success": False, "error": "bad command"}

    assert "bad command" in generate_pi_rpc_traces.event_error_message(event)


def test_reconstruct_session_file_from_rpc_message_events(tmp_path: Path) -> None:
    events_path = tmp_path / "rpc-events" / "task.jsonl"
    events_path.parent.mkdir()
    events = [
        {
            "type": "message_end",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "Read repo"}],
                "timestamp": 1777802277996,
            },
        },
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Done"}],
                "stopReason": "stop",
                "timestamp": 1777802278030,
            },
        },
        {"type": "agent_end"},
    ]
    events_path.write_text("\n".join(json.dumps(event) for event in events) + "\n")

    session_path = generate_pi_rpc_traces.reconstruct_session_file(
        events_path,
        tmp_path / "sessions",
        task_slug="task",
        cwd="/repo",
        provider="openrouter",
        model="qwen/qwen3.5-27b",
        thinking_level="medium",
        session_name="etpi_task",
        started_at="2026-05-03T10:00:00+00:00",
    )

    assert session_path is not None
    records = [json.loads(line) for line in session_path.read_text().splitlines()]
    assert [record["type"] for record in records] == [
        "session",
        "model_change",
        "thinking_level_change",
        "session_info",
        "message",
        "message",
    ]
    assert records[0]["cwd"] == "/repo"
    assert records[2]["thinkingLevel"] == "medium"
    assert [record["message"]["role"] for record in records if record["type"] == "message"] == [
        "user",
        "assistant",
    ]


def test_repair_rows_backfills_session_file(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    events_path = trace_dir / "rpc-events" / "task.jsonl"
    events_path.parent.mkdir(parents=True)
    events_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "message_end",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": "Read repo"}],
                            "timestamp": 1777802277996,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message_end",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Done"}],
                            "stopReason": "stop",
                            "timestamp": 1777802278030,
                        },
                    }
                ),
            ]
        )
        + "\n"
    )
    rows = [
        {
            "task_id": "task",
            "status": "ok",
            "cwd": "/repo",
            "session_file": None,
            "rpc_events_file": str(events_path),
            "started_at": "2026-05-03T10:00:00+00:00",
            "pi_command": ["pi", "--provider", "openrouter", "--model", "qwen/qwen3.5-27b"],
            "task": {"thinking_level": "minimal"},
        }
    ]

    repaired = repair_pi_trace_sessions.repair_rows(rows, trace_dir=trace_dir, dry_run=False)

    assert repaired == 1
    assert rows[0]["session_file"]
    assert Path(rows[0]["session_file"]).exists()


def test_run_lock_rejects_second_writer(tmp_path: Path) -> None:
    lock_path, fd = generate_pi_rpc_traces.acquire_run_lock(tmp_path)

    try:
        try:
            generate_pi_rpc_traces.acquire_run_lock(tmp_path)
        except RuntimeError as exc:
            assert "already locked" in str(exc)
        else:
            raise AssertionError("second lock acquisition should fail")
    finally:
        generate_pi_rpc_traces.release_run_lock(lock_path, fd)

    assert not lock_path.exists()


def test_run_lock_reclaims_dead_pid_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / generate_pi_rpc_traces.RUN_LOCK_NAME
    lock_path.write_text(json.dumps({"pid": 99999999, "started_at": "old"}) + "\n")

    acquired_path, fd = generate_pi_rpc_traces.acquire_run_lock(tmp_path)

    try:
        assert acquired_path == lock_path
        assert json.loads(lock_path.read_text())["pid"] != 99999999
    finally:
        generate_pi_rpc_traces.release_run_lock(acquired_path, fd)
