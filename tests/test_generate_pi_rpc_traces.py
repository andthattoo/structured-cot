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
        model="qwen/qwen3.6-27b",
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
        "qwen/qwen3.6-27b",
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
