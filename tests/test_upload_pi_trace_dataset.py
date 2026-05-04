from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "upload_pi_trace_dataset.py"
SPEC = importlib.util.spec_from_file_location("upload_pi_trace_dataset", SCRIPT_PATH)
assert SPEC is not None
upload_pi_trace_dataset = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = upload_pi_trace_dataset
SPEC.loader.exec_module(upload_pi_trace_dataset)


def test_build_upload_tree_indexes_successful_sessions(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    session_dir = trace_dir / "sessions" / "reconstructed"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "task_a.jsonl"
    session_file.write_text(
        json.dumps({"type": "session"}) + "\n"
        + json.dumps({"type": "message", "role": "assistant", "content": "Done"}) + "\n"
    )
    rpc_dir = trace_dir / "rpc-events"
    rpc_dir.mkdir()
    rpc_file = rpc_dir / "task_a.jsonl"
    rpc_file.write_text(json.dumps({"type": "agent_end"}) + "\n")
    stderr_dir = trace_dir / "stderr"
    stderr_dir.mkdir()
    stderr_file = stderr_dir / "task_a.log"
    stderr_file.write_text("")
    manifest = trace_dir / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "task_id": "task_a",
                "status": "ok",
                "session_file": str(session_file),
                "rpc_events_file": str(rpc_file),
                "stderr_file": str(stderr_file),
                "elapsed_sec": 3.5,
                "started_at": "2026-05-04T00:00:00+00:00",
                "ended_at": "2026-05-04T00:00:03+00:00",
                "error": None,
                "pi_command": ["pi", "--provider", "openrouter", "--model", "qwen/qwen3.5-27b"],
                "task": {
                    "prompt": "Do the task.",
                    "cwd": "/tmp/work",
                    "thinking_level": "minimal",
                    "metadata": {
                        "base_task_id": "task",
                        "repo_id": "repo",
                        "repo_name": "Repo",
                        "domain": "python",
                        "source": "test",
                        "verifiable": False,
                        "persona_id": "persona_a",
                        "intent": "build",
                        "language": "python",
                        "needs_workspace": True,
                        "difficulty": "easy",
                        "generator_model": "task-gen",
                        "task_generation_fallback": False,
                    },
                },
            }
        )
        + "\n"
    )
    staging = tmp_path / "staging"

    stats = upload_pi_trace_dataset.build_upload_tree(
        trace_dir,
        staging,
        run_id="run_1",
        repo_id="user/repo",
        include_rpc=True,
        include_stderr=True,
    )

    assert stats["rows"] == 1
    assert stats["indexed_rows"] == 1
    assert (staging / "runs/run_1/manifest.jsonl").exists()
    assert (staging / "runs/run_1/sessions/reconstructed/task_a.jsonl").exists()
    assert (staging / "runs/run_1/rpc-events/task_a.jsonl").exists()
    index_rows = [
        json.loads(line)
        for line in (staging / "index/run_1.jsonl").read_text().splitlines()
    ]
    assert index_rows[0]["task_id"] == "task_a"
    assert index_rows[0]["session_path"] == "runs/run_1/sessions/reconstructed/task_a.jsonl"
    assert index_rows[0]["thinking_level"] == "minimal"
    assert index_rows[0]["prompt"] == "Do the task."
    assert index_rows[0]["provider"] == "openrouter"
    assert index_rows[0]["model"] == "qwen/qwen3.5-27b"
    assert index_rows[0]["intent"] == "build"
    assert index_rows[0]["language"] == "python"
    assert index_rows[0]["persona_id"] == "persona_a"
    trajectory = json.loads(index_rows[0]["trajectory_json"])
    assert trajectory == [
        {"type": "session"},
        {"type": "message", "role": "assistant", "content": "Done"},
    ]
    assert all(value is not None for value in index_rows[0].values())
    assert set(index_rows[0]) == set(upload_pi_trace_dataset.INDEX_COLUMNS)


def test_dataset_card_points_viewer_at_index_only() -> None:
    card = upload_pi_trace_dataset.dataset_card("user/repo")

    assert "configs:" in card
    assert "path: index/*.jsonl" in card
    assert "trajectory_json" in card
    assert "sessions" in card
    assert "stable, non-null schema" in card


def test_build_upload_tree_skips_errors_by_default(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "manifest.jsonl").write_text(
        json.dumps({"task_id": "bad", "status": "error", "error": "timeout", "task": {}})
        + "\n"
    )

    stats = upload_pi_trace_dataset.build_upload_tree(
        trace_dir,
        tmp_path / "staging",
        run_id="run_1",
        repo_id="user/repo",
    )

    assert stats["rows"] == 1
    assert stats["indexed_rows"] == 0


def test_normalize_index_file_adds_stable_schema(tmp_path: Path) -> None:
    index_path = tmp_path / "old.jsonl"
    index_path.write_text(
        json.dumps(
            {
                "run_id": "old",
                "task_id": "task_a",
                "status": "ok",
                "session_path": "runs/old/sessions/task_a.jsonl",
                "rpc_events_path": None,
                "stderr_path": None,
                "elapsed_sec": "4.25",
                "error": None,
                "verifiable": None,
            }
        )
        + "\n"
    )

    upload_pi_trace_dataset.normalize_index_file(index_path)

    row = json.loads(index_path.read_text())
    assert set(row) == set(upload_pi_trace_dataset.INDEX_COLUMNS)
    assert row["rpc_events_path"] == ""
    assert row["stderr_path"] == ""
    assert row["error"] == ""
    assert row["elapsed_sec"] == 4.25
    assert row["verifiable"] is False
    assert row["persona_id"] == ""
    assert row["trajectory_json"] == ""
