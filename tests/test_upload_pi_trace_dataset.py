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
    session_file.write_text(json.dumps({"type": "session"}) + "\n")
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
                "error": None,
                "task": {
                    "thinking_level": "minimal",
                    "metadata": {
                        "base_task_id": "task",
                        "repo_id": "repo",
                        "repo_name": "Repo",
                        "domain": "python",
                        "source": "test",
                        "verifiable": False,
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
