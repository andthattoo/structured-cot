from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "make_etpi_step_dataset.py"
SPEC = importlib.util.spec_from_file_location("make_etpi_step_dataset", SCRIPT_PATH)
assert SPEC is not None
make_etpi_step_dataset = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = make_etpi_step_dataset
SPEC.loader.exec_module(make_etpi_step_dataset)


def trace_row() -> dict[str, object]:
    records = [
        {
            "type": "message",
            "id": "u1",
            "message": {"role": "user", "content": [{"type": "text", "text": "Read repo"}]},
        },
        {
            "type": "message",
            "id": "a1",
            "message": {
                "role": "assistant",
                "stopReason": "toolUse",
                "content": [
                    {"type": "thinking", "thinking": "Need the README."},
                    {"type": "toolCall", "id": "call_1", "name": "read", "arguments": {"path": "README.md"}},
                ],
            },
        },
        {
            "type": "message",
            "id": "t1",
            "message": {"role": "toolResult", "content": [{"type": "text", "text": "# Project"}]},
        },
        {
            "type": "message",
            "id": "a2",
            "message": {
                "role": "assistant",
                "stopReason": "stop",
                "content": [{"type": "text", "text": "Done."}],
            },
        },
    ]
    return {
        "run_id": "run_a",
        "task_id": "task_a",
        "source": "test_source",
        "status": "ok",
        "thinking_level": "medium",
        "repo_id": "repo_a",
        "task_kind": "architecture",
        "trajectory_json": json.dumps(records),
    }


def test_step_rows_slice_assistant_decisions() -> None:
    steps = make_etpi_step_dataset.step_rows_for_trace(trace_row())

    assert len(steps) == 2
    assert steps[0]["id"] == "run_a/task_a/assistant_0000"
    assert steps[0]["raw_thinking"] == "Need the README."
    assert steps[0]["target_assistant"]["tool_calls"] == [
        {"id": "call_1", "name": "read", "arguments": {"path": "README.md"}}
    ]
    assert steps[0]["state_messages"][0]["role"] == "user"
    assert steps[1]["state_messages"][2]["role"] == "tool"
    assert steps[1]["target_assistant"]["text"] == "Done."
    assert steps[1]["reward_features"]["has_tool_call"] is False
    assert steps[1]["metadata"]["task_kind"] == "architecture"


def test_main_writes_steps_stats_and_canary(tmp_path: Path) -> None:
    input_path = tmp_path / "index.jsonl"
    input_path.write_text(json.dumps(trace_row()) + "\n")
    out = tmp_path / "steps.jsonl"
    stats_out = tmp_path / "stats.json"
    canary_out = tmp_path / "canary.jsonl"

    rc = make_etpi_step_dataset.main(
        [
            "--input-index",
            str(input_path),
            "--out",
            str(out),
            "--stats-out",
            str(stats_out),
            "--canary-out",
            str(canary_out),
            "--canary-size",
            "1",
        ]
    )

    assert rc == 0
    assert len(out.read_text().splitlines()) == 2
    assert len(canary_out.read_text().splitlines()) == 1
    stats = json.loads(stats_out.read_text())
    assert stats["trajectories"] == 1
    assert stats["steps"] == 2
    assert stats["tool_call_steps"] == 1
    assert stats["sources"] == {"test_source": 2}
