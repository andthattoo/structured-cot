from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "extract_thinking_hidden_trajectory.py"
SPEC = importlib.util.spec_from_file_location("extract_thinking_hidden_trajectory", SCRIPT_PATH)
assert SPEC is not None
extract_thinking_hidden_trajectory = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = extract_thinking_hidden_trajectory
SPEC.loader.exec_module(extract_thinking_hidden_trajectory)


def step_row() -> dict[str, object]:
    return {
        "id": "run/task/assistant_0000",
        "run_id": "run",
        "task_id": "task",
        "source": "test",
        "thinking_level": "medium",
        "state_messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Read the repo."}],
                "tool_calls": [],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Need context."},
                    {"type": "toolCall", "name": "read", "arguments": {"path": "README.md"}},
                ],
                "tool_calls": [{"name": "read", "arguments": {"path": "README.md"}}],
            },
            {
                "role": "tool",
                "content": [{"type": "text", "text": "# Project"}],
                "tool_calls": [],
            },
        ],
        "raw_thinking": "Now answer from the README.",
    }


def test_select_step_by_index_and_id(tmp_path: Path) -> None:
    path = tmp_path / "steps.jsonl"
    rows = [{**step_row(), "id": "first"}, {**step_row(), "id": "second"}]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    by_index = extract_thinking_hidden_trajectory.select_step(path, step_index=1, step_id=None)
    by_id = extract_thinking_hidden_trajectory.select_step(path, step_index=None, step_id="first")

    assert by_index["id"] == "second"
    assert by_id["id"] == "first"


def test_render_step_prefix_and_target_thinking() -> None:
    step = step_row()

    prefix = extract_thinking_hidden_trajectory.render_step_prefix(step)
    target = extract_thinking_hidden_trajectory.target_thinking_text(step, allow_empty=False)

    assert "<|im_start|>user" in prefix
    assert "<think>\nNeed context.\n</think>" in prefix
    assert 'TOOL_CALL read {"path": "README.md"}' in prefix
    assert prefix.endswith("<|im_start|>assistant\n<think>\n")
    assert target == "Now answer from the README.\n</think>"


def test_pca_and_distances_are_well_formed() -> None:
    hidden = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    token_rows = [
        {"position_index": 0, "segment": "prefill"},
        {"position_index": 1, "segment": "thinking"},
        {"position_index": 2, "segment": "think_end"},
    ]

    coords = extract_thinking_hidden_trajectory.pca_2d(hidden)
    rows = extract_thinking_hidden_trajectory.distance_rows(hidden, token_rows)

    assert coords.shape == (3, 2)
    assert rows[0]["step_l2"] == 0.0
    assert rows[1]["l2_to_prefill"] == 1.0
    assert rows[2]["cumulative_l2"] == 2.0
