from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "summarize_pi_trace.py"
SPEC = importlib.util.spec_from_file_location("summarize_pi_trace", SCRIPT_PATH)
assert SPEC is not None
summarize_pi_trace = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = summarize_pi_trace
SPEC.loader.exec_module(summarize_pi_trace)


def test_summarize_pi_trace_counts_assistant_content_and_errors(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    rows = [
        {"type": "session", "id": "s"},
        {
            "type": "message",
            "id": "u1",
            "message": {"role": "user", "content": [{"type": "text", "text": "Task"}]},
        },
        {
            "type": "message",
            "id": "a1",
            "message": {
                "role": "assistant",
                "stopReason": "toolUse",
                "content": [
                    {"type": "thinking", "thinking": "Need context."},
                    {"type": "toolCall", "name": "read", "arguments": {"path": "README.md"}},
                ],
                "usage": {"input": 10, "output": 5, "totalTokens": 15, "cost": {"total": 0.01}},
            },
        },
        {
            "type": "message",
            "id": "a2",
            "message": {
                "role": "assistant",
                "stopReason": "error",
                "errorMessage": "401 User not found.",
                "content": [],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    summary = summarize_pi_trace.summarize(path)

    assert summary["records"] == 4
    assert summary["assistant_count"] == 2
    assert summary["tool_call_count"] == 1
    assert summary["thinking_chars"] == len("Need context.")
    assert summary["input_tokens"] == 10
    assert summary["output_tokens"] == 5
    assert summary["cost"] == 0.01
    assert summary["errors"][0]["error"] == "401 User not found."
