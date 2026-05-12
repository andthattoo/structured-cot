from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "import_tracejepa_qwen27.py"
SPEC = importlib.util.spec_from_file_location("import_tracejepa_qwen27", SCRIPT_PATH)
assert SPEC is not None
import_tracejepa_qwen27 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules[SPEC.name] = import_tracejepa_qwen27
SPEC.loader.exec_module(import_tracejepa_qwen27)


def test_row_matches_qwen27_from_model_field() -> None:
    pattern = import_tracejepa_qwen27.re.compile(
        import_tracejepa_qwen27.DEFAULT_MODEL_PATTERN,
        import_tracejepa_qwen27.re.IGNORECASE,
    )

    assert import_tracejepa_qwen27.row_matches_model({"model": "Qwen/Qwen3.6-27B"}, pattern)
    assert import_tracejepa_qwen27.row_matches_model({"model": "qwen/qwen3.5-27b"}, pattern)
    assert not import_tracejepa_qwen27.row_matches_model({"model": "Qwen/Qwen2.5-Coder-7B"}, pattern)


def test_normalize_row_outputs_etpi_schema() -> None:
    row = {
        "task_id": "task-a",
        "prompt": "Do the task",
        "cwd": "/work",
        "model": "Qwen/Qwen3.6-27B",
        "trajectory": [{"type": "session"}, {"type": "message", "message": {"role": "assistant"}}],
        "metadata": {"source": "tracejepa", "language": "python", "verifiable": True},
    }

    normalized = import_tracejepa_qwen27.normalize_tracejepa_row(
        row,
        run_id="tracejepa_qwen27",
        row_index=1,
        source_dataset="driaforall/tracejepa-pi-2500-v1",
    )

    assert normalized["run_id"] == "tracejepa_qwen27"
    assert normalized["task_id"] == "task-a"
    assert normalized["prompt"] == "Do the task"
    assert normalized["cwd"] == "/work"
    assert normalized["model"] == "Qwen/Qwen3.6-27B"
    assert normalized["source"] == "tracejepa"
    assert normalized["language"] == "python"
    assert normalized["verifiable"] is True
    assert json.loads(normalized["trajectory_json"])[0]["type"] == "session"


def test_select_rows_filters_and_normalizes() -> None:
    rows = [
        {"task_id": "small", "model": "Qwen/Qwen2.5-7B", "trajectory": []},
        {"task_id": "big", "model": "openrouter/qwen3.5-27b", "messages": [{"role": "user"}]},
    ]

    selected, stats = import_tracejepa_qwen27.select_rows(
        rows,
        run_id="run",
        source_dataset="source/repo",
        model_pattern=import_tracejepa_qwen27.DEFAULT_MODEL_PATTERN,
    )

    assert stats == {"scanned": 2, "selected": 1, "skipped": 1, "duplicate_transitions": 0}
    assert selected[0]["task_id"] == "big"
    assert json.loads(selected[0]["trajectory_json"]) == [{"role": "user"}]


def test_select_rows_dedupes_transition_rows_and_converts_goal_events() -> None:
    event = {"type": "instruction", "text": "Build the thing", "event_id": "e0", "metadata": {}}
    rows = [
        {
            "task_id": "task-a",
            "trajectory_id": "traj-a",
            "model": "qwen/qwen3.6-27b",
            "goal_events": [event],
            "step_index": 0,
        },
        {
            "task_id": "task-a",
            "trajectory_id": "traj-a",
            "model": "qwen/qwen3.6-27b",
            "goal_events": [event],
            "step_index": 1,
        },
    ]

    selected, stats = import_tracejepa_qwen27.select_rows(
        rows,
        run_id="run",
        source_dataset="source/repo",
        model_pattern=import_tracejepa_qwen27.DEFAULT_MODEL_PATTERN,
    )

    assert stats["selected"] == 1
    assert stats["duplicate_transitions"] == 1
    trajectory = json.loads(selected[0]["trajectory_json"])
    assert trajectory[0]["type"] == "session"
    assert trajectory[1]["message"]["role"] == "user"
    assert trajectory[1]["message"]["content"][0]["text"] == "Build the thing"


def test_tracejepa_events_to_pi_records_maps_actions_to_tool_calls() -> None:
    records = import_tracejepa_qwen27.tracejepa_events_to_pi_records(
        [
            {"type": "instruction", "text": "Inspect.", "event_id": "u0", "metadata": {}},
            {"type": "action", "text": "ls -la", "event_id": "a0", "metadata": {"tool_name": "bash"}},
            {"type": "observation", "text": "total 0", "event_id": "o0", "metadata": {"tool_name": "bash"}},
        ],
        task_id="task-a",
    )

    assert records[1]["message"]["role"] == "user"
    assert records[2]["message"]["role"] == "assistant"
    assert records[2]["message"]["content"][0]["type"] == "toolCall"
    assert records[2]["message"]["content"][0]["arguments"] == {"command": "ls -la"}
    assert records[3]["message"]["role"] == "toolResult"


def test_main_writes_local_index_from_jsonl(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source.jsonl"
    out = tmp_path / "index.jsonl"
    source.write_text(
        json.dumps({"task_id": "one", "model": "Qwen/Qwen3.6-27B", "trajectory_json": "[]"}) + "\n"
    )

    rc = import_tracejepa_qwen27.main(
        [
            "--source-jsonl",
            str(source),
            "--out-index",
            str(out),
            "--run-id",
            "local_run",
        ]
    )

    assert rc == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["selected"] == 1
    assert out.exists()
    row = json.loads(out.read_text())
    assert row["run_id"] == "local_run"


def test_parse_args_defaults_to_transitions_config() -> None:
    args = import_tracejepa_qwen27.parse_args(
        [
            "--source-jsonl",
            "source.jsonl",
            "--out-index",
            "out.jsonl",
        ]
    )

    assert args.config == "transitions"


def test_resolve_hf_token_prefers_env(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "env-token")

    assert import_tracejepa_qwen27.resolve_hf_token("HF_TOKEN") == "env-token"
