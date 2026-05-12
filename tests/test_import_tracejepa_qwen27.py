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

    assert stats == {"scanned": 2, "selected": 1, "skipped": 1}
    assert selected[0]["task_id"] == "big"
    assert json.loads(selected[0]["trajectory_json"]) == [{"role": "user"}]


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
