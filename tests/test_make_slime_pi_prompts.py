from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "make_slime_pi_prompts.py"
SPEC = importlib.util.spec_from_file_location("make_slime_pi_prompts", SCRIPT_PATH)
assert SPEC is not None
make_slime_pi_prompts = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = make_slime_pi_prompts
SPEC.loader.exec_module(make_slime_pi_prompts)


def test_convert_tasks_writes_prompt_messages_and_metadata(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps(
            {
                "task_id": "repo__smoke",
                "prompt": "Inspect the repo.",
                "cwd": "/tmp/repo",
                "verify_commands": ["pytest -q"],
                "source": "etpi_public_repos_v1",
            }
        )
        + "\n"
    )

    rows = make_slime_pi_prompts.convert_tasks(tasks, system_prompt="SYSTEM")

    assert len(rows) == 1
    row = rows[0]
    assert row["prompt"] == "SYSTEM\n\nTask:\nInspect the repo."
    assert row["messages"] == [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "Inspect the repo."},
    ]
    assert row["metadata"]["task_id"] == "repo__smoke"
    assert row["metadata"]["cwd"] == "/tmp/repo"
    assert row["metadata"]["verify_commands"] == ["pytest -q"]
    assert row["metadata"]["source"] == "etpi_public_repos_v1"


def test_main_writes_jsonl(tmp_path: Path, capsys) -> None:
    tasks = tmp_path / "tasks.jsonl"
    out = tmp_path / "slime.jsonl"
    tasks.write_text(json.dumps({"instruction": "Do the work."}) + "\n")

    rc = make_slime_pi_prompts.main(["--tasks", str(tasks), "--out", str(out)])  # type: ignore[arg-type]

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"] == 1
    assert out.exists()
    row = json.loads(out.read_text())
    assert row["metadata"]["task_id"] == "task_00001"
