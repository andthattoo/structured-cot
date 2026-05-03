from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "generate_persona_pi_tasks.py"
SPEC = importlib.util.spec_from_file_location("generate_persona_pi_tasks", SCRIPT_PATH)
assert SPEC is not None
generate_persona_pi_tasks = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = generate_persona_pi_tasks
SPEC.loader.exec_module(generate_persona_pi_tasks)


def persona_row() -> dict[str, object]:
    return {
        "uuid": "abc123",
        "professional_persona": "A meticulous community event planner who coordinates volunteers and budgets.",
        "persona": "Practical, organized, and direct.",
        "occupation": "event_planner",
        "skills_and_expertise": ["budgeting", "scheduling", "spreadsheet cleanup"],
        "career_goals_and_ambitions": "Wants to run larger community festivals with fewer logistics mistakes.",
    }


def test_persona_brief_extracts_work_relevant_fields() -> None:
    persona = generate_persona_pi_tasks.PersonaRecord(persona_id="abc123", row=persona_row())

    brief = generate_persona_pi_tasks.persona_brief(persona)

    assert brief["persona_id"] == "abc123"
    assert "event planner" in brief["professional_persona"]
    assert "budgeting" in brief["skills_and_expertise"]


def test_extract_json_object_accepts_fenced_json() -> None:
    text = """```json
{"title":"CSV helper","intent":"build","language":"python","needs_workspace":true,"user_request":"Build a CSV helper.","expected_artifacts":["README.md"],"verify_commands":["python3 -m pytest -q"],"difficulty":"easy"}
```"""

    payload = generate_persona_pi_tasks.extract_json_object(text)

    assert payload["intent"] == "build"
    assert payload["language"] == "python"


def test_task_row_is_pi_compatible_and_empty_workspace_prompt(tmp_path: Path) -> None:
    persona = generate_persona_pi_tasks.PersonaRecord(persona_id="abc123", row=persona_row())
    generated = {
        "title": "Volunteer schedule checker",
        "intent": "build",
        "language": "python",
        "needs_workspace": True,
        "user_request": "Build a small volunteer schedule checker.",
        "expected_artifacts": ["README.md", "tests/"],
        "verify_commands": ["python3 -m pytest -q"],
        "difficulty": "easy",
    }

    row = generate_persona_pi_tasks.task_row(
        persona,
        generated,
        index=1,
        root_dir=tmp_path,
        source="test_source",
        generator_model="test-model",
        intents=["build"],
        languages=["python"],
    )

    assert row["task_id"].startswith("persona_000001_abc123_build_python")
    assert row["cwd"].startswith(str(tmp_path))
    assert "empty directory" in row["prompt"]
    assert row["source"] == "test_source"
    assert row["persona_id"] == "abc123"
    assert row["verifiable"] is True


def test_dry_run_main_writes_tasks_and_workspaces(tmp_path: Path) -> None:
    personas = tmp_path / "personas.jsonl"
    personas.write_text(json.dumps(persona_row()) + "\n")
    out = tmp_path / "tasks.jsonl"
    root_dir = tmp_path / "workspaces"

    rc = generate_persona_pi_tasks.main(
        [
            "--personas-file",
            str(personas),
            "--sample-size",
            "1",
            "--root-dir",
            str(root_dir),
            "--out",
            str(out),
            "--create-workspaces",
            "--dry-run",
        ]
    )

    assert rc == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 1
    assert Path(rows[0]["cwd"]).exists()
    assert rows[0]["prompt"]
