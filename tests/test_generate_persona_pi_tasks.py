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


def test_default_persona_generator_model_is_nemotron() -> None:
    assert (
        generate_persona_pi_tasks.DEFAULT_MODEL
        == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    )


def test_progress_falls_back_without_tqdm(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "tqdm.auto", None)
    values = [1, 2, 3]

    assert list(generate_persona_pi_tasks.progress(values, total=3)) == values


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


def test_extract_json_object_rejects_none() -> None:
    try:
        generate_persona_pi_tasks.extract_json_object(None)  # type: ignore[arg-type]
    except ValueError as exc:
        assert "must be a string" in str(exc)
        pass
    else:
        raise AssertionError("None content should fail before normalization")


def test_task_response_schema_is_strict_and_enum_bounded() -> None:
    schema = generate_persona_pi_tasks.task_response_schema(
        intents=["build", "review"],
        languages=["python", "none"],
    )

    assert schema["additionalProperties"] is False
    assert "user_request" in schema["required"]
    assert schema["properties"]["intent"]["enum"] == ["build", "review"]
    assert schema["properties"]["language"]["enum"] == ["python", "none"]


def test_openrouter_chat_sends_structured_output_schema(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "{\"title\":\"ok\"}"}}]}).encode()

    def fake_urlopen(request, *, timeout):
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode())
        return FakeResponse()

    monkeypatch.setattr(generate_persona_pi_tasks.urllib.request, "urlopen", fake_urlopen)

    schema = generate_persona_pi_tasks.task_response_schema(
        intents=["build"],
        languages=["python"],
    )
    content = generate_persona_pi_tasks.openrouter_chat(
        model="test-model",
        messages=[{"role": "user", "content": "make task"}],
        api_key="sk-test",
        temperature=0.1,
        max_tokens=100,
        json_mode=True,
        structured_schema=schema,
        timeout_sec=12.0,
    )

    assert content == "{\"title\":\"ok\"}"
    assert captured["timeout"] == 12.0
    assert captured["body"]["response_format"]["type"] == "json_schema"
    assert captured["body"]["response_format"]["json_schema"]["strict"] is True
    assert captured["body"]["response_format"]["json_schema"]["schema"] == schema


def test_openrouter_chat_can_fall_back_to_json_object_mode(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "{\"title\":\"ok\"}"}}]}).encode()

    def fake_urlopen(request, *, timeout):
        captured["body"] = json.loads(request.data.decode())
        return FakeResponse()

    monkeypatch.setattr(generate_persona_pi_tasks.urllib.request, "urlopen", fake_urlopen)

    generate_persona_pi_tasks.openrouter_chat(
        model="test-model",
        messages=[{"role": "user", "content": "make task"}],
        api_key="sk-test",
        temperature=0.1,
        max_tokens=100,
        json_mode=True,
        structured_schema=None,
        timeout_sec=12.0,
    )

    assert captured["body"]["response_format"] == {"type": "json_object"}


def test_quality_gate_rejects_persona_dump_template() -> None:
    errors = generate_persona_pi_tasks.generated_task_quality_errors(
        {
            "intent": "automation",
            "language": "none",
            "needs_workspace": True,
            "user_request": "I work as a not_in_workforce and need a small none tool related to a pasted biography.",
            "expected_artifacts": [],
            "verify_commands": [],
        }
    )

    assert errors
    assert any("low-quality phrase" in error for error in errors)
    assert any("language=none" in error for error in errors)


def test_deterministic_task_avoids_raw_persona_dump_for_not_in_workforce() -> None:
    persona = generate_persona_pi_tasks.PersonaRecord(
        persona_id="abc123",
        row={
            **persona_row(),
            "occupation": "not_in_workforce",
            "skills_and_expertise": "Lifelong humanities research, oral-history interviews, and community workshop planning.",
        },
    )

    task = generate_persona_pi_tasks.deterministic_task(
        persona,
        intents=["build"],
        languages=["none", "python"],
        index=0,
    )

    assert task["needs_workspace"] is True
    assert task["language"] == "python"
    assert "not_in_workforce" not in task["user_request"]
    assert "related to" not in task["user_request"]
    assert "none tool" not in task["user_request"]
    assert not generate_persona_pi_tasks.generated_task_quality_errors(task)


def test_generate_rows_can_fallback_on_generation_error(tmp_path: Path, monkeypatch) -> None:
    persona = generate_persona_pi_tasks.PersonaRecord(persona_id="abc123", row=persona_row())

    def fake_openrouter_chat(**kwargs):
        raise RuntimeError("empty message")

    monkeypatch.setattr(generate_persona_pi_tasks, "openrouter_chat", fake_openrouter_chat)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    args = type(
        "Args",
        (),
        {
            "intents": "build",
            "languages": "python",
            "provider": "openrouter",
            "dry_run": False,
            "api_key_env": "OPENROUTER_API_KEY",
            "retries": 0,
            "fallback_on_error": True,
            "model": "test-model",
            "temperature": 0.1,
            "max_tokens": 100,
            "no_json_mode": False,
            "structured_output": True,
            "request_timeout_sec": 1.0,
            "retry_sleep_sec": 0.0,
            "start_index": 0,
            "root_dir": tmp_path,
            "source": "test",
        },
    )()

    rows = generate_persona_pi_tasks.generate_rows(args, [persona])

    assert len(rows) == 1
    assert rows[0]["intent"] == "build"
    assert rows[0]["language"] == "python"
    assert rows[0]["task_generation_fallback"] is True
    assert "empty message" in rows[0]["task_generation_error"]


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


def test_dataset_sampling_uses_shuffled_take_without_full_scan(monkeypatch) -> None:
    class FakeStream:
        def __init__(self) -> None:
            self.iterated = 0

        def shuffle(self, *, seed: int, buffer_size: int) -> "FakeStream":
            assert seed == 7
            assert buffer_size == 10000
            return self

        def __iter__(self):
            for index in range(1_000_000):
                self.iterated += 1
                yield {"uuid": f"row-{index}", "occupation": "tester"}

    fake_stream = FakeStream()

    def fake_load_dataset(dataset: str, *, split: str, streaming: bool) -> FakeStream:
        assert dataset == "fake/personas"
        assert split == "train"
        assert streaming is True
        return fake_stream

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        type("FakeDatasets", (), {"load_dataset": staticmethod(fake_load_dataset)}),
    )

    personas = generate_persona_pi_tasks.load_personas_from_dataset(
        "fake/personas",
        split="train",
        sample_size=3,
        seed=7,
        max_source_rows=None,
        shuffle_buffer=10000,
    )

    assert len(personas) == 3
    assert fake_stream.iterated == 3
