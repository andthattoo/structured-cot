from __future__ import annotations

import importlib.util
import json
import sys
import time
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


def test_generation_target_balances_intent_and_compatible_language() -> None:
    intents = ["build", "automation", "debug", "design", "review", "how_to", "data_transform"]
    languages = ["python", "javascript", "typescript", "cpp", "shell", "mixed", "none"]

    targets = [
        generate_persona_pi_tasks.generation_target(index, intents=intents, languages=languages)
        for index in range(1, 8)
    ]

    assert [target.intent for target in targets] == intents
    assert [target.needs_workspace for target in targets] == [True, True, False, False, False, False, True]
    assert targets[0].language == "python"
    assert targets[1].language == "shell"
    assert targets[2].language == "typescript"
    assert targets[3].language == "cpp"
    assert targets[4].language == "mixed"
    assert targets[5].language == "none"
    assert targets[6].language == "python"


def test_generation_messages_include_target_contract() -> None:
    persona = generate_persona_pi_tasks.PersonaRecord(persona_id="abc123", row=persona_row())
    target = generate_persona_pi_tasks.TaskTarget(intent="review", language="mixed", needs_workspace=False)

    messages = generate_persona_pi_tasks.generation_messages(
        persona,
        intents=["review"],
        languages=["mixed"],
        target=target,
    )
    payload = json.loads(messages[1]["content"])

    assert payload["target"] == {
        "intent": "review",
        "language": "mixed",
        "needs_workspace": False,
    }
    assert "Set intent exactly to review." in payload["style_rules"]


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
    assert "Lifelong humanities research" not in task["user_request"]
    assert len(task["user_request"]) < 450
    assert not generate_persona_pi_tasks.generated_task_quality_errors(task)


def test_persona_focus_maps_biography_to_domain_phrase() -> None:
    focus = generate_persona_pi_tasks.persona_focus(
        {
            "skills_and_expertise": (
                "Yang Yoshiaki Skubis is a licensed civil engineer with a graduate STEM education "
                "and eight years of experience on both public and private infrastructure projects."
            )
        },
        occupation="civil engineer",
    )

    assert focus == "construction material quantities and inspection notes"
    assert "Yang" not in focus
    assert "Skubis" not in focus


def test_quality_gate_rejects_low_signal_keyword_triples() -> None:
    errors = generate_persona_pi_tasks.generated_task_quality_errors(
        {
            "intent": "debug",
            "language": "typescript",
            "needs_workspace": False,
            "user_request": "I need help debugging a small typescript script that processes arts, humanities, and equipped.",
            "expected_artifacts": [],
            "verify_commands": [],
        }
    )

    assert any("low-signal keyword triple" in error for error in errors)


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
            "balance_targets": True,
            "concurrency": 1,
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
    assert generate_persona_pi_tasks.fallback_count(rows) == 1
    assert generate_persona_pi_tasks.fallback_rate(rows) == 1.0


def test_enforce_fallback_rate_rejects_bad_batches() -> None:
    rows = [
        {"task_generation_fallback": True},
        {"task_generation_fallback": False},
    ]

    try:
        generate_persona_pi_tasks.enforce_fallback_rate(rows, max_fallback_rate=0.25)
    except RuntimeError as exc:
        assert "fallback rate" in str(exc)
    else:
        raise AssertionError("fallback rate guard should reject excessive fallback rows")


def test_enforce_fallback_rate_accepts_none() -> None:
    generate_persona_pi_tasks.enforce_fallback_rate(
        [{"task_generation_fallback": True}],
        max_fallback_rate=None,
    )


def test_validate_model_name_rejects_placeholder() -> None:
    try:
        generate_persona_pi_tasks.validate_model_name("<your-alt-model>")
    except SystemExit as exc:
        assert "replace the placeholder" in str(exc)
    else:
        raise AssertionError("placeholder model id should be rejected")


def test_generate_rows_dry_run_uses_balanced_targets(tmp_path: Path) -> None:
    personas = [
        generate_persona_pi_tasks.PersonaRecord(persona_id=f"abc{index}", row=persona_row())
        for index in range(7)
    ]
    args = type(
        "Args",
        (),
        {
            "intents": "build,automation,debug,design,review,how_to,data_transform",
            "languages": "python,javascript,typescript,cpp,shell,mixed,none",
            "provider": "openrouter",
            "dry_run": True,
            "api_key_env": "OPENROUTER_API_KEY",
            "retries": 0,
            "fallback_on_error": True,
            "model": "test-model",
            "temperature": 0.1,
            "max_tokens": 100,
            "no_json_mode": False,
            "structured_output": True,
            "balance_targets": True,
            "concurrency": 1,
            "max_fallback_rate": 1.0,
            "request_timeout_sec": 1.0,
            "retry_sleep_sec": 0.0,
            "start_index": 0,
            "root_dir": tmp_path,
            "source": "test",
        },
    )()

    rows = generate_persona_pi_tasks.generate_rows(args, personas)

    assert [row["intent"] for row in rows] == [
        "build",
        "automation",
        "debug",
        "design",
        "review",
        "how_to",
        "data_transform",
    ]
    assert [row["language"] for row in rows] == [
        "python",
        "shell",
        "typescript",
        "cpp",
        "mixed",
        "none",
        "python",
    ]


def test_generate_rows_parallel_preserves_output_order(tmp_path: Path, monkeypatch) -> None:
    personas = [
        generate_persona_pi_tasks.PersonaRecord(persona_id=f"abc{index}", row=persona_row())
        for index in range(4)
    ]

    def fake_openrouter_chat(**kwargs):
        target = json.loads(kwargs["messages"][1]["content"])["target"]
        if target["intent"] == "build":
            time.sleep(0.03)
        expected_artifacts = ["README.md"] if target["needs_workspace"] else []
        request = (
            "Please build a small local tool with README and tests."
            if target["needs_workspace"]
            else "I need a concrete coding-agent plan with assumptions and a small example."
        )
        return json.dumps(
            {
                "title": f"{target['intent']} task",
                "intent": target["intent"],
                "language": target["language"],
                "needs_workspace": target["needs_workspace"],
                "user_request": request,
                "expected_artifacts": expected_artifacts,
                "verify_commands": [],
                "difficulty": "easy",
            }
        )

    monkeypatch.setattr(generate_persona_pi_tasks, "openrouter_chat", fake_openrouter_chat)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    args = type(
        "Args",
        (),
        {
            "intents": "build,automation,debug,design",
            "languages": "python,javascript,typescript,cpp,shell,mixed,none",
            "provider": "openrouter",
            "dry_run": False,
            "api_key_env": "OPENROUTER_API_KEY",
            "retries": 0,
            "fallback_on_error": False,
            "model": "test-model",
            "temperature": 0.1,
            "max_tokens": 100,
            "no_json_mode": False,
            "structured_output": True,
            "balance_targets": True,
            "concurrency": 4,
            "request_timeout_sec": 1.0,
            "retry_sleep_sec": 0.0,
            "start_index": 0,
            "root_dir": tmp_path,
            "source": "test",
        },
    )()

    rows = generate_persona_pi_tasks.generate_rows(args, personas)

    assert [row["intent"] for row in rows] == ["build", "automation", "debug", "design"]
    assert [row["language"] for row in rows] == ["python", "shell", "typescript", "cpp"]
    assert [row["task_id"].split("_")[1] for row in rows] == ["000001", "000002", "000003", "000004"]


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
