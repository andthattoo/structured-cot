import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_strict_dsl_sft.py"
SPEC = importlib.util.spec_from_file_location("prepare_strict_dsl_sft", SCRIPT)
strict = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = strict
SPEC.loader.exec_module(strict)


def _toy_row():
    return {
        "id": "toy",
        "is_resolved": True,
        "messages": [
            {"role": "user", "content": "Create hello.txt"},
            {
                "role": "assistant",
                "content": (
                    "I should create the file.\n</think>\n"
                    '<tool_call>{"name":"run_shell","arguments":{"command":'
                    '"echo \\"Hello, world!\\" > hello.txt"}}</tool_call>'
                ),
            },
            {"role": "tool", "content": "ok"},
            {
                "role": "assistant",
                "content": (
                    "Now verify the contents.\n</think>\n"
                    '<tool_call>{"name":"run_shell","arguments":{"command":'
                    '"cat hello.txt"}}</tool_call>'
                ),
            },
            {"role": "tool", "content": "Hello, world!\n"},
            {
                "role": "assistant",
                "content": (
                    "Done.\n</think>\n"
                    "<tool_call>\n<function=finish>\n<parameter=summary>\n"
                    "Created hello.txt.\n</parameter>\n</function>\n</tool_call>"
                ),
            },
        ],
    }


def test_strict_preserved_row_rewrites_every_assistant_turn_to_dsl_qwen_xml():
    rows = strict.convert_row(
        _toy_row(),
        variants={"preserved"},
        tool_format="qwen_xml",
        add_prompt=True,
        include_text=True,
        min_assistant_turns=1,
    )

    assert len(rows) == 1
    row = rows[0]
    assistant_messages = [m for m in row["messages"] if m["role"] == "assistant"]
    assert len(assistant_messages) == 3
    for message in assistant_messages:
        assert message["content"].startswith("<think>\nPLAN: ")
        assert "<function=" in message["content"]
    assert "<function=run_shell>" in assistant_messages[0]["content"]
    assert "<function=finish>" in assistant_messages[-1]["content"]
    assert row["dsl_stats"]["variant"] == "preserved"
    assert row["dsl_stats"]["assistant_targets"] == 3
    assert "Qwen XML tool-call format" in row["messages"][0]["content"]


def test_stripped_variants_emit_one_row_per_target_with_prior_think_removed():
    rows = strict.convert_row(
        _toy_row(),
        variants={"strip_prior_think", "action_only_prior"},
        tool_format="qwen_xml",
        add_prompt=False,
        include_text=False,
        min_assistant_turns=1,
    )

    assert len(rows) == 6
    second_target_rows = [
        row
        for row in rows
        if row["dsl_stats"]["target_message_index"] == 3
    ]
    assert {row["dsl_stats"]["variant"] for row in second_target_rows} == {
        "strip_prior_think",
        "action_only_prior",
    }
    for row in second_target_rows:
        prior_assistant = row["messages"][1]["content"]
        target_assistant = row["messages"][-1]["content"]
        assert "<think>" not in prior_assistant
        assert target_assistant.startswith("<think>\nPLAN: ")
    action_only = [
        row
        for row in second_target_rows
        if row["dsl_stats"]["variant"] == "action_only_prior"
    ][0]
    assert action_only["messages"][1]["content"].startswith("<tool_call>")


def test_cli_writes_jsonl_summary(tmp_path, capsys):
    input_path = tmp_path / "rows.jsonl"
    output_path = tmp_path / "strict.jsonl"
    input_path.write_text(json.dumps(_toy_row()) + "\n")

    old_argv = sys.argv
    try:
        sys.argv = [
            "prepare_strict_dsl_sft.py",
            "--input",
            str(input_path),
            "--out",
            str(output_path),
            "--variants",
            "preserved",
        ]
        strict.main()
    finally:
        sys.argv = old_argv

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(rows) == 1
    captured = capsys.readouterr()
    assert "output_rows_written" in captured.err
