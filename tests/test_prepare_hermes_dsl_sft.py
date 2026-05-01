import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_hermes_dsl_sft.py"
SPEC = importlib.util.spec_from_file_location("prepare_hermes_dsl_sft", SCRIPT)
prepare = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)


def test_convert_tool_calls_to_qwen_xml_from_hermes_json():
    value = (
        "<think>\nPLAN: seq(observe,act,verify,finish)\nSTATE: need_action\n"
        "RISK: none\nNEXT: tool_call\n</think>\n"
        '<tool_call>{"name":"run_shell","arguments":{"command":"printf '
        "'Hello, world!\\n' > hello.txt\",\"max_timeout_sec\":5}}</tool_call>"
    )

    converted, count = prepare.convert_tool_calls_to_qwen_xml(value)

    assert count == 1
    assert "<function=run_shell>" in converted
    assert "<parameter=command>" in converted
    assert "printf 'Hello, world!" in converted
    assert "' > hello.txt" in converted
    assert "<parameter=max_timeout_sec>\n5\n</parameter>" in converted
    assert '"name":"run_shell"' not in converted


def test_convert_row_can_emit_qwen_xml_tool_format():
    row = {
        "id": "toy",
        "conversations": [
            {
                "from": "human",
                "value": "Create hello.txt",
            },
            {
                "from": "gpt",
                "value": (
                    "<think>I should write the file.</think>\n"
                    '<tool_call>{"name":"run_shell","arguments":{"command":"printf '
                    "'Hello, world!\\n' > hello.txt\"}}</tool_call>"
                ),
            },
        ],
    }

    converted = prepare.convert_row(
        row,
        add_system_prompt=True,
        include_text=True,
        strip_prose_before_tool=True,
        tool_format="qwen_xml",
    )

    assistant = converted["messages"][-1]["content"]
    assert "<function=run_shell>" in assistant
    assert "<parameter=command>" in assistant
    assert converted["dsl_stats"]["tool_format"] == "qwen_xml"
    assert converted["dsl_stats"]["converted_tool_calls"] == 1
    assert "Qwen XML tool-call format" in converted["messages"][0]["content"]
