import sys
import types


def _install_terminal_bench_stubs() -> None:
    terminal_bench = types.ModuleType("terminal_bench")
    agents = types.ModuleType("terminal_bench.agents")
    base_agent = types.ModuleType("terminal_bench.agents.base_agent")
    failure_mode = types.ModuleType("terminal_bench.agents.failure_mode")
    terminal = types.ModuleType("terminal_bench.terminal")
    tmux_session = types.ModuleType("terminal_bench.terminal.tmux_session")

    base_agent.AgentResult = object
    base_agent.BaseAgent = type("BaseAgent", (), {"__init__": lambda self, **kwargs: None})
    failure_mode.FailureMode = type("FailureMode", (), {"NONE": "none"})
    tmux_session.TmuxSession = object

    sys.modules.setdefault("terminal_bench", terminal_bench)
    sys.modules.setdefault("terminal_bench.agents", agents)
    sys.modules.setdefault("terminal_bench.agents.base_agent", base_agent)
    sys.modules.setdefault("terminal_bench.agents.failure_mode", failure_mode)
    sys.modules.setdefault("terminal_bench.terminal", terminal)
    sys.modules.setdefault("terminal_bench.terminal.tmux_session", tmux_session)


_install_terminal_bench_stubs()

from terminal_bench_structured_cot_agent import StructuredCotTerminalAgent


def test_recovers_assigned_xml_parameter_command() -> None:
    agent = StructuredCotTerminalAgent()
    content = """<tool_call>
<function=run_shell>
<parameter=command="curl -s -X POST http://api:8000/spreadsheets/ -H "Content-Type: application/json" -d '{"title": "Financial Report"}'
</parameter>
</function>
</tool_call>"""

    _, tool_calls = agent._fallback_tool_calls_from_content(content)

    assert len(tool_calls) == 1
    arguments = tool_calls[0]["function"]["arguments"]
    assert "Financial Report" in arguments
    assert "Content-Type: application/json" in arguments


def test_recovers_qwen_xml_command() -> None:
    agent = StructuredCotTerminalAgent()
    content = """<think>
PLAN: seq(observe,act,verify,finish)
STATE: need_action
RISK: none
NEXT: tool_call
</think>
<tool_call>
<function=run_shell>
<parameter=command>
printf 'Hello, world!\\n' > hello.txt
</parameter>
</function>
</tool_call>"""

    _, tool_calls = agent._fallback_tool_calls_from_content(content)

    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "run_shell"
    assert "printf" in tool_calls[0]["function"]["arguments"]


def test_qwen_xml_auto_prompt_uses_leo_contract_before_task_appendix() -> None:
    agent = StructuredCotTerminalAgent(
        model="test-model",
        tool_mode="qwen_xml",
        grammar_mode="none",
    )

    prompt = agent._system_prompt()
    payload = agent._make_payload([], rerank=False)

    assert prompt.startswith("When thinking before tool use")
    assert "PLAN: one symbolic control-flow plan" in prompt
    assert prompt.index("When calling tools, use Qwen XML") < prompt.index(
        "Terminal-Bench task-specific instructions"
    )
    assert "<function=run_shell>" in prompt
    assert "<function=finish>" in prompt
    assert "grammar" not in payload


def test_qwen_xml_can_use_default_terminal_prompt() -> None:
    agent = StructuredCotTerminalAgent(
        model="test-model",
        tool_mode="qwen_xml",
        prompt_profile="default",
    )

    prompt = agent._system_prompt()

    assert prompt.startswith("You are solving a Terminal-Bench task")
    assert "When calling tools, use Qwen XML" in prompt


def test_recovers_hybrid_finish_call() -> None:
    agent = StructuredCotTerminalAgent()
    content = """<tool_call>
<function=finish,"arguments":{"summary":"Task complete."}}
</tool_call>"""

    _, tool_calls = agent._fallback_tool_calls_from_content(content)

    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "finish"


def test_tmux_safe_command_wraps_only_multiline_commands() -> None:
    agent = StructuredCotTerminalAgent()

    assert agent._tmux_safe_command("echo ok") == "echo ok"

    wrapped = agent._tmux_safe_command("cat << EOF\nhello\nEOF")
    assert "\n" not in wrapped
    assert "base64 -d" in wrapped
    assert "structured_cot_cmd_" in wrapped


def test_mqe_payload_requests_multiple_candidates() -> None:
    agent = StructuredCotTerminalAgent(model="test-model", mqe_mode="rerank", mqe_candidates=4)

    payload = agent._make_payload([], rerank=True)

    assert payload["n"] == 4
    assert payload["temperature"] == agent.mqe_temperature
    assert payload["top_p"] == agent.mqe_top_p


def test_mqe_rerank_selects_highest_scoring_command() -> None:
    agent = StructuredCotTerminalAgent(model="test-model", mqe_mode="rerank", mqe_candidates=2)
    agent._score_mqe_actions = lambda **kwargs: [0.1, 0.9]  # type: ignore[method-assign]
    choices = [
        {
            "message": {
                "content": '<tool_call>{"name":"run_shell","arguments":{"command":"echo bad"}}</tool_call>'
            }
        },
        {
            "message": {
                "content": '<tool_call>{"name":"run_shell","arguments":{"command":"echo good"}}</tool_call>'
            }
        },
    ]

    _, _, tool_calls = agent._select_choice_with_mqe(choices, [], "task", None, 1)

    assert "echo good" in tool_calls[0]["function"]["arguments"]


def test_thinking_context_default_preserves_full_history() -> None:
    agent = StructuredCotTerminalAgent(model="test-model")
    messages = [
        {"role": "assistant", "content": "PLAN: keep\n</think>", "tool_calls": []},
    ]

    assert agent._messages_for_request(messages) is messages


def test_latest_thinking_context_keeps_only_latest_assistant_thought() -> None:
    agent = StructuredCotTerminalAgent(model="test-model", thinking_context="latest")
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "PLAN: old\nSTATE: old\n</think>",
            "tool_calls": [{"id": "old", "function": {"name": "run_shell", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "old", "content": "{}"},
        {
            "role": "assistant",
            "content": "PLAN: current\nSTATE: current\n</think>",
            "tool_calls": [{"id": "current", "function": {"name": "run_shell", "arguments": "{}"}}],
        },
    ]

    request_messages = agent._messages_for_request(messages)

    assert request_messages is not messages
    assert request_messages[2]["content"] == ""
    assert request_messages[2]["tool_calls"] == messages[2]["tool_calls"]
    assert "PLAN: current" in request_messages[4]["content"]
    assert messages[2]["content"].startswith("PLAN: old")


def test_none_thinking_context_strips_all_assistant_thoughts() -> None:
    agent = StructuredCotTerminalAgent(model="test-model", thinking_context="none")
    messages = [
        {
            "role": "assistant",
            "content": "<think>\nPLAN: first\n</think>",
            "tool_calls": [{"id": "first", "function": {"name": "run_shell", "arguments": "{}"}}],
        },
        {
            "role": "assistant",
            "content": "PLAN: second\nSTATE: second\n</think>",
            "tool_calls": [{"id": "second", "function": {"name": "run_shell", "arguments": "{}"}}],
        },
    ]

    request_messages = agent._messages_for_request(messages)

    assert [message["content"] for message in request_messages] == ["", ""]


def test_strip_thinking_content_keeps_visible_non_thought_text() -> None:
    agent = StructuredCotTerminalAgent(model="test-model", thinking_context="latest")

    content = agent._strip_thinking_content(
        "<think>\nPLAN: hidden\n</think>\nVisible answer",
        has_tool_calls=False,
    )

    assert content == "Visible answer"
