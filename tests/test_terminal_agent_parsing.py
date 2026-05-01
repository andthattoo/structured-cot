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
