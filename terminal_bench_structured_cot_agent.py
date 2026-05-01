#!/usr/bin/env python3
"""Terminal-Bench agent for local structured-CoT experiments.

The agent talks to an OpenAI-compatible chat endpoint and exposes the tmux
terminal as two tools: run_shell and finish. With patched llama.cpp, the
grammar modes apply small GBNF grammars only to reasoning_content, then let
llama.cpp's normal lazy tool grammar parse shell tool calls.
"""

from __future__ import annotations

import base64
import json
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from terminal_bench.agents.base_agent import AgentResult, BaseAgent
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.terminal.tmux_session import TmuxSession


REASONING_INNER_GRAMMAR = r'''
root ::= "STEP: " step "\n"
step ::= "inspect" | "edit" | "test" | "verify" | "finish"
'''


STEP_STATUS_GRAMMAR = r'''
root ::= "STEP: " step "\n" "STATUS: " status "\n"
step ::= "inspect" | "edit" | "test" | "debug" | "verify" | "finish"
status ::= "unknown" | "in_progress" | "blocked" | "failing" | "passed" | "done"
'''


PHASE_GRAMMAR = r'''
root ::= "PHASE: " phase "\n" "CHECK: " check "\n" "NEXT: " next "\n"
phase ::= "inspect" | "plan" | "edit" | "test" | "debug" | "verify" | "finish"
check ::= "unknown" | "file_state" | "test_failure" | "command_output" | "requirements" | "done"
next ::= "inspect" | "edit" | "run_tests" | "debug" | "verify" | "finish"
'''


DSL_GRAMMAR = r'''
root ::= "PLAN: " plan "\n" "STATE: " state "\n" "RISK: " risk "\n" "NEXT: " next "\n"
plan ::= "seq(inspect,act,verify,finish)" | "seq(inspect,edit,verify,finish)" | "seq(inspect,test,debug,verify,finish)" | "fallback(verify,debug,finish)"
state ::= "need_context" | "need_action" | "need_fix" | "need_verify" | "blocked" | "ready"
risk ::= "none" | "missing_context" | "bad_tool_args" | "wrong_target" | "test_failure" | "premature_finish" | "repeat_loop"
next ::= "run_shell" | "finish"
'''


DSL_REASONING_RE = re.compile(
    r"\APLAN: "
    r"(seq\(inspect,act,verify,finish\)|"
    r"seq\(inspect,edit,verify,finish\)|"
    r"seq\(inspect,test,debug,verify,finish\)|"
    r"fallback\(verify,debug,finish\))\n"
    r"STATE: "
    r"(need_context|need_action|need_fix|need_verify|blocked|ready)\n"
    r"RISK: "
    r"(none|missing_context|bad_tool_args|wrong_target|test_failure|"
    r"premature_finish|repeat_loop)\n"
    r"NEXT: "
    r"(run_shell|finish)\n?\Z"
)


TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(.*?)(?:\s*</tool_call>|\s*\Z)",
    re.DOTALL | re.IGNORECASE,
)
TOOL_NAME_RE = re.compile(
    r'"name"\s*:\s*"(?P<json_name>[A-Za-z_][\w-]*)"|'
    r'"function=(?P<quoted_function>[A-Za-z_][\w-]*)"|'
    r"function\s*=\s*(?P<bare_function>[A-Za-z_][\w-]*)|"
    r"<function\s*=\s*(?P<tag_function>[A-Za-z_][\w-]*)",
    re.IGNORECASE,
)
TOOL_VALUE_RE = re.compile(
    r'"(?P<key>command|cmd|keystrokes|summary|final_answer)"\s*:\s*'
    r'(?P<value>"(?:\\.|[^"\\])*"|.*?)(?=,\s*"[A-Za-z_][\w-]*"\s*:|\}\s*\}?|</(?:parameter|function|tool_call)>|\Z)',
    re.DOTALL | re.IGNORECASE,
)
XML_PARAMETER_RE = re.compile(
    r"<parameter\s*=\s*\"?(?P<key>command|cmd|keystrokes|summary|final_answer)\"?\s*>\s*"
    r"(?P<value>.*?)(?:\s*</parameter>|\s*</function>|\s*</tool_call>|\s*\Z)",
    re.DOTALL | re.IGNORECASE,
)


RUN_SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "run_shell",
        "description": "Run one non-interactive shell command in the task terminal.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "A single shell command to execute.",
                },
                "max_timeout_sec": {
                    "type": "number",
                    "description": "Optional command timeout in seconds.",
                },
            },
            "required": ["command"],
        },
    },
}


FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "End the task when the requested work is complete.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of what was completed.",
                }
            },
            "required": ["summary"],
        },
    },
}


class StructuredCotTerminalAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "structured-cot-terminal"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str | None = None,
        grammar_mode: str = "none",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_turns: int = 40,
        request_timeout_sec: int = 300,
        command_timeout_sec: float = 180.0,
        observation_chars: int = 12000,
        tool_mode: str = "native",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.grammar_mode = grammar_mode
        if tool_mode not in {"native", "text"}:
            raise ValueError("tool_mode must be 'native' or 'text'")
        self.tool_mode = tool_mode
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.max_turns = int(max_turns)
        self.request_timeout_sec = int(request_timeout_sec)
        self.command_timeout_sec = float(command_timeout_sec)
        self.observation_chars = int(observation_chars)
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _system_prompt(self) -> str:
        prompt = (
            "You are solving a Terminal-Bench task in a Linux shell. "
            "Use run_shell to inspect files, edit files, run tests, and "
            "verify your work. Use one non-interactive shell command per "
            "tool call. Use finish only after the task is complete."
        )
        if self.tool_mode == "text":
            prompt += (
                " Emit tool calls as plain text using exactly this format: "
                "<tool_call>{\"name\":\"run_shell\",\"arguments\":{\"command\":\"...\"}}</tool_call>. "
                "When the task is complete, emit "
                "<tool_call>{\"name\":\"finish\",\"arguments\":{\"summary\":\"...\"}}</tool_call>. "
                "Do not wrap tool calls in markdown."
            )
        if self.grammar_mode == "dsl":
            prompt += (
                " Your reasoning is constrained to a compact PLAN/STATE/RISK/NEXT "
                "DSL. Treat the DSL as a decision record for the next tool call, "
                "not as decorative filler. Use active states and NEXT: run_shell "
                "while more evidence, edits, tests, or verification are needed. "
                "Use RISK: premature_finish when completion has not been verified. "
                "Use RISK: repeat_loop when you are about to repeat a prior check; "
                "then choose a different useful command or finish if the prior "
                "check already proved success. Use STATE: ready, RISK: none, and "
                "NEXT: finish only after command output or tests confirm the "
                "requested task is complete."
            )
        return prompt

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer local",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.request_timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))

    def _get_default_model(self) -> str:
        if self.model:
            return self.model

        req = urllib.request.Request(
            self.base_url + "/models",
            headers={"Authorization": "Bearer local"},
        )
        with urllib.request.urlopen(req, timeout=self.request_timeout_sec) as response:
            data = json.loads(response.read().decode("utf-8"))
        models = [item.get("id") for item in data.get("data", []) if item.get("id")]
        if not models:
            raise RuntimeError("No models returned by server /v1/models")
        return models[0]

    def _make_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._get_default_model(),
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.tool_mode == "native":
            payload["tools"] = [RUN_SHELL_TOOL, FINISH_TOOL]
            payload["tool_choice"] = "auto"
        if self.grammar_mode == "reasoning":
            payload["grammar"] = REASONING_INNER_GRAMMAR
        elif self.grammar_mode == "step_status":
            payload["grammar"] = STEP_STATUS_GRAMMAR
        elif self.grammar_mode == "phase":
            payload["grammar"] = PHASE_GRAMMAR
        elif self.grammar_mode == "dsl":
            payload["grammar"] = DSL_GRAMMAR
        elif self.grammar_mode not in {"none", "free"}:
            raise ValueError(
                "grammar_mode must be 'none', 'free', 'reasoning', "
                "'step_status', 'phase', or 'dsl', "
                f"got {self.grammar_mode!r}"
            )
        return payload

    def _record_usage(self, response: dict[str, Any]) -> None:
        usage = response.get("usage") or {}
        self.total_input_tokens += int(usage.get("prompt_tokens") or 0)
        self.total_output_tokens += int(usage.get("completion_tokens") or 0)

    def _write_jsonl(
        self,
        logging_dir: Path | None,
        name: str,
        record: dict[str, Any],
    ) -> None:
        if logging_dir is None:
            return
        path = logging_dir / name
        with path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _tool_result_message(
        self,
        tool_call_id: str,
        payload: dict[str, Any],
    ) -> dict[str, str]:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(payload, ensure_ascii=False),
        }

    def _reasoning_content(self, message: dict[str, Any]) -> str:
        reasoning = message.get("reasoning_content")
        if reasoning is None:
            reasoning = message.get("reasoning")
        if reasoning is None:
            return ""
        return str(reasoning)

    def _coerce_tool_args(self, args: Any) -> dict[str, Any]:
        if isinstance(args, dict):
            return dict(args)
        if isinstance(args, str):
            try:
                loaded = json.loads(args)
            except json.JSONDecodeError:
                return {"command": args}
            if isinstance(loaded, dict):
                return dict(loaded)
        return {}

    def _decode_tool_value(self, value: str) -> str:
        value = value.strip()
        if value.startswith('"'):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                decoded = value.strip('"')
            return str(decoded).strip()
        value = re.split(r"</(?:parameter|function|tool_call)>", value, maxsplit=1, flags=re.IGNORECASE)[0]
        return value.strip().strip('"').strip()

    def _recover_malformed_tool_call(self, text: str, index: int) -> dict[str, Any] | None:
        name = ""
        name_match = TOOL_NAME_RE.search(text)
        if name_match:
            name = next(group for group in name_match.groups() if group)
        lowered = text.lower()
        if not name:
            if "task_complete" in lowered or "finish" in lowered:
                name = "finish"
            elif "run_shell" in lowered or '"command"' in lowered or '"keystrokes"' in lowered:
                name = "run_shell"

        values: dict[str, str] = {}
        for match in TOOL_VALUE_RE.finditer(text):
            values[match.group("key").lower()] = self._decode_tool_value(match.group("value"))
        for match in XML_PARAMETER_RE.finditer(text):
            values[match.group("key").lower()] = self._decode_tool_value(match.group("value"))

        if name == "run_shell":
            command = values.get("command") or values.get("cmd") or values.get("keystrokes")
            if not command:
                return None
            return self._normalize_tool_call(
                {"name": "run_shell", "arguments": {"command": command}},
                index,
            )
        if name == "finish":
            summary = values.get("summary") or values.get("final_answer") or "Task complete."
            return self._normalize_tool_call(
                {"name": "finish", "arguments": {"summary": summary}},
                index,
            )
        return None

    def _normalize_tool_call(self, raw: dict[str, Any], index: int) -> dict[str, Any] | None:
        function = raw.get("function") if isinstance(raw.get("function"), dict) else None
        name = str(raw.get("name") or raw.get("tool") or (function or {}).get("name") or "")
        args = self._coerce_tool_args(raw.get("arguments") if "arguments" in raw else (function or {}).get("arguments"))

        if not name and "command" in args:
            name = "run_shell"

        shell_aliases = {"terminal", "shell", "bash", "run_command", "run_shell"}
        if name in shell_aliases:
            command = args.get("command") or args.get("cmd") or args.get("keystrokes")
            timeout = args.get("max_timeout_sec")
            if isinstance(command, list):
                command = "\n".join(str(item).rstrip("\n") for item in command)
            if command is None and isinstance(args.get("commands"), list):
                command = "\n".join(
                    str(item.get("keystrokes") or item.get("command") or "").rstrip("\n")
                    for item in args["commands"]
                    if isinstance(item, dict)
                )
            args = {"command": str(command or "").strip()}
            if timeout is not None:
                args["max_timeout_sec"] = timeout
            name = "run_shell"
        elif name == "finish" or bool(args.get("task_complete")):
            name = "finish"
            args = {
                "summary": str(
                    args.get("summary")
                    or args.get("final_answer")
                    or args.get("analysis")
                    or "Task complete."
                )
            }
        else:
            return None

        return {
            "id": str(raw.get("id") or f"fallback_tool_call_{index}"),
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        }

    def _fallback_tool_calls_from_content(
        self,
        content: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        tool_calls: list[dict[str, Any]] = []

        for match in TOOL_CALL_BLOCK_RE.finditer(content):
            raw_text = match.group(1).strip()
            try:
                raw = json.loads(raw_text)
            except json.JSONDecodeError:
                call = self._recover_malformed_tool_call(raw_text, len(tool_calls))
                if call is not None:
                    tool_calls.append(call)
                continue
            if isinstance(raw, dict):
                call = self._normalize_tool_call(raw, len(tool_calls))
                if call is not None:
                    tool_calls.append(call)

        if tool_calls:
            return TOOL_CALL_BLOCK_RE.sub("", content).strip(), tool_calls

        stripped = content.strip()
        if not stripped.startswith("{"):
            return content, []
        try:
            raw_json = json.loads(stripped)
        except json.JSONDecodeError:
            return content, []
        if not isinstance(raw_json, dict):
            return content, []

        if isinstance(raw_json.get("commands"), list):
            for command_item in raw_json["commands"]:
                if not isinstance(command_item, dict):
                    continue
                command = str(command_item.get("keystrokes") or command_item.get("command") or "").strip()
                if not command:
                    continue
                call = self._normalize_tool_call(
                    {"name": "run_shell", "arguments": {"command": command}},
                    len(tool_calls),
                )
                if call is not None:
                    tool_calls.append(call)
            if tool_calls:
                return "", tool_calls

        call = self._normalize_tool_call(raw_json, 0)
        if call is None:
            return content, []
        return "", [call]

    def _shell_syntax_error(self, command: str) -> str | None:
        try:
            completed = subprocess.run(
                ["bash", "-n", "-c", command],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except FileNotFoundError:
            return None
        except subprocess.TimeoutExpired:
            return "shell syntax preflight timed out"

        if completed.returncode == 0:
            return None
        message = (completed.stderr or completed.stdout or "").strip()
        return message or f"bash -n exited with status {completed.returncode}"

    def _tmux_safe_command(self, command: str) -> str:
        """Encode multiline commands so tmux's wait marker cannot corrupt them."""
        command = command.replace("\r\n", "\n").replace("\r", "\n")
        if "\n" not in command:
            return command

        encoded = base64.b64encode(command.encode("utf-8")).decode("ascii")
        script_path = f"/tmp/structured_cot_cmd_{uuid.uuid4().hex}.sh"
        quoted_path = shlex.quote(script_path)
        return (
            f"printf %s {shlex.quote(encoded)} | base64 -d > {quoted_path}; "
            f"bash {quoted_path}; rc=$?; rm -f {quoted_path}; test $rc -eq 0"
        )

    def _run_shell(
        self,
        session: TmuxSession,
        command: str,
        max_timeout_sec: float,
    ) -> dict[str, Any]:
        started = time.time()
        syntax_error = self._shell_syntax_error(command)
        if syntax_error is not None:
            return {
                "status": "error",
                "elapsed_sec": round(time.time() - started, 3),
                "command": command,
                "error": f"shell syntax preflight failed: {syntax_error}",
                "output": session.capture_pane(),
            }

        try:
            tmux_command = self._tmux_safe_command(command)
            session.send_keys(
                [tmux_command, "Enter"],
                block=True,
                max_timeout_sec=max_timeout_sec,
            )
            status = "ok"
            error = None
        except TimeoutError as exc:
            status = "timeout"
            error = str(exc)
        except Exception as exc:
            status = "error"
            error = repr(exc)

        output = session.get_incremental_output()
        if len(output) > self.observation_chars:
            output = output[-self.observation_chars :]

        return {
            "status": status,
            "elapsed_sec": round(time.time() - started, 3),
            "command": command,
            "error": error,
            "output": output,
        }

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        if logging_dir is not None:
            logging_dir.mkdir(parents=True, exist_ok=True)
            (logging_dir / "instruction.txt").write_text(instruction)

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": self._system_prompt(),
            },
            {
                "role": "user",
                "content": f"Task:\n{self._render_instruction(instruction)}",
            },
        ]

        try:
            session.get_incremental_output()
        except Exception:
            pass

        for turn in range(1, self.max_turns + 1):
            payload = self._make_payload(messages)
            self._write_jsonl(
                logging_dir,
                "requests.jsonl",
                {"turn": turn, "payload": payload},
            )

            try:
                response = self._post_json("/chat/completions", payload)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                self._write_jsonl(
                    logging_dir,
                    "errors.jsonl",
                    {"turn": turn, "error": body},
                )
                return AgentResult(
                    total_input_tokens=self.total_input_tokens,
                    total_output_tokens=self.total_output_tokens,
                    failure_mode=FailureMode.UNKNOWN_AGENT_ERROR,
                )

            self._record_usage(response)
            self._write_jsonl(
                logging_dir,
                "responses.jsonl",
                {"turn": turn, "response": response},
            )

            choices = response.get("choices") or []
            if not choices:
                return AgentResult(
                    total_input_tokens=self.total_input_tokens,
                    total_output_tokens=self.total_output_tokens,
                    failure_mode=FailureMode.UNKNOWN_AGENT_ERROR,
                )

            message = choices[0].get("message") or {}
            tool_calls = message.get("tool_calls") or []
            content = message.get("content") or ""
            if not tool_calls:
                content, tool_calls = self._fallback_tool_calls_from_content(str(content or ""))
            if self.grammar_mode == "dsl":
                reasoning = self._reasoning_content(message)
                if not DSL_REASONING_RE.fullmatch(reasoning):
                    self._write_jsonl(
                        logging_dir,
                        "grammar_violations.jsonl",
                        {
                            "turn": turn,
                            "grammar_mode": self.grammar_mode,
                            "reasoning_content": reasoning,
                        },
                    )

            if not tool_calls:
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": "Call run_shell for the next command, or finish if done.",
                    }
                )
                continue

            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                }
            )

            for tool_call in tool_calls:
                function = tool_call.get("function") or {}
                name = function.get("name")
                raw_args = function.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}

                tool_call_id = tool_call.get("id") or f"call_{turn}"

                if name == "finish":
                    messages.append(
                        self._tool_result_message(
                            tool_call_id,
                            {"status": "ok", "summary": args.get("summary", "")},
                        )
                    )
                    self._write_jsonl(
                        logging_dir,
                        "final.jsonl",
                        {"turn": turn, "messages": messages[-3:]},
                    )
                    return AgentResult(
                        total_input_tokens=self.total_input_tokens,
                        total_output_tokens=self.total_output_tokens,
                        failure_mode=FailureMode.NONE,
                    )

                if name != "run_shell":
                    result = {
                        "status": "error",
                        "error": f"unknown tool: {name}",
                    }
                else:
                    command = str(args.get("command") or "").strip()
                    timeout = float(
                        args.get("max_timeout_sec") or self.command_timeout_sec
                    )
                    if not command:
                        result = {
                            "status": "error",
                            "error": "empty command",
                            "output": session.capture_pane(),
                        }
                    else:
                        result = self._run_shell(session, command, timeout)

                messages.append(self._tool_result_message(tool_call_id, result))

        return AgentResult(
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            failure_mode=FailureMode.AGENT_TIMEOUT,
        )
