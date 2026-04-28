#!/usr/bin/env python3
"""Terminal-Bench agent for local structured-CoT experiments.

The agent talks to an OpenAI-compatible chat endpoint and exposes the tmux
terminal as two tools: run_shell and finish. With patched llama.cpp,
grammar_mode=reasoning applies a small GBNF grammar only to reasoning_content,
then lets llama.cpp's normal lazy tool grammar parse shell tool calls.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from terminal_bench.agents.base_agent import AgentResult, BaseAgent
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.terminal.tmux_session import TmuxSession


REASONING_INNER_GRAMMAR = r'''
root ::= "GOAL: " line "TOOL: " tool "\n" "WHY: " line
tool ::= "run_shell" | "finish"
line ::= [^\n]+ "\n"
'''


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
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.grammar_mode = grammar_mode
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.max_turns = int(max_turns)
        self.request_timeout_sec = int(request_timeout_sec)
        self.command_timeout_sec = float(command_timeout_sec)
        self.observation_chars = int(observation_chars)
        self.total_input_tokens = 0
        self.total_output_tokens = 0

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
            "tools": [RUN_SHELL_TOOL, FINISH_TOOL],
            "tool_choice": "auto",
        }
        if self.grammar_mode == "reasoning":
            payload["grammar"] = REASONING_INNER_GRAMMAR
        elif self.grammar_mode not in {"none", "free"}:
            raise ValueError(
                "grammar_mode must be 'none', 'free', or 'reasoning', "
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

    def _run_shell(
        self,
        session: TmuxSession,
        command: str,
        max_timeout_sec: float,
    ) -> dict[str, Any]:
        started = time.time()
        try:
            session.send_keys(
                [command, "Enter"],
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
                "content": (
                    "You are solving a Terminal-Bench task in a Linux shell. "
                    "Use run_shell to inspect files, edit files, run tests, and "
                    "verify your work. Use one non-interactive shell command per "
                    "tool call. Use finish only after the task is complete."
                ),
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
