"""slime custom generation and reward hooks for ETPI terminal rollouts.

This module is intentionally dependency-light and duck-typed so it can be
imported in CPU-only tests without slime installed. In a slime run, pass:

    --custom-generate-function-path slime_plugins.etpi_pi_harness.generate
    --custom-rm-path slime_plugins.etpi_pi_harness.reward_func

The rollout protocol is simple:

    <bash>pytest -q</bash>
    <final>Summary of work and verification.</final>

Shell execution is disabled unless ``ETPI_SLIME_ENABLE_SHELL=1`` is set.
Use it only inside disposable rollout workspaces.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOKENIZER: Any | None = None

BASH_RE = re.compile(r"<bash>(?P<body>.*?)</bash>", re.DOTALL | re.IGNORECASE)
FINAL_RE = re.compile(r"<final>(?P<body>.*?)</final>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class ParsedAction:
    kind: str
    body: str


@dataclass(frozen=True)
class CommandResult:
    command: str
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    disabled: bool = False


def parse_action(text: str) -> ParsedAction:
    final_match = FINAL_RE.search(text)
    if final_match:
        return ParsedAction("final", final_match.group("body").strip())
    bash_match = BASH_RE.search(text)
    if bash_match:
        return ParsedAction("bash", bash_match.group("body").strip())
    return ParsedAction("final", text.strip())


def sample_metadata(sample: Any) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def command_timeout(args: Any) -> float:
    return float(getattr(args, "etpi_command_timeout_sec", 30.0))


def max_turns(args: Any) -> int:
    return int(getattr(args, "etpi_max_turns", 8))


def max_observation_chars(args: Any) -> int:
    return int(getattr(args, "etpi_max_observation_chars", 6000))


def shell_enabled() -> bool:
    return os.environ.get("ETPI_SLIME_ENABLE_SHELL") == "1"


def workspace_from_metadata(metadata: dict[str, Any]) -> Path:
    cwd = metadata.get("cwd")
    if isinstance(cwd, str) and cwd.strip():
        return Path(cwd).expanduser()
    return Path.cwd()


def truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + f"\n...[truncated {omitted} chars]"


def run_bash(command: str, *, cwd: Path, timeout_sec: float, char_limit: int) -> CommandResult:
    if not shell_enabled():
        return CommandResult(
            command=command,
            returncode=126,
            stdout="",
            stderr="Shell execution disabled. Set ETPI_SLIME_ENABLE_SHELL=1 in the rollout container.",
            disabled=True,
        )
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            shell=True,
            executable="/bin/bash",
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
        return CommandResult(
            command=command,
            returncode=proc.returncode,
            stdout=truncate_text(proc.stdout, char_limit),
            stderr=truncate_text(proc.stderr, char_limit),
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            command=command,
            returncode=124,
            stdout=truncate_text(exc.stdout or "", char_limit),
            stderr=truncate_text(exc.stderr or f"Command timed out after {timeout_sec:.1f}s", char_limit),
            timed_out=True,
        )


def format_observation(result: CommandResult) -> str:
    payload = {
        "command": result.command,
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "disabled": result.disabled,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    return "<observation>\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n</observation>\n"


def tokenizer_for(args: Any) -> Any:
    global TOKENIZER
    if TOKENIZER is not None:
        return TOKENIZER
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required inside slime to tokenize rollout samples") from exc
    checkpoint = getattr(args, "hf_checkpoint", None)
    if not checkpoint:
        raise RuntimeError("args.hf_checkpoint is required for ETPI slime tokenization")
    TOKENIZER = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    return TOKENIZER


def encode_text(args: Any, text: str) -> list[int]:
    tokenizer = tokenizer_for(args)
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


async def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    def _post() -> dict[str, Any]:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as response:
            loaded = json.loads(response.read().decode())
            if not isinstance(loaded, dict):
                raise RuntimeError(f"unexpected SGLang response type: {type(loaded).__name__}")
            return loaded

    return await asyncio.to_thread(_post)


async def call_sglang(args: Any, prompt: str, sampling_params: dict[str, Any]) -> str:
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    payload = {"text": prompt, "sampling_params": sampling_params}
    response = await post_json(url, payload)
    text = response.get("text")
    if isinstance(text, str):
        return text
    raise RuntimeError(f"SGLang response missing text: {response}")


def set_sample_status(sample: Any, name: str) -> None:
    status_owner = getattr(sample, "Status", None) or getattr(type(sample), "Status", None)
    if status_owner is not None and hasattr(status_owner, name):
        setattr(sample, "status", getattr(status_owner, name))
    else:
        setattr(sample, "status", name)


def append_segment(
    args: Any,
    response_parts: list[str],
    response_token_ids: list[int],
    loss_mask: list[int],
    text: str,
    *,
    train_on: bool,
) -> None:
    token_ids = encode_text(args, text)
    response_parts.append(text)
    response_token_ids.extend(token_ids)
    loss_mask.extend([1 if train_on else 0] * len(token_ids))


async def generate(args: Any, sample: Any, sampling_params: dict[str, Any]) -> Any:
    """slime ``--custom-generate-function-path`` hook."""

    metadata = sample_metadata(sample)
    cwd = workspace_from_metadata(metadata)
    context = str(getattr(sample, "prompt"))
    response_parts: list[str] = []
    response_token_ids: list[int] = []
    loss_mask: list[int] = []
    command_results: list[dict[str, Any]] = []

    for _turn in range(max_turns(args)):
        model_text = await call_sglang(args, context, sampling_params)
        append_segment(args, response_parts, response_token_ids, loss_mask, model_text, train_on=True)
        action = parse_action(model_text)
        if action.kind == "final":
            break
        result = run_bash(
            action.body,
            cwd=cwd,
            timeout_sec=command_timeout(args),
            char_limit=max_observation_chars(args),
        )
        command_results.append(
            {
                "command": result.command,
                "returncode": result.returncode,
                "timed_out": result.timed_out,
                "disabled": result.disabled,
            }
        )
        observation = format_observation(result)
        append_segment(args, response_parts, response_token_ids, loss_mask, observation, train_on=False)
        context += model_text + observation
        if result.disabled:
            break

    prompt_token_ids = encode_text(args, str(getattr(sample, "prompt")))
    sample.response = "".join(response_parts)
    sample.tokens = prompt_token_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.loss_mask = loss_mask
    sample.metadata = dict(metadata)
    sample.metadata["etpi_command_results"] = command_results
    set_sample_status(sample, "COMPLETED")
    return sample


def verify_commands(metadata: dict[str, Any]) -> list[str]:
    commands = metadata.get("verify_commands")
    if isinstance(commands, list):
        return [command for command in commands if isinstance(command, str) and command.strip()]
    if isinstance(commands, str) and commands.strip():
        return [commands]
    return []


async def reward_func(args: Any, sample: Any, **_kwargs: Any) -> float:
    """slime ``--custom-rm-path`` hook."""

    metadata = sample_metadata(sample)
    commands = verify_commands(metadata)
    if commands:
        cwd = workspace_from_metadata(metadata)
        rewards: list[float] = []
        for command in commands:
            result = run_bash(
                command,
                cwd=cwd,
                timeout_sec=command_timeout(args),
                char_limit=max_observation_chars(args),
            )
            rewards.append(1.0 if result.returncode == 0 else 0.0)
        return sum(rewards) / len(rewards)

    response = str(getattr(sample, "response", ""))
    action = parse_action(response)
    if action.kind == "final" and action.body:
        return 0.2
    return 0.0
