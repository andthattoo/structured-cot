#!/usr/bin/env python3
"""Prepare Hermes agent traces for compact-DSL SFT.

The source dataset contains verbose <think> blocks before tool calls. This
script rewrites those blocks into a small existing-token DSL while preserving
the original tool calls and tool responses:

    <think>
    PLAN: seq(observe,act,verify,finish)
    STATE: need_action
    RISK: bad_tool_args
    NEXT: tool_call
    </think>
    <tool_call>...</tool_call>

The goal is not to treat the teacher's prose as ground truth. The teacher trace
provides the action/observation trajectory; the DSL is a compact supervised
state label for the next assistant action.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET = "DJLougen/hermes-agent-traces-filtered"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DSL_SYSTEM_PROMPT = (
    "When thinking before tool use, use this compact DSL inside <think> tags:\n"
    "PLAN: one symbolic control-flow plan\n"
    "STATE: current state\n"
    "RISK: main risk to avoid\n"
    "NEXT: tool_call or final\n"
    "Use the DSL as a decision record for the next action, then emit the "
    "tool call or final answer normally."
)

DSL_PLANS = [
    "seq(observe,act,verify,finish)",
    "seq(observe,act,verify,repair,finish)",
    "seq(observe,verify,finish)",
]
DSL_STATES = ["need_context", "need_action", "need_fix", "need_verify", "blocked", "ready"]
DSL_RISKS = [
    "none",
    "missing_context",
    "bad_tool_args",
    "wrong_target",
    "tool_failure",
    "premature_final",
    "repeat_loop",
]
DSL_NEXTS = ["tool_call", "final"]

DSL_LABEL_GRAMMAR = r'''
root ::= "PLAN: " plan "\n" "STATE: " state "\n" "RISK: " risk "\n" "NEXT: " next "\n"
plan ::= "seq(observe,act,verify,finish)" | "seq(observe,act,verify,repair,finish)" | "seq(observe,verify,finish)"
state ::= "need_context" | "need_action" | "need_fix" | "need_verify" | "blocked" | "ready"
risk ::= "none" | "missing_context" | "bad_tool_args" | "wrong_target" | "tool_failure" | "premature_final" | "repeat_loop"
next ::= "tool_call" | "final"
'''

THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE
)
THINK_BLOCK_RE = re.compile(
    r"(<think>\s*.*?\s*</think>)", re.DOTALL | re.IGNORECASE
)

ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "tool": "tool",
    "function": "tool",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite Hermes agent reasoning traces into compact DSL SFT data."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Hugging Face dataset id to load. Default: {DEFAULT_DATASET}",
    )
    source.add_argument(
        "--input",
        type=Path,
        help="Local JSON/JSONL file with Hermes-style rows.",
    )
    parser.add_argument("--split", default="train", help="Dataset split to load.")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSONL path. The repo ignores *.jsonl by default.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit.")
    parser.add_argument(
        "--min-rewrites",
        type=int,
        default=1,
        help="Skip rows with fewer rewritten assistant think blocks.",
    )
    parser.add_argument(
        "--no-system-dsl-prompt",
        action="store_true",
        help="Do not append the DSL instruction to the first system message.",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Do not include a rendered ChatML-ish text field.",
    )
    parser.add_argument(
        "--keep-pre-tool-prose",
        action="store_true",
        help=(
            "Keep natural-language assistant prose between </think> and "
            "<tool_call>. By default it is stripped to teach abstract reasoning "
            "followed directly by tool calls."
        ),
    )
    parser.add_argument(
        "--labeler",
        choices=["heuristic", "local_grammar", "openrouter"],
        default="heuristic",
        help=(
            "How to label each think block. heuristic is offline. local_grammar "
            "uses an OpenAI-compatible local server with a GBNF grammar. "
            "openrouter uses JSON Schema structured outputs."
        ),
    )
    parser.add_argument(
        "--labeler-model",
        default=None,
        help="Model id for local_grammar/openrouter. Defaults depend on backend.",
    )
    parser.add_argument(
        "--labeler-base-url",
        default=None,
        help=(
            "OpenAI-compatible base URL. Defaults to http://127.0.0.1:8000/v1 "
            "for local_grammar and https://openrouter.ai/api/v1 for openrouter."
        ),
    )
    parser.add_argument(
        "--labeler-api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable containing API key for openrouter.",
    )
    parser.add_argument(
        "--labeler-timeout-sec",
        type=int,
        default=120,
        help="HTTP timeout for model labeling calls.",
    )
    parser.add_argument(
        "--labeler-context-messages",
        type=int,
        default=6,
        help="Number of previous conversation messages to show the labeler.",
    )
    parser.add_argument(
        "--labeler-max-chars",
        type=int,
        default=6000,
        help="Maximum rendered context characters passed to the labeler.",
    )
    parser.add_argument(
        "--labeler-fallback",
        choices=["heuristic", "skip", "raise"],
        default="heuristic",
        help="What to do if the model labeler fails.",
    )
    return parser.parse_args()


def iter_local_rows(path: Path) -> Iterable[dict[str, Any]]:
    text = path.read_text()
    stripped = text.lstrip()
    if stripped.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path} JSON root must be a list or JSONL rows")
        yield from data
        return

    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: row must be an object")
        yield row


def iter_hf_rows(dataset_id: str, split: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install with `uv sync` or "
            "`uv pip install datasets>=3,<4`."
        ) from exc

    yield from load_dataset(dataset_id, split=split)


def normalize_conversations(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("conversations must be a list or JSON string list")

    conversations: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        speaker = str(item.get("from") or item.get("role") or "").strip()
        content = item.get("value", item.get("content", ""))
        conversations.append({"from": speaker, "value": str(content)})
    return conversations


def compact_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def has_any(text: str, words: Iterable[str]) -> bool:
    return any(word in text for word in words)


def has_tool_call(assistant_text: str) -> bool:
    return TOOL_CALL_RE.search(assistant_text) is not None


def build_labeler_context(
    history: list[dict[str, str]],
    assistant_value: str,
    *,
    max_messages: int,
    max_chars: int,
) -> str:
    start = max(0, len(history) - max_messages)
    parts: list[str] = []
    for item in history[start:]:
        role = ROLE_MAP.get(item["from"], item["from"])
        parts.append(f"{role.upper()}:\n{item['value']}")
    parts.append("ASSISTANT_TO_COMPRESS:\n" + assistant_value)
    text = "\n\n---\n\n".join(parts)
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


def parse_dsl_text(text: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for line in text.strip().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().upper()
        value = value.strip()
        if key in {"PLAN", "STATE", "RISK", "NEXT"}:
            labels[key] = value
    validate_dsl(labels)
    return labels


def validate_dsl(dsl: dict[str, str]) -> None:
    missing = [key for key in ["PLAN", "STATE", "RISK", "NEXT"] if key not in dsl]
    if missing:
        raise ValueError(f"DSL label missing fields: {missing}")
    if dsl["PLAN"] not in DSL_PLANS:
        raise ValueError(f"invalid PLAN: {dsl['PLAN']!r}")
    if dsl["STATE"] not in DSL_STATES:
        raise ValueError(f"invalid STATE: {dsl['STATE']!r}")
    if dsl["RISK"] not in DSL_RISKS:
        raise ValueError(f"invalid RISK: {dsl['RISK']!r}")
    if dsl["NEXT"] not in DSL_NEXTS:
        raise ValueError(f"invalid NEXT: {dsl['NEXT']!r}")


class DslLabeler:
    def __init__(
        self,
        *,
        mode: str,
        model: str | None,
        base_url: str | None,
        api_key_env: str,
        timeout_sec: int,
        fallback: str,
    ):
        self.mode = mode
        self.model = model
        self.base_url = (base_url or self._default_base_url()).rstrip("/")
        self.api_key_env = api_key_env
        self.timeout_sec = timeout_sec
        self.fallback = fallback

    def _default_base_url(self) -> str:
        if self.mode == "openrouter":
            return DEFAULT_OPENROUTER_BASE_URL
        return "http://127.0.0.1:8000/v1"

    def label(
        self,
        *,
        think: str,
        assistant_text: str,
        prior_tool_text: str,
        context: str,
    ) -> dict[str, str]:
        if self.mode == "heuristic":
            return infer_dsl(
                think,
                assistant_text,
                prior_tool_text=prior_tool_text,
            )
        try:
            if self.mode == "local_grammar":
                return self._label_local_grammar(
                    assistant_text=assistant_text,
                    context=context,
                )
            if self.mode == "openrouter":
                return self._label_openrouter(
                    assistant_text=assistant_text,
                    context=context,
                )
        except Exception:
            if self.fallback == "heuristic":
                return infer_dsl(
                    think,
                    assistant_text,
                    prior_tool_text=prior_tool_text,
                )
            if self.fallback == "skip":
                raise
            raise
        raise ValueError(f"unknown labeler mode: {self.mode}")

    def _messages(self, context: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You compress assistant thinking into a tiny symbolic DSL. "
                    "Return exactly one label block. Do not explain. Choose only "
                    "from the allowed values.\n\n"
                    "Allowed PLAN values:\n"
                    + "\n".join(f"- {item}" for item in DSL_PLANS)
                    + "\n\nAllowed STATE values: "
                    + ", ".join(DSL_STATES)
                    + "\nAllowed RISK values: "
                    + ", ".join(DSL_RISKS)
                    + "\nAllowed NEXT values: "
                    + ", ".join(DSL_NEXTS)
                    + "\n\nLabel semantics:\n"
                    "- NEXT is tool_call if the assistant emits tool_call tags, else final.\n"
                    "- Use tool_failure only when the previous tool response itself failed, e.g. success=false, exit_code nonzero, or an explicit tool error.\n"
                    "- total_count=0 is missing_context, not tool_failure, unless the tool also errored.\n"
                    "- Successful file reads/searches are usually need_context or need_action, not need_fix.\n"
                    "- Use premature_final when the assistant is about to finish without enough verification."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Compress this turn into the DSL.\n\n"
                    f"{context}\n\n"
                    "Return format:\n"
                    "PLAN: ...\nSTATE: ...\nRISK: ...\nNEXT: ...\n"
                ),
            },
        ]

    def _post_chat(self, payload: dict[str, Any], *, api_key: str | None) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        elif self.mode == "local_grammar":
            headers["Authorization"] = "Bearer local"
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))

    def _label_local_grammar(self, *, assistant_text: str, context: str) -> dict[str, str]:
        payload = {
            "model": self.model or self._get_default_model(),
            "messages": self._messages(context),
            "temperature": 0.0,
            "max_tokens": 80,
            "grammar": DSL_LABEL_GRAMMAR,
        }
        response = self._post_chat(payload, api_key=None)
        content = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        dsl = parse_dsl_text(content)
        expected_next = "tool_call" if has_tool_call(assistant_text) else "final"
        dsl["NEXT"] = expected_next
        return dsl

    def _label_openrouter(self, *, assistant_text: str, context: str) -> dict[str, str]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is not set")
        payload = {
            "model": self.model or "openai/gpt-4o-mini",
            "messages": self._messages(context),
            "temperature": 0.0,
            "max_tokens": 120,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "dsl_label",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "PLAN": {"type": "string", "enum": DSL_PLANS},
                            "STATE": {"type": "string", "enum": DSL_STATES},
                            "RISK": {"type": "string", "enum": DSL_RISKS},
                            "NEXT": {"type": "string", "enum": DSL_NEXTS},
                        },
                        "required": ["PLAN", "STATE", "RISK", "NEXT"],
                        "additionalProperties": False,
                    },
                },
            },
        }
        response = self._post_chat(payload, api_key=api_key)
        content = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "{}")
        )
        dsl = json.loads(content)
        validate_dsl(dsl)
        expected_next = "tool_call" if has_tool_call(assistant_text) else "final"
        dsl["NEXT"] = expected_next
        return dsl

    def _get_default_model(self) -> str:
        req = urllib.request.Request(
            self.base_url + "/models",
            headers={"Authorization": "Bearer local"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
            data = json.loads(response.read().decode("utf-8"))
        models = [item.get("id") for item in data.get("data", []) if item.get("id")]
        if not models:
            raise RuntimeError("No models returned by server /v1/models")
        return models[0]


def infer_dsl(
    think: str,
    assistant_text: str,
    *,
    prior_tool_text: str = "",
) -> dict[str, str]:
    text = compact_ws(think)
    prior = compact_ws(prior_tool_text)
    combined = compact_ws(think + " " + prior_tool_text)
    next_action = "tool_call" if has_tool_call(assistant_text) else "final"

    if next_action == "final":
        return {
            "PLAN": "seq(observe,verify,finish)",
            "STATE": "ready",
            "RISK": "none",
            "NEXT": "final",
        }

    repair_markers = [
        "error",
        "failed",
        "failure",
        "exception",
        "bug",
        "fix",
        "debug",
        "retry",
        "wrong",
        "not available",
        "unavailable",
        "success false",
        '"success": false',
    ]
    verify_markers = [
        "verify",
        "validate",
        "confirm",
        "check",
        "test",
        "ensure",
    ]
    context_markers = [
        "inspect",
        "look",
        "read",
        "search",
        "find",
        "explore",
        "list",
        "schema",
        "docs",
    ]
    arg_markers = [
        "argument",
        "parameter",
        "json",
        "schema",
        "format",
        "quote",
        "malformed",
    ]

    prior_failed = has_any(
        prior,
        [
            "error",
            "failed",
            "failure",
            "exception",
            "not available",
            "unavailable",
            "success false",
            '"success": false',
            "exit_code\": 1",
        ],
    )

    if prior_failed or has_any(combined, repair_markers):
        plan = "seq(observe,act,verify,repair,finish)"
        state = "need_fix"
    elif has_any(combined, verify_markers):
        plan = "seq(observe,act,verify,finish)"
        state = "need_verify"
    elif has_any(combined, context_markers):
        plan = "seq(observe,act,verify,finish)"
        state = "need_context"
    else:
        plan = "seq(observe,act,verify,finish)"
        state = "need_action"

    if prior_failed:
        risk = "tool_failure"
    elif has_any(combined, ["repeat", "again", "same command", "loop"]):
        risk = "repeat_loop"
    elif has_any(combined, arg_markers):
        risk = "bad_tool_args"
    elif has_any(combined, ["ambiguous", "unclear", "missing", "not enough", "unknown"]):
        risk = "missing_context"
    elif has_any(combined, repair_markers):
        risk = "tool_failure"
    elif "finish" in text or "complete" in text or "done" in text:
        risk = "premature_final"
    else:
        risk = "none"

    return {"PLAN": plan, "STATE": state, "RISK": risk, "NEXT": next_action}


def render_dsl(dsl: dict[str, str]) -> str:
    return (
        f"PLAN: {dsl['PLAN']}\n"
        f"STATE: {dsl['STATE']}\n"
        f"RISK: {dsl['RISK']}\n"
        f"NEXT: {dsl['NEXT']}"
    )


def strip_pre_tool_prose(value: str) -> str:
    if not has_tool_call(value):
        return value
    think_match = THINK_BLOCK_RE.search(value)
    tool_match = TOOL_CALL_RE.search(value)
    if not think_match or not tool_match or tool_match.start() < think_match.end():
        return value
    return value[: think_match.end()].rstrip() + "\n" + value[tool_match.start() :].lstrip()


def rewrite_assistant_value(
    value: str,
    *,
    prior_tool_text: str = "",
    strip_prose_before_tool: bool = True,
    labeler: DslLabeler | None = None,
    context: str = "",
) -> tuple[str, int, list[dict[str, str]]]:
    rewrites: list[dict[str, str]] = []

    def replace(match: re.Match[str]) -> str:
        think = match.group(1)
        if labeler is None:
            dsl = infer_dsl(think, value, prior_tool_text=prior_tool_text)
        else:
            dsl = labeler.label(
                think=think,
                assistant_text=value,
                prior_tool_text=prior_tool_text,
                context=context,
            )
        rewrites.append(dsl)
        return "<think>\n" + render_dsl(dsl) + "\n</think>"

    rewritten, count = THINK_RE.subn(replace, value, count=1)
    if count and strip_prose_before_tool:
        rewritten = strip_pre_tool_prose(rewritten)
    return rewritten, count, rewrites


def append_dsl_system_prompt(conversations: list[dict[str, str]]) -> None:
    for item in conversations:
        if ROLE_MAP.get(item["from"], item["from"]) == "system":
            item["value"] = item["value"].rstrip() + "\n\n" + DSL_SYSTEM_PROMPT
            return
    conversations.insert(0, {"from": "system", "value": DSL_SYSTEM_PROMPT})


def to_messages(conversations: list[dict[str, str]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in conversations:
        role = ROLE_MAP.get(item["from"], item["from"])
        messages.append({"role": role, "content": item["value"]})
    return messages


def render_chatml(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = message["role"]
        content = message["content"].rstrip()
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts) + "\n"


def convert_row(
    row: dict[str, Any],
    *,
    add_system_prompt: bool,
    include_text: bool,
    strip_prose_before_tool: bool,
    labeler: DslLabeler | None = None,
    labeler_context_messages: int = 6,
    labeler_max_chars: int = 6000,
) -> dict[str, Any]:
    conversations = normalize_conversations(row.get("conversations", []))
    rewritten_conversations: list[dict[str, str]] = []
    rewritten_blocks = 0
    tool_turns = 0
    labels: list[dict[str, str]] = []
    prior_tool_text = ""
    history: list[dict[str, str]] = []

    for item in conversations:
        new_item = dict(item)
        role = ROLE_MAP.get(new_item["from"], new_item["from"])
        if role == "assistant":
            context = build_labeler_context(
                history,
                new_item["value"],
                max_messages=labeler_context_messages,
                max_chars=labeler_max_chars,
            )
            new_value, count, row_labels = rewrite_assistant_value(
                new_item["value"],
                prior_tool_text=prior_tool_text,
                strip_prose_before_tool=strip_prose_before_tool,
                labeler=labeler,
                context=context,
            )
            new_item["value"] = new_value
            rewritten_blocks += count
            labels.extend(row_labels)
            if has_tool_call(new_value):
                tool_turns += 1
        elif role == "tool":
            prior_tool_text = new_item["value"]
        if role != "tool" and role != "assistant":
            prior_tool_text = ""
        rewritten_conversations.append(new_item)
        history.append(new_item)

    if add_system_prompt:
        append_dsl_system_prompt(rewritten_conversations)

    messages = to_messages(rewritten_conversations)
    out = {
        "id": row.get("id"),
        "category": row.get("category"),
        "subcategory": row.get("subcategory"),
        "task": row.get("task"),
        "tools": row.get("tools"),
        "conversations": rewritten_conversations,
        "messages": messages,
        "dsl_stats": {
            "rewritten_think_blocks": rewritten_blocks,
            "assistant_tool_turns": tool_turns,
            "labels": labels,
        },
    }
    if include_text:
        out["text"] = render_chatml(messages)
    return out


def main() -> None:
    args = parse_args()
    rows: Iterable[dict[str, Any]]
    if args.input:
        rows = iter_local_rows(args.input)
    else:
        rows = iter_hf_rows(args.dataset, args.split)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    labeler = DslLabeler(
        mode=args.labeler,
        model=args.labeler_model,
        base_url=args.labeler_base_url,
        api_key_env=args.labeler_api_key_env,
        timeout_sec=args.labeler_timeout_sec,
        fallback=args.labeler_fallback,
    )

    seen = 0
    written = 0
    rewritten_total = 0
    skipped = 0
    with args.out.open("w") as f:
        for row in rows:
            if args.limit is not None and seen >= args.limit:
                break
            seen += 1
            try:
                converted = convert_row(
                    row,
                    add_system_prompt=not args.no_system_dsl_prompt,
                    include_text=not args.no_text,
                    strip_prose_before_tool=not args.keep_pre_tool_prose,
                    labeler=labeler,
                    labeler_context_messages=args.labeler_context_messages,
                    labeler_max_chars=args.labeler_max_chars,
                )
            except Exception:
                if args.labeler_fallback == "skip":
                    skipped += 1
                    continue
                raise
            rewrites = int(converted["dsl_stats"]["rewritten_think_blocks"])
            if rewrites < args.min_rewrites:
                continue
            rewritten_total += rewrites
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")
            written += 1

    print(
        json.dumps(
            {
                "input_rows_seen": seen,
                "output_rows_written": written,
                "rows_skipped": skipped,
                "rewritten_think_blocks": rewritten_total,
                "labeler": args.labeler,
                "out": str(args.out),
            },
            indent=2,
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
