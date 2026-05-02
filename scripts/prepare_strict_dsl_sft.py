#!/usr/bin/env python3
"""Build strict multi-turn compact-DSL SFT data from successful agent traces.

This script keeps the original actions/tool calls and replaces assistant
reasoning with a validated symbolic DSL:

    <think>
    PLAN: seq(observe,act,verify,finish)
    STATE: need_action
    RISK: none
    NEXT: tool_call
    </think>
    <tool_call>...</tool_call>

It emits both preserved-context rows and per-turn rows whose prior assistant
thinking has been stripped. The latter teaches the policy to continue from
compact production contexts where old thoughts are not retained.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_hermes_dsl_sft import (  # noqa: E402
    DSL_RISKS,
    DSL_STATES,
    ROLE_MAP,
    THINK_BLOCK_RE,
    TOOL_CALL_RE,
    append_dsl_system_prompt,
    compact_ws,
    convert_tool_calls_to_qwen_xml,
    iter_hf_rows,
    iter_local_rows,
    normalize_conversations,
    parse_tool_call_payload,
    render_chatml,
    render_dsl,
    tool_call_name_and_args,
    to_messages,
    tool_response_failed,
    validate_dsl,
)


STRICT_PLANS = [
    "seq(observe,act,verify,finish)",
    "seq(observe,act,verify,repair,finish)",
    "seq(observe,verify,finish)",
]
STRICT_NEXTS = ["tool_call", "final"]
QWEN_XML_FUNCTION_RE = re.compile(
    r"<function\s*=\s*\"?(?P<name>[A-Za-z_][\w-]*)\"?\s*>",
    re.IGNORECASE,
)
QWEN_XML_PARAMETER_RE = re.compile(
    r"<parameter\s*=\s*\"?(?P<key>[A-Za-z_][\w-]*)\"?\s*>\s*"
    r"(?P<value>.*?)(?:\s*</parameter>|\s*</function>|\s*</tool_call>|\s*\Z)",
    re.DOTALL | re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate strict multi-turn compact-DSL SFT rows."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Local JSON/JSONL trace file.")
    source.add_argument("--dataset", help="Hugging Face dataset id.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--variants",
        default="preserved,strip_prior_think,action_only_prior",
        help=(
            "Comma-separated variants: preserved, strip_prior_think, "
            "action_only_prior."
        ),
    )
    parser.add_argument(
        "--tool-format",
        choices=["qwen_xml", "preserve"],
        default="qwen_xml",
        help="Canonicalize tool calls to Qwen XML or preserve source blocks.",
    )
    parser.add_argument(
        "--require-success",
        action="store_true",
        help="Skip rows that look unresolved/failed.",
    )
    parser.add_argument(
        "--no-system-dsl-prompt",
        action="store_true",
        help="Do not prepend/append the compact DSL system prompt.",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Do not include rendered ChatML text.",
    )
    parser.add_argument(
        "--min-assistant-turns",
        type=int,
        default=1,
        help="Skip traces with fewer strict assistant targets.",
    )
    return parser.parse_args()


def row_is_success(row: dict[str, Any]) -> bool:
    for key in ("is_resolved", "resolved", "success", "passed"):
        if key in row:
            return bool(row[key])
    result = row.get("result")
    if isinstance(result, dict):
        for key in ("is_resolved", "resolved", "success", "passed"):
            if key in result:
                return bool(result[key])
    return True


def normalize_row_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    if "conversations" in row:
        conversations = normalize_conversations(row["conversations"])
        return to_messages(conversations)

    raw_messages = row.get("messages")
    if isinstance(raw_messages, str):
        raw_messages = json.loads(raw_messages)
    if not isinstance(raw_messages, list):
        raise ValueError("row must contain conversations or messages")

    messages: list[dict[str, str]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or item.get("from") or "").strip()
        role = ROLE_MAP.get(role, role)
        content = item.get("content", item.get("value", ""))
        messages.append({"role": role, "content": str(content)})
    return messages


def message_to_conversation(message: dict[str, str]) -> dict[str, str]:
    role = message["role"]
    from_value = "gpt" if role == "assistant" else role
    if role == "user":
        from_value = "human"
    return {"from": from_value, "value": message["content"]}


def strip_think(text: str) -> str:
    return THINK_BLOCK_RE.sub("", text).strip()


def first_tool_call(text: str) -> str:
    match = TOOL_CALL_RE.search(text)
    return match.group(0).strip() if match else ""


def action_only_text(text: str) -> str:
    return first_tool_call(text) or strip_think(text)


def strip_prior_text(text: str) -> str:
    stripped = strip_think(text)
    return stripped.strip()


def qwen_xml_name_args(block: str) -> tuple[str, dict[str, str]] | None:
    name_match = QWEN_XML_FUNCTION_RE.search(block)
    if not name_match:
        return None
    args = {
        match.group("key"): match.group("value").strip()
        for match in QWEN_XML_PARAMETER_RE.finditer(block)
    }
    return name_match.group("name"), args


def canonical_tool_call(block: str, *, tool_format: str) -> tuple[str, str, dict[str, Any]]:
    if tool_format == "preserve":
        parsed = qwen_xml_name_args(block)
        if parsed is not None:
            name, args = parsed
            return block.strip(), name, args
        payload = parse_tool_call_payload(TOOL_CALL_RE.sub(r"\1", block).strip())
        if payload is not None:
            parsed_json = tool_call_name_and_args(payload)
            if parsed_json is not None:
                name, args = parsed_json
                return block.strip(), name, args
        return block.strip(), "unknown", {}

    converted, count = convert_tool_calls_to_qwen_xml(block)
    if count:
        parsed = qwen_xml_name_args(converted)
        if parsed is None:
            raise ValueError("converted Qwen XML tool call could not be parsed")
        name, args = parsed
        return converted.strip(), name, args

    parsed = qwen_xml_name_args(block)
    if parsed is not None:
        name, args = parsed
        return block.strip(), name, args

    raise ValueError("assistant tool call is neither JSON nor Qwen XML")


def command_from_action(name: str, args: dict[str, Any]) -> str:
    if name not in {"run_shell", "terminal", "shell", "bash"}:
        return ""
    for key in ("command", "cmd", "keystrokes"):
        value = args.get(key)
        if value is not None:
            return str(value)
    return ""


def command_kind(command: str) -> str:
    text = compact_ws(command)
    if not text:
        return "action"
    if any(word in text for word in ["pytest", "run-tests", " test", "npm test"]):
        return "verify"
    if any(word in text for word in ["cat ", "ls", "find ", "grep", "rg ", "docs/json"]):
        return "observe"
    if any(word in text for word in [">", "curl -x post", "curl -x put", "apply_patch"]):
        return "act"
    if any(word in text for word in ["python", "jq", "json.tool"]):
        return "observe"
    return "act"


def infer_strict_dsl(
    *,
    assistant_text: str,
    action_name: str,
    action_args: dict[str, Any],
    prior_tool_text: str,
    prior_commands: list[str],
) -> dict[str, str]:
    has_action = bool(first_tool_call(assistant_text))
    if not has_action or action_name == "finish":
        dsl = {
            "PLAN": "seq(observe,verify,finish)",
            "STATE": "ready",
            "RISK": "none",
            "NEXT": "final",
        }
        validate_dsl(dsl)
        return dsl

    command = command_from_action(action_name, action_args)
    kind = command_kind(command)
    prior_failed = tool_response_failed(prior_tool_text)
    repeated = bool(command and command in prior_commands)

    if prior_failed:
        plan = "seq(observe,act,verify,repair,finish)"
        state = "need_fix"
        risk = "tool_failure"
    elif repeated:
        plan = "seq(observe,act,verify,repair,finish)"
        state = "need_fix"
        risk = "repeat_loop"
    elif kind == "observe":
        plan = "seq(observe,act,verify,finish)"
        state = "need_context"
        risk = "none"
    elif kind == "verify":
        plan = "seq(observe,act,verify,finish)"
        state = "need_verify"
        risk = "none"
    else:
        plan = "seq(observe,act,verify,finish)"
        state = "need_action"
        risk = "none"

    text = compact_ws(assistant_text + " " + command)
    if risk == "none" and any(token in text for token in ["schema", "format", "quote"]):
        risk = "bad_tool_args"
    if risk == "none" and any(token in text for token in ["unknown", "unclear", "missing"]):
        risk = "missing_context"
    if risk == "none" and any(token in text for token in ["wrong target", "wrong file"]):
        risk = "wrong_target"

    dsl = {"PLAN": plan, "STATE": state, "RISK": risk, "NEXT": "tool_call"}
    validate_dsl(dsl)
    return dsl


def render_strict_assistant(dsl: dict[str, str], action_block: str, *, final_text: str = "") -> str:
    validate_dsl(dsl)
    if dsl["PLAN"] not in STRICT_PLANS:
        raise ValueError(f"invalid strict PLAN: {dsl['PLAN']}")
    if dsl["STATE"] not in DSL_STATES:
        raise ValueError(f"invalid strict STATE: {dsl['STATE']}")
    if dsl["RISK"] not in DSL_RISKS:
        raise ValueError(f"invalid strict RISK: {dsl['RISK']}")
    if dsl["NEXT"] not in STRICT_NEXTS:
        raise ValueError(f"invalid strict NEXT: {dsl['NEXT']}")
    body = "<think>\n" + render_dsl(dsl) + "\n</think>"
    if action_block:
        return body + "\n" + action_block.strip()
    if final_text.strip():
        return body + "\n" + final_text.strip()
    return body


def strictify_messages(
    messages: list[dict[str, str]],
    *,
    tool_format: str,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    strict_messages: list[dict[str, str]] = []
    labels: list[dict[str, Any]] = []
    prior_tool_text = ""
    prior_commands: list[str] = []

    for index, message in enumerate(messages):
        role = message["role"]
        content = message["content"]
        if role != "assistant":
            strict_messages.append(dict(message))
            if role == "tool":
                prior_tool_text = content
            elif role in {"user", "system"}:
                prior_tool_text = ""
            continue

        block = first_tool_call(content)
        action_block = ""
        action_name = "finish"
        action_args: dict[str, Any] = {}
        if block:
            action_block, action_name, action_args = canonical_tool_call(
                block,
                tool_format=tool_format,
            )
        dsl = infer_strict_dsl(
            assistant_text=content,
            action_name=action_name,
            action_args=action_args,
            prior_tool_text=prior_tool_text,
            prior_commands=prior_commands,
        )
        final_text = "" if block else strip_think(content)
        strict_content = render_strict_assistant(dsl, action_block, final_text=final_text)
        strict_messages.append({"role": role, "content": strict_content})
        label = dict(dsl)
        label["_source"] = "strict_heuristic"
        label["_message_index"] = index
        label["_action_name"] = action_name
        labels.append(label)
        command = command_from_action(action_name, action_args)
        if command:
            prior_commands.append(command)
        prior_tool_text = ""

    return strict_messages, labels


def add_system_prompt(messages: list[dict[str, str]], *, tool_format: str) -> list[dict[str, str]]:
    conversations = [message_to_conversation(message) for message in messages]
    append_dsl_system_prompt(conversations, tool_format="qwen_xml" if tool_format == "qwen_xml" else "hermes_json")
    return to_messages(conversations)


def assistant_indices(messages: list[dict[str, str]]) -> list[int]:
    return [index for index, message in enumerate(messages) if message["role"] == "assistant" and "<think>" in message["content"]]


def strip_context_for_target(
    messages: list[dict[str, str]],
    *,
    target_index: int,
    mode: str,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for index, message in enumerate(messages[: target_index + 1]):
        if message["role"] != "assistant" or index == target_index:
            out.append(dict(message))
            continue
        if mode == "strip_prior_think":
            content = strip_prior_text(message["content"])
        elif mode == "action_only_prior":
            content = action_only_text(message["content"])
        else:
            raise ValueError(f"unknown strip mode: {mode}")
        out.append({"role": "assistant", "content": content})
    return out


def make_row(
    source_row: dict[str, Any],
    messages: list[dict[str, str]],
    labels: list[dict[str, Any]],
    *,
    variant: str,
    include_text: bool,
    tool_format: str,
    target_message_index: int | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": source_row.get("id"),
        "source_id": source_row.get("id"),
        "task": source_row.get("task") or source_row.get("instruction"),
        "messages": messages,
        "conversations": [message_to_conversation(message) for message in messages],
        "dsl_stats": {
            "variant": variant,
            "tool_format": tool_format,
            "strict": True,
            "labels": labels,
            "assistant_targets": len(assistant_indices(messages)),
        },
    }
    if target_message_index is not None:
        out["dsl_stats"]["target_message_index"] = target_message_index
    if include_text:
        out["text"] = render_chatml(messages)
    return out


def convert_row(
    row: dict[str, Any],
    *,
    variants: set[str],
    tool_format: str,
    add_prompt: bool,
    include_text: bool,
    min_assistant_turns: int,
) -> list[dict[str, Any]]:
    messages = normalize_row_messages(row)
    if add_prompt:
        messages = add_system_prompt(messages, tool_format=tool_format)
    strict_messages, labels = strictify_messages(messages, tool_format=tool_format)
    targets = assistant_indices(strict_messages)
    if len(targets) < min_assistant_turns:
        return []

    rows: list[dict[str, Any]] = []
    if "preserved" in variants:
        rows.append(
            make_row(
                row,
                strict_messages,
                labels,
                variant="preserved",
                include_text=include_text,
                tool_format=tool_format,
            )
        )

    for variant in ("strip_prior_think", "action_only_prior"):
        if variant not in variants:
            continue
        for target_index in targets:
            context_messages = strip_context_for_target(
                strict_messages,
                target_index=target_index,
                mode=variant,
            )
            target_labels = [
                label
                for label in labels
                if label.get("_message_index") == target_index
            ]
            rows.append(
                make_row(
                    row,
                    context_messages,
                    target_labels,
                    variant=variant,
                    include_text=include_text,
                    tool_format=tool_format,
                    target_message_index=target_index,
                )
            )
    return rows


def parse_variants(value: str) -> set[str]:
    variants = {item.strip() for item in value.split(",") if item.strip()}
    allowed = {"preserved", "strip_prior_think", "action_only_prior"}
    unknown = variants - allowed
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")
    if not variants:
        raise ValueError("at least one variant is required")
    return variants


def iter_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.input is not None:
        yield from iter_local_rows(args.input)
        return
    yield from iter_hf_rows(args.dataset, args.split)


def main() -> None:
    args = parse_args()
    variants = parse_variants(args.variants)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    seen = 0
    written = 0
    skipped_failed = 0
    skipped_empty = 0
    variant_counts: dict[str, int] = {}

    with args.out.open("w") as f:
        for row in iter_rows(args):
            if args.limit is not None and seen >= args.limit:
                break
            seen += 1
            if args.require_success and not row_is_success(row):
                skipped_failed += 1
                continue
            out_rows = convert_row(
                row,
                variants=variants,
                tool_format=args.tool_format,
                add_prompt=not args.no_system_dsl_prompt,
                include_text=not args.no_text,
                min_assistant_turns=args.min_assistant_turns,
            )
            if not out_rows:
                skipped_empty += 1
                continue
            for out_row in out_rows:
                f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                written += 1
                variant = str(out_row["dsl_stats"]["variant"])
                variant_counts[variant] = variant_counts.get(variant, 0) + 1

    print(
        json.dumps(
            {
                "input_rows_seen": seen,
                "output_rows_written": written,
                "skipped_failed": skipped_failed,
                "skipped_empty": skipped_empty,
                "variants": sorted(variants),
                "variant_counts": variant_counts,
                "tool_format": args.tool_format,
                "out": str(args.out),
            },
            indent=2,
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
