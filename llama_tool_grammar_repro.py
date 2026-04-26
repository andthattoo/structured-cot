#!/usr/bin/env python3
"""Minimal tool-calling + grammar repro matrix.

This intentionally avoids the OpenAI Python SDK so it can run on a fresh GPU
box with only Python available. It targets OpenAI-compatible chat completions
endpoints such as llama.cpp's llama-server and vLLM's OpenAI server.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


ANSWER_GRAMMAR = r'''
root ::= "ANSWER: " city "\n"
city ::= "Paris" | "London" | "Tokyo"
'''


TOOL_CALL_XML_GRAMMAR = r'''
root ::= "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
'''


TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name.",
                    }
                },
                "required": ["city"],
            },
        },
    }
]


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer local",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_models(base_url: str, timeout: int) -> list[str]:
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={"Authorization": "Bearer local"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []
    return [item.get("id", "") for item in data.get("data", []) if item.get("id")]


def message_text(choice: dict[str, Any]) -> str:
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def tool_calls(choice: dict[str, Any]) -> list[Any]:
    message = choice.get("message") or {}
    calls = message.get("tool_calls")
    return calls if isinstance(calls, list) else []


def classify(case_name: str, response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return "no choices"

    choice = choices[0]
    text = message_text(choice)
    calls = tool_calls(choice)

    if "tools" in case_name:
        if calls:
            return "tool_call parsed"
        if "<tool_call>" in text:
            return "tool_call text only"
        return "no tool call"

    if "grammar" in case_name:
        if text.startswith("ANSWER: "):
            return "grammar-shaped answer"
        return "not grammar-shaped"

    return "ok"


def add_grammar(payload: dict[str, Any], grammar: str, server: str) -> None:
    if server == "llama_cpp":
        payload["grammar"] = grammar
    elif server == "vllm":
        payload["structured_outputs"] = {"grammar": grammar}
    elif server == "vllm_legacy":
        payload["guided_grammar"] = grammar
    else:
        raise ValueError(f"unknown server kind: {server}")


def has_grammar(payload: dict[str, Any]) -> bool:
    return any(key in payload for key in ("grammar", "structured_outputs", "guided_grammar"))


def make_payload(
    case_name: str,
    model: str,
    max_tokens: int,
    temperature: float,
    server: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Use the get_weather tool for Paris. If no tool is available, "
                    "answer exactly: ANSWER: Paris"
                ),
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if case_name in {"tools_only", "tools_plus_answer_grammar", "tools_plus_tool_xml_grammar"}:
        payload["tools"] = TOOL_SCHEMA
        payload["tool_choice"] = "auto"

    if case_name == "grammar_only":
        add_grammar(payload, ANSWER_GRAMMAR, server)
    elif case_name == "tools_plus_answer_grammar":
        add_grammar(payload, ANSWER_GRAMMAR, server)
    elif case_name == "tools_plus_tool_xml_grammar":
        add_grammar(payload, TOOL_CALL_XML_GRAMMAR, server)

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--server",
        choices=["llama_cpp", "vllm", "vllm_legacy"],
        default="llama_cpp",
        help=(
            "Grammar request shape to use. llama_cpp sends top-level 'grammar'; "
            "vllm sends top-level 'structured_outputs'; vllm_legacy sends "
            "top-level 'guided_grammar'."
        ),
    )
    parser.add_argument("--case", action="append", choices=[
        "tools_only",
        "grammar_only",
        "tools_plus_answer_grammar",
        "tools_plus_tool_xml_grammar",
    ])
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--jsonl", help="Optional path to write raw responses as JSONL.")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    model = args.model
    if not model:
        models = get_models(base_url, args.timeout)
        model = models[0] if models else "local-model"

    cases = args.case or [
        "tools_only",
        "grammar_only",
        "tools_plus_answer_grammar",
        "tools_plus_tool_xml_grammar",
    ]

    chat_url = base_url + "/chat/completions"
    out_fh = open(args.jsonl, "a", encoding="utf-8") if args.jsonl else None

    try:
        for i, case_name in enumerate(cases, start=1):
            if i > 1 and args.sleep > 0:
                time.sleep(args.sleep)

            payload = make_payload(case_name, model, args.max_tokens, args.temperature, args.server)
            print(f"\n=== {case_name} ===", flush=True)
            print(f"request: tools={'tools' in payload} grammar={has_grammar(payload)}", flush=True)

            record: dict[str, Any] = {
                "case": case_name,
                "request": payload,
                "ok": False,
                "response": None,
                "error": None,
            }

            try:
                response = post_json(chat_url, payload, args.timeout)
                record["ok"] = True
                record["response"] = response
                verdict = classify(case_name, response)
                choice = (response.get("choices") or [{}])[0]
                print(f"verdict: {verdict}", flush=True)
                print(f"finish_reason: {choice.get('finish_reason')}", flush=True)
                print(f"tool_calls: {json.dumps(tool_calls(choice), ensure_ascii=False)}", flush=True)
                text = message_text(choice)
                print("content:")
                print(text)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                record["error"] = f"HTTP {exc.code}: {body}"
                print(record["error"], file=sys.stderr)
            except Exception as exc:
                record["error"] = repr(exc)
                print(record["error"], file=sys.stderr)

            if out_fh:
                out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_fh.flush()
    finally:
        if out_fh:
            out_fh.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
