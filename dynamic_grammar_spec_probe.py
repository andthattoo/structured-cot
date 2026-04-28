#!/usr/bin/env python3
"""Probe grammar-in-grammar via a constrained next-grammar spec.

Step 1 is constrained by a meta-grammar. The model emits:
  NEXT_GRAMMAR / ALLOWED_CITY / EDGE / GOAL / STATE

Step 2 compiles that safe spec into real GBNF and sends it with tools enabled
to patched llama.cpp. This tests dynamic next-step grammar selection without
letting the model emit arbitrary GBNF.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from typing import Any


META_GRAMMAR = r'''
root ::= "NEXT_GRAMMAR: " grammar_id "\n" "ALLOWED_CITY: " city "\n" "EDGE: " line "GOAL: " line "STATE: " line
line ::= [^\n]+ "\n"
grammar_id ::= "tool_plan" | "tool_plan_city"
city ::= "Paris" | "London" | "Tokyo"
'''


CITY_TOOL_PLAN_TEMPLATE = r'''
root ::= "GOAL: " line "TOOL: get_weather\n" "ARGS: city=" city "\n"
line ::= [^\n]+ "\n"
city ::= "__CITY__"
'''


GENERIC_TOOL_PLAN_GRAMMAR = r'''
root ::= "GOAL: " line "TOOL: " line "ARGS: " line
line ::= [^\n]+ "\n"
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
                    "city": {"type": "string", "description": "City name."}
                },
                "required": ["city"],
            },
        },
    }
]


DEFAULT_TASK = "Use the get_weather tool for Paris."


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
    req = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"Authorization": "Bearer local"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []
    return [item.get("id", "") for item in data.get("data", []) if item.get("id")]


def first_choice(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    return choices[0] if choices else {}


def first_message(response: dict[str, Any]) -> dict[str, Any]:
    choice = first_choice(response)
    return choice.get("message") or {}


def message_text(message: dict[str, Any]) -> str:
    for key in ("content", "reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def parse_spec(text: str) -> dict[str, str]:
    spec: dict[str, str] = {}
    for line in text.splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        spec[key.strip()] = value.strip()

    missing = [key for key in ("GOAL", "STATE", "EDGE", "NEXT_GRAMMAR", "ALLOWED_CITY") if key not in spec]
    if missing:
        raise ValueError(f"spec missing fields {missing}: {text!r}")

    if spec["NEXT_GRAMMAR"] not in {"tool_plan", "tool_plan_city"}:
        raise ValueError(f"unknown NEXT_GRAMMAR: {spec['NEXT_GRAMMAR']!r}")

    if spec["ALLOWED_CITY"] not in {"Paris", "London", "Tokyo"}:
        raise ValueError(f"unknown ALLOWED_CITY: {spec['ALLOWED_CITY']!r}")

    return spec


def compile_spec_to_grammar(spec: dict[str, str]) -> str:
    if spec["NEXT_GRAMMAR"] == "tool_plan_city":
        city = spec["ALLOWED_CITY"]
        return CITY_TOOL_PLAN_TEMPLATE.replace("__CITY__", city).strip() + "\n"
    return GENERIC_TOOL_PLAN_GRAMMAR.strip() + "\n"


def tool_calls(response: dict[str, Any]) -> list[Any]:
    calls = first_message(response).get("tool_calls")
    return calls if isinstance(calls, list) else []


def generate_spec(
    base_url: str,
    model: str,
    task: str,
    timeout: int,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    prompt = f"""Choose the next-step reasoning grammar spec for this task.

Task:
{task}

Rules:
- NEXT_GRAMMAR should be tool_plan_city when the city is clear.
- ALLOWED_CITY should be the city from the task.
- EDGE should mention the main risk in one short sentence.
- GOAL should say what the next reasoning step should accomplish.
- STATE should summarize known task state in one short sentence.
"""

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "grammar": META_GRAMMAR,
        "reasoning": {"effort": "none"},
        "reasoning_budget": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    return post_json(base_url.rstrip("/") + "/chat/completions", payload, timeout)


def run_task(
    base_url: str,
    model: str,
    task: str,
    grammar: str,
    timeout: int,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": task}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "tools": TOOL_SCHEMA,
        "tool_choice": "auto",
        "grammar": grammar,
    }
    return post_json(base_url.rstrip("/") + "/chat/completions", payload, timeout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--spec-max-tokens", type=int, default=256)
    parser.add_argument("--run-max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--json", help="Optional path to write full probe record.")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    model = args.model
    if not model:
        models = get_models(base_url, args.timeout)
        model = models[0] if models else "local-model"

    record: dict[str, Any] = {
        "task": args.task,
        "model": model,
        "meta_grammar": META_GRAMMAR,
        "spec_response": None,
        "spec": None,
        "compiled_grammar": None,
        "run_response": None,
        "error": None,
    }

    try:
        spec_response = generate_spec(
            base_url=base_url,
            model=model,
            task=args.task,
            timeout=args.timeout,
            max_tokens=args.spec_max_tokens,
            temperature=args.temperature,
        )
        record["spec_response"] = spec_response
        spec_text = message_text(first_message(spec_response))
        spec = parse_spec(spec_text)
        grammar = compile_spec_to_grammar(spec)
        record["spec"] = spec
        record["compiled_grammar"] = grammar

        print("=== generated spec ===")
        print(spec_text)
        print("\n=== compiled grammar ===")
        print(grammar)

        run_response = run_task(
            base_url=base_url,
            model=model,
            task=args.task,
            grammar=grammar,
            timeout=args.timeout,
            max_tokens=args.run_max_tokens,
            temperature=args.temperature,
        )
        record["run_response"] = run_response

        message = first_message(run_response)
        usage = run_response.get("usage") or {}
        print("\n=== constrained run ===")
        print(f"finish_reason: {first_choice(run_response).get('finish_reason')}")
        print(f"completion_tokens: {usage.get('completion_tokens')}")
        print(f"tool_calls: {json.dumps(tool_calls(run_response), ensure_ascii=False)}")
        print("reasoning_content:")
        print(message.get("reasoning_content") or message.get("reasoning") or "")
        print("content:")
        print(message.get("content") or "")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        record["error"] = f"HTTP {exc.code}: {body}"
        print(record["error"])
    except Exception as exc:
        record["error"] = repr(exc)
        print(record["error"])

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)
            fh.write("\n")

    return 1 if record["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
