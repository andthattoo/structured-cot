#!/usr/bin/env python3
"""Probe model-generated next-step grammars with patched llama.cpp.

Step 1 asks the model to generate a small GBNF grammar for the next reasoning
step. Step 2 sends that grammar with tools enabled and checks whether patched
llama.cpp constrains `reasoning_content` while preserving parsed tool calls.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from typing import Any


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


def first_choice_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    if not choices:
        return {}
    return choices[0].get("message") or {}


def collect_text(message: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("content", "reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    return "\n".join(parts)


def choice_finish_reason(response: dict[str, Any]) -> str | None:
    choices = response.get("choices") or []
    if not choices:
        return None
    value = choices[0].get("finish_reason")
    return value if isinstance(value, str) else None


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in model output:\n{text}")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("top-level JSON value is not an object")
    return parsed


def validate_generated_grammar(data: dict[str, Any]) -> str:
    grammar = data.get("grammar")
    if not isinstance(grammar, str) or "root ::=" not in grammar:
        raise ValueError(f"generated JSON does not contain a GBNF grammar: {data}")
    if "..." in grammar:
        raise ValueError(f"generated grammar still contains a placeholder: {grammar!r}")
    if "<think>" in grammar or "</think>" in grammar:
        raise ValueError(f"generated grammar must not include think tags: {grammar!r}")
    if "<tool_call>" in grammar or "tool_call" in grammar:
        raise ValueError(f"generated grammar must not include tool-call syntax: {grammar!r}")
    return grammar


def tool_calls(response: dict[str, Any]) -> list[Any]:
    message = first_choice_message(response)
    calls = message.get("tool_calls")
    return calls if isinstance(calls, list) else []


def generate_grammar(
    base_url: str,
    model: str,
    task: str,
    timeout: int,
    max_tokens: int,
    temperature: float,
    grammar_reasoning_budget: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = f"""Generate one small llama.cpp GBNF grammar.

The grammar is for the NEXT assistant reasoning step only.
Important constraints:
- The grammar is applied to message.reasoning_content only.
- Do not include <think> or </think> tags.
- Do not include tool-call syntax.
- Keep it short: 2 to 5 labeled lines.
- It must end in a terminal state after the final newline.
- Use llama.cpp GBNF syntax.
- Use only simple rules like: line ::= [^\\n]+ "\\n"
- Do not explain. Do not include markdown.

Next task:
{task}

Return only compact JSON:
{{"name":"tool_plan","grammar":"root ::= \\"GOAL: \\" line \\"TOOL: \\" line \\"ARGS: \\" line\\nline ::= [^\\\\n]+ \\"\\\\n\\""}}
"""

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if grammar_reasoning_budget is not None:
        payload["reasoning_budget"] = grammar_reasoning_budget

    response = post_json(base_url.rstrip("/") + "/chat/completions", payload, timeout)
    finish_reason = choice_finish_reason(response)
    if finish_reason == "length":
        raise ValueError("grammar-generation call hit max_tokens before producing content")

    message = first_choice_message(response)
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        text = collect_text(message)
        raise ValueError(f"grammar-generation call produced no assistant content; got:\n{text}")

    data = extract_json_object(content)
    validate_generated_grammar(data)
    return data, response


def run_with_generated_grammar(
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
    parser.add_argument("--grammar-max-tokens", type=int, default=512)
    parser.add_argument("--run-max-tokens", type=int, default=1024)
    parser.add_argument(
        "--grammar-reasoning-budget",
        type=int,
        default=0,
        help="Reasoning budget for the grammar-generation call; use -1 to omit.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--json", help="Optional path to write the full probe record.")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    model = args.model
    if not model:
        models = get_models(base_url, args.timeout)
        model = models[0] if models else "local-model"

    record: dict[str, Any] = {
        "task": args.task,
        "model": model,
        "generated": None,
        "grammar_response": None,
        "run_response": None,
        "error": None,
    }

    try:
        generated, grammar_response = generate_grammar(
            base_url=base_url,
            model=model,
            task=args.task,
            timeout=args.timeout,
            max_tokens=args.grammar_max_tokens,
            temperature=args.temperature,
            grammar_reasoning_budget=None
            if args.grammar_reasoning_budget < 0
            else args.grammar_reasoning_budget,
        )
        grammar = validate_generated_grammar(generated)
        record["generated"] = generated
        record["grammar_response"] = grammar_response

        print("=== generated grammar ===")
        print(f"name: {generated.get('name', '<unnamed>')}")
        print(grammar)

        run_response = run_with_generated_grammar(
            base_url=base_url,
            model=model,
            task=args.task,
            grammar=grammar,
            timeout=args.timeout,
            max_tokens=args.run_max_tokens,
            temperature=args.temperature,
        )
        record["run_response"] = run_response

        message = first_choice_message(run_response)
        calls = tool_calls(run_response)
        usage = run_response.get("usage") or {}

        print("\n=== constrained run ===")
        print(f"finish_reason: {(run_response.get('choices') or [{}])[0].get('finish_reason')}")
        print(f"completion_tokens: {usage.get('completion_tokens')}")
        print(f"tool_calls: {json.dumps(calls, ensure_ascii=False)}")
        print("reasoning_content:")
        print(message.get("reasoning_content") or message.get("reasoning") or "")
        print("content:")
        print(message.get("content") or "")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        record["error"] = f"HTTP {exc.code}: {body}"
        print(record["error"], file=sys.stderr)
    except Exception as exc:
        record["error"] = repr(exc)
        print(record["error"], file=sys.stderr)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)
            fh.write("\n")

    return 1 if record["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
