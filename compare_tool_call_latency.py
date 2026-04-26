#!/usr/bin/env python3
"""Compare free vs grammar-constrained Hermes tool-call generation latency.

Targets an OpenAI-compatible chat-completions server such as vLLM. The
constrained case uses vLLM's `structured_outputs.grammar` request shape.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from typing import Any


HERMES_TOOL_GRAMMAR = r'''
root ::= "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>"
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


def tool_call_count(response: dict[str, Any]) -> int:
    choices = response.get("choices") or []
    if not choices:
        return 0
    message = choices[0].get("message") or {}
    calls = message.get("tool_calls")
    return len(calls) if isinstance(calls, list) else 0


def completion_tokens(response: dict[str, Any]) -> int | None:
    usage = response.get("usage") or {}
    value = usage.get("completion_tokens")
    return value if isinstance(value, int) else None


def make_payload(mode: str, model: str, max_tokens: int, temperature: float) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Use the get_weather tool for Paris. Do not answer in text.",
            }
        ],
        "tools": TOOL_SCHEMA,
        "tool_choice": "auto",
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if mode == "fsm":
        payload["structured_outputs"] = {"grammar": HERMES_TOOL_GRAMMAR}

    return payload


def summarize(values: list[float]) -> str:
    if not values:
        return "-"
    mean = statistics.mean(values)
    med = statistics.median(values)
    return f"mean={mean:.3f}s median={med:.3f}s min={min(values):.3f}s max={max(values):.3f}s"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--jsonl", help="Optional path to write per-run raw responses.")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    model = args.model
    if not model:
        models = get_models(base_url, args.timeout)
        model = models[0] if models else "local-model"

    chat_url = base_url + "/chat/completions"
    out_fh = open(args.jsonl, "a", encoding="utf-8") if args.jsonl else None
    records: list[dict[str, Any]] = []

    try:
        for run_idx in range(args.runs):
            for mode in ("free", "fsm"):
                payload = make_payload(mode, model, args.max_tokens, args.temperature)
                start = time.perf_counter()
                error = None
                response = None
                try:
                    response = post_json(chat_url, payload, args.timeout)
                except urllib.error.HTTPError as exc:
                    error = f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}"
                except Exception as exc:
                    error = repr(exc)
                elapsed = time.perf_counter() - start

                record = {
                    "run": run_idx + 1,
                    "mode": mode,
                    "elapsed_s": elapsed,
                    "completion_tokens": completion_tokens(response or {}),
                    "tool_call_count": tool_call_count(response or {}),
                    "ok": error is None,
                    "error": error,
                    "response": response,
                }
                records.append(record)

                tok = record["completion_tokens"]
                tok_text = str(tok) if tok is not None else "?"
                status = "ok" if error is None else error
                print(
                    f"{mode:4s} run={run_idx + 1:02d} "
                    f"time={elapsed:.3f}s completion_tokens={tok_text} "
                    f"tool_calls={record['tool_call_count']} {status}",
                    flush=True,
                )

                if out_fh:
                    out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_fh.flush()
    finally:
        if out_fh:
            out_fh.close()

    print()
    for mode in ("free", "fsm"):
        subset = [r for r in records if r["mode"] == mode and r["ok"]]
        times = [float(r["elapsed_s"]) for r in subset]
        toks = [int(r["completion_tokens"]) for r in subset if isinstance(r["completion_tokens"], int)]
        successes = sum(1 for r in subset if r["tool_call_count"] > 0)
        print(f"{mode.upper()}:")
        print(f"  tool-call success: {successes}/{args.runs}")
        print(f"  latency: {summarize(times)}")
        if toks:
            print(
                "  completion tokens: "
                f"mean={statistics.mean(toks):.1f} median={statistics.median(toks):.1f} "
                f"min={min(toks)} max={max(toks)}"
            )
        else:
            print("  completion tokens: -")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
