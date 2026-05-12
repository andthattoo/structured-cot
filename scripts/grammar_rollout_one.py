"""Single grammar-constrained rollout against the Pi harness (dry-run).

Talks to either the llama.cpp OpenAI-compatible server (run_server.sh)
or the SGLang server (run_sglang_server.sh). Both accept GBNF — only
the sampling-param field name differs:

    llama.cpp : extra_body={"grammar": <gbnf>}
    sglang    : extra_body={"ebnf":    <gbnf>}

Does NOT execute bash commands — observations are stubbed so we can
inspect grammar mechanics + IR shape without sandboxing. Swap
stub_observation() for real execution once the IR looks right.

Usage:
    # llama.cpp (default)
    python scripts/grammar_rollout_one.py --task-idx 0

    # SGLang on 2x A100 (port 30000)
    python scripts/grammar_rollout_one.py --backend sglang --task-idx 0
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = REPO_ROOT / "data" / "pi_tasks" / "tracejepa_pi_2500_v1_qwen27.index.jsonl"
GRAMMAR_PATH = REPO_ROOT / "grammars" / "fsm_grammar_pi_turn.gbnf"
HF_DATASET = "andthattoo/etpi-pi-traces"

BACKEND_DEFAULTS = {
    "llamacpp": {"base_url": "http://127.0.0.1:8000/v1", "grammar_field": "grammar"},
    "sglang":   {"base_url": "http://127.0.0.1:30000/v1", "grammar_field": "ebnf"},
}

SYSTEM_PROMPT = (
    "You are an autonomous shell agent solving a task by issuing bash commands.\n"
    "Each turn, output EXACTLY this structure and nothing else:\n"
    "<think>\n"
    "STATE: <one short line: current workspace state>\n"
    "ACTION: <one short line: what you'll do next and why>\n"
    "EXPECT: <one short line: what you'll see in the observation if it works>\n"
    "</think>\n"
    "<bash>SINGLE_COMMAND</bash>   to run one bash command and read its output\n"
    "OR\n"
    "<final>SUMMARY</final>        to declare the task complete\n"
)

BASH_RE = re.compile(r"<bash>(.*?)</bash>", re.DOTALL)
FINAL_RE = re.compile(r"<final>(.*?)</final>", re.DOTALL)


def load_task_local(idx: int) -> dict:
    with INDEX_PATH.open() as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)
    raise IndexError(f"task index {idx} not found in {INDEX_PATH}")


def load_task_hf(idx: int, run_id: str | None) -> dict:
    from datasets import load_dataset
    ds = load_dataset(HF_DATASET, split="train", streaming=True)
    if run_id:
        ds = ds.filter(lambda r: r.get("run_id") == run_id)
    for i, row in enumerate(ds):
        if i == idx:
            return dict(row)
    raise IndexError(
        f"task index {idx} not found in {HF_DATASET} (run_id={run_id})"
    )


def load_task(source: str, idx: int, run_id: str | None) -> dict:
    if source == "local":
        return load_task_local(idx)
    if source == "hf":
        return load_task_hf(idx, run_id)
    raise ValueError(f"unknown source: {source}")


def stub_observation(command: str) -> str:
    payload = {
        "command": command,
        "returncode": 0,
        "stdout": "(simulated: no output — this is a dry-run)",
        "stderr": "",
        "note": "DRY-RUN: bash was not executed. Pretend it succeeded and continue.",
    }
    return json.dumps(payload, indent=2)


def run_rollout(
    task: dict,
    grammar: str,
    max_turns: int,
    model: str,
    base_url: str,
    max_tokens_per_turn: int,
    grammar_field: str,
) -> dict:
    client = OpenAI(base_url=base_url, api_key="not-needed")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task["prompt"]},
    ]
    turns: list[dict] = []

    for turn_idx in range(max_turns):
        r = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens_per_turn,
            temperature=0.0,
            extra_body={grammar_field: grammar},
        )
        text = r.choices[0].message.content or ""
        completion_tokens = r.usage.completion_tokens if r.usage else None
        turn: dict = {
            "turn": turn_idx,
            "text": text,
            "completion_tokens": completion_tokens,
        }

        final_m = FINAL_RE.search(text)
        if final_m:
            turn["action"] = {"kind": "final", "body": final_m.group(1).strip()}
            turns.append(turn)
            break

        bash_m = BASH_RE.search(text)
        if bash_m:
            command = bash_m.group(1).strip()
            obs = stub_observation(command)
            obs_block = f"<observation>\n{obs}\n</observation>"
            turn["action"] = {"kind": "bash", "body": command}
            turn["observation"] = obs
            turns.append(turn)
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": obs_block})
            continue

        turn["action"] = {"kind": "unparseable", "body": text}
        turns.append(turn)
        break

    return {
        "task_id": task.get("task_id"),
        "prompt": task["prompt"],
        "turns": turns,
        "total_completion_tokens": sum(
            (t.get("completion_tokens") or 0) for t in turns
        ),
        "ended": turns[-1]["action"]["kind"] if turns else "no_turns",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task-idx", type=int, default=0)
    p.add_argument("--max-turns", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--backend", choices=list(BACKEND_DEFAULTS), default="llamacpp")
    p.add_argument("--base-url", default=None,
                   help="override; defaults follow --backend")
    p.add_argument("--source", choices=["local", "hf"], default="hf",
                   help="task source: local jsonl or HF dataset")
    p.add_argument("--run-id", default=None,
                   help="HF only: filter to a specific run_id "
                        "(e.g. tracejepa_pi_2500_v1_qwen27)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    defaults = BACKEND_DEFAULTS[args.backend]
    base_url = args.base_url or defaults["base_url"]
    grammar_field = defaults["grammar_field"]

    task = load_task(args.source, args.task_idx, args.run_id)
    grammar = GRAMMAR_PATH.read_text()

    result = run_rollout(
        task=task,
        grammar=grammar,
        max_turns=args.max_turns,
        model=args.model,
        base_url=base_url,
        max_tokens_per_turn=args.max_tokens,
        grammar_field=grammar_field,
    )

    out_path = (
        Path(args.out)
        if args.out
        else REPO_ROOT / f"rollout_task{args.task_idx}.json"
    )
    out_path.write_text(json.dumps(result, indent=2))

    print(f"=== Task {args.task_idx}  id={task.get('task_id')} ===")
    print(f"Turns: {len(result['turns'])}  "
          f"Total completion tokens: {result['total_completion_tokens']}  "
          f"Ended: {result['ended']}")
    for t in result["turns"]:
        print(f"\n--- turn {t['turn']}  ({t['completion_tokens']} tokens) ---")
        print(t["text"])
        if "observation" in t:
            print(f"\n[stub observation returned to model]:")
            print(t["observation"])
    print(f"\nFull trace written to: {out_path}")


if __name__ == "__main__":
    main()
