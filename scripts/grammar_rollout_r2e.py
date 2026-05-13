"""Grammar-constrained R2E-Gym rollout for SFT data generation.

For each R2E-Gym-Lite task:
  1. Spin up the env's Docker container via RepoEnv
  2. Run a multi-turn rollout against SGLang with our IR grammar applied
  3. Per turn: model emits <think>STATE/ACTION/EXPECT</think><bash>cmd</bash>
     or <final>...</final>; we route to env.step(Action(...))
  4. After loop or <final>: env.compute_reward() to get pass/fail
  5. Save full trace + reward to JSON

Must run under R2E-Gym's venv so r2egym + datasets are importable:

    cd ~/R2E-Gym
    uv run python ~/structured-cot/scripts/grammar_rollout_r2e.py --task-idx 0

Hits SGLang directly (not the grammar proxy) — this script injects ebnf
itself via extra_body, so the proxy isn't needed here. Proxy is Pi-only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset
from openai import OpenAI

from r2egym.agenthub.action.action import Action
from r2egym.agenthub.environment.env import EnvArgs, RepoEnv


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
GRAMMAR_PATH = REPO_ROOT / "grammars" / "fsm_grammar_pi_turn.gbnf"

BASH_RE = re.compile(r"<bash>(.*?)</bash>", re.DOTALL)
FINAL_RE = re.compile(r"<final>(.*?)</final>", re.DOTALL)

SYSTEM_PROMPT = """\
You are a software engineering agent solving a GitHub issue in a Linux sandbox.
Your workspace is /testbed. Tests live at /testbed/r2e_tests/ — DO NOT modify them.

Each turn output EXACTLY this structure and nothing else:

<think>
STATE: <one short line: current state of your investigation/fix>
ACTION: <one short line: what you will do next and why>
EXPECT: <one short line: what you expect the observation to look like>
</think>
<bash>SINGLE_COMMAND</bash>     to run one bash command
OR
<final>SHORT_SUMMARY</final>    to declare the task complete

Rules:
- Each bash command runs in /testbed.
- These first-tokens are BLOCKED: git, ipython, jupyter, nohup.
- File edits via heredoc, sed, awk, or python scripts are fine.
- Make minimal changes to non-test files to fix the issue.
- Use <final> only when you've verified the fix works.
"""


def extract_action(text: str) -> dict | None:
    final_m = FINAL_RE.search(text)
    if final_m:
        return {"kind": "final", "body": final_m.group(1).strip()}
    bash_m = BASH_RE.search(text)
    if bash_m:
        return {"kind": "bash", "body": bash_m.group(1).strip()}
    return None


def observation_to_text(obs: Any, char_limit: int = 8000) -> str:
    """Best-effort render of an r2egym Observation back to a string."""
    for attr in ("output", "obs", "content", "stdout"):
        val = getattr(obs, attr, None)
        if isinstance(val, str) and val:
            text = val
            break
    else:
        text = obs if isinstance(obs, str) else repr(obs)
    if len(text) > char_limit:
        text = text[:char_limit] + f"\n...[truncated {len(text) - char_limit} chars]"
    return text


def format_observation_block(obs: Any) -> str:
    return f"<observation>\n{observation_to_text(obs)}\n</observation>"


def run_one_rollout(
    *,
    task_row: dict,
    grammar: str,
    base_url: str,
    model: str,
    max_turns: int,
    max_tokens: int,
    step_timeout: int,
    reward_timeout: int,
) -> dict:
    env = RepoEnv(
        EnvArgs(ds=task_row),
        verbose=False,
        step_timeout=step_timeout,
        reward_timeout=reward_timeout,
    )
    task_instruction = env.get_task_instruction()
    client = OpenAI(base_url=base_url, api_key="not-needed")

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_instruction},
    ]
    turns: list[dict] = []
    ended = "max_turns"

    try:
        for turn_idx in range(max_turns):
            r = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
                extra_body={"ebnf": grammar},
            )
            text = r.choices[0].message.content or ""
            completion_tokens = r.usage.completion_tokens if r.usage else None
            turn: dict = {
                "turn": turn_idx,
                "text": text,
                "completion_tokens": completion_tokens,
            }

            action_spec = extract_action(text)
            if not action_spec:
                turn["error"] = "no parseable <bash> or <final> tag in response"
                turns.append(turn)
                ended = "unparseable"
                break
            turn["action"] = action_spec

            if action_spec["kind"] == "final":
                try:
                    obs, _r, _done, _info = env.step(
                        Action(
                            "finish",
                            {"command": "submit", "result": action_spec["body"][:1000]},
                        ),
                        timeout=step_timeout,
                    )
                    turn["observation"] = observation_to_text(obs, char_limit=2000)
                except Exception as e:
                    turn["observation_error"] = f"{type(e).__name__}: {e}"
                turns.append(turn)
                ended = "final"
                break

            cmd = action_spec["body"]
            try:
                obs, _step_reward, done, info = env.step(
                    Action("execute_bash", {"cmd": cmd}),
                    timeout=step_timeout,
                )
                obs_block = format_observation_block(obs)
                turn["observation"] = obs_block
            except Exception as e:
                obs_block = (
                    f"<observation>\n[execution error] "
                    f"{type(e).__name__}: {e}\n</observation>"
                )
                turn["observation"] = obs_block
                done = False

            turns.append(turn)
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": obs_block})
            if done:
                ended = "env_done"
                break

        # Run the verifier
        final_reward: float | None
        reward_error: str | None
        try:
            final_reward = float(env.compute_reward(timeout=reward_timeout))
            reward_error = None
        except Exception as e:
            final_reward = None
            reward_error = f"{type(e).__name__}: {e}"

        return {
            "task_id": task_row.get("commit_hash") or task_row.get("docker_image"),
            "repo_name": task_row.get("repo_name"),
            "docker_image": task_row.get("docker_image"),
            "turns": turns,
            "ended": ended,
            "final_reward": final_reward,
            "reward_error": reward_error,
            "total_completion_tokens": sum(
                (t.get("completion_tokens") or 0) for t in turns
            ),
            "problem_statement_preview": (task_row.get("problem_statement") or "")[
                :400
            ],
        }
    finally:
        try:
            env.close()
        except Exception:
            pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task-idx", type=int, default=0)
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--model", default="Qwen/Qwen3.6-27B")
    p.add_argument(
        "--base-url",
        default="http://127.0.0.1:30000/v1",
        help="SGLang URL. Script injects grammar via extra_body so the grammar "
             "proxy is NOT in this path — point at SGLang directly.",
    )
    p.add_argument("--dataset", default="R2E-Gym/R2E-Gym-Lite")
    p.add_argument("--split", default="train")
    p.add_argument("--step-timeout", type=int, default=90)
    p.add_argument("--reward-timeout", type=int, default=300)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if not GRAMMAR_PATH.exists():
        sys.exit(f"grammar not found: {GRAMMAR_PATH}")
    grammar = GRAMMAR_PATH.read_text()
    print(f"loaded grammar: {GRAMMAR_PATH} ({len(grammar)} chars)")

    print(f"loading dataset {args.dataset} split={args.split}")
    ds = load_dataset(args.dataset, split=args.split)
    print(f"  rows: {len(ds)}")
    task_row = ds[args.task_idx]
    print(f"task idx={args.task_idx}  repo={task_row.get('repo_name')}  "
          f"image={task_row.get('docker_image')}")

    result = run_one_rollout(
        task_row=task_row,
        grammar=grammar,
        base_url=args.base_url,
        model=args.model,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        step_timeout=args.step_timeout,
        reward_timeout=args.reward_timeout,
    )

    out_path = (
        Path(args.out)
        if args.out
        else REPO_ROOT / f"r2e_rollout_idx{args.task_idx}.json"
    )
    out_path.write_text(json.dumps(result, indent=2, default=str))

    print()
    print(f"=== summary ===")
    print(f"ended:         {result['ended']}")
    print(f"turns:         {len(result['turns'])}")
    print(f"final_reward:  {result['final_reward']}")
    print(f"reward_error:  {result['reward_error']}")
    print(f"tokens:        {result['total_completion_tokens']}")
    print(f"trace:         {out_path}")


if __name__ == "__main__":
    main()
