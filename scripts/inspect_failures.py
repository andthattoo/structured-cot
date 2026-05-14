"""Inspect failure modes in rollouts that came back reward=0.

Categorizes by `ended` field (max_turns / final / unparseable / env_done /
error) and prints example trajectories per category so we can spot
patterns: premature <final>, runaway loops, bash-quoting errors, etc.

Usage:
    uv run python scripts/inspect_failures.py \\
        --traces-dir ~/structured-cot/traces --recurse

    # focus on one category, more examples:
    uv run python scripts/inspect_failures.py \\
        --traces-dir ~/structured-cot/traces --recurse \\
        --only-ending max_turns --per-category 10 --last-turns 4
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


BASH_RE = re.compile(r"<bash>(.*?)</bash>", re.DOTALL)
FINAL_RE = re.compile(r"<final>(.*?)</final>", re.DOTALL)


def trace_summary(t: dict) -> dict:
    turns = t.get("turns", [])
    n = len(turns)
    last = turns[-1] if turns else {}
    bash_commands = []
    for turn in turns:
        m = BASH_RE.search(turn.get("text", ""))
        if m:
            bash_commands.append(m.group(1).strip())
    last_bash = bash_commands[-1] if bash_commands else None
    # crude "looping" detection: same first-token in 3+ consecutive bash actions
    first_tokens = [b.strip().split()[0] if b.strip() else "" for b in bash_commands]
    max_consecutive = 1
    cur = 1
    for i in range(1, len(first_tokens)):
        if first_tokens[i] == first_tokens[i - 1]:
            cur += 1
            max_consecutive = max(max_consecutive, cur)
        else:
            cur = 1
    return {
        "n_turns": n,
        "n_bash": len(bash_commands),
        "last_ending_action": last.get("action", {}).get("kind"),
        "max_consecutive_same_cmd": max_consecutive,
        "first_token_distribution": Counter(first_tokens).most_common(3),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--traces-dir", required=True)
    p.add_argument("--recurse", action="store_true")
    p.add_argument("--per-category", type=int, default=3)
    p.add_argument("--last-turns", type=int, default=2)
    p.add_argument("--only-ending", default=None,
                   help="show only one ending category (max_turns/final/...)")
    p.add_argument("--text-chars", type=int, default=500)
    p.add_argument("--obs-chars", type=int, default=200)
    args = p.parse_args()

    traces_dir = Path(args.traces_dir)
    paths = (
        traces_dir.glob("**/*.json") if args.recurse else traces_dir.glob("*.json")
    )

    by_ending: dict[str, list[dict]] = defaultdict(list)
    all_summaries: list[dict] = []
    for path in paths:
        if path.name.startswith("_"):
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if data.get("status") != "done":
            continue
        if data.get("final_reward") == 1.0:
            continue
        ending = data.get("ended") or "?"
        by_ending[ending].append(data)
        s = trace_summary(data)
        s["ended"] = ending
        all_summaries.append(s)

    print("=== aggregate failure breakdown ===")
    total = sum(len(v) for v in by_ending.values())
    for ending, traces in sorted(by_ending.items(), key=lambda x: -len(x[1])):
        pct = len(traces) / total * 100 if total else 0
        print(f"  {ending:12s}: {len(traces):5d}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':12s}: {total}")
    print()

    looping = sum(1 for s in all_summaries if s["max_consecutive_same_cmd"] >= 3)
    print(f"trajectories with >=3 consecutive same-first-token bash commands: "
          f"{looping} ({looping/total*100:.1f}%)" if total else "no failures")
    print()
    avg_turns_by_ending = {
        k: round(sum(len(t.get("turns", [])) for t in v) / len(v), 1)
        for k, v in by_ending.items() if v
    }
    print("avg turns per failure mode:")
    for k, v in sorted(avg_turns_by_ending.items()):
        print(f"  {k:12s}: {v}")
    print()

    for ending, traces in sorted(by_ending.items(), key=lambda x: -len(x[1])):
        if args.only_ending and ending != args.only_ending:
            continue
        print(f"\n{'='*70}")
        print(f"  ENDING: {ending}  ({len(traces)} traces)")
        print(f"{'='*70}")
        for t in traces[: args.per_category]:
            turns = t.get("turns", [])
            print(f"\n--- task {t.get('task_id')}  reward={t.get('final_reward')}  "
                  f"turns={len(turns)} ---")
            problem = t.get("problem_statement_preview", "")
            if problem:
                print(f"problem: {problem[:200]}...")
            for turn in turns[-args.last_turns :]:
                print(f"\n  ## turn {turn.get('turn')}  "
                      f"({turn.get('completion_tokens')} toks)  "
                      f"action={turn.get('action',{}).get('kind')}")
                print(f"  text: {turn.get('text', '')[: args.text_chars]}")
                obs = turn.get("observation")
                if obs:
                    print(f"  obs:  {obs[: args.obs_chars]}")


if __name__ == "__main__":
    main()
