"""Build SFT and DPO datasets from a directory of grammar_rollout_batch.py traces.

For each task (grouped by row_idx) with at least one reward=1.0 trace:
  - SFT: emit one record using the *shortest* successful trace (rewards
    brevity-given-correctness, matching the broader project goal).
  - DPO: pair each successful trace with each failed trace from the same
    task (up to --max-dpo-pairs-per-task per task).

Output is JSONL using the messages-list format that HuggingFace TRL's
SFTTrainer and DPOTrainer accept.

Usage:
    uv run python scripts/build_sft_dpo_dataset.py \\
        --traces-dir traces/20260513_102140 \\
        --out-dir datasets/20260513_102140

Inputs assumed:
  - traces dir contains per-rollout *.json files with fields:
      task_id, row_idx, sample_idx, status, final_reward, turns, ended
  - R2E-Gym/R2E-Gym-Lite split=train must be loadable (for the
    canonical task_instruction text — traces only save a preview).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from grammar_rollout_r2e import SYSTEM_PROMPT  # noqa: E402


def load_traces(traces_dir: Path) -> list[dict]:
    out = []
    for path in sorted(traces_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue  # _status.json
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"[warn] skipping unreadable: {path}", file=sys.stderr)
            continue
        if data.get("status") == "error":
            continue
        if data.get("status") == "skipped":
            continue
        out.append(data)
    return out


def turn_count(trace: dict) -> int:
    return len(trace.get("turns", []))


def build_messages(trace: dict, task_instruction: str) -> list[dict]:
    """Reconstruct the chat-format messages list from a trace.

    Format: [system, user, (assistant, user-obs)*] where each assistant
    message is the model's grammar-shaped output and each user-obs is the
    sandbox observation block we fed back into context.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_instruction},
    ]
    for turn in trace.get("turns", []):
        text = turn.get("text", "")
        messages.append({"role": "assistant", "content": text})
        obs = turn.get("observation")
        # The last turn that emitted <final> has no following observation
        # (we record the finish observation as 'observation' on that turn,
        # but R2E returns "<<<Finished>>>" and we typically don't want to
        # extend the trajectory beyond that). Same applies if the turn was
        # the last and we hit max_turns mid-rollout.
        if obs and turn.get("action", {}).get("kind") != "final":
            messages.append({"role": "user", "content": obs})
    return messages


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--traces-dir", required=True)
    p.add_argument("--out-dir", default=None,
                   help="default: datasets/<traces-dir-basename>")
    p.add_argument("--dataset", default="R2E-Gym/R2E-Gym-Lite")
    p.add_argument("--split", default="train")
    p.add_argument("--max-dpo-pairs-per-task", type=int, default=4,
                   help="cap on (pos, neg) pairs emitted per task")
    p.add_argument("--max-trace-chars", type=int, default=120000,
                   help="drop traces whose joined text is longer than this "
                        "(rare runaway rollouts)")
    args = p.parse_args()

    traces_dir = Path(args.traces_dir)
    if not traces_dir.exists():
        sys.exit(f"traces dir not found: {traces_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO_ROOT / "datasets" / traces_dir.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build] traces: {traces_dir}")
    print(f"[build] out:    {out_dir}")
    print(f"[build] loading dataset {args.dataset} split={args.split}")
    ds = load_dataset(args.dataset, split=args.split)
    print(f"[build] {len(ds)} dataset rows")

    print("[build] reading traces...")
    traces = load_traces(traces_dir)
    print(f"[build] {len(traces)} usable trace files")

    # Group by row_idx
    by_task: dict[int, list[dict]] = defaultdict(list)
    for t in traces:
        ri = t.get("row_idx")
        if ri is None:
            continue
        by_task[ri].append(t)

    bucket_counts: Counter = Counter()
    sft_records: list[dict] = []
    dpo_records: list[dict] = []
    dropped_too_long = 0

    for row_idx, samples in sorted(by_task.items()):
        positives = [s for s in samples if s.get("final_reward") == 1.0]
        negatives = [s for s in samples if s.get("final_reward") == 0.0]
        n_pos, n_neg = len(positives), len(negatives)
        bucket = f"{n_pos}pos_{n_neg}neg"
        bucket_counts[bucket] += 1

        if not positives:
            continue  # no SFT, no DPO

        task_instruction = ds[row_idx].get("prompt") or ds[row_idx].get(
            "problem_statement", ""
        )
        if not task_instruction:
            continue

        # SFT: shortest positive
        best_pos = min(positives, key=turn_count)
        messages = build_messages(best_pos, task_instruction)
        joined_chars = sum(len(m["content"]) for m in messages)
        if joined_chars > args.max_trace_chars:
            dropped_too_long += 1
        else:
            sft_records.append({
                "task_id": best_pos.get("task_id"),
                "row_idx": row_idx,
                "sample_idx": best_pos.get("sample_idx"),
                "turns": turn_count(best_pos),
                "ended": best_pos.get("ended"),
                "messages": messages,
            })

        # DPO: positives × negatives, capped
        if negatives:
            pairs = []
            for pos in positives:
                for neg in negatives:
                    pairs.append((pos, neg))
                    if len(pairs) >= args.max_dpo_pairs_per_task:
                        break
                if len(pairs) >= args.max_dpo_pairs_per_task:
                    break

            for pos, neg in pairs:
                prompt_msgs = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task_instruction},
                ]
                chosen_msgs = build_messages(pos, task_instruction)[2:]
                rejected_msgs = build_messages(neg, task_instruction)[2:]
                if not chosen_msgs or not rejected_msgs:
                    continue
                ch_chars = sum(len(m["content"]) for m in chosen_msgs)
                rj_chars = sum(len(m["content"]) for m in rejected_msgs)
                if max(ch_chars, rj_chars) > args.max_trace_chars:
                    dropped_too_long += 1
                    continue
                dpo_records.append({
                    "task_id": pos.get("task_id"),
                    "row_idx": row_idx,
                    "pos_sample_idx": pos.get("sample_idx"),
                    "neg_sample_idx": neg.get("sample_idx"),
                    "chosen_turns": turn_count(pos),
                    "rejected_turns": turn_count(neg),
                    "prompt": prompt_msgs,
                    "chosen": chosen_msgs,
                    "rejected": rejected_msgs,
                })

    sft_path = out_dir / "sft.jsonl"
    dpo_path = out_dir / "dpo.jsonl"
    stats_path = out_dir / "stats.json"

    with sft_path.open("w") as f:
        for rec in sft_records:
            f.write(json.dumps(rec) + "\n")
    with dpo_path.open("w") as f:
        for rec in dpo_records:
            f.write(json.dumps(rec) + "\n")

    avg_sft_turns = (sum(r["turns"] for r in sft_records) /
                     len(sft_records)) if sft_records else 0
    avg_dpo_chosen_turns = (sum(r["chosen_turns"] for r in dpo_records) /
                            len(dpo_records)) if dpo_records else 0
    avg_dpo_rejected_turns = (sum(r["rejected_turns"] for r in dpo_records) /
                              len(dpo_records)) if dpo_records else 0
    stats = {
        "traces_dir": str(traces_dir),
        "tasks_seen": len(by_task),
        "buckets_pos_neg": dict(bucket_counts),
        "sft_records": len(sft_records),
        "dpo_records": len(dpo_records),
        "avg_sft_turns": round(avg_sft_turns, 2),
        "avg_dpo_chosen_turns": round(avg_dpo_chosen_turns, 2),
        "avg_dpo_rejected_turns": round(avg_dpo_rejected_turns, 2),
        "dropped_too_long": dropped_too_long,
        "max_dpo_pairs_per_task": args.max_dpo_pairs_per_task,
        "max_trace_chars": args.max_trace_chars,
    }
    stats_path.write_text(json.dumps(stats, indent=2))

    print()
    print("=== build summary ===")
    print(json.dumps(stats, indent=2))
    print()
    print(f"sft:   {sft_path}")
    print(f"dpo:   {dpo_path}")
    print(f"stats: {stats_path}")


if __name__ == "__main__":
    main()
