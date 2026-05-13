"""Batch grammar-constrained R2E-Gym rollout for SFT data generation.

For each (task, sample_idx) pair: spin up RepoEnv, run a grammar-shaped
multi-turn rollout, save trace + reward to a per-rollout JSON file.

Resume support: if a sample's output file already exists, it's skipped.
Status file at traces/<run_id>/_status.json updates every 30 seconds.

Designed to run unattended overnight via systemd (see systemd/*.service).

Must run under R2E-Gym's venv:

    cd ~/R2E-Gym
    uv run python ~/structured-cot/scripts/grammar_rollout_batch.py \\
        --n-tasks 200 --samples-per-task 4 --concurrency 4 \\
        --max-turns 30 --shuffle --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset

# Reuse the single-task implementation
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from grammar_rollout_r2e import (  # noqa: E402
    GRAMMAR_PATH,
    run_one_rollout,
)

REPO_ROOT = SCRIPT_DIR.parent


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_task_id(row: dict, idx: int) -> str:
    """Stable id for a dataset row. Uses commit_hash if present (short), else idx."""
    commit = (row.get("commit_hash") or "")[:12]
    repo = (row.get("repo_name") or "row").replace("/", "_")
    if commit:
        return f"{idx:05d}_{repo}_{commit}"
    return f"{idx:05d}_{repo}"


def sample_output_path(out_dir: Path, task_id: str, sample_idx: int) -> Path:
    return out_dir / f"{task_id}_s{sample_idx:02d}.json"


def run_one_with_dirs(
    *,
    task_row: dict,
    row_idx: int,
    sample_idx: int,
    grammar: str,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict:
    task_id = make_task_id(task_row, row_idx)
    out_path = sample_output_path(out_dir, task_id, sample_idx)
    if out_path.exists():
        return {
            "task_id": task_id,
            "row_idx": row_idx,
            "sample_idx": sample_idx,
            "status": "skipped",
        }

    t0 = time.time()
    try:
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
        result["task_id"] = task_id
        result["row_idx"] = row_idx
        result["sample_idx"] = sample_idx
        result["duration_sec"] = time.time() - t0
        result["status"] = "done"
        out_path.write_text(json.dumps(result, indent=2, default=str))
        return {
            "task_id": task_id,
            "row_idx": row_idx,
            "sample_idx": sample_idx,
            "status": "done",
            "reward": result.get("final_reward"),
            "ended": result.get("ended"),
            "turns": len(result.get("turns", [])),
            "duration_sec": result["duration_sec"],
        }
    except Exception as e:
        err = {
            "task_id": task_id,
            "row_idx": row_idx,
            "sample_idx": sample_idx,
            "status": "error",
            "error_type": type(e).__name__,
            "error": str(e)[:1000],
            "traceback": traceback.format_exc()[:4000],
            "duration_sec": time.time() - t0,
        }
        out_path.write_text(json.dumps(err, indent=2))
        return err


class StatusTracker:
    def __init__(self, out_dir: Path, total: int, args: argparse.Namespace) -> None:
        self.out_dir = out_dir
        self.total = total
        self.args = args
        self.lock = threading.Lock()
        self.counts: dict[str, int] = {
            "done": 0,
            "error": 0,
            "skipped": 0,
            "reward_1": 0,
            "reward_0": 0,
            "reward_other": 0,
        }
        self.start_time = time.time()
        self.recent: list[dict] = []
        self.path = out_dir / "_status.json"
        self._dump()

    def update(self, summary: dict) -> None:
        with self.lock:
            status = summary.get("status", "?")
            self.counts[status] = self.counts.get(status, 0) + 1
            if status == "done":
                r = summary.get("reward")
                if r == 1.0:
                    self.counts["reward_1"] += 1
                elif r == 0.0:
                    self.counts["reward_0"] += 1
                else:
                    self.counts["reward_other"] += 1
            self.recent.append(summary)
            if len(self.recent) > 20:
                self.recent = self.recent[-20:]
            self._dump()

    def _dump(self) -> None:
        completed = sum(v for k, v in self.counts.items()
                        if k in ("done", "error", "skipped"))
        elapsed = time.time() - self.start_time
        rate = completed / elapsed if elapsed > 0 else 0
        remaining = self.total - completed
        eta_sec = remaining / rate if rate > 0 else None
        payload = {
            "updated": now_iso(),
            "elapsed_sec": round(elapsed, 1),
            "rate_per_sec": round(rate, 4),
            "eta_sec": round(eta_sec, 1) if eta_sec else None,
            "total": self.total,
            "completed": completed,
            "remaining": remaining,
            "counts": dict(self.counts),
            "args": vars(self.args),
            "recent": self.recent,
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str))
        tmp.replace(self.path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-tasks", type=int, default=200)
    p.add_argument("--offset", type=int, default=0,
                   help="skip the first N indices of the (shuffled) task list. "
                        "Lets you generate a non-overlapping second batch by "
                        "re-running with --offset N where N is the prior run's "
                        "--n-tasks. Same --seed must be used both runs.")
    p.add_argument("--samples-per-task", type=int, default=4)
    p.add_argument("--concurrency", type=int, default=4,
                   help="parallel rollouts (each spins its own docker container)")
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    p.add_argument("--dataset", default="R2E-Gym/R2E-Gym-Lite")
    p.add_argument("--split", default="train")
    p.add_argument("--shuffle", action="store_true",
                   help="shuffle dataset row order before slicing")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--step-timeout", type=int, default=90)
    p.add_argument("--reward-timeout", type=int, default=300)
    p.add_argument("--run-id", default=None,
                   help="explicit run id. Default = local timestamp. "
                        "Re-use to resume a prior run.")
    args = p.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "traces" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[batch] run_id={run_id}")
    print(f"[batch] out_dir={out_dir}")

    if not GRAMMAR_PATH.exists():
        sys.exit(f"grammar not found: {GRAMMAR_PATH}")
    grammar = GRAMMAR_PATH.read_text()
    print(f"[batch] grammar: {GRAMMAR_PATH} ({len(grammar)} chars)")

    print(f"[batch] loading dataset {args.dataset} split={args.split}")
    ds = load_dataset(args.dataset, split=args.split)
    print(f"[batch] dataset rows: {len(ds)}")

    indices = list(range(len(ds)))
    if args.shuffle:
        random.Random(args.seed).shuffle(indices)
    indices = indices[args.offset : args.offset + args.n_tasks]
    print(f"[batch] selected {len(indices)} task indices "
          f"(shuffle={args.shuffle}, seed={args.seed}, offset={args.offset})")

    # Build the flat (row_idx, sample_idx) work list
    work: list[tuple[int, int]] = []
    for row_idx in indices:
        for sample_idx in range(args.samples_per_task):
            work.append((row_idx, sample_idx))
    total = len(work)
    print(f"[batch] total rollouts: {total} "
          f"({args.n_tasks} tasks x {args.samples_per_task} samples)")
    print(f"[batch] concurrency: {args.concurrency}")

    tracker = StatusTracker(out_dir, total, args)

    def worker(row_idx: int, sample_idx: int) -> dict:
        task_row = ds[row_idx]
        return run_one_with_dirs(
            task_row=task_row,
            row_idx=row_idx,
            sample_idx=sample_idx,
            grammar=grammar,
            out_dir=out_dir,
            args=args,
        )

    started_at = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {
            ex.submit(worker, row_idx, sample_idx): (row_idx, sample_idx)
            for row_idx, sample_idx in work
        }
        for fut in as_completed(futures):
            row_idx, sample_idx = futures[fut]
            try:
                summary = fut.result()
            except Exception as e:
                summary = {
                    "row_idx": row_idx,
                    "sample_idx": sample_idx,
                    "status": "error",
                    "error_type": type(e).__name__,
                    "error": str(e)[:500],
                }
            tracker.update(summary)
            completed += 1
            elapsed = time.time() - started_at
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total - completed) / rate if rate > 0 else float("inf")
            r = summary.get("reward")
            r_str = f"reward={r}" if r is not None else summary.get("status", "?")
            print(
                f"[batch] {completed}/{total}  "
                f"row={row_idx} s={sample_idx}  {r_str}  "
                f"rate={rate*60:.1f}/min  eta={eta/60:.1f}min",
                flush=True,
            )

    print()
    print("=== final ===")
    print(json.dumps(tracker.counts, indent=2))
    print(f"out_dir: {out_dir}")
    print(f"status:  {tracker.path}")


if __name__ == "__main__":
    main()
