#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


MODELS = {
    "qwen35": {
        "model_id": "Qwen/Qwen3.6-35B-A3B",
        "served_name": "qwen3.6-35b-a3b",
        "role": "headline",
    },
    "qwen8": {
        "model_id": "Qwen/Qwen3-8B",
        "served_name": "qwen3-8b",
        "role": "exploratory",
    },
    "deepseek_llama8": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "served_name": "deepseek-r1-distill-llama-8b",
        "role": "exploratory",
    },
}


@dataclass(frozen=True)
class Benchmark:
    alias: str
    args: tuple[str, ...]
    max_tokens: int
    key: bool = False


BENCHMARKS = (
    Benchmark("humaneval_full", ("--dataset", "humaneval", "--n-problems", "164"), 8192, True),
    Benchmark("mbpp_100", ("--dataset", "mbpp", "--n-problems", "100"), 8192, False),
    Benchmark(
        "lcb_v6_2025_50",
        (
            "--dataset",
            "livecodebench",
            "--lcb-version",
            "release_v6",
            "--date-cutoff",
            "2025-01-01",
            "--platform",
            "leetcode",
            "--n-problems",
            "50",
        ),
        16384,
        True,
    ),
    Benchmark(
        "lcb_v6_post_20260423_50",
        (
            "--dataset",
            "livecodebench",
            "--lcb-version",
            "release_v6",
            "--date-cutoff",
            "2026-04-23",
            "--platform",
            "leetcode",
            "--n-problems",
            "50",
        ),
        16384,
        False,
    ),
)


MODES = {
    "free": ("--only", "free"),
    "prompt_terse": ("--only", "prompt_terse"),
    "fsm_base": (
        "--only",
        "fsm",
        "--grammar-api",
        "vllm",
        "--grammar-file",
        "grammars/fsm_grammar.gbnf",
    ),
    "fsm_plan": (
        "--only",
        "fsm",
        "--grammar-api",
        "vllm",
        "--grammar-file",
        "grammars/fsm_grammar_lcb_plan.gbnf",
    ),
}


def run_json(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        return {"ok": False, "returncode": proc.returncode, "stderr": proc.stderr[-2000:]}
    try:
        return {"ok": True, "data": json.loads(proc.stdout)}
    except json.JSONDecodeError:
        return {"ok": True, "stdout": proc.stdout[-2000:]}


def benchmark_count(bench: Benchmark) -> int | None:
    code = r"""
import argparse, json
from fsm_vs_free_eval import load_benchmark
p = argparse.ArgumentParser()
p.add_argument('--dataset')
p.add_argument('--n-problems', type=int, default=0)
p.add_argument('--lcb-version', default='release_v5')
p.add_argument('--date-cutoff', default='')
p.add_argument('--platform', default='leetcode')
args = p.parse_args()
rows = load_benchmark(args.dataset, args.n_problems, args)
print(len(rows))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code, *bench.args],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        print(proc.stderr[-2000:], file=sys.stderr)
        return None
    try:
        return int(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-alias", choices=sorted(MODELS), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--experiment-root", default="experiments/vllm_matrix_20260426")
    parser.add_argument("--key-repeats", type=int, default=3)
    parser.add_argument("--exploratory-repeats", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--limit-cells", nargs="*", default=None)
    parser.add_argument("--limit-repeats", nargs="*", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_cfg = MODELS[args.model_alias]
    root = Path(args.experiment_root)
    run_root = root / "runs" / args.model_alias
    run_root.mkdir(parents=True, exist_ok=True)

    matrix_manifest = {
        "model_alias": args.model_alias,
        "model": model_cfg,
        "base_url": args.base_url,
        "key_repeats": args.key_repeats,
        "exploratory_repeats": args.exploratory_repeats,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmarks": [b.alias for b in BENCHMARKS],
        "modes": sorted(MODES),
    }
    write_json(run_root / "matrix_manifest.json", matrix_manifest)

    failures = 0
    for bench in BENCHMARKS:
        count = benchmark_count(bench)
        if not count:
            skip_dir = run_root / bench.alias
            write_json(
                skip_dir / "SKIPPED.json",
                {
                    "benchmark": bench.alias,
                    "reason": "no problems after filtering" if count == 0 else "inventory failed",
                    "count": count,
                },
            )
            print(f"[skip] {args.model_alias} {bench.alias}: count={count}", flush=True)
            continue

        repeats = args.key_repeats if (model_cfg["role"] == "headline" and bench.key) else args.exploratory_repeats
        for mode, mode_args in MODES.items():
            cell_name = f"{bench.alias}/{mode}"
            if args.limit_cells and cell_name not in args.limit_cells:
                continue
            for repeat in range(repeats):
                if args.limit_repeats is not None and repeat not in args.limit_repeats:
                    continue
                out_dir = run_root / bench.alias / mode / f"repeat_{repeat}"
                manifest_path = out_dir / "run_manifest.json"
                summary_path = out_dir / "summary.json"
                if summary_path.exists() and not args.force:
                    print(f"[skip-existing] {args.model_alias} {cell_name} r{repeat}", flush=True)
                    continue
                if manifest_path.exists() and not args.force:
                    try:
                        prior_manifest = json.loads(manifest_path.read_text())
                    except Exception:
                        prior_manifest = {}
                    if "returncode" not in prior_manifest:
                        print(f"[skip-in-progress] {args.model_alias} {cell_name} r{repeat}", flush=True)
                        continue
                out_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    "fsm_vs_free_eval.py",
                    "--base-url",
                    args.base_url,
                    "--model",
                    model_cfg["served_name"],
                    "--tokenizer",
                    model_cfg["model_id"],
                    "--max-tokens",
                    str(bench.max_tokens),
                    "--request-timeout",
                    str(args.request_timeout),
                    "--out-dir",
                    str(out_dir),
                    *bench.args,
                    *mode_args,
                ]
                manifest = {
                    "model_alias": args.model_alias,
                    "model_id": model_cfg["model_id"],
                    "served_model_name": model_cfg["served_name"],
                    "benchmark": bench.alias,
                    "benchmark_args": bench.args,
                    "mode": mode,
                    "mode_args": mode_args,
                    "repeat": repeat,
                    "problem_count": count,
                    "command": cmd,
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                write_json(manifest_path, manifest)
                print(f"[run] {args.model_alias} {cell_name} r{repeat}", flush=True)
                t0 = time.time()
                with (out_dir / "stdout.log").open("w") as stdout, (out_dir / "stderr.log").open("w") as stderr:
                    proc = subprocess.run(cmd, text=True, stdout=stdout, stderr=stderr)
                manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                manifest["elapsed_sec"] = time.time() - t0
                manifest["returncode"] = proc.returncode
                manifest["summary_exists"] = summary_path.exists()
                write_json(manifest_path, manifest)
                if proc.returncode != 0 or not summary_path.exists():
                    failures += 1
                    print(f"[fail] {args.model_alias} {cell_name} r{repeat} rc={proc.returncode}", flush=True)
                else:
                    print(f"[done] {args.model_alias} {cell_name} r{repeat} {manifest['elapsed_sec']:.0f}s", flush=True)

    write_json(
        run_root / "matrix_status.json",
        {
            "model_alias": args.model_alias,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "failures": failures,
        },
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
