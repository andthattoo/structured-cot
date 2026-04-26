#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from run_vllm_matrix import BENCHMARKS, MODELS, MODES, benchmark_count


def expected_repeats(model_alias: str, bench_alias: str) -> int:
    model_cfg = MODELS[model_alias]
    bench = next(b for b in BENCHMARKS if b.alias == bench_alias)
    if model_cfg["role"] == "headline" and bench.key:
        return 3
    return 1


def cell_complete(root: Path, model_alias: str, cell: str) -> bool:
    bench_alias, mode = cell.split("/", 1)
    repeats = expected_repeats(model_alias, bench_alias)
    for repeat in range(repeats):
        summary = root / "runs" / model_alias / bench_alias / mode / f"repeat_{repeat}" / "summary.json"
        if not summary.exists():
            return False
    return True


def repeat_complete(root: Path, model_alias: str, bench_alias: str, mode: str, repeat: int) -> bool:
    summary = root / "runs" / model_alias / bench_alias / mode / f"repeat_{repeat}" / "summary.json"
    return summary.exists()


def repeat_in_progress(root: Path, model_alias: str, bench_alias: str, mode: str, repeat: int) -> bool:
    manifest = root / "runs" / model_alias / bench_alias / mode / f"repeat_{repeat}" / "run_manifest.json"
    summary = manifest.parent / "summary.json"
    if summary.exists() or not manifest.exists():
        return False
    try:
        data = json.loads(manifest.read_text())
    except Exception:
        return True
    return "returncode" not in data


def discover_units(model_alias: str, root: Path) -> list[str]:
    units: list[str] = []
    for bench in BENCHMARKS:
        count = benchmark_count(bench)
        if not count:
            continue
        for mode in sorted(MODES):
            repeats = expected_repeats(model_alias, bench.alias)
            for repeat in range(repeats):
                if repeat_complete(root, model_alias, bench.alias, mode, repeat):
                    continue
                if repeat_in_progress(root, model_alias, bench.alias, mode, repeat):
                    continue
                units.append(f"{bench.alias}/{mode}/r{repeat}")
    return units


def launch_cell(
    model_alias: str,
    unit: str,
    root: Path,
    base_url: str,
    request_timeout: float,
    logs_dir: Path,
) -> subprocess.Popen:
    bench, mode, repeat_text = unit.split("/", 2)
    repeat = int(repeat_text.removeprefix("r"))
    cell = f"{bench}/{mode}"
    log_stem = f"{model_alias}__{bench}__{mode}__r{repeat}"
    stdout_path = logs_dir / f"{log_stem}.stdout.log"
    stderr_path = logs_dir / f"{log_stem}.stderr.log"
    cmd = [
        sys.executable,
        "scripts/run_vllm_matrix.py",
        "--model-alias",
        model_alias,
        "--experiment-root",
        str(root),
        "--base-url",
        base_url,
        "--request-timeout",
        str(request_timeout),
        "--limit-cells",
        cell,
        "--limit-repeats",
        str(repeat),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    stdout = stdout_path.open("w")
    stderr = stderr_path.open("w")
    proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, text=True, env=env)
    # The file handles are intentionally owned by the child process after fork.
    stdout.close()
    stderr.close()
    print(f"[launch] pid={proc.pid} unit={unit} base_url={base_url} stdout={stdout_path}", flush=True)
    return proc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-alias", choices=sorted(MODELS), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--base-urls", nargs="+", default=None)
    parser.add_argument("--experiment-root", default="experiments/vllm_matrix_20260426")
    parser.add_argument("--max-jobs", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--cells", nargs="*", default=None)
    parser.add_argument("--units", nargs="*", default=None)
    args = parser.parse_args()

    root = Path(args.experiment_root)
    logs_dir = root / "logs" / "cell_parallel"
    logs_dir.mkdir(parents=True, exist_ok=True)

    base_urls = args.base_urls or [args.base_url]
    units = args.units
    if units is None and args.cells:
        units = []
        for cell in args.cells:
            bench, mode = cell.split("/", 1)
            for repeat in range(expected_repeats(args.model_alias, bench)):
                units.append(f"{bench}/{mode}/r{repeat}")
    if units is None:
        units = discover_units(args.model_alias, root)
    filtered_units = []
    for unit in units:
        bench, mode, repeat_text = unit.split("/", 2)
        repeat = int(repeat_text.removeprefix("r"))
        if repeat_complete(root, args.model_alias, bench, mode, repeat):
            continue
        if repeat_in_progress(root, args.model_alias, bench, mode, repeat):
            continue
        filtered_units.append(unit)
    units = filtered_units
    if not units:
        print(f"[done] no pending units for {args.model_alias}", flush=True)
        return 0

    print(
        f"[plan] model={args.model_alias} units={len(units)} "
        f"max_jobs={args.max_jobs} endpoints={len(base_urls)}",
        flush=True,
    )
    for url in base_urls:
        print(f"[endpoint] {url}", flush=True)
    for unit in units:
        print(f"[pending] {unit}", flush=True)

    failures = 0
    pending = list(units)
    running: dict[subprocess.Popen, str] = {}
    launches = 0
    while pending or running:
        while pending and len(running) < args.max_jobs:
            unit = pending.pop(0)
            base_url = base_urls[launches % len(base_urls)]
            launches += 1
            running[launch_cell(args.model_alias, unit, root, base_url, args.request_timeout, logs_dir)] = unit

        time.sleep(5)
        for proc, cell in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del running[proc]
            bench, mode, repeat_text = cell.split("/", 2)
            repeat = int(repeat_text.removeprefix("r"))
            complete = repeat_complete(root, args.model_alias, bench, mode, repeat)
            if rc != 0 or not complete:
                failures += 1
                print(f"[fail] unit={cell} rc={rc} complete={complete}", flush=True)
            else:
                print(f"[done] unit={cell} rc=0", flush=True)

    if failures:
        print(f"[matrix-fail] model={args.model_alias} failures={failures}", flush=True)
        return 1
    print(f"[matrix-done] model={args.model_alias}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
