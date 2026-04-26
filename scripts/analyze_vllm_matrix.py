#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


MODE_KEY = {
    "free": "free",
    "prompt_terse": "prompt_terse",
    "fsm_base": "fsm",
    "fsm_plan": "fsm",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def stddev(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def fmt(x: float, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{digits}f}"


def paired_vs_free(rows: list[dict]) -> list[dict]:
    completed = [r for r in rows if "pass_rate" in r]
    by_key = {(r["model_alias"], r["benchmark"], r["mode"], r["repeat"]): r for r in completed}
    pairs = []
    for model, bench, repeat in sorted({(r["model_alias"], r["benchmark"], r["repeat"]) for r in completed}):
        free = by_key.get((model, bench, "free", repeat))
        if not free:
            continue
        for mode in ("prompt_terse", "fsm_base", "fsm_plan"):
            other = by_key.get((model, bench, mode, repeat))
            if not other:
                continue
            pairs.append(
                {
                    "model_alias": model,
                    "benchmark": bench,
                    "mode": mode,
                    "repeat": repeat,
                    "pass_delta_pp": (other["pass_rate"] - free["pass_rate"]) * 100,
                    "think_compression": free["think_tokens_mean"] / max(other["think_tokens_mean"], 1),
                    "total_compression": free["total_tokens_mean"] / max(other["total_tokens_mean"], 1),
                    "free_pass_rate": free["pass_rate"],
                    "mode_pass_rate": other["pass_rate"],
                    "free_think": free["think_tokens_mean"],
                    "mode_think": other["think_tokens_mean"],
                }
            )
    return pairs


def collect(root: Path):
    rows = []
    task_rows = []
    skipped = []
    for manifest_path in root.glob("runs/*/*/*/*/run_manifest.json"):
        manifest = load_json(manifest_path)
        out_dir = manifest_path.parent
        summary_path = out_dir / "summary.json"
        result_path = out_dir / "results.jsonl"
        row = {
            "model_alias": manifest["model_alias"],
            "model_id": manifest["model_id"],
            "benchmark": manifest["benchmark"],
            "mode": manifest["mode"],
            "repeat": manifest["repeat"],
            "returncode": manifest.get("returncode"),
            "elapsed_sec": manifest.get("elapsed_sec"),
            "problem_count": manifest.get("problem_count"),
            "out_dir": str(out_dir),
        }
        if summary_path.exists():
            summary = load_json(summary_path)
            key = MODE_KEY[manifest["mode"]]
            if key in summary:
                row.update(summary[key])
            row["n"] = summary.get("n")
        rows.append(row)
        if result_path.exists():
            key = MODE_KEY[manifest["mode"]]
            for line in result_path.read_text().splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                if key not in item:
                    continue
                d = item[key]
                task_rows.append(
                    {
                        "model_alias": manifest["model_alias"],
                        "benchmark": manifest["benchmark"],
                        "mode": manifest["mode"],
                        "repeat": manifest["repeat"],
                        "task_id": item["task_id"],
                        "pass": bool(d.get("pass")),
                        "think_tokens": d.get("think_tokens", 0),
                        "total_tokens": d.get("total_tokens", 0),
                        "post_think_tokens": d.get("post_think_tokens", 0),
                        "failure_type": d.get("failure_type", "unknown"),
                        "extraction_issue": d.get("extraction_issue", "unknown"),
                    }
                )
    for skip_path in root.glob("runs/*/*/SKIPPED.json"):
        skipped.append(load_json(skip_path) | {"path": str(skip_path)})
    return rows, task_rows, skipped


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = sorted({k for row in rows for k in row})
    lines = [",".join(keys)]
    for row in rows:
        vals = []
        for key in keys:
            val = row.get(key, "")
            text = json.dumps(val) if isinstance(val, (dict, list)) else str(val)
            vals.append('"' + text.replace('"', '""') + '"')
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n")


def try_charts(report_dir: Path, rows: list[dict]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    completed = [r for r in rows if "pass_rate" in r]
    if not completed:
        return []
    groups = defaultdict(list)
    for r in completed:
        groups[(r["model_alias"], r["benchmark"], r["mode"])].append(r)
    agg = []
    for (model, bench, mode), xs in sorted(groups.items()):
        agg.append(
            {
                "label": f"{model}\n{bench}\n{mode}",
                "model": model,
                "benchmark": bench,
                "mode": mode,
                "pass_rate": mean([x["pass_rate"] for x in xs]) * 100,
                "think": mean([x["think_tokens_mean"] for x in xs]),
                "total": mean([x["total_tokens_mean"] for x in xs]),
            }
        )
    charts = []
    for metric, ylabel, filename in [
        ("pass_rate", "pass@1 (%)", "pass_rate_by_cell.png"),
        ("think", "mean think tokens", "think_tokens_by_cell.png"),
        ("total", "mean total tokens", "total_tokens_by_cell.png"),
    ]:
        fig_w = max(12, len(agg) * 0.32)
        fig, ax = plt.subplots(figsize=(fig_w, 7))
        vals = [a[metric] for a in agg]
        ax.bar(range(len(vals)), vals)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([a["label"] for a in agg], rotation=80, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        out = report_dir / filename
        fig.savefig(out, dpi=180)
        plt.close(fig)
        charts.append(filename)

    pair_rows = paired_vs_free(rows)
    if pair_rows:
        labels = [f"{p['model_alias']}\n{p['benchmark']}\n{p['mode']}\nr{p['repeat']}" for p in pair_rows]
        fig_w = max(12, len(pair_rows) * 0.34)
        for metric, ylabel, filename, baseline in [
            ("pass_delta_pp", "pass@1 delta vs FREE (pp)", "pass_delta_vs_free.png", 0),
            ("think_compression", "FREE / mode mean think tokens", "think_compression_vs_free.png", 1),
            ("total_compression", "FREE / mode mean total tokens", "total_compression_vs_free.png", 1),
        ]:
            fig, ax = plt.subplots(figsize=(fig_w, 7))
            vals = [p[metric] for p in pair_rows]
            ax.bar(range(len(vals)), vals)
            ax.axhline(baseline, color="black", linewidth=0.8, alpha=0.55)
            ax.set_ylabel(ylabel)
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(labels, rotation=80, ha="right", fontsize=7)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            out = report_dir / filename
            fig.savefig(out, dpi=180)
            plt.close(fig)
            charts.append(filename)
    return charts


def build_report(root: Path, report_dir: Path, rows: list[dict], task_rows: list[dict], skipped: list[dict], charts: list[str]) -> str:
    completed = [r for r in rows if "pass_rate" in r]
    groups = defaultdict(list)
    for r in completed:
        groups[(r["model_alias"], r["benchmark"], r["mode"])].append(r)
    pair_rows = paired_vs_free(rows)
    pair_groups = defaultdict(list)
    for p in pair_rows:
        pair_groups[(p["model_alias"], p["benchmark"], p["mode"])].append(p)

    lines = []
    lines.append("# vLLM Structured-CoT Matrix Findings\n")
    lines.append("## Scope\n")
    lines.append("This report summarizes the vLLM-only matrix artifacts under `" + str(root) + "`.\n")
    lines.append("The server configuration keeps vLLM defaults, no quantization, and no runtime tuning beyond tensor parallelism needed to serve the model.\n")
    if skipped:
        lines.append("## Skipped Cells\n")
        for s in skipped:
            lines.append(f"- `{s.get('path')}`: {s.get('reason')} (count={s.get('count')})")
        lines.append("")
    lines.append("## Charts\n")
    for c in charts:
        lines.append(f"![{c}]({c})")
    lines.append("")
    lines.append("## Cell Summary\n")
    lines.append("| Model | Benchmark | Mode | Repeats | pass@1 mean | pass@1 sd | Think mean | Total mean | Post-think mean |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for (model, bench, mode), xs in sorted(groups.items()):
        lines.append(
            f"| {model} | {bench} | {mode} | {len(xs)} | "
            f"{fmt(mean([x['pass_rate'] for x in xs]) * 100)} | "
            f"{fmt(stddev([x['pass_rate'] * 100 for x in xs]))} | "
            f"{fmt(mean([x['think_tokens_mean'] for x in xs]), 0)} | "
            f"{fmt(mean([x['total_tokens_mean'] for x in xs]), 0)} | "
            f"{fmt(mean([x['post_think_tokens_mean'] for x in xs]), 0)} |"
        )
    lines.append("")
    lines.append("## Paired Deltas vs FREE\n")
    lines.append("| Model | Benchmark | Mode | Repeats | pass delta pp mean | pass delta pp sd | Think compression | Total compression |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for (model, bench, mode), xs in sorted(pair_groups.items()):
        lines.append(
            f"| {model} | {bench} | {mode} | {len(xs)} | "
            f"{fmt(mean([x['pass_delta_pp'] for x in xs]))} | "
            f"{fmt(stddev([x['pass_delta_pp'] for x in xs]))} | "
            f"{fmt(mean([x['think_compression'] for x in xs]), 2)}x | "
            f"{fmt(mean([x['total_compression'] for x in xs]), 2)}x |"
        )
    lines.append("")
    lines.append("## Claim Checks\n")
    fsm_pairs = [p for p in pair_rows if p["mode"].startswith("fsm_")]
    prompt_pairs = [p for p in pair_rows if p["mode"] == "prompt_terse"]
    if fsm_pairs:
        lines.append(
            "- FSM-constrained cells completed so far: "
            f"{len(fsm_pairs)} repeat-pairs; mean think compression "
            f"{fmt(mean([p['think_compression'] for p in fsm_pairs]), 2)}x; "
            f"mean pass delta {fmt(mean([p['pass_delta_pp'] for p in fsm_pairs]))} pp."
        )
    if prompt_pairs:
        lines.append(
            "- Prompt-only terse control cells completed so far: "
            f"{len(prompt_pairs)} repeat-pairs; mean think compression "
            f"{fmt(mean([p['think_compression'] for p in prompt_pairs]), 2)}x; "
            f"mean pass delta {fmt(mean([p['pass_delta_pp'] for p in prompt_pairs]))} pp."
        )
    models_done = sorted({r["model_alias"] for r in completed})
    benches_done = sorted({r["benchmark"] for r in completed})
    lines.append(f"- Completed model coverage in this report: {', '.join(models_done) if models_done else '-'}")
    lines.append(f"- Completed benchmark coverage in this report: {', '.join(benches_done) if benches_done else '-'}")
    lines.append("")
    lines.append("## Failure Accounting\n")
    fail_counts = defaultdict(int)
    for r in task_rows:
        if not r["pass"]:
            fail_counts[(r["model_alias"], r["benchmark"], r["mode"], r["failure_type"])] += 1
    if fail_counts:
        lines.append("| Model | Benchmark | Mode | Failure type | Count |")
        lines.append("| --- | --- | --- | --- | ---: |")
        for (model, bench, mode, failure), count in sorted(fail_counts.items()):
            lines.append(f"| {model} | {bench} | {mode} | {failure} | {count} |")
    else:
        lines.append("No failures were recorded in completed cells.")
    lines.append("")
    lines.append("## Notes\n")
    lines.append("- This is a vLLM replication/generalization study, not a bit-for-bit llama.cpp/GGUF reproduction.")
    lines.append("- The faithful vLLM Structured-CoT setup does not use `--reasoning-parser`; with the parser enabled, vLLM masks only after reasoning by default.")
    lines.append("- LiveCodeBench numbers use the repo's public-test harness and should not be presented as official hidden-test LCB scores.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", default="experiments/vllm_matrix_20260426")
    args = parser.parse_args()
    root = Path(args.experiment_root)
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    rows, task_rows, skipped = collect(root)
    write_csv(report_dir / "run_summary.csv", rows)
    write_csv(report_dir / "task_results.csv", task_rows)
    charts = try_charts(report_dir, rows)
    report = build_report(root, report_dir, rows, task_rows, skipped, charts)
    (report_dir / "FINDINGS.md").write_text(report)
    print(report_dir / "FINDINGS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
