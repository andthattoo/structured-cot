#!/usr/bin/env python3
"""Create Pi task JSONL from the ETPI public repo manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("data/etpi_repos/public_repos.json")


TASK_TEMPLATES = [
    (
        "architecture",
        "Read this repository and summarize its architecture. Identify the main modules, the core execution path, and the files a new contributor should inspect first.",
    ),
    (
        "reproduce",
        "Inspect this repository and explain how to install dependencies, run the most relevant tests, and reproduce a small local smoke check. Do not edit files.",
    ),
    (
        "core_flow",
        "Find the implementation of the repository's core feature and trace the control/data flow through the most important files. Cite concrete file paths.",
    ),
    (
        "test_map",
        "Map the test structure of this repository. Explain which tests are fastest and which files they exercise. Do not run long test suites.",
    ),
    (
        "critic",
        "Review one central module in this repository for likely bugs, edge cases, or maintainability risks. Give findings with file paths and explain the evidence.",
    ),
    (
        "debug_plan",
        "Assume a recent change caused a regression in this repository. Inspect the project and propose a minimal debugging plan with the first three commands/files you would check.",
    ),
    (
        "small_edit_plan",
        "Identify a small, safe improvement that could be made in this repository, such as a clearer error message, docs clarification, or focused test. Explain the exact files you would edit, but do not modify files.",
    ),
    (
        "api_surface",
        "Find and explain the public API surface for this repository. Include important exported functions/classes/commands and where they are defined.",
    ),
]


DOMAIN_EXTRA_TASKS = {
    "python": [
        (
            "python_packaging",
            "Inspect the Python packaging and test configuration. Explain dependencies, optional extras, and how local development is intended to work.",
        )
    ],
    "webdev": [
        (
            "web_build_pipeline",
            "Inspect the JavaScript/TypeScript build pipeline. Explain package structure, scripts, test tooling, and where plugin/component behavior is implemented.",
        )
    ],
    "cpp": [
        (
            "cpp_build",
            "Inspect the CMake/build setup. Explain the main targets, test targets, and the smallest useful build or test command.",
        )
    ],
}


def load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list")
    return [row for row in data if isinstance(row, dict)]


def task_rows(repos: list[dict[str, Any]], root_dir: Path, *, include_extra: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repo in repos:
        repo_id = str(repo["repo_id"])
        domain = str(repo["domain"])
        cwd = str((root_dir / repo_id).resolve())
        templates = list(TASK_TEMPLATES)
        if include_extra:
            templates.extend(DOMAIN_EXTRA_TASKS.get(domain, []))
        for task_kind, prompt in templates:
            rows.append(
                {
                    "task_id": f"{repo_id}__{task_kind}",
                    "cwd": cwd,
                    "prompt": prompt,
                    "repo_id": repo_id,
                    "repo_name": repo.get("name"),
                    "domain": domain,
                    "source": "etpi_public_repos_v1",
                    "verifiable": False,
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ETPI repo task JSONL.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root-dir", type=Path, default=Path("/root/etpi-repos"))
    parser.add_argument("--out", type=Path, default=Path("data/pi_tasks/etpi_public_repo_tasks.jsonl"))
    parser.add_argument("--no-extra", action="store_true", help="Disable one domain-specific extra task per repo.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repos = load_manifest(args.manifest)
    rows = task_rows(repos, args.root_dir.expanduser(), include_extra=not args.no_extra)
    write_jsonl(args.out, rows)
    print(json.dumps({"out": str(args.out), "repos": len(repos), "tasks": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
