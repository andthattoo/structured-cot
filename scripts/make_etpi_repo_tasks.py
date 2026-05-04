#!/usr/bin/env python3
"""Create Pi task JSONL from the ETPI public repo manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("data/etpi_repos/public_repos.json")


TaskTemplate = tuple[str, str]


TASK_TEMPLATES: list[TaskTemplate] = [
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

EXPANDED_TASK_TEMPLATES: list[TaskTemplate] = [
    (
        "dependency_map",
        "Inspect this repository and map its major internal and external dependencies. Explain which dependencies are core to runtime behavior versus development or testing. Do not edit files.",
    ),
    (
        "config_map",
        "Inspect the repository configuration files and explain how formatting, linting, typing, building, testing, and packaging are configured. Cite concrete file paths. Do not edit files.",
    ),
    (
        "extension_points",
        "Find the main extension points or customization hooks in this repository. Explain who would use them, where they are defined, and one minimal example of how they fit into the codebase. Do not edit files.",
    ),
    (
        "error_handling",
        "Trace how this repository reports and handles errors in a central workflow. Identify the key exception/result types, user-facing messages, and tests that cover failure cases. Do not edit files.",
    ),
    (
        "test_gap_analysis",
        "Inspect a central feature and its nearby tests. Identify two plausible missing edge-case tests and explain exactly where they would belong. Do not run long test suites or edit files.",
    ),
    (
        "issue_triage",
        "Pretend a user reported a confusing behavior in this repository but gave no reproduction. Inspect the codebase and write a concrete triage checklist with the files, commands, and clues you would check first. Do not edit files.",
    ),
    (
        "minimal_example",
        "Find a small public feature in this repository and write a minimal usage example for it, explaining which source files implement the behavior. Do not edit files.",
    ),
    (
        "docs_alignment",
        "Compare the README or docs entry points with the implementation of one central feature. Note any documentation assumptions a new contributor should verify in code. Do not edit files.",
    ),
    (
        "release_risk",
        "Inspect this repository as if preparing a small release. Identify likely risk areas, the shortest useful validation commands, and the files that would need extra attention. Do not edit files.",
    ),
    (
        "performance_scan",
        "Look for one performance-sensitive path in this repository. Explain why it might matter, where it is implemented, and what evidence or tests you would collect before changing it. Do not edit files.",
    ),
    (
        "state_model",
        "Identify an important state model, lifecycle, or data structure in this repository. Explain the invariants it appears to maintain and where those invariants are enforced. Do not edit files.",
    ),
    (
        "contributor_route",
        "Create a practical onboarding route for a new contributor who wants to make a small first change in this repository. Include the files to read, likely tests, and one safe change idea. Do not edit files.",
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


EXPANDED_DOMAIN_EXTRA_TASKS = {
    "python": [
        (
            "python_imports",
            "Inspect the Python package layout and import boundaries. Explain which modules form the public API, which are internal helpers, and how imports avoid cycles. Do not edit files.",
        ),
        (
            "python_types",
            "Inspect how this Python project uses type hints, validation, and runtime checks. Identify the most important typed interfaces and any places where dynamic behavior matters. Do not edit files.",
        ),
        (
            "python_fixtures",
            "Inspect the Python test fixtures and helper utilities. Explain how tests construct inputs, isolate side effects, and which fixture would be useful for a new focused test. Do not edit files.",
        )
    ],
    "webdev": [
        (
            "web_package_boundaries",
            "Inspect the package or workspace boundaries in this JavaScript/TypeScript repository. Explain the main packages, their responsibilities, and how they depend on each other. Do not edit files.",
        ),
        (
            "web_runtime_flow",
            "Trace one runtime flow in this JavaScript/TypeScript repository from public entry point to internal implementation. Cite concrete files and note the likely tests. Do not edit files.",
        ),
        (
            "web_types_contracts",
            "Inspect important TypeScript types, interfaces, or component contracts in this repository. Explain how they protect callers and where the contracts are tested. Do not edit files.",
        )
    ],
    "cpp": [
        (
            "cpp_headers",
            "Inspect the public header/API layout for this C++ repository. Explain which headers are central, which details are internal, and how users are expected to include the library. Do not edit files.",
        ),
        (
            "cpp_tests",
            "Inspect the C++ tests and examples. Explain how test cases are organized, which tests are fastest to run, and where a focused regression test would belong. Do not edit files.",
        ),
        (
            "cpp_compatibility",
            "Inspect how this C++ repository handles compiler, standard-library, or platform compatibility. Identify key macros, build options, and tests that protect compatibility. Do not edit files.",
        )
    ],
}


def load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list")
    return [row for row in data if isinstance(row, dict)]


def templates_for_repo(domain: str, *, profile: str, include_extra: bool) -> list[TaskTemplate]:
    templates = list(TASK_TEMPLATES)
    if profile == "expanded":
        templates.extend(EXPANDED_TASK_TEMPLATES)
    if include_extra:
        templates.extend(DOMAIN_EXTRA_TASKS.get(domain, []))
        if profile == "expanded":
            templates.extend(EXPANDED_DOMAIN_EXTRA_TASKS.get(domain, []))
    return templates


def task_rows(
    repos: list[dict[str, Any]],
    root_dir: Path,
    *,
    include_extra: bool,
    profile: str = "core",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repo in repos:
        repo_id = str(repo["repo_id"])
        domain = str(repo["domain"])
        cwd = str((root_dir / repo_id).resolve())
        templates = templates_for_repo(domain, profile=profile, include_extra=include_extra)
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
                    "task_kind": task_kind,
                    "task_profile": profile,
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
    parser.add_argument(
        "--profile",
        choices=["core", "expanded"],
        default="core",
        help="Task taxonomy to generate. 'core' preserves the original small set; 'expanded' adds broader read-only repo tasks.",
    )
    parser.add_argument("--no-extra", action="store_true", help="Disable one domain-specific extra task per repo.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repos = load_manifest(args.manifest)
    rows = task_rows(
        repos,
        args.root_dir.expanduser(),
        include_extra=not args.no_extra,
        profile=args.profile,
    )
    write_jsonl(args.out, rows)
    print(
        json.dumps(
            {"out": str(args.out), "profile": args.profile, "repos": len(repos), "tasks": len(rows)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
