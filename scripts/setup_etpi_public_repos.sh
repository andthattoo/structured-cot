#!/usr/bin/env bash
# Clone and optionally set up the public ETPI repo pool on a CPU machine.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/etpi-repos}"
MANIFEST="${MANIFEST:-data/etpi_repos/public_repos.json}"
MODE="${MODE:-clone}" # clone | setup | smoke | all
JOBS="${JOBS:-2}"

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is required" >&2
    exit 1
fi
if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git is required" >&2
    exit 1
fi

mkdir -p "${ROOT_DIR}"

python3 - "$MANIFEST" "$ROOT_DIR" "$MODE" "$JOBS" <<'PY'
import json
import shlex
import subprocess
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
root = Path(sys.argv[2]).expanduser().resolve()
mode = sys.argv[3]
jobs = sys.argv[4]

repos = json.loads(manifest.read_text())
if mode not in {"clone", "setup", "smoke", "all"}:
    raise SystemExit("MODE must be clone, setup, smoke, or all")


def run(command, cwd=None):
    print(f"+ {command}", flush=True)
    subprocess.run(command, shell=True, cwd=cwd, check=True)


for repo in repos:
    repo_id = repo["repo_id"]
    dest = root / repo_id
    print(f"\n== {repo_id} ({repo['domain']}) -> {dest}", flush=True)

    if mode in {"clone", "all"}:
        if dest.exists():
            run("git fetch --all --tags --prune", cwd=dest)
        else:
            run(f"git clone {shlex.quote(repo['url'])} {shlex.quote(str(dest))}")
        checkout = repo.get("checkout")
        if checkout:
            run(f"git checkout {shlex.quote(str(checkout))}", cwd=dest)

    if mode in {"setup", "all"}:
        if not dest.exists():
            print(f"skip setup, missing {dest}", flush=True)
            continue
        for command in repo.get("setup") or []:
            run(command.replace("-j2", f"-j{jobs}"), cwd=dest)

    if mode == "smoke":
        if not dest.exists():
            print(f"skip smoke, missing {dest}", flush=True)
            continue
        for command in repo.get("smoke_tests") or []:
            run(command.replace("-j2", f"-j{jobs}"), cwd=dest)
PY
