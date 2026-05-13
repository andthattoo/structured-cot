"""Upload SFT and DPO JSONL files to HuggingFace as public datasets.

Pre-req: `huggingface-cli login` once, or set HF_TOKEN env var.

Usage:
    uv run python scripts/upload_datasets.py \\
        --sft datasets/20260513_102140/sft.jsonl \\
        --dpo datasets/20260513_102140/dpo.jsonl

Skip one with --skip-sft / --skip-dpo. Override repo names with
--sft-repo / --dpo-repo.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset


def push_one(jsonl_path: Path, repo_id: str, kind: str) -> None:
    if not jsonl_path.exists():
        print(f"[upload] WARN: {jsonl_path} not found, skipping {kind}")
        return
    n = sum(1 for _ in jsonl_path.open())
    if n == 0:
        print(f"[upload] WARN: {jsonl_path} is empty, skipping {kind}")
        return
    print(f"[upload] {kind}: {jsonl_path} ({n} examples) -> {repo_id}")
    ds = load_dataset("json", data_files=str(jsonl_path), split="train")
    print(f"[upload] loaded: {ds}")
    print(f"[upload] pushing to hub (public)...")
    ds.push_to_hub(
        repo_id,
        private=False,
        commit_message=f"Upload from {jsonl_path.name}",
    )
    print(f"[upload] ✓ https://huggingface.co/datasets/{repo_id}")
    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sft", required=True, help="path to sft.jsonl")
    p.add_argument("--dpo", required=True, help="path to dpo.jsonl")
    p.add_argument("--sft-repo", default="andthattoo/etpi-sft")
    p.add_argument("--dpo-repo", default="andthattoo/etpi-dpo")
    p.add_argument("--skip-sft", action="store_true")
    p.add_argument("--skip-dpo", action="store_true")
    args = p.parse_args()

    if not args.skip_sft:
        push_one(Path(args.sft), args.sft_repo, "SFT")
    if not args.skip_dpo:
        push_one(Path(args.dpo), args.dpo_repo, "DPO")


if __name__ == "__main__":
    main()
