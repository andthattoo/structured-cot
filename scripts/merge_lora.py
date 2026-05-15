"""Merge a PEFT LoRA adapter into a base HuggingFace causal LM checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-shard-size", default="5GB")
    args = p.parse_args()

    adapter_dir = Path(args.adapter_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[merge] loading base model: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    print(f"[merge] loading adapter: {adapter_dir}")
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    print("[merge] merging adapter into base weights")
    merged = peft_model.merge_and_unload()
    print(f"[merge] saving merged checkpoint: {out_dir}")
    merged.save_pretrained(
        str(out_dir),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    tokenizer.save_pretrained(str(out_dir))
    print("[merge] done")


if __name__ == "__main__":
    main()
