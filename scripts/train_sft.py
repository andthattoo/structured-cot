"""SFT-seed Qwen 3.6 27B on grammar-shaped R2E trajectories.

LoRA training (rank 128, all-linear) for memory efficiency, then merges
the adapter into base weights. Output is a normal HuggingFace checkpoint
with no adapter file — downstream RL loads it like any other model.

Run on the 2x A100 box:

    pip install "trl>=0.12" "peft>=0.10" "bitsandbytes>=0.43" \\
                "transformers>=4.45" "accelerate>=0.34" datasets
    # optionally: pip install flash-attn --no-build-isolation

    # single-GPU debug (cuts seq_len, slow):
    python scripts/train_sft.py \\
        --sft-jsonl datasets/20260513_102140/sft.jsonl \\
        --out-dir models/sft_20260513_102140

    # 2-GPU DDP (recommended for 27B + LoRA — both A100s, near-linear scale):
    accelerate launch --num_processes 2 --mixed_precision bf16 \\
        scripts/train_sft.py \\
        --sft-jsonl datasets/20260513_102140/sft.jsonl \\
        --out-dir models/sft_20260513_102140
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sft-jsonl", required=True)
    p.add_argument("--base-model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4,
                   help="LoRA LR (higher than full FT — LoRA params start "
                        "at zero and need stronger updates to move)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-rank", type=int, default=128)
    p.add_argument("--lora-alpha", type=int, default=256)
    p.add_argument("--max-seq-length", type=int, default=8192,
                   help="trajectories longer than this get truncated. "
                        "Most R2E trajectories fit; the few long ones "
                        "(30 turns with big heredocs) we accept losing.")
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--no-flash-attn", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    adapter_dir = out_dir / "adapter"
    merged_dir = out_dir / "merged"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[sft] loading base model: {args.base_model}")
    attn_impl = "flash_attention_2"
    if args.no_flash_attn:
        attn_impl = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print(f"[sft] loading dataset: {args.sft_jsonl}")
    ds = load_dataset("json", data_files=args.sft_jsonl, split="train")
    print(f"[sft] examples: {len(ds)}")

    sft_config = SFTConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        max_seq_length=args.max_seq_length,
        packing=False,
        # Mask loss on system + user messages; only train on assistant tokens.
        # Critical for agent trajectories: we don't want to learn to predict
        # bash observations (those are environment-provided, not policy output).
        assistant_only_loss=True,
        dataset_kwargs={"skip_prepare_dataset": False},
        report_to="none",
        remove_unused_columns=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        seed=0,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=sft_config,
    )

    print("[sft] starting training...")
    trainer.train()
    print(f"[sft] saving adapter to: {adapter_dir}")
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # ---- merge into base weights ----
    # Free the training model + clear cache before reloading for merge.
    del trainer
    del model
    torch.cuda.empty_cache()

    print("[sft] reloading base model for merge...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    print(f"[sft] applying adapter from: {adapter_dir}")
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    print("[sft] merging adapter into base weights (this can take a couple minutes)...")
    merged = peft_model.merge_and_unload()
    print(f"[sft] saving merged checkpoint to: {merged_dir}")
    merged.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(str(merged_dir))

    print()
    print("=== done ===")
    print(f"adapter (small):  {adapter_dir}")
    print(f"merged checkpoint (full model, ~54 GB bf16): {merged_dir}")
    print()
    print("Load downstream with no PEFT dependency:")
    print(f"  AutoModelForCausalLM.from_pretrained('{merged_dir}')")


if __name__ == "__main__":
    main()
