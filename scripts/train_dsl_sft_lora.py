#!/usr/bin/env python3
"""LoRA SFT for compact-DSL agent traces.

The trainer accepts the rich JSONL produced by prepare_hermes_dsl_sft.py and
creates one training example per assistant turn. Only the current assistant
target span is trained; all prior context tokens are masked with -100.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter on DSL SFT JSONL.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--model", required=True, help="HF Transformers checkpoint id/path.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--context-messages", type=int, default=12)
    parser.add_argument("--limit-examples", type=int, default=None)
    parser.add_argument(
        "--objective",
        choices=["assistant_turn", "factorized_bottleneck"],
        default="assistant_turn",
        help=(
            "assistant_turn trains DSL+action as one assistant message. "
            "factorized_bottleneck creates two examples per turn: "
            "teacher-think -> DSL, then DSL -> action."
        ),
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target module names.",
    )
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit QLoRA loading.")
    parser.add_argument("--no-bf16", action="store_true", help="Disable bf16 training.")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    return parser.parse_args()


def render_chatml(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = message["role"]
        content = message["content"].rstrip()
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts) + "\n"


def select_context(messages: list[dict[str, str]], assistant_index: int, context_messages: int) -> list[dict[str, str]]:
    system = [m for m in messages[:assistant_index] if m["role"] == "system"][:1]
    prior_non_system = [m for m in messages[:assistant_index] if m["role"] != "system"]
    return system + prior_non_system[-context_messages:] + [messages[assistant_index]]


THINK_BLOCK_RE = re.compile(r"(<think>\s*.*?\s*</think>)", re.DOTALL | re.IGNORECASE)


def assistant_span(text: str) -> tuple[int, int]:
    marker = "<|im_start|>assistant\n"
    start = text.rfind(marker)
    if start < 0:
        raise ValueError("assistant marker not found")
    content_start = start + len(marker)
    end_marker = "<|im_end|>"
    end = text.find(end_marker, content_start)
    if end < 0:
        raise ValueError("assistant end marker not found")
    return content_start, end + len(end_marker)


def dsl_and_action(content: str) -> tuple[str, str] | None:
    match = THINK_BLOCK_RE.search(content)
    if not match:
        return None
    dsl = match.group(1).strip()
    action = content[match.end() :].lstrip()
    return dsl, action


def target_span(text: str, target: str) -> tuple[int, int]:
    start = text.rfind(target)
    if start < 0:
        raise ValueError("target text not found")
    return start, start + len(target)


def load_turn_examples(
    path: Path,
    *,
    context_messages: int,
    limit_examples: int | None,
    objective: str,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages") or []
            labels = row.get("dsl_stats", {}).get("labels") or []
            label_index = 0
            for i, message in enumerate(messages):
                if message.get("role") != "assistant":
                    continue
                content = str(message.get("content") or "")
                if "<think>" not in content:
                    continue
                label = labels[label_index] if label_index < len(labels) else {}
                label_index += 1
                selected = select_context(messages, i, context_messages)

                if objective == "assistant_turn":
                    text = render_chatml(selected)
                    span = assistant_span(text)
                    examples.append({"text": text, "target_span": span, "kind": "assistant_turn"})
                elif objective == "factorized_bottleneck":
                    parsed = dsl_and_action(content)
                    original_think = str(label.get("_original_think") or "").strip()
                    if parsed is None or not original_think:
                        continue
                    dsl, action = parsed
                    context = selected[:-1]
                    context_text = render_chatml(context)

                    compression_target = dsl + "<|im_end|>"
                    compression_text = (
                        context_text
                        + "<|im_start|>assistant\n"
                        + "<teacher_think>\n"
                        + original_think
                        + "\n</teacher_think>\n"
                        + compression_target
                        + "\n"
                    )
                    examples.append(
                        {
                            "text": compression_text,
                            "target_span": target_span(compression_text, compression_target),
                            "kind": "compress_to_dsl",
                        }
                    )

                    action_target = action + "<|im_end|>"
                    action_text = (
                        context_text
                        + "<|im_start|>assistant\n"
                        + dsl
                        + "\n"
                        + action_target
                        + "\n"
                    )
                    examples.append(
                        {
                            "text": action_text,
                            "target_span": target_span(action_text, action_target),
                            "kind": "dsl_to_action",
                        }
                    )
                else:
                    raise ValueError(f"unknown objective: {objective}")

                if limit_examples is not None and len(examples) >= limit_examples:
                    return examples
    return examples


class DslTurnDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list[dict[str, Any]], tokenizer: Any, max_seq_len: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.examples[idx]
        text = example["text"]
        span_start, span_end = example["target_span"]
        encoded = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            truncation=True,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        offsets = encoded["offset_mapping"]
        labels = []
        for token_id, (start, end) in zip(input_ids, offsets):
            if end > span_start and start < span_end:
                labels.append(token_id)
            else:
                labels.append(-100)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass
class DataCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].shape[0] for item in features)
        pad_id = self.tokenizer.pad_token_id
        batch: dict[str, list[torch.Tensor]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            pad = max_len - item["input_ids"].shape[0]
            batch["input_ids"].append(torch.nn.functional.pad(item["input_ids"], (0, pad), value=pad_id))
            batch["attention_mask"].append(torch.nn.functional.pad(item["attention_mask"], (0, pad), value=0))
            batch["labels"].append(torch.nn.functional.pad(item["labels"], (0, pad), value=-100))
        return {key: torch.stack(value) for key, value in batch.items()}


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise SystemExit(
            "Missing training deps. Install on the training box with:\n"
            "  uv pip install 'peft>=0.13' 'accelerate>=0.34' bitsandbytes\n"
            "or run with:\n"
            "  uv run --with peft --with accelerate --with bitsandbytes python scripts/train_dsl_sft_lora.py ...\n"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    use_bf16 = (not args.no_bf16) and (not args.fp16)

    quant_config = None
    if not args.no_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        quantization_config=quant_config,
    )
    model.config.use_cache = False
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    examples = load_turn_examples(
        args.train_jsonl,
        context_messages=args.context_messages,
        limit_examples=args.limit_examples,
        objective=args.objective,
    )
    if not examples:
        raise SystemExit("No assistant-turn examples found in training JSONL.")
    kind_counts: dict[str, int] = {}
    for example in examples:
        kind = example.get("kind", "unknown")
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    print(f"Loaded {len(examples)} examples: {kind_counts}")

    dataset = DslTurnDataset(examples, tokenizer, args.max_seq_len)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=args.fp16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if not args.no_4bit else "adamw_torch",
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollator(tokenizer),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
