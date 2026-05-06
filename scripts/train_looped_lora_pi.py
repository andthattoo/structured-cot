#!/usr/bin/env python3
"""Train a tail-looped LoRA adapter on ETPI assistant steps.

This is a small experimental trainer for ELT-style recurrent-depth adaptation:
freeze the base LM, add LoRA to a tail block, run that tail block multiple
times, and train intermediate loop exits to imitate a max-loop teacher.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_thinking_hidden_trajectory import hf_rows
from extract_thinking_hidden_trajectory import part_text
from extract_thinking_hidden_trajectory import steps_from_trace_row
from extract_thinking_hidden_trajectory import torch_dtype


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield row


def step_stream(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.steps:
        yield from read_jsonl(args.steps)
        return

    for trace in hf_rows(args.hf_dataset, args.split):
        yield from steps_from_trace_row(trace, include_empty_assistant=args.include_empty_assistant)


def render_parts_without_thinking(parts: list[dict[str, Any]], tool_calls: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for part in parts:
        kind = part.get("type")
        if kind == "thinking":
            continue
        if kind == "toolCall":
            name = "" if part.get("name") is None else str(part.get("name"))
            args = part.get("arguments") or {}
            chunks.append(f"TOOL_CALL {name} {json.dumps(args, ensure_ascii=False, sort_keys=True)}")
            continue
        text = part_text(part)
        if text:
            chunks.append(text)

    for call in tool_calls:
        name = "" if call.get("name") is None else str(call.get("name"))
        args = call.get("arguments") or {}
        line = f"TOOL_CALL {name} {json.dumps(args, ensure_ascii=False, sort_keys=True)}"
        if line not in chunks:
            chunks.append(line)

    return "\n".join(chunks).strip()


def render_message_without_thinking(message: dict[str, Any]) -> str:
    role = str(message.get("role") or "user")
    rendered_role = "tool" if role == "tool" else role
    parts = message.get("content") if isinstance(message.get("content"), list) else []
    calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    content = render_parts_without_thinking(parts, calls)
    return f"<|im_start|>{rendered_role}\n{content}\n<|im_end|>"


def visible_target_text(step: dict[str, Any]) -> str:
    target = step.get("target_assistant")
    if not isinstance(target, dict):
        return ""
    parts = target.get("content") if isinstance(target.get("content"), list) else []
    calls = target.get("tool_calls") if isinstance(target.get("tool_calls"), list) else []
    return render_parts_without_thinking(parts, calls)


def render_training_text(step: dict[str, Any]) -> tuple[str, str]:
    state = step.get("state_messages")
    if not isinstance(state, list):
        state = []
    rendered = [render_message_without_thinking(message) for message in state if isinstance(message, dict)]
    prompt = "\n".join(rendered + ["<|im_start|>assistant\n"])
    target = visible_target_text(step)
    if not target:
        return prompt, ""
    return prompt, f"{target}\n<|im_end|>"


def tokenize_example(
    step: dict[str, Any],
    tokenizer: Any,
    *,
    max_total_tokens: int | None,
    max_target_tokens: int | None,
) -> dict[str, Any] | None:
    prompt, target = render_training_text(step)
    if not target:
        return None

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if not prompt_ids or not target_ids:
        return None
    if max_target_tokens is not None and len(target_ids) > max_target_tokens:
        return None

    input_ids = prompt_ids + target_ids
    if max_total_tokens is not None and len(input_ids) > max_total_tokens:
        return None

    return {
        "id": step.get("id") or "",
        "run_id": step.get("run_id") or "",
        "task_id": step.get("task_id") or "",
        "source": step.get("source") or "",
        "input_ids": input_ids,
        "labels": [-100] * len(prompt_ids) + target_ids,
        "prompt_tokens": len(prompt_ids),
        "target_tokens": len(target_ids),
        "has_tool_call": bool(step.get("reward_features", {}).get("has_tool_call"))
        if isinstance(step.get("reward_features"), dict)
        else False,
    }


def load_examples(args: argparse.Namespace, tokenizer: Any) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    skipped = Counter()
    for step in step_stream(args):
        example = tokenize_example(
            step,
            tokenizer,
            max_total_tokens=args.max_total_tokens,
            max_target_tokens=args.max_target_tokens,
        )
        if example is None:
            skipped["filtered"] += 1
            continue
        examples.append(example)
        if args.max_examples is not None and len(examples) >= args.max_examples:
            break
    if not examples:
        raise ValueError("no trainable examples after filtering")
    print(json.dumps({"examples": len(examples), "skipped": dict(skipped)}, sort_keys=True), flush=True)
    return examples


def collate_batch(examples: list[dict[str, Any]], *, pad_token_id: int) -> dict[str, Any]:
    import torch

    max_len = max(len(example["input_ids"]) for example in examples)
    input_ids = torch.full((len(examples), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
    labels = torch.full((len(examples), max_len), -100, dtype=torch.long)

    for row, example in enumerate(examples):
        width = len(example["input_ids"])
        input_ids[row, :width] = torch.tensor(example["input_ids"], dtype=torch.long)
        attention_mask[row, :width] = 1
        labels[row, :width] = torch.tensor(example["labels"], dtype=torch.long)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def find_layers_container(model: Any) -> tuple[Any, str, Any]:
    import torch.nn as nn

    for module in model.modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            return module, "layers", layers
    raise ValueError("could not find transformer ModuleList named 'layers'")


@contextlib.contextmanager
def loop_tail_layers(model: Any, *, tail_layers: int, loops: int):
    import torch.nn as nn

    container, attr, original_layers = find_layers_container(model)
    if loops < 1:
        raise ValueError("loops must be at least 1")
    if tail_layers < 1 or tail_layers > len(original_layers):
        raise ValueError(f"tail_layers must be in 1..{len(original_layers)}")

    prefix = list(original_layers[:-tail_layers])
    tail = list(original_layers[-tail_layers:])
    setattr(container, attr, nn.ModuleList(prefix + tail * loops))
    try:
        yield
    finally:
        setattr(container, attr, original_layers)


def masked_kl_loss(student_logits: Any, teacher_logits: Any, labels: Any, *, temperature: float) -> Any:
    import torch
    import torch.nn.functional as F

    mask = labels.ne(-100)
    if not torch.any(mask):
        return student_logits.new_zeros(())

    student = student_logits[mask] / temperature
    teacher = teacher_logits[mask].detach() / temperature
    teacher_probs = F.softmax(teacher, dim=-1)
    student_log_probs = F.log_softmax(student, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)


def lambda_for_step(step: int, *, total_steps: int, final_lambda: float) -> float:
    if total_steps <= 1:
        return final_lambda
    progress = min(1.0, max(0.0, step / float(total_steps - 1)))
    return (1.0 - progress) + final_lambda * progress


def batch_iterator(examples: list[dict[str, Any]], *, batch_size: int, rng: random.Random):
    order = list(range(len(examples)))
    while True:
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            yield [examples[index] for index in order[start : start + batch_size]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tail-looped LoRA adapter on ETPI Pi traces.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hf-dataset", help="HF trajectory dataset repo id.")
    source.add_argument("--steps", type=Path, help="Local ETPI step JSONL.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--include-empty-assistant", action="store_true")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-total-tokens", type=int, default=8192)
    parser.add_argument("--max-target-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--tail-layers", type=int, default=4)
    parser.add_argument("--max-loops", type=int, default=4)
    parser.add_argument("--min-student-loops", type=int, default=1)
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument("--final-lambda", type=float, default=0.0)
    parser.add_argument("--student-loss-weight", type=float, default=1.0)
    parser.add_argument("--distill-loss-weight", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated PEFT target module names.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_loops < 2:
        raise ValueError("--max-loops must be at least 2")
    if args.min_student_loops < 1 or args.min_student_loops >= args.max_loops:
        raise ValueError("--min-student-loops must be in 1..max_loops-1")

    import torch
    from peft import LoraConfig
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype(args.dtype),
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device_map and args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    container, _attr, layers = find_layers_container(model)
    total_layers = len(layers)
    if args.tail_layers < 1 or args.tail_layers > total_layers:
        raise ValueError(f"--tail-layers must be in 1..{total_layers}")
    first_tail_layer = total_layers - args.tail_layers
    target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        layers_to_transform=list(range(first_tail_layer, total_layers)),
        layers_pattern="layers",
    )
    del container
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.train()

    examples = load_examples(args, tokenizer)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    iterator = batch_iterator(examples, batch_size=args.batch_size, rng=rng)
    optimizer.zero_grad(set_to_none=True)

    for step in range(args.max_steps):
        lambda_value = lambda_for_step(step, total_steps=args.max_steps, final_lambda=args.final_lambda)
        student_loops = rng.randint(args.min_student_loops, args.max_loops - 1)
        accum_loss = 0.0
        accum_teacher = 0.0
        accum_student = 0.0
        accum_kl = 0.0

        for _micro in range(args.grad_accum_steps):
            raw_batch = next(iterator)
            batch = collate_batch(raw_batch, pad_token_id=pad_token_id)
            batch = {key: value.to(device) for key, value in batch.items()}

            with loop_tail_layers(model, tail_layers=args.tail_layers, loops=args.max_loops):
                teacher_outputs = model(**batch, use_cache=False)
            with loop_tail_layers(model, tail_layers=args.tail_layers, loops=student_loops):
                student_outputs = model(**batch, use_cache=False)

            teacher_loss = teacher_outputs.loss
            student_ce = student_outputs.loss
            distill_coeff = args.distill_loss_weight * (1.0 - lambda_value)
            if distill_coeff:
                kl = masked_kl_loss(
                    student_outputs.logits,
                    teacher_outputs.logits,
                    batch["labels"],
                    temperature=args.distill_temperature,
                )
            else:
                kl = student_ce.new_zeros(())
            student_loss = lambda_value * student_ce + distill_coeff * kl
            loss = teacher_loss + args.student_loss_weight * student_loss

            (loss / args.grad_accum_steps).backward()
            accum_loss += float(loss.detach().cpu())
            accum_teacher += float(teacher_loss.detach().cpu())
            accum_student += float(student_ce.detach().cpu())
            accum_kl += float(kl.detach().cpu())

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if args.log_every and (step + 1) % args.log_every == 0:
            payload = {
                "step": step + 1,
                "loss": accum_loss / args.grad_accum_steps,
                "teacher_ce": accum_teacher / args.grad_accum_steps,
                "student_ce": accum_student / args.grad_accum_steps,
                "kl": accum_kl / args.grad_accum_steps,
                "lambda": lambda_value,
                "student_loops": student_loops,
                "max_loops": args.max_loops,
            }
            print(json.dumps(payload, sort_keys=True), flush=True)

        if args.save_every and (step + 1) % args.save_every == 0:
            checkpoint_dir = args.out_dir / f"checkpoint-{step + 1:06d}"
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    summary = {
        "model": args.model,
        "examples": len(examples),
        "max_steps": args.max_steps,
        "tail_layers": args.tail_layers,
        "max_loops": args.max_loops,
        "min_student_loops": args.min_student_loops,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_targets": target_modules,
        "max_total_tokens": args.max_total_tokens,
        "max_target_tokens": args.max_target_tokens,
    }
    (args.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out_dir": str(args.out_dir), **summary}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
