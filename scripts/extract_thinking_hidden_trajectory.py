#!/usr/bin/env python3
"""Extract hidden-state trajectories for one ETPI thinking step.

This script teacher-forces an existing step-dataset row instead of sampling new
tokens. It renders the state up to an assistant thinking block, feeds the
recorded `raw_thinking` plus `</think>`, captures selected transformer layer
states, and writes vectors plus simple PCA/distance summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


TOOL_RESULT_ROLES = {"tool", "toolResult"}
TEXT_KEYS = ("text", "thinking", "content")


def read_jsonl(path: Path):
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield line_no - 1, row


def select_step(path: Path, *, step_index: int | None, step_id: str | None) -> dict[str, Any]:
    if step_index is None and step_id is None:
        raise ValueError("pass --step-index or --step-id")
    for index, row in read_jsonl(path):
        if step_index is not None and index == step_index:
            return row
        if step_id is not None and row.get("id") == step_id:
            return row
    if step_id is not None:
        raise ValueError(f"{path}: no step with id {step_id!r}")
    raise ValueError(f"{path}: no step at index {step_index}")


def hf_rows(dataset: str, split: str):
    from datasets import load_dataset

    yield from load_dataset(dataset, split=split, streaming=True)


def content_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return [part for part in content if isinstance(part, dict)]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return []


def extract_tool_calls(parts: list[dict[str, Any]], message: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for part in parts:
        if part.get("type") != "toolCall":
            continue
        calls.append(
            {
                "id": "" if part.get("id") is None else str(part.get("id")),
                "name": "" if part.get("name") is None else str(part.get("name")),
                "arguments": part.get("arguments") or {},
            }
        )
    raw_calls = message.get("tool_calls")
    if isinstance(raw_calls, list):
        for call in raw_calls:
            if not isinstance(call, dict):
                continue
            calls.append(
                {
                    "id": "" if call.get("id") is None else str(call.get("id")),
                    "name": "" if call.get("name") is None else str(call.get("name")),
                    "arguments": call.get("arguments") or {},
                }
            )
    return calls


def text_from_parts(parts: list[dict[str, Any]], *, include_thinking: bool = True) -> str:
    chunks: list[str] = []
    for part in parts:
        if not include_thinking and part.get("type") == "thinking":
            continue
        text = part_text(part)
        if text:
            chunks.append(text)
    return "\n".join(chunks)


def thinking_from_parts(parts: list[dict[str, Any]]) -> str:
    return "\n".join(part_text(part) for part in parts if part.get("type") == "thinking" and part_text(part))


def trajectory_records(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("trajectory_json")
    if not isinstance(value, str) or not value.strip():
        return []
    records = json.loads(value)
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def trajectory_messages(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for record in records:
        if record.get("type") != "message" and not ("role" in record and "content" in record):
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            message = record
        role = message.get("role")
        if not isinstance(role, str):
            continue
        role = "tool" if role in TOOL_RESULT_ROLES else role
        parts = content_parts(message.get("content"))
        messages.append(
            {
                "id": "" if record.get("id") is None else str(record.get("id")),
                "role": role,
                "raw_role": message.get("role"),
                "content": parts,
                "text": text_from_parts(parts),
                "text_without_thinking": text_from_parts(parts, include_thinking=False),
                "tool_calls": extract_tool_calls(parts, message),
                "stop_reason": "" if message.get("stopReason") is None else str(message.get("stopReason")),
                "error": "" if message.get("errorMessage") is None else str(message.get("errorMessage")),
                "usage": message.get("usage") if isinstance(message.get("usage"), dict) else {},
            }
        )
    return messages


def metadata_from_trace_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "run_id",
        "task_id",
        "source",
        "thinking_level",
        "base_task_id",
        "repo_id",
        "repo_name",
        "domain",
        "persona_id",
        "intent",
        "language",
        "needs_workspace",
        "task_kind",
        "task_profile",
    ]
    return {key: row.get(key) for key in keys if key in row}


def steps_from_trace_row(row: dict[str, Any], *, include_empty_assistant: bool = False) -> list[dict[str, Any]]:
    messages = trajectory_messages(trajectory_records(row))
    state: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    run_id = str(row.get("run_id") or "")
    task_id = str(row.get("task_id") or "")
    metadata = metadata_from_trace_row(row)
    assistant_index = 0

    for message in messages:
        if message["role"] != "assistant":
            state.append(message)
            continue
        if not include_empty_assistant and not message["content"] and not message["tool_calls"]:
            state.append(message)
            continue
        raw_thinking = thinking_from_parts(message["content"])
        calls = message["tool_calls"]
        steps.append(
            {
                "id": f"{run_id}/{task_id}/assistant_{assistant_index:04d}",
                "run_id": run_id,
                "task_id": task_id,
                "source": row.get("source") or "",
                "thinking_level": row.get("thinking_level") or "",
                "state_messages": list(state),
                "target_assistant": message,
                "raw_thinking": raw_thinking,
                "compressed_thinking": None,
                "loss_mask": {
                    "state": False,
                    "tool_results": False,
                    "assistant_action": True,
                    "raw_verbose_thinking": False,
                    "compressed_thinking": False,
                },
                "reward_features": {
                    "trace_success": str(row.get("status") or "ok") == "ok",
                    "step_index": len(steps),
                    "assistant_index": assistant_index,
                    "state_messages": len(state),
                    "has_tool_call": bool(calls),
                    "tool_names": [call.get("name") or "" for call in calls],
                    "stop_reason": message.get("stop_reason") or "",
                    "raw_thinking_chars": len(raw_thinking),
                    "target_text_chars": len(message.get("text_without_thinking") or ""),
                },
                "metadata": metadata,
            }
        )
        assistant_index += 1
        state.append(message)

    return steps


def select_step_from_hf(
    dataset: str,
    *,
    split: str,
    step_index: int | None,
    step_id: str | None,
    include_empty_assistant: bool = False,
) -> dict[str, Any]:
    if step_index is None and step_id is None:
        raise ValueError("pass --step-index or --step-id")

    global_index = 0
    for trace in hf_rows(dataset, split):
        for step in steps_from_trace_row(trace, include_empty_assistant=include_empty_assistant):
            if step_index is not None and global_index == step_index:
                return step
            if step_id is not None and step.get("id") == step_id:
                return step
            global_index += 1

    if step_id is not None:
        raise ValueError(f"{dataset}/{split}: no step with id {step_id!r}")
    raise ValueError(f"{dataset}/{split}: no step at global index {step_index}")


def part_text(part: dict[str, Any]) -> str:
    for key in TEXT_KEYS:
        value = part.get(key)
        if value is not None:
            return str(value)
    return ""


def render_content_parts(parts: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for part in parts:
        kind = part.get("type")
        if kind == "thinking":
            text = part_text(part)
            if text:
                chunks.append(f"<think>\n{text}\n</think>")
        elif kind == "toolCall":
            name = "" if part.get("name") is None else str(part.get("name"))
            args = part.get("arguments") or {}
            chunks.append(f"TOOL_CALL {name} {json.dumps(args, ensure_ascii=False, sort_keys=True)}")
        else:
            text = part_text(part)
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def render_message(message: dict[str, Any]) -> str:
    role = str(message.get("role") or "user")
    rendered_role = "tool" if role == "tool" else role
    parts = message.get("content") if isinstance(message.get("content"), list) else []
    content = render_content_parts(parts)
    calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    for call in calls:
        name = "" if call.get("name") is None else str(call.get("name"))
        args = call.get("arguments") or {}
        line = f"TOOL_CALL {name} {json.dumps(args, ensure_ascii=False, sort_keys=True)}"
        if line not in content:
            content = f"{content}\n{line}".strip()
    return f"<|im_start|>{rendered_role}\n{content}\n<|im_end|>"


def render_step_prefix(step: dict[str, Any]) -> str:
    state = step.get("state_messages")
    if not isinstance(state, list):
        state = []
    rendered = [render_message(message) for message in state if isinstance(message, dict)]
    rendered.append("<|im_start|>assistant\n<think>\n")
    return "\n".join(rendered)


def target_thinking_text(step: dict[str, Any], *, allow_empty: bool) -> str:
    raw = step.get("raw_thinking")
    if not isinstance(raw, str):
        raw = ""
    raw = raw.strip()
    if not raw and not allow_empty:
        raise ValueError(f"step {step.get('id')} has empty raw_thinking; pass --allow-empty-thinking")
    return f"{raw}\n</think>" if raw else "</think>"


def raw_thinking_text(step: dict[str, Any], *, allow_empty: bool) -> str:
    raw = step.get("raw_thinking")
    if not isinstance(raw, str):
        raw = ""
    raw = raw.strip()
    if not raw and not allow_empty:
        raise ValueError(f"step {step.get('id')} has empty raw_thinking; pass --allow-empty-thinking")
    return raw


def filler_token_ids(tokenizer: Any, count: int, *, filler_text: str) -> list[int]:
    if count <= 0:
        return []
    seed_ids = tokenizer.encode(filler_text, add_special_tokens=False)
    if not seed_ids:
        seed_ids = tokenizer.encode(" therefore", add_special_tokens=False)
    if not seed_ids:
        raise ValueError("filler text produced no tokens")
    repeats = (count // len(seed_ids)) + 1
    return (seed_ids * repeats)[:count]


def target_sequence(
    tokenizer: Any,
    raw_thinking: str,
    *,
    mode: str,
    seed: int,
    filler_text: str,
    override_file: Path | None = None,
) -> tuple[list[int], list[int], list[int], str]:
    recorded_ids = tokenizer.encode(raw_thinking, add_special_tokens=False) if raw_thinking else []

    if override_file is not None:
        thinking_text = override_file.read_text().strip()
        thinking_ids = tokenizer.encode(thinking_text, add_special_tokens=False) if thinking_text else []
    elif mode == "recorded":
        thinking_ids = list(recorded_ids)
        thinking_text = raw_thinking
    elif mode == "empty":
        thinking_ids = []
        thinking_text = ""
    elif mode == "filler":
        thinking_ids = filler_token_ids(tokenizer, len(recorded_ids), filler_text=filler_text)
        thinking_text = tokenizer.decode(thinking_ids, clean_up_tokenization_spaces=False)
    elif mode == "shuffle":
        thinking_ids = list(recorded_ids)
        rng = random.Random(seed)
        rng.shuffle(thinking_ids)
        thinking_text = tokenizer.decode(thinking_ids, clean_up_tokenization_spaces=False)
    else:
        raise ValueError(f"unknown thinking mode {mode!r}")

    end_text = "\n</think>" if thinking_ids else "</think>"
    end_ids = tokenizer.encode(end_text, add_special_tokens=False)
    if not end_ids:
        raise ValueError("end marker produced no tokens")
    target_ids = thinking_ids + end_ids
    target_text = f"{thinking_text}{end_text}" if thinking_text else end_text
    return target_ids, thinking_ids, end_ids, target_text


def parse_layers(value: str, layer_count: int) -> list[int]:
    if value.strip().lower() in {"last", "-1"}:
        return [layer_count - 1]
    layers: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        index = int(item)
        if index < 0:
            index = layer_count + index
        if index < 0 or index >= layer_count:
            raise ValueError(f"layer {item} resolves to {index}, expected 0..{layer_count - 1}")
        layers.append(index)
    if not layers:
        raise ValueError("no layers selected")
    return sorted(set(layers))


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if matrix.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    coords = centered @ basis
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])
    return coords.astype(np.float32)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return 1.0 - float(np.dot(a, b) / denom)


def distance_rows(hidden: np.ndarray, token_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prefill = hidden[0]
    rows: list[dict[str, Any]] = []
    cumulative = 0.0
    previous = prefill
    for index, token_row in enumerate(token_rows):
        vector = hidden[index]
        step_l2 = float(np.linalg.norm(vector - previous)) if index else 0.0
        cumulative += step_l2
        rows.append(
            {
                **token_row,
                "hidden_norm": float(np.linalg.norm(vector)),
                "l2_to_prefill": float(np.linalg.norm(vector - prefill)),
                "cosine_to_prefill": cosine_distance(vector, prefill),
                "step_l2": step_l2,
                "cumulative_l2": cumulative,
            }
        )
        previous = vector
    return rows


def transformer_layers(model: Any) -> list[Any]:
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
        getattr(model, "layers", None),
    ]
    for candidate in candidates:
        if candidate is not None:
            return list(candidate)
    raise ValueError("could not locate transformer layers on model")


def backbone_model(model: Any) -> Any:
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "forward"):
        return inner
    return model


def torch_dtype(name: str) -> Any:
    import torch

    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unknown dtype {name!r}")


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=args.device_map,
        torch_dtype=torch_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    return model, tokenizer


def token_strings(tokenizer: Any, ids: list[int]) -> list[str]:
    return [tokenizer.decode([token_id], clean_up_tokenization_spaces=False) for token_id in ids]


def capture_hidden_states(
    model: Any,
    *,
    input_ids: Any,
    attention_mask: Any,
    layers: list[int],
    positions: list[int],
) -> dict[int, np.ndarray]:
    import torch

    all_layers = transformer_layers(model)
    captured: dict[int, np.ndarray] = {}
    handles = []

    def make_hook(layer_index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            index_tensor = torch.tensor(positions, device=hidden.device, dtype=torch.long)
            captured[layer_index] = hidden[0].index_select(0, index_tensor).detach().float().cpu().numpy()

        return hook

    for layer_index in layers:
        handles.append(all_layers[layer_index].register_forward_hook(make_hook(layer_index)))

    try:
        with torch.no_grad():
            backbone_model(model)(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    missing = [layer for layer in layers if layer not in captured]
    if missing:
        raise RuntimeError(f"failed to capture hidden states for layers {missing}")
    return captured


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract hidden states for one ETPI thinking step.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--steps", type=Path, help="Step dataset JSONL.")
    source.add_argument("--hf-dataset", help="HF trajectory dataset repo id. Steps are sliced on the fly.")
    parser.add_argument("--split", default="train")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--step-index", type=int)
    group.add_argument("--step-id")
    parser.add_argument("--model", required=True, help="HF model id or local model path.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layers", default="-1", help="Comma-separated layer indices, negative allowed; default last.")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--allow-empty-thinking", action="store_true")
    parser.add_argument("--include-empty-assistant", action="store_true")
    parser.add_argument(
        "--thinking-mode",
        choices=["recorded", "empty", "filler", "shuffle"],
        default="recorded",
        help="Teacher-forced thinking control. Filler/shuffle preserve recorded thinking token count.",
    )
    parser.add_argument("--thinking-override-file", type=Path, default=None)
    parser.add_argument("--control-seed", type=int, default=0)
    parser.add_argument("--filler-text", default=" therefore")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps is not None:
        step = select_step(args.steps, step_index=args.step_index, step_id=args.step_id)
    else:
        step = select_step_from_hf(
            args.hf_dataset,
            split=args.split,
            step_index=args.step_index,
            step_id=args.step_id,
            include_empty_assistant=args.include_empty_assistant,
        )
    prefix = render_step_prefix(step)
    thinking_raw = raw_thinking_text(step, allow_empty=args.allow_empty_thinking)

    model, tokenizer = load_model_and_tokenizer(args)
    layers = parse_layers(args.layers, len(transformer_layers(model)))

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    target_ids, thinking_ids, end_ids, target = target_sequence(
        tokenizer,
        thinking_raw,
        mode=args.thinking_mode,
        seed=args.control_seed,
        filler_text=args.filler_text,
        override_file=args.thinking_override_file,
    )
    input_token_ids = prefix_ids + target_ids

    if not prefix_ids:
        raise ValueError("rendered prefix produced no tokens")
    if not target_ids:
        raise ValueError("target produced no tokens")
    if args.max_input_tokens is not None and len(input_token_ids) > args.max_input_tokens:
        raise ValueError(
            f"input has {len(input_token_ids)} tokens, above --max-input-tokens {args.max_input_tokens}"
        )

    import torch

    input_device = next(model.parameters()).device
    input_ids = torch.tensor([input_token_ids], dtype=torch.long, device=input_device)
    attention_mask = torch.ones_like(input_ids)

    positions = [len(prefix_ids) - 1] + [len(prefix_ids) + i for i in range(len(target_ids))]
    token_rows = [
        {
            "position_index": 0,
            "sequence_position": len(prefix_ids) - 1,
            "segment": "prefill",
            "token_id": "",
            "token_text": "",
        }
    ]
    for offset, (token_id, token_text) in enumerate(zip(target_ids, token_strings(tokenizer, target_ids)), 1):
        segment = "thinking" if offset <= len(thinking_ids) else "think_end"
        token_rows.append(
            {
                "position_index": offset,
                "sequence_position": len(prefix_ids) + offset - 1,
                "segment": segment,
                "token_id": token_id,
                "token_text": token_text,
            }
        )

    captured = capture_hidden_states(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        layers=layers,
        positions=positions,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "hidden_states.npz",
        **{f"layer_{layer}": matrix for layer, matrix in captured.items()},
        positions=np.array(positions, dtype=np.int64),
        token_ids=np.array([-1] + target_ids, dtype=np.int64),
    )

    metadata = {
        "step_id": step.get("id"),
        "run_id": step.get("run_id"),
        "task_id": step.get("task_id"),
        "source": step.get("source"),
        "thinking_level": step.get("thinking_level"),
        "model": args.model,
        "layers": layers,
        "thinking_mode": args.thinking_mode,
        "thinking_override_file": str(args.thinking_override_file) if args.thinking_override_file else None,
        "control_seed": args.control_seed,
        "prefix_tokens": len(prefix_ids),
        "recorded_thinking_tokens": len(tokenizer.encode(thinking_raw, add_special_tokens=False)) if thinking_raw else 0,
        "thinking_tokens": len(thinking_ids),
        "end_tokens": len(end_ids),
        "total_tokens": len(input_token_ids),
        "raw_thinking_chars": len(thinking_raw),
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    (args.out_dir / "prompt_prefix.txt").write_text(prefix)
    (args.out_dir / "target_thinking.txt").write_text(target)

    for layer, matrix in captured.items():
        rows = distance_rows(matrix, token_rows)
        write_csv(args.out_dir / f"distances_layer_{layer}.csv", rows)
        coords = pca_2d(matrix)
        projection_rows = [
            {**row, "pca_x": float(coord[0]), "pca_y": float(coord[1])}
            for row, coord in zip(rows, coords)
        ]
        write_csv(args.out_dir / f"pca_layer_{layer}.csv", projection_rows)

    print(json.dumps({"out_dir": str(args.out_dir), **metadata}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
