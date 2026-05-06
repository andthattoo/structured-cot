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
from pathlib import Path
from typing import Any

import numpy as np


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


def part_text(part: dict[str, Any]) -> str:
    for key in ("text", "thinking", "content"):
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
    parser.add_argument("--steps", type=Path, required=True, help="Step dataset JSONL.")
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    step = select_step(args.steps, step_index=args.step_index, step_id=args.step_id)
    prefix = render_step_prefix(step)
    target = target_thinking_text(step, allow_empty=args.allow_empty_thinking)

    model, tokenizer = load_model_and_tokenizer(args)
    layers = parse_layers(args.layers, len(transformer_layers(model)))

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    thinking_raw = str(step.get("raw_thinking") or "").strip()
    thinking_ids = tokenizer.encode(thinking_raw, add_special_tokens=False) if thinking_raw else []
    end_ids = tokenizer.encode("\n</think>" if thinking_raw else "</think>", add_special_tokens=False)
    target_ids = thinking_ids + end_ids
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
        "prefix_tokens": len(prefix_ids),
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
