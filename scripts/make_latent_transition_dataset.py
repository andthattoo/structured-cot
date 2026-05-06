#!/usr/bin/env python3
"""Build latent pre/post-thinking transition pairs from ETPI traces."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_thinking_hidden_trajectory import capture_hidden_states
from extract_thinking_hidden_trajectory import hf_rows
from extract_thinking_hidden_trajectory import load_model_and_tokenizer
from extract_thinking_hidden_trajectory import parse_layers
from extract_thinking_hidden_trajectory import raw_thinking_text
from extract_thinking_hidden_trajectory import render_step_prefix
from extract_thinking_hidden_trajectory import steps_from_trace_row
from extract_thinking_hidden_trajectory import target_sequence
from extract_thinking_hidden_trajectory import transformer_layers


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


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return 1.0 - float(np.dot(a, b) / denom)


def pair_inputs(
    step: dict[str, Any],
    tokenizer: Any,
    *,
    allow_empty_thinking: bool,
    min_thinking_tokens: int,
    max_prefix_tokens: int | None,
    max_total_tokens: int | None,
) -> tuple[list[int], list[int], list[int], list[int], list[int], str, str | None]:
    try:
        raw = raw_thinking_text(step, allow_empty=allow_empty_thinking)
    except ValueError as exc:
        return [], [], [], [], [], "", str(exc)

    prefix = render_step_prefix(step)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    target_ids, thinking_ids, end_ids, _target_text = target_sequence(
        tokenizer,
        raw,
        mode="recorded",
        seed=0,
        filler_text=" therefore",
    )
    input_ids = prefix_ids + target_ids

    if len(thinking_ids) < min_thinking_tokens:
        return prefix_ids, thinking_ids, end_ids, target_ids, input_ids, raw, "below_min_thinking_tokens"
    if max_prefix_tokens is not None and len(prefix_ids) > max_prefix_tokens:
        return prefix_ids, thinking_ids, end_ids, target_ids, input_ids, raw, "above_max_prefix_tokens"
    if max_total_tokens is not None and len(input_ids) > max_total_tokens:
        return prefix_ids, thinking_ids, end_ids, target_ids, input_ids, raw, "above_max_total_tokens"
    if not prefix_ids:
        return prefix_ids, thinking_ids, end_ids, target_ids, input_ids, raw, "empty_prefix"
    if not target_ids:
        return prefix_ids, thinking_ids, end_ids, target_ids, input_ids, raw, "empty_target"

    positions = [len(prefix_ids) - 1, len(input_ids) - 1]
    return prefix_ids, thinking_ids, end_ids, target_ids, input_ids, raw, None


class ShardWriter:
    def __init__(self, out_dir: Path, *, shard_size: int, compress: bool) -> None:
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.compress = compress
        self.shard_index = 0
        self.row_index = 0
        self.x_rows: list[np.ndarray] = []
        self.y_rows: list[np.ndarray] = []

    def add(self, x: np.ndarray, y: np.ndarray) -> tuple[str, int]:
        if len(self.x_rows) >= self.shard_size:
            self.flush()
        shard_name = f"shard_{self.shard_index:05d}.npz"
        row = self.row_index
        self.x_rows.append(x.astype(np.float32, copy=False))
        self.y_rows.append(y.astype(np.float32, copy=False))
        self.row_index += 1
        return f"shards/{shard_name}", row

    def flush(self) -> None:
        if not self.x_rows:
            return
        shard_dir = self.out_dir / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        path = shard_dir / f"shard_{self.shard_index:05d}.npz"
        x = np.stack(self.x_rows)
        y = np.stack(self.y_rows)
        saver = np.savez_compressed if self.compress else np.savez
        saver(path, x=x, y=y, delta=y - x)
        self.shard_index += 1
        self.row_index = 0
        self.x_rows = []
        self.y_rows = []


def pair_metadata(
    step: dict[str, Any],
    *,
    pair_index: int,
    shard_path: str,
    row_in_shard: int,
    layers: list[int],
    prefix_tokens: int,
    thinking_tokens: int,
    end_tokens: int,
    total_tokens: int,
    raw_thinking_chars: int,
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    target = step.get("target_assistant") if isinstance(step.get("target_assistant"), dict) else {}
    reward = step.get("reward_features") if isinstance(step.get("reward_features"), dict) else {}
    tool_names = reward.get("tool_names") if isinstance(reward.get("tool_names"), list) else []
    delta = y - x
    layer_metrics = []
    for offset, layer in enumerate(layers):
        layer_metrics.append(
            {
                "layer": layer,
                "l2_delta": float(np.linalg.norm(delta[offset])),
                "cosine_distance": cosine_distance(x[offset], y[offset]),
            }
        )
    return {
        "pair_index": pair_index,
        "id": step.get("id") or f"pair_{pair_index:08d}",
        "run_id": step.get("run_id") or "",
        "task_id": step.get("task_id") or "",
        "source": step.get("source") or "",
        "thinking_level": step.get("thinking_level") or "",
        "shard_path": shard_path,
        "row_in_shard": row_in_shard,
        "layers": layers,
        "prefix_tokens": prefix_tokens,
        "thinking_tokens": thinking_tokens,
        "end_tokens": end_tokens,
        "total_tokens": total_tokens,
        "raw_thinking_chars": raw_thinking_chars,
        "target_stop_reason": target.get("stop_reason") or reward.get("stop_reason") or "",
        "target_has_tool_call": bool(reward.get("has_tool_call")),
        "tool_names": tool_names,
        "layer_metrics": layer_metrics,
    }


def write_stats(path: Path, stats: dict[str, Any]) -> None:
    payload = {
        **{key: value for key, value in stats.items() if not isinstance(value, Counter)},
        "skip_reasons": dict(sorted(stats["skip_reasons"].items())),
        "runs": dict(sorted(stats["runs"].items())),
        "sources": dict(sorted(stats["sources"].items())),
        "tool_names": dict(sorted(stats["tool_names"].items())),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build latent h_before -> h_after thinking transition pairs.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hf-dataset", help="HF trajectory dataset repo id.")
    source.add_argument("--steps", type=Path, help="Local ETPI step JSONL.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layers", default="-1")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--min-thinking-tokens", type=int, default=1)
    parser.add_argument("--max-prefix-tokens", type=int, default=None)
    parser.add_argument("--max-total-tokens", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--allow-empty-thinking", action="store_true")
    parser.add_argument("--include-empty-assistant", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(args)
    layers = parse_layers(args.layers, len(transformer_layers(model)))

    import torch

    input_device = next(model.parameters()).device
    writer = ShardWriter(args.out_dir, shard_size=args.shard_size, compress=args.compress)
    pairs_path = args.out_dir / "pairs.jsonl"
    stats: dict[str, Any] = {
        "model": args.model,
        "layers": layers,
        "pairs": 0,
        "seen_steps": 0,
        "skipped_steps": 0,
        "skip_reasons": Counter(),
        "runs": Counter(),
        "sources": Counter(),
        "tool_names": Counter(),
    }

    with pairs_path.open("w") as pairs_file:
        for step in step_stream(args):
            stats["seen_steps"] += 1
            (
                prefix_ids,
                thinking_ids,
                end_ids,
                _target_ids,
                input_token_ids,
                raw,
                skip_reason,
            ) = pair_inputs(
                step,
                tokenizer,
                allow_empty_thinking=args.allow_empty_thinking,
                min_thinking_tokens=args.min_thinking_tokens,
                max_prefix_tokens=args.max_prefix_tokens,
                max_total_tokens=args.max_total_tokens,
            )
            if skip_reason is not None:
                stats["skipped_steps"] += 1
                stats["skip_reasons"][skip_reason] += 1
                continue

            positions = [len(prefix_ids) - 1, len(input_token_ids) - 1]
            input_ids = torch.tensor([input_token_ids], dtype=torch.long, device=input_device)
            attention_mask = torch.ones_like(input_ids)
            captured = capture_hidden_states(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                layers=layers,
                positions=positions,
            )
            x = np.stack([captured[layer][0] for layer in layers])
            y = np.stack([captured[layer][1] for layer in layers])
            shard_path, row_in_shard = writer.add(x, y)
            metadata = pair_metadata(
                step,
                pair_index=stats["pairs"],
                shard_path=shard_path,
                row_in_shard=row_in_shard,
                layers=layers,
                prefix_tokens=len(prefix_ids),
                thinking_tokens=len(thinking_ids),
                end_tokens=len(end_ids),
                total_tokens=len(input_token_ids),
                raw_thinking_chars=len(raw),
                x=x,
                y=y,
            )
            pairs_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")

            stats["pairs"] += 1
            stats["runs"][metadata["run_id"]] += 1
            stats["sources"][metadata["source"]] += 1
            stats["tool_names"].update(metadata["tool_names"])
            if args.max_pairs is not None and stats["pairs"] >= args.max_pairs:
                break

    writer.flush()
    write_stats(args.out_dir / "stats.json", stats)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "pairs": stats["pairs"],
                "seen_steps": stats["seen_steps"],
                "skipped_steps": stats["skipped_steps"],
                "layers": layers,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
