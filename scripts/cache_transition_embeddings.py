#!/usr/bin/env python3
"""Cache transition encoder embeddings for MQE critic training."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from train_transition_encoder import (
    ACTION_TYPES,
    action_feature_dim,
    action_features,
    action_text,
    build_transition_head_class,
    encode_texts,
    pad_features,
    read_jsonl,
    resolve_device,
)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_states(data_dir: Path) -> dict[str, str]:
    path = data_dir / "states.jsonl"
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            state_hash = str(row.get("state_hash") or "")
            text = str(row.get("serialized_state") or "")
            if state_hash and text:
                values[state_hash] = text
    return values


def goal_text(row: dict[str, Any], states: dict[str, str]) -> str:
    concrete = states.get(str(row.get("goal_state_hash") or ""))
    if concrete:
        return concrete
    return "\n".join(
        part
        for part in [
            str(row.get("goal_prompt") or "").strip(),
            str(row.get("goal_policy_text") or "").strip(),
        ]
        if part
    )


def row_id(row: dict[str, Any], index: int, split: str) -> str:
    return str(row.get("example_id") or f"{split}:{index}")


def load_encoder_for_cache(encoder_dir: Path, device: str, dtype_name: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: peft. Run with:\n"
            "  uv run --with torch --with transformers --with peft python scripts/cache_transition_embeddings.py ..."
        ) from exc

    config_path = encoder_dir / "transition_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing transition config: {config_path}")
    transition_config = json.loads(config_path.read_text())
    base_model = transition_config["base_model"]
    dtype = torch.float32
    if dtype_name == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_name == "float16":
        dtype = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(encoder_dir, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    encoder = AutoModel.from_pretrained(base_model, torch_dtype=dtype, trust_remote_code=True).to(device)
    adapter_dir = encoder_dir / "encoder_adapter"
    if (adapter_dir / "adapter_config.json").exists():
        encoder = PeftModel.from_pretrained(encoder, adapter_dir).to(device)
    encoder.eval()
    head_blob = torch.load(encoder_dir / "transition_head.pt", map_location="cpu")
    TransitionHead = build_transition_head_class()
    head = TransitionHead(
        int(head_blob["embedding_dim"]),
        int(head_blob["action_feature_dim"]),
        int(head_blob["config"]["transition_hidden_dim"]),
        len(head_blob.get("action_types") or ACTION_TYPES),
    ).to(device)
    head.load_state_dict(head_blob["transition_head_state_dict"])
    head.eval()
    return encoder, tokenizer, head, transition_config, head_blob


def cache_split(
    *,
    split: str,
    rows: list[dict[str, Any]],
    states: dict[str, str],
    encoder,
    tokenizer,
    head,
    feature_dim: int,
    max_state_tokens: int,
    max_action_tokens: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    import torch

    row_ids: list[str] = []
    text_hashes = {"state": [], "action": [], "next_state": [], "goal": []}
    z_state_parts = []
    z_action_parts = []
    z_next_parts = []
    z_goal_parts = []
    z_state_action_parts = []
    feature_parts = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            state_texts = [str(row.get("state_text") or "") for row in batch_rows]
            action_texts = [action_text(row) for row in batch_rows]
            next_state_texts = [str(row.get("next_state_text") or "") for row in batch_rows]
            goal_texts = [goal_text(row, states) for row in batch_rows]
            features = torch.tensor(
                [pad_features(action_features(row), feature_dim) for row in batch_rows],
                dtype=torch.float32,
                device=device,
            )
            z_state = encode_texts(
                encoder,
                tokenizer,
                state_texts,
                max_length=max_state_tokens,
                device=device,
                truncation_side="left",
            )
            z_action = encode_texts(
                encoder,
                tokenizer,
                action_texts,
                max_length=max_action_tokens,
                device=device,
                truncation_side="right",
            )
            z_next = encode_texts(
                encoder,
                tokenizer,
                next_state_texts,
                max_length=max_state_tokens,
                device=device,
                truncation_side="left",
            )
            z_goal = encode_texts(
                encoder,
                tokenizer,
                goal_texts,
                max_length=max_state_tokens,
                device=device,
                truncation_side="left",
            )
            z_state_action, _ = head(z_state, z_action, features)
            z_state_parts.append(z_state.cpu())
            z_action_parts.append(z_action.cpu())
            z_next_parts.append(z_next.cpu())
            z_goal_parts.append(z_goal.cpu())
            z_state_action_parts.append(z_state_action.cpu())
            feature_parts.append(features.cpu())
            for offset, row in enumerate(batch_rows):
                row_ids.append(row_id(row, start + offset, split))
            text_hashes["state"].extend(stable_hash(text) for text in state_texts)
            text_hashes["action"].extend(stable_hash(text) for text in action_texts)
            text_hashes["next_state"].extend(stable_hash(text) for text in next_state_texts)
            text_hashes["goal"].extend(stable_hash(text) for text in goal_texts)
    return {
        "row_ids": row_ids,
        "z_state": torch.cat(z_state_parts, dim=0) if z_state_parts else torch.empty((0, 0)),
        "z_action": torch.cat(z_action_parts, dim=0) if z_action_parts else torch.empty((0, 0)),
        "z_next": torch.cat(z_next_parts, dim=0) if z_next_parts else torch.empty((0, 0)),
        "z_goal": torch.cat(z_goal_parts, dim=0) if z_goal_parts else torch.empty((0, 0)),
        "z_state_action": torch.cat(z_state_action_parts, dim=0) if z_state_action_parts else torch.empty((0, 0)),
        "action_features": torch.cat(feature_parts, dim=0) if feature_parts else torch.empty((0, feature_dim)),
        "text_hashes": text_hashes,
    }


def encode_unique_texts(
    *,
    label: str,
    texts: list[str],
    encoder,
    tokenizer,
    max_length: int,
    truncation_side: str,
    batch_size: int,
    device: str,
    progress_every: int,
) -> tuple[Any, list[str]]:
    import torch

    text_to_index: dict[str, int] = {}
    unique_texts: list[str] = []
    indices: list[int] = []
    for text in texts:
        index = text_to_index.get(text)
        if index is None:
            index = len(unique_texts)
            text_to_index[text] = index
            unique_texts.append(text)
        indices.append(index)

    started = time.time()
    print(
        json.dumps(
            {
                "phase": "encode_unique",
                "label": label,
                "rows": len(texts),
                "unique_texts": len(unique_texts),
                "batch_size": batch_size,
                "max_length": max_length,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    parts = []
    total_batches = (len(unique_texts) + batch_size - 1) // batch_size
    with torch.no_grad():
        for batch_no, start in enumerate(range(0, len(unique_texts), batch_size), start=1):
            batch_texts = unique_texts[start : start + batch_size]
            encoded = encode_texts(
                encoder,
                tokenizer,
                batch_texts,
                max_length=max_length,
                device=device,
                truncation_side=truncation_side,
            )
            parts.append(encoded.cpu())
            if progress_every > 0 and (batch_no == 1 or batch_no % progress_every == 0 or batch_no == total_batches):
                elapsed = max(1e-6, time.time() - started)
                done = min(start + batch_size, len(unique_texts))
                print(
                    json.dumps(
                        {
                            "phase": "encode_unique_progress",
                            "label": label,
                            "batch": batch_no,
                            "batches": total_batches,
                            "done_unique": done,
                            "unique_texts": len(unique_texts),
                            "unique_per_sec": round(done / elapsed, 3),
                            "elapsed_sec": round(elapsed, 1),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    if not parts:
        return torch.empty((0, 0)), []
    unique_tensor = torch.cat(parts, dim=0)
    index_tensor = torch.tensor(indices, dtype=torch.long)
    return unique_tensor[index_tensor], [stable_hash(text) for text in texts]


def cache_split_deduped(
    *,
    split: str,
    rows: list[dict[str, Any]],
    states: dict[str, str],
    encoder,
    tokenizer,
    head,
    feature_dim: int,
    max_state_tokens: int,
    max_action_tokens: int,
    batch_size: int,
    head_batch_size: int,
    device: str,
    progress_every: int,
) -> dict[str, Any]:
    import torch

    print(
        json.dumps(
            {
                "phase": "cache_split",
                "split": split,
                "rows": len(rows),
                "feature_dim": feature_dim,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    row_ids = [row_id(row, index, split) for index, row in enumerate(rows)]
    state_texts = [str(row.get("state_text") or "") for row in rows]
    action_texts = [action_text(row) for row in rows]
    next_state_texts = [str(row.get("next_state_text") or "") for row in rows]
    goal_texts = [goal_text(row, states) for row in rows]

    state_like_texts = state_texts + next_state_texts + goal_texts
    z_state_like, state_like_hashes = encode_unique_texts(
        label=f"{split}:state_next_goal",
        texts=state_like_texts,
        encoder=encoder,
        tokenizer=tokenizer,
        max_length=max_state_tokens,
        truncation_side="left",
        batch_size=batch_size,
        device=device,
        progress_every=progress_every,
    )
    row_count = len(rows)
    z_state = z_state_like[:row_count]
    z_next = z_state_like[row_count : 2 * row_count]
    z_goal = z_state_like[2 * row_count :]
    state_hashes = state_like_hashes[:row_count]
    next_hashes = state_like_hashes[row_count : 2 * row_count]
    goal_hashes = state_like_hashes[2 * row_count :]
    z_action, action_hashes = encode_unique_texts(
        label=f"{split}:action",
        texts=action_texts,
        encoder=encoder,
        tokenizer=tokenizer,
        max_length=max_action_tokens,
        truncation_side="right",
        batch_size=batch_size,
        device=device,
        progress_every=progress_every,
    )

    features = torch.tensor(
        [pad_features(action_features(row), feature_dim) for row in rows],
        dtype=torch.float32,
    )
    state_action_parts = []
    started = time.time()
    total_batches = (len(rows) + head_batch_size - 1) // head_batch_size
    with torch.no_grad():
        for batch_no, start in enumerate(range(0, len(rows), head_batch_size), start=1):
            end = min(start + head_batch_size, len(rows))
            pred, _ = head(
                z_state[start:end].to(device),
                z_action[start:end].to(device),
                features[start:end].to(device),
            )
            state_action_parts.append(pred.cpu())
            if progress_every > 0 and (batch_no == 1 or batch_no % progress_every == 0 or batch_no == total_batches):
                elapsed = max(1e-6, time.time() - started)
                print(
                    json.dumps(
                        {
                            "phase": "transition_head_progress",
                            "split": split,
                            "batch": batch_no,
                            "batches": total_batches,
                            "done_rows": end,
                            "rows": len(rows),
                            "rows_per_sec": round(end / elapsed, 3),
                            "elapsed_sec": round(elapsed, 1),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    return {
        "row_ids": row_ids,
        "z_state": z_state,
        "z_action": z_action,
        "z_next": z_next,
        "z_goal": z_goal,
        "z_state_action": torch.cat(state_action_parts, dim=0) if state_action_parts else torch.empty((0, 0)),
        "action_features": features,
        "text_hashes": {
            "state": state_hashes,
            "action": action_hashes,
            "next_state": next_hashes,
            "goal": goal_hashes,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache transition encoder embeddings for MQE.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--encoder-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-state-tokens", type=int, default=None)
    parser.add_argument("--max-action-tokens", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-batch-size", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    encoder, tokenizer, head, transition_config, head_blob = load_encoder_for_cache(args.encoder_dir, device, args.dtype)
    train_rows = read_jsonl(args.data_dir / "train.jsonl", args.max_train_rows)
    val_rows = read_jsonl(args.data_dir / "val.jsonl", args.max_val_rows)
    feature_dim = int(head_blob["action_feature_dim"])
    train_config = head_blob["config"]
    max_state_tokens = args.max_state_tokens or int(train_config["max_state_tokens"])
    max_action_tokens = args.max_action_tokens or int(train_config["max_action_tokens"])
    states = load_states(args.data_dir)
    cache_fn = cache_split if args.no_dedupe else cache_split_deduped
    cache_kwargs = {
        "states": states,
        "encoder": encoder,
        "tokenizer": tokenizer,
        "head": head,
        "feature_dim": feature_dim,
        "max_state_tokens": max_state_tokens,
        "max_action_tokens": max_action_tokens,
        "batch_size": args.batch_size,
        "device": device,
    }
    if args.no_dedupe:
        print(
            json.dumps({"phase": "cache_mode", "mode": "legacy_no_dedupe"}, sort_keys=True),
            flush=True,
        )
    else:
        cache_kwargs["head_batch_size"] = args.head_batch_size
        cache_kwargs["progress_every"] = args.progress_every
        print(
            json.dumps({"phase": "cache_mode", "mode": "dedupe"}, sort_keys=True),
            flush=True,
        )

    train_cache = cache_fn(
        split="train",
        rows=train_rows,
        **cache_kwargs,
    )
    val_cache = cache_fn(
        split="val",
        rows=val_rows,
        **cache_kwargs,
    )
    payload = {
        "format": "transition-cache-v1",
        "metadata": {
            "data_dir": str(args.data_dir),
            "encoder_dir": str(args.encoder_dir),
            "base_model": transition_config["base_model"],
            "embedding_dim": int(head_blob["embedding_dim"]),
            "action_feature_dim": feature_dim,
            "max_state_tokens": max_state_tokens,
            "max_action_tokens": max_action_tokens,
        },
        "splits": {
            "train": train_cache,
            "val": val_cache,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(payload, args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "embedding_dim": int(head_blob["embedding_dim"]),
                "action_feature_dim": feature_dim,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
