#!/usr/bin/env python3
"""Train a goal-free state/action -> next-state transition encoder.

The input data is the JSONL produced by ``prepare_agenttrove_mqe.py``. This
script fine-tunes a shared text encoder with LoRA and trains a small transition
head so ``T(E(state), E(action), action_features)`` retrieves ``E(next_state)``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ACTION_TYPES = ["act", "read", "write", "test", "install", "git", "network", "database", "final"]


@dataclass(frozen=True)
class TransitionTrainConfig:
    data_dir: str = "data/mqe/agenttrove_50k_features"
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    output_dir: str = "outputs/transition_encoder/qwen06b_agenttrove_v0"
    max_state_tokens: int = 2048
    max_action_tokens: int = 512
    batch_size: int = 8
    eval_batch_size: int = 16
    grad_accum: int = 4
    epochs: int = 1
    lr: float = 2e-4
    weight_decay: float = 0.01
    temperature: float = 0.05
    transition_hidden_dim: int = 1024
    max_hard_negatives: int = 4
    hard_next_weight: float = 1.0
    hard_action_weight: float = 0.25
    hard_action_margin: float = 0.1
    aux_action_type_weight: float = 0.05
    gradient_checkpointing: bool = True
    target_next_grad: bool = False
    hard_negative_grad: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    bf16: bool = False
    fp16: bool = False
    seed: int = 17
    device: str = "auto"
    max_train_rows: int | None = None
    max_val_rows: int | None = None
    max_steps: int | None = None
    eval_every: int | None = None
    save_every: int | None = None
    keep_checkpoints: int = 3
    save_optimizer: bool = False
    log_every: int = 10


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def action_text(row: dict[str, Any]) -> str:
    if row.get("action_text"):
        return str(row["action_text"])
    action = row.get("action") or {}
    return str(action.get("canonical_str") or action.get("raw_json") or "")


def action_features(row: dict[str, Any]) -> list[float]:
    value = row.get("action_features") or []
    if not isinstance(value, list):
        return []
    out: list[float] = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def action_type_index(row: dict[str, Any]) -> int:
    value = str(row.get("action_type") or "act")
    if value not in ACTION_TYPES:
        value = "act"
    return ACTION_TYPES.index(value)


def action_feature_dim(rows: list[dict[str, Any]]) -> int:
    return max((len(action_features(row)) for row in rows), default=0)


def pad_features(values: list[float], dim: int) -> list[float]:
    values = values[:dim]
    if len(values) < dim:
        values = [*values, *([0.0] * (dim - len(values)))]
    return values


def row_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("example_id") or f"row:{index}")


class TransitionRows:
    def __init__(self, rows: list[dict[str, Any]], feature_dim: int):
        self.rows = rows
        self.feature_dim = feature_dim

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "index": index,
            "row_id": row_id(row, index),
            "state_text": str(row.get("state_text") or ""),
            "action_text": action_text(row),
            "next_state_text": str(row.get("next_state_text") or ""),
            "action_features": pad_features(action_features(row), self.feature_dim),
            "action_type": action_type_index(row),
            "negative_next_indices": [int(i) for i in row.get("negative_next_indices") or []],
            "negative_action_indices": [int(i) for i in row.get("negative_action_indices") or []],
        }


class TransitionCollator:
    def __init__(self, all_rows: list[dict[str, Any]], feature_dim: int, max_hard_negatives: int):
        self.all_rows = all_rows
        self.feature_dim = feature_dim
        self.max_hard_negatives = max_hard_negatives

    def negative_category(self, source_row: dict[str, Any], neg_row: dict[str, Any]) -> str:
        if neg_row.get("rollout_id") == source_row.get("rollout_id"):
            return "same_rollout"
        if neg_row.get("action_family_signature") == source_row.get("action_family_signature"):
            return "same_signature"
        return "other"

    def select_balanced_negatives(self, source_row: dict[str, Any], indices: list[int]) -> list[tuple[int, str]]:
        buckets = {"same_rollout": [], "same_signature": [], "other": []}
        for neg_index in indices:
            if not (0 <= neg_index < len(self.all_rows)):
                continue
            category = self.negative_category(source_row, self.all_rows[neg_index])
            buckets[category].append(neg_index)

        selected: list[tuple[int, str]] = []
        seen: set[int] = set()
        order = ["same_signature", "same_rollout", "other"]
        while len(selected) < self.max_hard_negatives:
            added = False
            for category in order:
                while buckets[category] and buckets[category][0] in seen:
                    buckets[category].pop(0)
                if buckets[category]:
                    neg_index = buckets[category].pop(0)
                    selected.append((neg_index, category))
                    seen.add(neg_index)
                    added = True
                    if len(selected) >= self.max_hard_negatives:
                        break
            if not added:
                break
        return selected

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        hard_next_texts: list[str] = []
        hard_next_rows: list[int] = []
        hard_next_categories: list[str] = []
        hard_action_texts: list[str] = []
        hard_action_features: list[list[float]] = []
        hard_action_rows: list[int] = []

        for batch_row, item in enumerate(items):
            source_row = self.all_rows[item["index"]]
            for neg_index, category in self.select_balanced_negatives(source_row, item["negative_next_indices"]):
                neg = self.all_rows[neg_index]
                hard_next_texts.append(str(neg.get("next_state_text") or ""))
                hard_next_rows.append(batch_row)
                hard_next_categories.append(category)
            for neg_index, _ in self.select_balanced_negatives(source_row, item["negative_action_indices"]):
                neg = self.all_rows[neg_index]
                hard_action_texts.append(action_text(neg))
                hard_action_features.append(pad_features(action_features(neg), self.feature_dim))
                hard_action_rows.append(batch_row)

        return {
            "indices": torch.tensor([item["index"] for item in items], dtype=torch.long),
            "row_ids": [item["row_id"] for item in items],
            "state_texts": [item["state_text"] for item in items],
            "action_texts": [item["action_text"] for item in items],
            "next_state_texts": [item["next_state_text"] for item in items],
            "action_features": torch.tensor([item["action_features"] for item in items], dtype=torch.float32),
            "action_types": torch.tensor([item["action_type"] for item in items], dtype=torch.long),
            "hard_next_texts": hard_next_texts,
            "hard_next_rows": torch.tensor(hard_next_rows, dtype=torch.long),
            "hard_next_categories": hard_next_categories,
            "hard_action_texts": hard_action_texts,
            "hard_action_features": torch.tensor(hard_action_features, dtype=torch.float32)
            if hard_action_features
            else torch.empty((0, self.feature_dim), dtype=torch.float32),
            "hard_action_rows": torch.tensor(hard_action_rows, dtype=torch.long),
        }


def require_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    return torch, nn, F, DataLoader


def build_transition_head_class():
    torch, nn, F, _ = require_torch()

    class TransitionHead(nn.Module):
        def __init__(self, embedding_dim: int, action_feature_dim: int, hidden_dim: int, n_action_types: int):
            super().__init__()
            self.embedding_dim = int(embedding_dim)
            self.action_feature_dim = int(action_feature_dim)
            self.transition = nn.Sequential(
                nn.Linear(embedding_dim * 2 + action_feature_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, embedding_dim),
            )
            self.action_type_head = nn.Linear(embedding_dim, n_action_types)

        def forward(self, state_z, action_z, action_feature_tensor):
            if self.action_feature_dim:
                if action_feature_tensor.shape[-1] < self.action_feature_dim:
                    pad = state_z.new_zeros((*action_feature_tensor.shape[:-1], self.action_feature_dim - action_feature_tensor.shape[-1]))
                    action_feature_tensor = torch.cat([action_feature_tensor, pad], dim=-1)
                elif action_feature_tensor.shape[-1] > self.action_feature_dim:
                    action_feature_tensor = action_feature_tensor[..., : self.action_feature_dim]
                inputs = torch.cat([state_z, action_z, action_feature_tensor.to(state_z.dtype)], dim=-1)
            else:
                inputs = torch.cat([state_z, action_z], dim=-1)
            pred = F.normalize(self.transition(inputs), p=2, dim=-1)
            return pred, self.action_type_head(pred)

    return TransitionHead


def last_token_pool(hidden_states, attention_mask):
    import torch

    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).unsqueeze(0)
    last_indices = (positions * attention_mask).argmax(dim=1)
    return hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), last_indices]


def tokenize(tokenizer, texts: list[str], *, max_length: int, device: str, truncation_side: str):
    original = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = truncation_side
    try:
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    finally:
        tokenizer.truncation_side = original
    return {key: value.to(device) for key, value in batch.items()}


def encode_texts(encoder, tokenizer, texts: list[str], *, max_length: int, device: str, truncation_side: str):
    import torch.nn.functional as F

    if not texts:
        return None
    batch = tokenize(tokenizer, texts, max_length=max_length, device=device, truncation_side=truncation_side)
    output = encoder(**batch)
    pooled = last_token_pool(output.last_hidden_state, batch["attention_mask"])
    return F.normalize(pooled.float(), p=2, dim=-1)


def load_lora_encoder(config: TransitionTrainConfig, device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: peft. Run with:\n"
            "  uv run --with torch --with transformers --with peft --with accelerate python scripts/train_transition_encoder.py ..."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    dtype = torch.float32
    if config.bf16:
        dtype = torch.bfloat16
    elif config.fp16:
        dtype = torch.float16
    encoder = AutoModel.from_pretrained(config.model, torch_dtype=dtype, trust_remote_code=True).to(device)
    encoder.config.pad_token_id = tokenizer.pad_token_id
    if config.gradient_checkpointing:
        if hasattr(encoder.config, "use_cache"):
            encoder.config.use_cache = False
        if hasattr(encoder, "gradient_checkpointing_enable"):
            encoder.gradient_checkpointing_enable()
        if hasattr(encoder, "enable_input_require_grads"):
            encoder.enable_input_require_grads()
    if config.lora_r > 0:
        lora = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=[item.strip() for item in config.lora_targets.split(",") if item.strip()],
            bias="none",
        )
        encoder = get_peft_model(encoder, lora)
        encoder.print_trainable_parameters()
    return encoder, tokenizer


def retrieval_loss(pred, next_z, hard_next_z, hard_next_rows, temperature: float):
    torch, _, F, _ = require_torch()

    logits = pred @ next_z.T / temperature
    labels = torch.arange(pred.shape[0], device=pred.device)
    if hard_next_z is not None and hard_next_z.shape[0] > 0:
        hard_logits = pred[hard_next_rows] * hard_next_z
        hard_scores = hard_logits.sum(dim=-1) / temperature
        hard_matrix = pred.new_full((pred.shape[0], max(1, int((hard_next_rows == 0).sum().item()))), -1e4)
        offsets = [0] * pred.shape[0]
        for score, row_index in zip(hard_scores, hard_next_rows.tolist()):
            col = offsets[row_index]
            if col >= hard_matrix.shape[1]:
                extra = pred.new_full((pred.shape[0], 1), -1e4)
                hard_matrix = torch.cat([hard_matrix, extra], dim=1)
            hard_matrix[row_index, col] = score
            offsets[row_index] += 1
        logits = torch.cat([logits, hard_matrix], dim=1)
    return F.cross_entropy(logits, labels), logits[:, : pred.shape[0]]


def hard_action_loss(pred, state_z, next_z, hard_action_z, hard_action_features, hard_action_rows, head, config: TransitionTrainConfig):
    torch, _, F, _ = require_torch()

    if hard_action_z is None or hard_action_z.shape[0] == 0:
        return pred.sum() * 0.0
    hard_pred, _ = head(state_z[hard_action_rows], hard_action_z, hard_action_features.to(state_z.device))
    positive = (pred[hard_action_rows] * next_z[hard_action_rows]).sum(dim=-1)
    negative = (hard_pred * next_z[hard_action_rows]).sum(dim=-1)
    return F.relu(negative - positive + config.hard_action_margin).mean()


def evaluate(encoder, tokenizer, head, rows: list[dict[str, Any]], config: TransitionTrainConfig, feature_dim: int, device: str) -> dict[str, float]:
    torch, _, F, DataLoader = require_torch()

    dataset = TransitionRows(rows, feature_dim)
    loader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=TransitionCollator(rows, feature_dim, config.max_hard_negatives),
    )
    encoder.eval()
    head.eval()
    total = 0
    acc1 = 0
    acc5 = 0
    aux_correct = 0
    margins: list[float] = []
    hard_total = {"same_rollout": 0, "same_signature": 0}
    hard_correct = {"same_rollout": 0, "same_signature": 0}
    with torch.no_grad():
        for batch in loader:
            state_z = encode_texts(
                encoder,
                tokenizer,
                batch["state_texts"],
                max_length=config.max_state_tokens,
                device=device,
                truncation_side="left",
            )
            action_z = encode_texts(
                encoder,
                tokenizer,
                batch["action_texts"],
                max_length=config.max_action_tokens,
                device=device,
                truncation_side="right",
            )
            next_z = encode_texts(
                encoder,
                tokenizer,
                batch["next_state_texts"],
                max_length=config.max_state_tokens,
                device=device,
                truncation_side="left",
            )
            features = batch["action_features"].to(device)
            labels = batch["action_types"].to(device)
            pred, aux_logits = head(state_z, action_z, features)
            logits = pred @ next_z.T
            ranks = logits.argsort(dim=1, descending=True)
            target = torch.arange(logits.shape[0], device=device)
            total += logits.shape[0]
            acc1 += int((ranks[:, 0] == target).sum().detach().cpu())
            acc5 += int((ranks[:, : min(5, logits.shape[1])] == target[:, None]).any(dim=1).sum().detach().cpu())
            aux_correct += int((aux_logits.argmax(dim=-1) == labels).sum().detach().cpu())
            pos = logits.diag()
            masked = logits.masked_fill(torch.eye(logits.shape[0], device=device, dtype=torch.bool), -1e9)
            margins.extend((pos - masked.max(dim=1).values).detach().cpu().tolist())
            hard_next_z = encode_texts(
                encoder,
                tokenizer,
                batch["hard_next_texts"],
                max_length=config.max_state_tokens,
                device=device,
                truncation_side="left",
            )
            if hard_next_z is not None and hard_next_z.shape[0] > 0:
                rows_for_hard = batch["hard_next_rows"].to(device)
                hard_scores = (pred[rows_for_hard] * hard_next_z).sum(dim=-1)
                pos_scores = pos[rows_for_hard]
                for ok, category in zip((pos_scores > hard_scores).detach().cpu().tolist(), batch["hard_next_categories"]):
                    if category in hard_total:
                        hard_total[category] += 1
                        hard_correct[category] += int(bool(ok))

    return {
        "next_retrieval_acc@1": acc1 / max(1, total),
        "next_retrieval_acc@5": acc5 / max(1, total),
        "action_type_acc": aux_correct / max(1, total),
        "mean_positive_margin": sum(margins) / max(1, len(margins)),
        "same_rollout_hard_negative_acc": hard_correct["same_rollout"] / max(1, hard_total["same_rollout"]),
        "same_signature_hard_negative_acc": hard_correct["same_signature"] / max(1, hard_total["same_signature"]),
        "same_rollout_hard_negative_count": float(hard_total["same_rollout"]),
        "same_signature_hard_negative_count": float(hard_total["same_signature"]),
    }


def averaged(values: dict[str, list[float]]) -> dict[str, float]:
    return {key: sum(items) / max(1, len(items)) for key, items in values.items()}


def checkpoint_sort_key(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[-1])
    except ValueError:
        return -1


def prune_checkpoints(output_dir: Path, keep: int) -> None:
    if keep <= 0:
        return
    checkpoints = sorted(output_dir.glob("checkpoint-step-*"), key=checkpoint_sort_key)
    for path in checkpoints[:-keep]:
        shutil.rmtree(path, ignore_errors=True)


def write_transition_artifacts(
    *,
    target_dir: Path,
    tokenizer,
    encoder,
    head,
    optimizer,
    config: TransitionTrainConfig,
    embedding_dim: int,
    feature_dim: int,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    global_step: int,
) -> None:
    import torch

    target_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(target_dir)
    if config.lora_r > 0:
        encoder.save_pretrained(target_dir / "encoder_adapter")
    else:
        encoder.save_pretrained(target_dir)
    torch.save(
        {
            "transition_head_state_dict": head.state_dict(),
            "embedding_dim": embedding_dim,
            "action_feature_dim": feature_dim,
            "action_types": ACTION_TYPES,
            "config": asdict(config),
            "metrics": metrics,
            "global_step": global_step,
        },
        target_dir / "transition_head.pt",
    )
    (target_dir / "transition_config.json").write_text(
        json.dumps(
            {
                "base_model": config.model,
                "embedding_dim": embedding_dim,
                "action_feature_dim": feature_dim,
                "action_types": ACTION_TYPES,
                "config": asdict(config),
                "global_step": global_step,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    summary = {
        "output_dir": str(target_dir),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "embedding_dim": embedding_dim,
        "action_feature_dim": feature_dim,
        "global_step": global_step,
        "metrics": metrics,
    }
    (target_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if config.save_optimizer:
        torch.save(
            {
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "config": asdict(config),
            },
            target_dir / "trainer_state.pt",
        )


def append_eval_row(
    *,
    encoder,
    tokenizer,
    head,
    val_rows: list[dict[str, Any]],
    config: TransitionTrainConfig,
    feature_dim: int,
    device: str,
    epoch: int,
    global_step: int,
    running: dict[str, list[float]],
    metrics: list[dict[str, Any]],
    phase: str,
) -> dict[str, Any]:
    val_metrics = evaluate(encoder, tokenizer, head, val_rows, config, feature_dim, device)
    row = {
        "epoch": epoch,
        "step": global_step,
        "phase": phase,
        **{f"train_{key}": value for key, value in averaged(running).items()},
        **{f"val_{key}": value for key, value in val_metrics.items()},
    }
    metrics.append(row)
    print(json.dumps(row, sort_keys=True), flush=True)
    encoder.train()
    head.train()
    return row


def train_transition_encoder(config: TransitionTrainConfig) -> dict[str, Any]:
    torch, _, F, DataLoader = require_torch()

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    data_dir = Path(config.data_dir)
    train_rows = read_jsonl(data_dir / "train.jsonl", config.max_train_rows)
    val_rows = read_jsonl(data_dir / "val.jsonl", config.max_val_rows)
    if not train_rows:
        raise ValueError(f"No train rows found in {data_dir / 'train.jsonl'}")
    if not val_rows:
        raise ValueError(f"No val rows found in {data_dir / 'val.jsonl'}")
    feature_dim = action_feature_dim(train_rows + val_rows)
    device = resolve_device(config.device)
    encoder, tokenizer = load_lora_encoder(config, device)
    TransitionHead = build_transition_head_class()
    embedding_dim = int(getattr(encoder.config, "hidden_size"))
    head = TransitionHead(embedding_dim, feature_dim, config.transition_hidden_dim, len(ACTION_TYPES)).to(device)
    trainable = [param for param in encoder.parameters() if param.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=config.weight_decay)
    dataset = TransitionRows(train_rows, feature_dim)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=TransitionCollator(train_rows, feature_dim, config.max_hard_negatives),
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, Any]] = []
    global_step = 0
    stop_training = False
    last_eval_step: int | None = None
    last_save_step: int | None = None

    for epoch in range(1, config.epochs + 1):
        encoder.train()
        head.train()
        running: dict[str, list[float]] = {}
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, start=1):
            if config.max_steps is not None and global_step >= config.max_steps:
                stop_training = True
                break
            state_z = encode_texts(
                encoder,
                tokenizer,
                batch["state_texts"],
                max_length=config.max_state_tokens,
                device=device,
                truncation_side="left",
            )
            action_z = encode_texts(
                encoder,
                tokenizer,
                batch["action_texts"],
                max_length=config.max_action_tokens,
                device=device,
                truncation_side="right",
            )
            next_z = encode_texts(
                encoder,
                tokenizer,
                batch["next_state_texts"],
                max_length=config.max_state_tokens,
                device=device,
                truncation_side="left",
            ) if config.target_next_grad else None
            if next_z is None:
                with torch.no_grad():
                    next_z = encode_texts(
                        encoder,
                        tokenizer,
                        batch["next_state_texts"],
                        max_length=config.max_state_tokens,
                        device=device,
                        truncation_side="left",
                    )
            features = batch["action_features"].to(device)
            labels = batch["action_types"].to(device)
            pred, aux_logits = head(state_z, action_z, features)
            if config.hard_negative_grad:
                hard_next_z = encode_texts(
                    encoder,
                    tokenizer,
                    batch["hard_next_texts"],
                    max_length=config.max_state_tokens,
                    device=device,
                    truncation_side="left",
                )
            else:
                with torch.no_grad():
                    hard_next_z = encode_texts(
                        encoder,
                        tokenizer,
                        batch["hard_next_texts"],
                        max_length=config.max_state_tokens,
                        device=device,
                        truncation_side="left",
                    )
            hard_next_rows = batch["hard_next_rows"].to(device)
            loss_retrieval, inbatch_logits = retrieval_loss(pred, next_z, hard_next_z, hard_next_rows, config.temperature)
            if config.hard_negative_grad:
                hard_action_z = encode_texts(
                    encoder,
                    tokenizer,
                    batch["hard_action_texts"],
                    max_length=config.max_action_tokens,
                    device=device,
                    truncation_side="right",
                )
            else:
                with torch.no_grad():
                    hard_action_z = encode_texts(
                        encoder,
                        tokenizer,
                        batch["hard_action_texts"],
                        max_length=config.max_action_tokens,
                        device=device,
                        truncation_side="right",
                    )
            hard_action_rows = batch["hard_action_rows"].to(device)
            hard_action_features = batch["hard_action_features"].to(device)
            loss_hard_action = hard_action_loss(
                pred,
                state_z,
                next_z,
                hard_action_z,
                hard_action_features,
                hard_action_rows,
                head,
                config,
            )
            loss_aux = F.cross_entropy(aux_logits, labels)
            loss = (
                loss_retrieval
                + config.hard_action_weight * loss_hard_action
                + config.aux_action_type_weight * loss_aux
            ) / max(1, config.grad_accum)
            loss.backward()
            if step % config.grad_accum == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            global_step += 1
            with torch.no_grad():
                target = torch.arange(inbatch_logits.shape[0], device=device)
                acc1 = (inbatch_logits.argmax(dim=-1) == target).to(torch.float32).mean()
            for name, value in {
                "loss": float((loss * max(1, config.grad_accum)).detach().cpu()),
                "loss_retrieval": float(loss_retrieval.detach().cpu()),
                "loss_hard_action": float(loss_hard_action.detach().cpu()),
                "loss_aux_action_type": float(loss_aux.detach().cpu()),
                "train_next_retrieval_acc@1": float(acc1.detach().cpu()),
            }.items():
                running.setdefault(name, []).append(value)
            if config.log_every > 0 and global_step % config.log_every == 0:
                print(json.dumps({"epoch": epoch, "step": global_step, **averaged(running)}, sort_keys=True), flush=True)
            if (
                config.eval_every is not None
                and config.eval_every > 0
                and global_step % config.eval_every == 0
                and last_eval_step != global_step
            ):
                append_eval_row(
                    encoder=encoder,
                    tokenizer=tokenizer,
                    head=head,
                    val_rows=val_rows,
                    config=config,
                    feature_dim=feature_dim,
                    device=device,
                    epoch=epoch,
                    global_step=global_step,
                    running=running,
                    metrics=metrics,
                    phase="periodic_eval",
                )
                last_eval_step = global_step
            if (
                config.save_every is not None
                and config.save_every > 0
                and global_step % config.save_every == 0
                and last_save_step != global_step
            ):
                checkpoint_dir = output_dir / f"checkpoint-step-{global_step}"
                write_transition_artifacts(
                    target_dir=checkpoint_dir,
                    tokenizer=tokenizer,
                    encoder=encoder,
                    head=head,
                    optimizer=optimizer,
                    config=config,
                    embedding_dim=embedding_dim,
                    feature_dim=feature_dim,
                    train_rows=train_rows,
                    val_rows=val_rows,
                    metrics=metrics,
                    global_step=global_step,
                )
                prune_checkpoints(output_dir, config.keep_checkpoints)
                last_save_step = global_step
                print(
                    json.dumps(
                        {"checkpoint": str(checkpoint_dir), "phase": "save_checkpoint", "step": global_step},
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if config.max_steps is not None and global_step >= config.max_steps:
                stop_training = True
                break

        if last_eval_step != global_step:
            append_eval_row(
                encoder=encoder,
                tokenizer=tokenizer,
                head=head,
                val_rows=val_rows,
                config=config,
                feature_dim=feature_dim,
                device=device,
                epoch=epoch,
                global_step=global_step,
                running=running,
                metrics=metrics,
                phase="epoch_end",
            )
            last_eval_step = global_step
        if stop_training:
            break

    write_transition_artifacts(
        target_dir=output_dir,
        tokenizer=tokenizer,
        encoder=encoder,
        head=head,
        optimizer=optimizer,
        config=config,
        embedding_dim=embedding_dim,
        feature_dim=feature_dim,
        train_rows=train_rows,
        val_rows=val_rows,
        metrics=metrics,
        global_step=global_step,
    )
    summary = {
        "output_dir": str(output_dir),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "embedding_dim": embedding_dim,
        "action_feature_dim": feature_dim,
        "global_step": global_step,
        "metrics": metrics,
    }
    return summary


def parse_args(argv: list[str] | None = None) -> TransitionTrainConfig:
    parser = argparse.ArgumentParser(description="Train a transition-aware encoder for MQE.")
    parser.add_argument("--data-dir", default=TransitionTrainConfig.data_dir)
    parser.add_argument("--model", default=TransitionTrainConfig.model)
    parser.add_argument("--output-dir", default=TransitionTrainConfig.output_dir)
    parser.add_argument("--max-state-tokens", type=int, default=TransitionTrainConfig.max_state_tokens)
    parser.add_argument("--max-action-tokens", type=int, default=TransitionTrainConfig.max_action_tokens)
    parser.add_argument("--batch-size", type=int, default=TransitionTrainConfig.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=TransitionTrainConfig.eval_batch_size)
    parser.add_argument("--grad-accum", type=int, default=TransitionTrainConfig.grad_accum)
    parser.add_argument("--epochs", type=int, default=TransitionTrainConfig.epochs)
    parser.add_argument("--lr", type=float, default=TransitionTrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TransitionTrainConfig.weight_decay)
    parser.add_argument("--temperature", type=float, default=TransitionTrainConfig.temperature)
    parser.add_argument("--transition-hidden-dim", type=int, default=TransitionTrainConfig.transition_hidden_dim)
    parser.add_argument("--max-hard-negatives", type=int, default=TransitionTrainConfig.max_hard_negatives)
    parser.add_argument("--hard-action-weight", type=float, default=TransitionTrainConfig.hard_action_weight)
    parser.add_argument("--hard-action-margin", type=float, default=TransitionTrainConfig.hard_action_margin)
    parser.add_argument("--aux-action-type-weight", type=float, default=TransitionTrainConfig.aux_action_type_weight)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-next-grad", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hard-negative-grad", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lora-r", type=int, default=TransitionTrainConfig.lora_r)
    parser.add_argument("--lora-alpha", type=int, default=TransitionTrainConfig.lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=TransitionTrainConfig.lora_dropout)
    parser.add_argument("--lora-targets", default=TransitionTrainConfig.lora_targets)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=TransitionTrainConfig.seed)
    parser.add_argument("--device", default=TransitionTrainConfig.device)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--keep-checkpoints", type=int, default=TransitionTrainConfig.keep_checkpoints)
    parser.add_argument("--save-optimizer", action="store_true")
    parser.add_argument("--log-every", type=int, default=TransitionTrainConfig.log_every)
    return TransitionTrainConfig(**vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> int:
    summary = train_transition_encoder(parse_args(argv))
    print(
        json.dumps(
            {
                "output_dir": summary["output_dir"],
                "train_rows": summary["train_rows"],
                "val_rows": summary["val_rows"],
                "embedding_dim": summary["embedding_dim"],
                "action_feature_dim": summary["action_feature_dim"],
                "final_metrics": summary["metrics"][-1] if summary["metrics"] else {},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
