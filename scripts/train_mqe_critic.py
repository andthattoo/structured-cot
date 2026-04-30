#!/usr/bin/env python3
"""Train a lightweight MQE-style directed distance critic.

This consumes the JSONL produced by scripts/prepare_agenttrove_mqe.py. The
critic predicts a directed distance-to-goal for states and state/action pairs,
then can score candidate actions by predicted distance reduction.

The default encoder is a deterministic hashing encoder so the first pilot can
run without a large embedding model. Use --encoder-backend sentence-transformers
for Qwen embedding experiments after the data path is validated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9_./:=?&%{}$'\"+\-]+")


@dataclass(frozen=True)
class MQETrainConfig:
    data_dir: str = "data/mqe/agenttrove"
    output_dir: str = "outputs/mqe/agenttrove_v0"
    encoder_backend: str = "hashing"
    encoder_model: str = "hashing"
    embedding_dim: int = 2048
    encoder_max_length: int = 8192
    encoder_prompt: str | None = None
    encoder_dtype: str = "auto"
    encoder_attn_implementation: str | None = None
    goal_text_mode: str = "concrete"
    latent_dim: int = 256
    mrn_components: int = 8
    hidden_dim: int = 512
    batch_size: int = 512
    embed_batch_size: int = 128
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    distance_scale_mode: str = "discount"
    distance_scale: float | None = None
    distance_regression_weight: float = 0.2
    action_loss_weight: float = 1.0
    invariance_loss_weight: float = 0.1
    monotonic_loss_weight: float = 0.1
    monotonic_margin: float = 0.02
    pairwise_loss_weight: float = 0.5
    pairwise_margin: float = 0.02
    action_contrastive_loss_weight: float = 0.5
    action_contrastive_margin: float = 0.02
    action_contrastive_negatives: int = 4
    action_choice_eval_negatives: int = 8
    use_action_features: bool = True
    multistep_loss_weight: float = 0.5
    multistep_batch_size: int = 1024
    multistep_max_terms: int = 50000
    waypoint_one_step_prob: float = 0.5
    waypoint_lambda: float = 0.95
    linex_clip: float = 20.0
    triangle_loss_weight: float = 0.05
    triangle_margin: float = 0.01
    triangle_batch_size: int = 1024
    triangle_max_terms: int = 50000
    dropout: float = 0.0
    checkpoint_metric: str = "val_spearman_state"
    max_train_rows: int | None = None
    max_val_rows: int | None = None
    seed: int = 17
    device: str = "auto"
    encoder_device: str = "auto"
    cache_path: str | None = None


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def _action_text(row: dict[str, Any]) -> str:
    action = row.get("action") or {}
    return str(
        action.get("canonical_str")
        or action.get("raw_bash")
        or action.get("raw_repl_block")
        or action.get("raw_json")
        or ""
    )


def _action_features(row: dict[str, Any]) -> list[float]:
    value = row.get("action_features")
    if value is None and isinstance(row.get("action"), dict):
        value = row["action"].get("features")
    if not isinstance(value, list):
        return []

    features: list[float] = []
    for item in value:
        try:
            features.append(float(item))
        except (TypeError, ValueError):
            features.append(0.0)
    return features


def _goal_text(row: dict[str, Any], mode: str = "concrete") -> str:
    concrete = str(row.get("goal_state_text") or "").strip()
    policy = "\n".join(
        part
        for part in [
            str(row.get("goal_prompt") or "").strip(),
            str(row.get("goal_policy_text") or "").strip(),
        ]
        if part
    )
    if mode == "concrete":
        return concrete or policy
    if mode == "policy":
        return policy
    if mode == "both":
        return "\n".join(part for part in [concrete, policy] if part)
    raise ValueError("goal_text_mode must be concrete, policy, or both")


def row_texts(row: dict[str, Any], goal_text_mode: str) -> tuple[str, str, str, str]:
    return (
        str(row.get("state_text") or ""),
        str(row.get("next_state_text") or ""),
        _goal_text(row, goal_text_mode),
        _action_text(row),
    )


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def attach_goal_state_texts(rows: list[dict[str, Any]], data_dir: Path) -> int:
    states_path = data_dir / "states.jsonl"
    if not states_path.exists():
        return 0
    by_hash: dict[str, str] = {}
    with states_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            state_hash = str(record.get("state_hash") or "")
            text = str(record.get("serialized_state") or "").strip()
            if state_hash and text:
                by_hash.setdefault(state_hash, text)

    attached = 0
    for row in rows:
        if row.get("goal_state_text"):
            continue
        text = by_hash.get(str(row.get("goal_state_hash") or ""))
        if text:
            row["goal_state_text"] = text
            attached += 1
    return attached


def hashing_encode(texts: list[str], dim: int):
    import torch

    vectors = torch.zeros((len(texts), dim), dtype=torch.float32)
    for row, text in enumerate(texts):
        tokens = TOKEN_RE.findall(text.lower())
        if not tokens:
            continue
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vectors[row, index] += sign
        norm = vectors[row].norm(p=2)
        if norm > 0:
            vectors[row] /= norm
    return vectors


def sentence_transformer_encode(
    texts: list[str],
    *,
    model_name: str,
    batch_size: int,
    device: str,
    max_length: int,
    prompt: str | None,
    dtype: str,
    attn_implementation: str | None,
):
    import torch
    from sentence_transformers import SentenceTransformer

    model_kwargs: dict[str, Any] = {}
    tokenizer_kwargs: dict[str, Any] = {"padding_side": "left"}
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    if dtype != "auto":
        if not hasattr(torch, dtype):
            raise ValueError(f"Unknown torch dtype {dtype!r}")
        model_kwargs["torch_dtype"] = getattr(torch, dtype)

    try:
        model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs=model_kwargs or None,
            tokenizer_kwargs=tokenizer_kwargs,
        )
    except TypeError:
        model = SentenceTransformer(model_name, device=device)
        if hasattr(model, "tokenizer"):
            model.tokenizer.padding_side = "left"

    if max_length > 0:
        model.max_seq_length = max_length

    kwargs: dict[str, Any] = {}
    if prompt:
        kwargs["prompt"] = prompt
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_tensor=True,
        show_progress_bar=True,
        **kwargs,
    )
    return embeddings.detach().cpu().to(torch.float32)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def encode_unique_texts(rows: list[dict[str, Any]], config: MQETrainConfig):
    import torch

    text_to_index: dict[str, int] = {}
    texts: list[str] = []
    for row in rows:
        for text in row_texts(row, config.goal_text_mode):
            if text not in text_to_index:
                text_to_index[text] = len(texts)
                texts.append(text)

    cache_path = Path(config.cache_path) if config.cache_path else None
    cache_key = {
        "encoder_backend": config.encoder_backend,
        "encoder_model": config.encoder_model,
        "embedding_dim": config.embedding_dim,
        "encoder_max_length": config.encoder_max_length,
        "encoder_prompt": config.encoder_prompt,
        "encoder_dtype": config.encoder_dtype,
        "encoder_attn_implementation": config.encoder_attn_implementation,
        "goal_text_mode": config.goal_text_mode,
        "text_hashes": [stable_hash(text) for text in texts],
    }
    if cache_path and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        if cached.get("cache_key") == cache_key:
            return text_to_index, cached["embeddings"]

    if config.encoder_backend == "hashing":
        embeddings = hashing_encode(texts, config.embedding_dim)
    elif config.encoder_backend == "sentence-transformers":
        embeddings = sentence_transformer_encode(
            texts,
            model_name=config.encoder_model,
            batch_size=config.embed_batch_size,
            device=resolve_device(config.encoder_device),
            max_length=config.encoder_max_length,
            prompt=config.encoder_prompt,
            dtype=config.encoder_dtype,
            attn_implementation=config.encoder_attn_implementation,
        )
    else:
        raise ValueError("encoder_backend must be hashing or sentence-transformers")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cache_key": cache_key, "embeddings": embeddings}, cache_path)
    return text_to_index, embeddings


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 1.0
    ordered = sorted(values)
    return float(ordered[round((len(ordered) - 1) * pct)])


def with_distance_scale(config: MQETrainConfig, train_rows: list[dict[str, Any]]) -> MQETrainConfig:
    if config.distance_scale is not None:
        return config
    if config.distance_scale_mode == "discount":
        gammas = [float(row.get("gamma") or 0.98) for row in train_rows]
        gamma = sum(gammas) / max(1, len(gammas))
        if not 0 < gamma < 1:
            raise ValueError(f"discount scale needs gamma in (0,1), got {gamma}")
        scale = 1.0 / (-math.log(gamma))
    elif config.distance_scale_mode == "percentile":
        scale = max(1.0, percentile([float(row["target_distance_steps"]) for row in train_rows], 0.95))
    else:
        raise ValueError("distance_scale_mode must be discount or percentile")
    return MQETrainConfig(**{**asdict(config), "distance_scale": scale})


def build_indexed_rows(
    rows: list[dict[str, Any]],
    text_to_index: dict[str, int],
    distance_scale: float,
    goal_text_mode: str,
) -> list[dict[str, Any]]:
    indexed: list[dict[str, Any]] = []
    for row in rows:
        state, next_state, goal, action = row_texts(row, goal_text_mode)
        target = float(row["target_distance_steps"]) / distance_scale
        next_target = float(row["next_distance_steps"]) / distance_scale
        rollout_key = str(row.get("rollout_id"))
        rollout_id = int(hashlib.sha256(rollout_key.encode("utf-8")).hexdigest()[:15], 16)
        goal_ordinal = int(row.get("goal_index") or 0)
        source_ordinal = int(round(goal_ordinal - float(row["target_distance_steps"])))
        group_key = f"{rollout_key}:{row.get('goal_state_hash')}:{goal_ordinal}"
        indexed.append(
            {
                "state_idx": text_to_index[state],
                "next_state_idx": text_to_index[next_state],
                "goal_idx": text_to_index[goal],
                "action_idx": text_to_index[action],
                "group_id": int(hashlib.sha256(group_key.encode("utf-8")).hexdigest()[:15], 16),
                "rollout_id": rollout_id,
                "source_ordinal": source_ordinal,
                "goal_ordinal": goal_ordinal,
                "target": target,
                "next_target": next_target,
                "raw_target": float(row["target_distance_steps"]),
                "raw_next_target": float(row["next_distance_steps"]),
                "action_features": _action_features(row),
                "example_id": row.get("example_id"),
                "challenge_id": row.get("challenge_id"),
            }
        )
    return indexed


def action_feature_dim(rows: list[dict[str, Any]], *, enabled: bool) -> int:
    if not enabled:
        return 0
    return max((len(row.get("action_features") or []) for row in rows), default=0)


def pad_action_features(rows: list[dict[str, Any]], dim: int) -> None:
    for row in rows:
        values = [float(value) for value in (row.get("action_features") or [])[:dim]]
        if len(values) < dim:
            values.extend([0.0] * (dim - len(values)))
        row["action_features"] = values


def require_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    return torch, nn, F, DataLoader, Dataset


def make_mlp(nn, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
    layers: list[Any] = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.extend([nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.extend([nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, out_dim)])
    return nn.Sequential(*layers)


def build_model_classes():
    torch, nn, F, _, _ = require_torch()

    class DirectedQuasimetricCritic(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            latent_dim: int,
            mrn_components: int,
            dropout: float = 0.0,
            action_feature_dim: int = 0,
        ):
            super().__init__()
            if latent_dim % mrn_components != 0:
                raise ValueError("latent_dim must be divisible by mrn_components")
            self.action_feature_dim = int(action_feature_dim)
            self.psi = make_mlp(nn, input_dim, hidden_dim, latent_dim, dropout)
            self.phi = make_mlp(nn, input_dim * 2 + self.action_feature_dim, hidden_dim, latent_dim, dropout)
            self.mrn_components = mrn_components
            self.mrn_part_dim = latent_dim // mrn_components

        def qdist(self, source, goal):
            diff = source - goal
            groups = diff.reshape(*diff.shape[:-1], self.mrn_components, self.mrn_part_dim)
            asymmetric = F.relu(groups).amax(dim=-1)
            symmetric = groups.norm(p=2, dim=-1)
            return (asymmetric + symmetric).mean(dim=-1)

        def embed_state(self, state):
            return self.psi(state)

        def embed_goal(self, goal):
            return self.psi(goal)

        def embed_state_action(self, state, action, action_features=None):
            pieces = [state, action]
            if self.action_feature_dim:
                if action_features is None:
                    action_features = state.new_zeros((*state.shape[:-1], self.action_feature_dim))
                else:
                    action_features = action_features.to(device=state.device, dtype=state.dtype)
                    if action_features.shape[-1] < self.action_feature_dim:
                        pad_shape = (*action_features.shape[:-1], self.action_feature_dim - action_features.shape[-1])
                        action_features = torch.cat([action_features, state.new_zeros(pad_shape)], dim=-1)
                    elif action_features.shape[-1] > self.action_feature_dim:
                        action_features = action_features[..., : self.action_feature_dim]
                pieces.append(action_features)
            return self.phi(torch.cat(pieces, dim=-1))

        def forward(self, state, next_state, goal, action, action_features=None):
            state_z = self.embed_state(state)
            next_z = self.embed_state(next_state)
            goal_z = self.embed_goal(goal)
            state_action_z = self.embed_state_action(state, action, action_features)
            return {
                "state_distance": self.qdist(state_z, goal_z),
                "next_distance": self.qdist(next_z, goal_z),
                "action_distance": self.qdist(state_action_z, goal_z),
                # Correct MQE invariance: action embedding should land near next state.
                "invariance_distance": self.qdist(state_action_z, next_z),
            }

    return DirectedQuasimetricCritic


def collate_batch(rows: list[dict[str, Any]], embeddings, device: str):
    import torch

    def idx(key: str):
        return torch.tensor([row[key] for row in rows], dtype=torch.long)

    feature_dim = len(rows[0].get("action_features") or []) if rows else 0
    if feature_dim:
        action_features = torch.tensor(
            [row.get("action_features") or [0.0] * feature_dim for row in rows],
            dtype=torch.float32,
            device=device,
        )
    else:
        action_features = torch.empty((len(rows), 0), dtype=torch.float32, device=device)

    return {
        "state": embeddings[idx("state_idx")].to(device),
        "next_state": embeddings[idx("next_state_idx")].to(device),
        "goal": embeddings[idx("goal_idx")].to(device),
        "action": embeddings[idx("action_idx")].to(device),
        "action_features": action_features,
        "group_id": torch.tensor([row["group_id"] for row in rows], dtype=torch.long, device=device),
        "rollout_id": torch.tensor([row["rollout_id"] for row in rows], dtype=torch.long, device=device),
        "source_ordinal": torch.tensor([row["source_ordinal"] for row in rows], dtype=torch.long, device=device),
        "goal_ordinal": torch.tensor([row["goal_ordinal"] for row in rows], dtype=torch.long, device=device),
        "target": torch.tensor([row["target"] for row in rows], dtype=torch.float32, device=device),
        "next_target": torch.tensor([row["next_target"] for row in rows], dtype=torch.float32, device=device),
        "raw_target": torch.tensor([row["raw_target"] for row in rows], dtype=torch.float32, device=device),
    }


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(rankdata(xs), rankdata(ys))


def pairwise_ranking_loss(distances, targets, group_ids, margin: float):
    _, _, F, _, _ = require_torch()

    closer_than = targets[:, None] < targets[None, :]
    same_group = group_ids[:, None] == group_ids[None, :]
    mask = same_group & closer_than
    if not bool(mask.any()):
        return distances.sum() * 0.0
    violations = F.relu(distances[:, None] - distances[None, :] + margin)
    return violations[mask].mean()


def linex_loss(pred, target, clip: float):
    delta = pred - target
    if clip > 0:
        delta = delta.clamp(min=-clip, max=clip)
    return (delta.exp() - delta - 1.0).mean()


def build_multistep_terms(rows: list[dict[str, Any]], config: MQETrainConfig) -> list[tuple[int, int, int]]:
    ordinal_state_idx: dict[tuple[int, int], int] = {}
    for row in rows:
        rollout_id = int(row["rollout_id"])
        source = int(row["source_ordinal"])
        ordinal_state_idx.setdefault((rollout_id, source), int(row["state_idx"]))
        ordinal_state_idx.setdefault((rollout_id, source + 1), int(row["next_state_idx"]))

    rng = random.Random(config.seed + 211)
    terms: list[tuple[int, int, int]] = []
    for direct_index, row in enumerate(rows):
        rollout_id = int(row["rollout_id"])
        source = int(row["source_ordinal"])
        goal = int(row["goal_ordinal"])
        max_offset = goal - source
        if max_offset < 1:
            continue
        offset = 1
        if max_offset > 1 and rng.random() >= config.waypoint_one_step_prob:
            while offset < max_offset and rng.random() < config.waypoint_lambda:
                offset += 1
        waypoint_idx = ordinal_state_idx.get((rollout_id, source + offset))
        if waypoint_idx is not None:
            terms.append((direct_index, waypoint_idx, offset))
    rng.shuffle(terms)
    return terms[: config.multistep_max_terms] if config.multistep_max_terms > 0 else terms


def sample_multistep_loss(model, rows, terms, embeddings, device: str, config: MQETrainConfig, rng: random.Random):
    import torch

    if not terms or config.multistep_batch_size <= 0:
        zero = next(model.parameters()).sum() * 0.0
        return zero, {"loss_multistep": 0.0, "multistep_terms": 0.0}
    selected = rng.sample(terms, min(len(terms), config.multistep_batch_size))
    direct_rows = [rows[index] for index, _, _ in selected]
    state_idx = torch.tensor([row["state_idx"] for row in direct_rows], dtype=torch.long)
    action_idx = torch.tensor([row["action_idx"] for row in direct_rows], dtype=torch.long)
    goal_idx = torch.tensor([row["goal_idx"] for row in direct_rows], dtype=torch.long)
    waypoint_idx = torch.tensor([waypoint for _, waypoint, _ in selected], dtype=torch.long)
    offsets = torch.tensor([offset for _, _, offset in selected], dtype=torch.float32, device=device)

    states = embeddings[state_idx].to(device)
    actions = embeddings[action_idx].to(device)
    goals = embeddings[goal_idx].to(device)
    waypoints = embeddings[waypoint_idx].to(device)
    action_features = None
    if getattr(model, "action_feature_dim", 0):
        action_features = torch.tensor(
            [row.get("action_features") or [] for row in direct_rows],
            dtype=torch.float32,
            device=device,
        )
    state_action_z = model.embed_state_action(states, actions, action_features)
    waypoint_z = model.embed_state(waypoints)
    goal_z = model.embed_goal(goals)
    pred = model.qdist(state_action_z, goal_z)
    waypoint_distance = model.qdist(waypoint_z, goal_z)
    # φ(s,a) is already one step after s, so offset=1 has no extra step cost.
    remaining_step_cost = torch.clamp(offsets - 1.0, min=0.0) / float(config.distance_scale or 1.0)
    target = waypoint_distance + remaining_step_cost
    loss = linex_loss(pred, target.detach(), config.linex_clip)
    return loss, {
        "loss_multistep": float(loss.detach().cpu()),
        "multistep_terms": float(len(selected)),
        "multistep_target_mean": float(target.detach().mean().cpu()),
        "multistep_pred_mean": float(pred.detach().mean().cpu()),
    }


def build_triangle_terms(rows: list[dict[str, Any]], max_terms: int, seed: int) -> list[tuple[int, int, int]]:
    pair_to_index: dict[tuple[int, int, int], int] = {}
    by_rollout_source: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for index, row in enumerate(rows):
        rollout_id = int(row["rollout_id"])
        source = int(row["source_ordinal"])
        goal = int(row["goal_ordinal"])
        if goal <= source:
            continue
        pair_to_index.setdefault((rollout_id, source, goal), index)
        by_rollout_source.setdefault((rollout_id, source), []).append((goal, index))

    terms: list[tuple[int, int, int]] = []
    for direct_index, row in enumerate(rows):
        rollout_id = int(row["rollout_id"])
        source = int(row["source_ordinal"])
        goal = int(row["goal_ordinal"])
        if goal - source <= 1:
            continue
        for middle, source_to_middle_index in by_rollout_source.get((rollout_id, source), []):
            if middle <= source or middle >= goal:
                continue
            middle_to_goal_index = pair_to_index.get((rollout_id, middle, goal))
            if middle_to_goal_index is not None:
                terms.append((direct_index, source_to_middle_index, middle_to_goal_index))
    rng = random.Random(seed)
    rng.shuffle(terms)
    return terms[:max_terms] if max_terms > 0 else terms


def sample_triangle_loss(model, rows, terms, embeddings, device: str, sample_size: int, margin: float, rng: random.Random):
    import torch
    _, _, F, _, _ = require_torch()

    if not terms or sample_size <= 0:
        zero = next(model.parameters()).sum() * 0.0
        return zero, {"loss_triangle": 0.0, "triangle_terms": 0.0}
    selected = rng.sample(terms, min(len(terms), sample_size))
    row_indices = sorted({index for term in selected for index in term})
    local = {row_index: offset for offset, row_index in enumerate(row_indices)}
    state_idx = torch.tensor([rows[index]["state_idx"] for index in row_indices], dtype=torch.long)
    goal_idx = torch.tensor([rows[index]["goal_idx"] for index in row_indices], dtype=torch.long)
    states = embeddings[state_idx].to(device)
    goals = embeddings[goal_idx].to(device)
    state_z = model.embed_state(states)
    goal_z = model.embed_goal(goals)
    distances = model.qdist(state_z, goal_z)
    direct = torch.stack([distances[local[term[0]]] for term in selected])
    left = torch.stack([distances[local[term[1]]] for term in selected])
    right = torch.stack([distances[local[term[2]]] for term in selected])
    residual = direct - left - right - margin
    loss = F.relu(residual).mean()
    return loss, {
        "loss_triangle": float(loss.detach().cpu()),
        "triangle_terms": float(len(selected)),
        "triangle_violation_rate": float((residual > 0).to(torch.float32).mean().detach().cpu()),
    }


def compute_loss(outputs, batch, config: MQETrainConfig):
    _, _, F, _, _ = require_torch()

    loss_state = F.smooth_l1_loss(outputs["state_distance"], batch["target"])
    loss_next = F.smooth_l1_loss(outputs["next_distance"], batch["next_target"])
    loss_action = F.smooth_l1_loss(outputs["action_distance"], batch["next_target"])
    invariance = outputs["invariance_distance"].pow(2).mean()
    monotonic = F.relu(outputs["next_distance"] - outputs["state_distance"] + config.monotonic_margin).pow(2).mean()
    rank_state = pairwise_ranking_loss(outputs["state_distance"], batch["target"], batch["group_id"], config.pairwise_margin)
    rank_action = pairwise_ranking_loss(
        outputs["action_distance"], batch["next_target"], batch["group_id"], config.pairwise_margin
    )
    loss = (
        config.distance_regression_weight * (loss_state + loss_next + config.action_loss_weight * loss_action)
        + config.invariance_loss_weight * invariance
        + config.monotonic_loss_weight * monotonic
        + config.pairwise_loss_weight * (rank_state + 0.5 * rank_action)
    )
    return loss, {
        "loss_state": float(loss_state.detach().cpu()),
        "loss_next": float(loss_next.detach().cpu()),
        "loss_action": float(loss_action.detach().cpu()),
        "loss_invariance": float(invariance.detach().cpu()),
        "loss_monotonic": float(monotonic.detach().cpu()),
        "loss_pairwise_state": float(rank_state.detach().cpu()),
        "loss_pairwise_action": float(rank_action.detach().cpu()),
    }


def action_contrastive_loss(model, batch, config: MQETrainConfig):
    """Rank the real action below shuffled negative actions for the same state/goal."""
    import torch
    _, _, F, _, _ = require_torch()

    negatives = int(config.action_contrastive_negatives)
    if negatives <= 0 or batch["action"].shape[0] < 2:
        zero = next(model.parameters()).sum() * 0.0
        return zero, {"loss_action_contrastive": 0.0, "action_contrastive_acc": 0.0}

    batch_size = batch["action"].shape[0]
    state_z = model.embed_state(batch["state"])
    goal_z = model.embed_goal(batch["goal"])
    positive_z = model.embed_state_action(batch["state"], batch["action"], batch["action_features"])
    positive = model.qdist(positive_z, goal_z)

    losses = []
    accuracies = []
    for _ in range(negatives):
        perm = torch.randperm(batch_size, device=batch["action"].device)
        if batch_size > 1:
            identity = perm == torch.arange(batch_size, device=batch["action"].device)
            perm[identity] = (perm[identity] + 1) % batch_size
        negative_z = model.embed_state_action(batch["state"], batch["action"][perm], batch["action_features"][perm])
        negative = model.qdist(negative_z, goal_z)
        losses.append(F.relu(positive - negative + config.action_contrastive_margin).mean())
        accuracies.append((positive < negative).to(torch.float32).mean())

    loss = torch.stack(losses).mean()
    acc = torch.stack(accuracies).mean()
    return loss, {
        "loss_action_contrastive": float(loss.detach().cpu()),
        "action_contrastive_acc": float(acc.detach().cpu()),
    }


def evaluate(model, rows, embeddings, config: MQETrainConfig, device: str) -> dict[str, float]:
    torch, _, _, DataLoader, _ = require_torch()

    model.eval()
    loader = DataLoader(
        rows,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, embeddings, device),
    )
    losses: list[float] = []
    preds: list[float] = []
    targets: list[float] = []
    action_preds: list[float] = []
    action_targets: list[float] = []
    choice_correct = 0
    choice_total = 0
    improvement_margins: list[float] = []
    rng = random.Random(config.seed + 7331)
    all_action_row_indices = list(range(len(rows)))
    scale = float(config.distance_scale or 1.0)
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["state"], batch["next_state"], batch["goal"], batch["action"], batch["action_features"])
            loss, _ = compute_loss(outputs, batch, config)
            losses.append(float(loss.detach().cpu()))
            preds.extend((outputs["state_distance"] * scale).detach().cpu().tolist())
            targets.extend(batch["raw_target"].detach().cpu().tolist())
            action_preds.extend((outputs["action_distance"] * scale).detach().cpu().tolist())
            action_targets.extend((batch["next_target"] * scale).detach().cpu().tolist())

        if config.action_choice_eval_negatives > 0 and len(all_action_row_indices) > 1:
            state_idx = torch.tensor([row["state_idx"] for row in rows], dtype=torch.long)
            goal_idx = torch.tensor([row["goal_idx"] for row in rows], dtype=torch.long)
            pos_action_idx = torch.tensor([row["action_idx"] for row in rows], dtype=torch.long)
            states = embeddings[state_idx].to(device)
            goals = embeddings[goal_idx].to(device)
            pos_actions = embeddings[pos_action_idx].to(device)
            state_z = model.embed_state(states)
            goal_z = model.embed_goal(goals)
            state_distance = model.qdist(state_z, goal_z)
            pos_features = torch.tensor(
                [row.get("action_features") or [] for row in rows],
                dtype=torch.float32,
                device=device,
            )
            pos_distance = model.qdist(model.embed_state_action(states, pos_actions, pos_features), goal_z)
            best_negative = None
            for _ in range(config.action_choice_eval_negatives):
                neg_indices = []
                neg_features = []
                for row_index, row in enumerate(rows):
                    candidate_index = row_index
                    while candidate_index == row_index:
                        candidate_index = rng.choice(all_action_row_indices)
                    candidate = rows[candidate_index]
                    neg_indices.append(candidate["action_idx"])
                    neg_features.append(candidate.get("action_features") or [])
                neg_actions = embeddings[torch.tensor(neg_indices, dtype=torch.long)].to(device)
                neg_feature_tensor = torch.tensor(neg_features, dtype=torch.float32, device=device)
                neg_distance = model.qdist(model.embed_state_action(states, neg_actions, neg_feature_tensor), goal_z)
                best_negative = neg_distance if best_negative is None else torch.minimum(best_negative, neg_distance)
            if best_negative is not None:
                choice_correct = int((pos_distance < best_negative).sum().detach().cpu())
                choice_total = len(rows)
                pos_improvement = state_distance - pos_distance
                neg_improvement = state_distance - best_negative
                improvement_margins = (pos_improvement - neg_improvement).detach().cpu().tolist()

    mae = sum(abs(p - t) for p, t in zip(preds, targets)) / max(1, len(targets))
    action_mae = sum(abs(p - t) for p, t in zip(action_preds, action_targets)) / max(1, len(action_targets))
    spearman_state = spearman(preds, targets)
    spearman_action = spearman(action_preds, action_targets)
    choice_acc = choice_correct / choice_total if choice_total else 0.0
    mean_margin = sum(improvement_margins) / max(1, len(improvement_margins))
    return {
        "loss": sum(losses) / max(1, len(losses)),
        "mae_steps": mae,
        "action_mae_steps": action_mae,
        "spearman_state": spearman_state,
        "spearman_action": spearman_action,
        "spearman_mean": (spearman_state + spearman_action) / 2.0,
        "action_choice_acc": choice_acc,
        "action_choice_margin": mean_margin,
    }


class IndexedMQEDataset:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def checkpoint_score(row: dict[str, Any], metric: str) -> tuple[float, bool]:
    value = row.get(metric)
    if value is None and not metric.startswith("val_"):
        value = row.get(f"val_{metric}")
    if value is None:
        raise ValueError(f"checkpoint metric {metric!r} not found")
    higher = any(token in metric for token in ("spearman", "accuracy", "acc", "rate"))
    return float(value), higher


def train_mqe(config: MQETrainConfig) -> dict[str, Any]:
    torch, _, _, DataLoader, _ = require_torch()

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    data_dir = Path(config.data_dir)
    train_raw = read_jsonl(data_dir / "train.jsonl", config.max_train_rows)
    val_raw = read_jsonl(data_dir / "val.jsonl", config.max_val_rows)
    if not train_raw:
        raise ValueError(f"No train rows found in {data_dir / 'train.jsonl'}")
    if not val_raw:
        raise ValueError(f"No val rows found in {data_dir / 'val.jsonl'}")

    train_goal_texts = attach_goal_state_texts(train_raw, data_dir)
    val_goal_texts = attach_goal_state_texts(val_raw, data_dir)
    config = with_distance_scale(config, train_raw)
    text_to_index, embeddings = encode_unique_texts(train_raw + val_raw, config)
    input_dim = int(embeddings.shape[1])
    train_rows = build_indexed_rows(train_raw, text_to_index, float(config.distance_scale), config.goal_text_mode)
    val_rows = build_indexed_rows(val_raw, text_to_index, float(config.distance_scale), config.goal_text_mode)
    action_dim = action_feature_dim(train_rows + val_rows, enabled=config.use_action_features)
    pad_action_features(train_rows, action_dim)
    pad_action_features(val_rows, action_dim)
    multistep_terms = build_multistep_terms(train_rows, config)
    triangle_terms = build_triangle_terms(train_rows, config.triangle_max_terms, config.seed)

    device = resolve_device(config.device)
    model_cls = build_model_classes()
    model = model_cls(input_dim, config.hidden_dim, config.latent_dim, config.mrn_components, config.dropout, action_dim).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    dataset = IndexedMQEDataset(train_rows)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score: float | None = None
    best_path = output_dir / "best.pt"
    metrics_history: list[dict[str, Any]] = []
    multistep_rng = random.Random(config.seed + 1217)
    triangle_rng = random.Random(config.seed + 1009)

    for epoch in range(1, config.epochs + 1):
        model.train()
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_batch(batch, embeddings, device),
        )
        losses: list[float] = []
        parts_by_name: dict[str, list[float]] = {}
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["state"], batch["next_state"], batch["goal"], batch["action"], batch["action_features"])
            loss, parts = compute_loss(outputs, batch, config)
            if config.action_contrastive_loss_weight > 0:
                contrastive_loss, contrastive_parts = action_contrastive_loss(model, batch, config)
                loss = loss + config.action_contrastive_loss_weight * contrastive_loss
                parts.update(contrastive_parts)
            if config.multistep_loss_weight > 0:
                multi_loss, multi_parts = sample_multistep_loss(
                    model, train_rows, multistep_terms, embeddings, device, config, multistep_rng
                )
                loss = loss + config.multistep_loss_weight * multi_loss
                parts.update(multi_parts)
            if config.triangle_loss_weight > 0:
                tri_loss, tri_parts = sample_triangle_loss(
                    model,
                    train_rows,
                    triangle_terms,
                    embeddings,
                    device,
                    config.triangle_batch_size,
                    config.triangle_margin,
                    triangle_rng,
                )
                loss = loss + config.triangle_loss_weight * tri_loss
                parts.update(tri_parts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            for name, value in parts.items():
                parts_by_name.setdefault(name, []).append(value)

        val_metrics = evaluate(model, val_rows, embeddings, config, device)
        row = {
            "epoch": epoch,
            "train_loss": sum(losses) / max(1, len(losses)),
            **{f"train_{name}": sum(values) / max(1, len(values)) for name, values in parts_by_name.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        metrics_history.append(row)
        print(json.dumps(row, sort_keys=True))
        score, higher = checkpoint_score(row, config.checkpoint_metric)
        if best_score is None or (higher and score > best_score) or ((not higher) and score < best_score):
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "input_dim": input_dim,
                    "action_feature_dim": action_dim,
                    "model_class": "DirectedQuasimetricCritic",
                    "distance_scale": config.distance_scale,
                    "best_metric": config.checkpoint_metric,
                    "best_score": best_score,
                },
                best_path,
            )

    summary = {
        "best_checkpoint": str(best_path),
        "config": asdict(config),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "unique_texts": len(text_to_index),
        "input_dim": input_dim,
        "action_feature_dim": action_dim,
        "goal_state_texts_attached": {"train": train_goal_texts, "val": val_goal_texts},
        "multistep_terms": len(multistep_terms),
        "triangle_terms": len(triangle_terms),
        "metrics": metrics_history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args(argv: list[str] | None = None) -> MQETrainConfig:
    parser = argparse.ArgumentParser(description="Train MQE directed-distance critic.")
    parser.add_argument("--data-dir", default=MQETrainConfig.data_dir)
    parser.add_argument("--output-dir", default=MQETrainConfig.output_dir)
    parser.add_argument("--encoder-backend", choices=["hashing", "sentence-transformers"], default="hashing")
    parser.add_argument("--encoder-model", default="hashing")
    parser.add_argument("--embedding-dim", type=int, default=2048)
    parser.add_argument("--encoder-max-length", type=int, default=8192)
    parser.add_argument("--encoder-prompt", default=None)
    parser.add_argument("--encoder-dtype", default="auto")
    parser.add_argument("--encoder-attn-implementation", default=None)
    parser.add_argument("--goal-text-mode", choices=["concrete", "policy", "both"], default="concrete")
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--mrn-components", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--embed-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--distance-scale-mode", choices=["percentile", "discount"], default="discount")
    parser.add_argument("--distance-scale", type=float, default=None)
    parser.add_argument("--distance-regression-weight", type=float, default=0.2)
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument("--invariance-loss-weight", type=float, default=0.1)
    parser.add_argument("--monotonic-loss-weight", type=float, default=0.1)
    parser.add_argument("--monotonic-margin", type=float, default=0.02)
    parser.add_argument("--pairwise-loss-weight", type=float, default=0.5)
    parser.add_argument("--pairwise-margin", type=float, default=0.02)
    parser.add_argument("--action-contrastive-loss-weight", type=float, default=0.5)
    parser.add_argument("--action-contrastive-margin", type=float, default=0.02)
    parser.add_argument("--action-contrastive-negatives", type=int, default=4)
    parser.add_argument("--action-choice-eval-negatives", type=int, default=8)
    parser.add_argument("--use-action-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multistep-loss-weight", type=float, default=0.5)
    parser.add_argument("--multistep-batch-size", type=int, default=1024)
    parser.add_argument("--multistep-max-terms", type=int, default=50000)
    parser.add_argument("--waypoint-one-step-prob", type=float, default=0.5)
    parser.add_argument("--waypoint-lambda", type=float, default=0.95)
    parser.add_argument("--linex-clip", type=float, default=20.0)
    parser.add_argument("--triangle-loss-weight", type=float, default=0.05)
    parser.add_argument("--triangle-margin", type=float, default=0.01)
    parser.add_argument("--triangle-batch-size", type=int, default=1024)
    parser.add_argument("--triangle-max-terms", type=int, default=50000)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--checkpoint-metric", default="val_spearman_state")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--encoder-device", default="auto")
    parser.add_argument("--cache-path", default=None)
    return MQETrainConfig(**vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    summary = train_mqe(config)
    print(
        json.dumps(
            {
                "best_checkpoint": summary["best_checkpoint"],
                "train_rows": summary["train_rows"],
                "val_rows": summary["val_rows"],
                "unique_texts": summary["unique_texts"],
                "input_dim": summary["input_dim"],
                "action_feature_dim": summary["action_feature_dim"],
                "multistep_terms": summary["multistep_terms"],
                "triangle_terms": summary["triangle_terms"],
                "final_metrics": summary["metrics"][-1] if summary["metrics"] else {},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
