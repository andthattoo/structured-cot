#!/usr/bin/env python3
"""Prepare AgentTrove Terminus-2 traces for MQE critic training.

The output schema matches scripts/train_mqe_critic.py:

  train.jsonl / val.jsonl:
    state_text, next_state_text, goal_prompt, goal_state_hash, goal_index,
    target_distance_steps, next_distance_steps, action, rollout_id, ...

  states.jsonl:
    state_hash, serialized_state

Each assistant JSON turn becomes one (state, action, next_state, goal, distance)
row. Distances are empirical remaining assistant action steps until the final
assistant action in that trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET = "open-thoughts/AgentTrove"
ACTION_HASH_DIM = 64
COMMAND_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:=?&%{}$'\"+\-]+")
PATH_TOKEN_RE = re.compile(r"(?:[./~]?[\w.-]+/)+[\w./-]+|[\w.-]+\.(?:py|json|ya?ml|md|txt|csv|tsv|db|sql|sh|toml|lock)")
FLAG_TOKEN_RE = re.compile(r"(?:^|\s)-{1,2}[A-Za-z0-9][A-Za-z0-9_-]*")

VERB_BUCKETS = [
    "ls",
    "cat",
    "grep",
    "rg",
    "find",
    "python",
    "pytest",
    "pip",
    "uv",
    "git",
    "curl",
    "wget",
    "mkdir",
    "cp",
    "mv",
    "rm",
    "chmod",
    "sed",
    "awk",
    "tar",
    "make",
    "npm",
    "node",
    "bash",
    "echo",
    "sqlite3",
    "psql",
    "docker",
    "unknown",
]

SCALAR_ACTION_FEATURE_NAMES = [
    "finish_true",
    "has_commands",
    "num_commands_log",
    "command_chars_log",
    "num_paths_log",
    "num_flags_log",
    "has_multiline",
    "has_shell_control",
    "reads_files",
    "writes_files",
    "runs_tests",
    "runs_python",
    "installs_deps",
    "git_op",
    "network_op",
    "destructive_op",
    "search_op",
    "list_op",
    "chmod_op",
    "mkdir_op",
    "archive_op",
    "config_op",
    "env_op",
    "database_op",
    "package_manager_op",
    "has_py_path",
    "has_json_path",
    "has_yaml_path",
    "has_md_path",
    "has_test_path",
    *[f"verb_{verb}" for verb in VERB_BUCKETS],
]
ACTION_FEATURE_NAMES = [
    *SCALAR_ACTION_FEATURE_NAMES,
    *[f"token_hash_{index:02d}" for index in range(ACTION_HASH_DIM)],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare AgentTrove rows for MQE critic training.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Local AgentTrove JSONL sample.")
    source.add_argument("--dataset", default=None, help="Hugging Face dataset id to stream.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out-dir", type=Path, default=Path("data/mqe/agenttrove"))
    parser.add_argument("--limit-rollouts", type=int, default=None)
    parser.add_argument("--limit-transitions", type=int, default=None)
    parser.add_argument("--scan-limit", type=int, default=200_000)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--source", action="append", default=[], help="Filter original_source. Can repeat.")
    parser.add_argument("--teacher", action="append", default=[], help="Filter original_teacher substring. Can repeat.")
    parser.add_argument("--min-actions", type=int, default=2)
    parser.add_argument("--max-actions", type=int, default=80)
    parser.add_argument("--max-state-chars", type=int, default=12_000)
    parser.add_argument("--max-action-chars", type=int, default=4_000)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--no-streaming", action="store_true")
    return parser.parse_args()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compact(text: Any, *, max_chars: int) -> str:
    value = "" if text is None else str(text)
    if len(value) > max_chars:
        return value[-max_chars:]
    return value


def iter_input_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.input is not None:
        with args.input.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Run with:\n"
            "  uv run --with datasets python scripts/prepare_agenttrove_mqe.py ..."
        ) from exc

    dataset = load_dataset(
        args.dataset or DEFAULT_DATASET,
        split=args.split,
        streaming=not args.no_streaming,
    )
    yield from dataset


def row_value(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def row_matches(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.source:
        source = row_value(row, "original_source", "source", "task")
        if source not in set(args.source):
            return False
    if args.teacher:
        teacher = row_value(row, "original_teacher", "teacher", "model").lower()
        if not any(item.lower() in teacher for item in args.teacher):
            return False
    return True


def get_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages") or row.get("conversations") or []
    return [message for message in messages if isinstance(message, dict)]


def role_of(message: dict[str, Any]) -> str:
    return str(message.get("role") or message.get("from") or "")


def content_of(message: dict[str, Any]) -> str:
    return str(message.get("content") if "content" in message else message.get("value") or "")


def parse_assistant_json(content: str) -> dict[str, Any] | None:
    text = content.strip()
    if not text.startswith("{"):
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    has_action = bool(value.get("commands")) or bool(value.get("task_complete"))
    if not has_action:
        return None
    return value


def action_text(action: dict[str, Any], *, max_chars: int) -> str:
    payload: dict[str, Any] = {}
    if "commands" in action:
        payload["commands"] = action.get("commands") or []
    if "task_complete" in action:
        payload["task_complete"] = action.get("task_complete")
    if "final_answer" in action:
        payload["final_answer"] = action.get("final_answer")
    if not payload:
        payload = action
    return compact(json.dumps(payload, ensure_ascii=False, sort_keys=True), max_chars=max_chars)


def action_command_texts(action: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for command in action.get("commands") or []:
        if isinstance(command, dict):
            value = (
                command.get("keystrokes")
                or command.get("command")
                or command.get("cmd")
                or command.get("raw")
                or command.get("canonical")
            )
        else:
            value = command
        if value not in (None, ""):
            values.append(str(value))
    return values


def normalize_verb(token: str) -> str:
    value = token.strip().lower().split("/")[-1]
    if value in {"sudo", "env", "command", "timeout"}:
        return ""
    if value.startswith("python"):
        return "python"
    if value in {"py.test", "pytest"}:
        return "pytest"
    if value in {"pip3"}:
        return "pip"
    if value in {"nodejs"}:
        return "node"
    if value in VERB_BUCKETS:
        return value
    return "unknown"


def action_verbs(command_texts: list[str]) -> set[str]:
    verbs: set[str] = set()
    for text in command_texts:
        for raw_line in text.splitlines()[:40]:
            line = raw_line.strip()
            if not line or line.startswith(">") or line in {"EOF", "PY", "JSON"}:
                continue
            for segment in re.split(r"\s*(?:&&|\|\||;|\|)\s*", line)[:4]:
                match = COMMAND_TOKEN_RE.search(segment)
                if not match:
                    continue
                verb = normalize_verb(match.group(0))
                if verb:
                    verbs.add(verb)
                    break
    return verbs or {"unknown"}


def log_feature(value: int | float, *, cap: int | float) -> float:
    if value <= 0:
        return 0.0
    return min(1.0, math.log1p(float(value)) / math.log1p(float(cap)))


def hashed_action_features(text: str) -> list[float]:
    values = [0.0] * ACTION_HASH_DIM
    for token in COMMAND_TOKEN_RE.findall(text.lower())[:512]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % ACTION_HASH_DIM
        sign = 1.0 if digest[4] & 1 else -1.0
        values[index] += sign
    norm = math.sqrt(sum(value * value for value in values))
    if norm > 0:
        values = [value / norm for value in values]
    return values


def extract_action_features(action: dict[str, Any], *, max_chars: int) -> list[float]:
    command_texts = action_command_texts(action)
    command_blob = compact("\n".join(command_texts) or action_text(action, max_chars=max_chars), max_chars=max_chars)
    lower = command_blob.lower()
    paths = PATH_TOKEN_RE.findall(command_blob)
    flags = FLAG_TOKEN_RE.findall(command_blob)
    verbs = action_verbs(command_texts or [command_blob])

    scalar: dict[str, float] = {
        "finish_true": 1.0 if action.get("task_complete") is True else 0.0,
        "has_commands": 1.0 if command_texts else 0.0,
        "num_commands_log": log_feature(len(command_texts), cap=8),
        "command_chars_log": log_feature(len(command_blob), cap=max_chars),
        "num_paths_log": log_feature(len(set(paths)), cap=32),
        "num_flags_log": log_feature(len(flags), cap=32),
        "has_multiline": 1.0 if "\n" in command_blob.strip() else 0.0,
        "has_shell_control": 1.0 if re.search(r"&&|\|\||[;|<>]", command_blob) else 0.0,
        "reads_files": 1.0 if re.search(r"\b(cat|head|tail|less|sed\s+-n|jq)\b", lower) else 0.0,
        "writes_files": 1.0
        if re.search(r">\s|\b(tee|touch|cat\s+>|sed\s+-i|cp|mv|mkdir|chmod|apply_patch)\b", lower)
        else 0.0,
        "runs_tests": 1.0 if re.search(r"\b(pytest|unittest|npm\s+test|cargo\s+test|go\s+test|make\s+test)\b", lower) else 0.0,
        "runs_python": 1.0 if re.search(r"\bpython[0-9.]*\b", lower) else 0.0,
        "installs_deps": 1.0 if re.search(r"\b(apt-get|apt|pip|uv|npm|pnpm|yarn)\s+(install|add|sync)\b", lower) else 0.0,
        "git_op": 1.0 if re.search(r"\bgit\b", lower) else 0.0,
        "network_op": 1.0 if re.search(r"\b(curl|wget|ssh|scp|http://|https://)\b", lower) else 0.0,
        "destructive_op": 1.0 if re.search(r"\b(rm|truncate|drop|delete|reset|clean)\b", lower) else 0.0,
        "search_op": 1.0 if re.search(r"\b(rg|grep|find|fd)\b", lower) else 0.0,
        "list_op": 1.0 if re.search(r"\b(ls|tree|find)\b", lower) else 0.0,
        "chmod_op": 1.0 if re.search(r"\bchmod\b", lower) else 0.0,
        "mkdir_op": 1.0 if re.search(r"\bmkdir\b", lower) else 0.0,
        "archive_op": 1.0 if re.search(r"\b(tar|zip|unzip|7z|gzip|gunzip)\b", lower) else 0.0,
        "config_op": 1.0 if re.search(r"\b(config|yaml|yml|toml|json|env)\b", lower) else 0.0,
        "env_op": 1.0 if re.search(r"\b(export|env|source|\.env)\b", lower) else 0.0,
        "database_op": 1.0 if re.search(r"\b(sqlite3|psql|mysql|duckdb|select|insert|update)\b", lower) else 0.0,
        "package_manager_op": 1.0 if re.search(r"\b(apt-get|pip|uv|npm|pnpm|yarn|cargo|go\s+mod)\b", lower) else 0.0,
        "has_py_path": 1.0 if re.search(r"\.py\b|tests?/", lower) else 0.0,
        "has_json_path": 1.0 if re.search(r"\.json\b", lower) else 0.0,
        "has_yaml_path": 1.0 if re.search(r"\.ya?ml\b", lower) else 0.0,
        "has_md_path": 1.0 if re.search(r"\.md\b", lower) else 0.0,
        "has_test_path": 1.0 if re.search(r"\b(test|tests|spec)\b", lower) else 0.0,
    }
    for verb in VERB_BUCKETS:
        scalar[f"verb_{verb}"] = 1.0 if verb in verbs else 0.0
    return [scalar[name] for name in SCALAR_ACTION_FEATURE_NAMES] + hashed_action_features(command_blob)


def action_feature_map(features: list[float]) -> dict[str, float]:
    return {
        name: float(features[index]) if index < len(features) else 0.0
        for index, name in enumerate(ACTION_FEATURE_NAMES)
    }


def action_type_from_features(features: list[float]) -> str:
    values = action_feature_map(features)
    if values["runs_tests"] > 0:
        return "test"
    if values["installs_deps"] > 0 or values["package_manager_op"] > 0:
        return "install"
    if values["git_op"] > 0:
        return "git"
    if values["network_op"] > 0:
        return "network"
    if values["database_op"] > 0:
        return "database"
    if values["writes_files"] > 0:
        return "write"
    if values["reads_files"] > 0 or values["search_op"] > 0 or values["list_op"] > 0:
        return "read"
    if values["finish_true"] > 0:
        return "final"
    return "act"


def short_hash(text: str, length: int = 12) -> str:
    return stable_hash(text)[:length]


def action_signature(action: dict[str, Any], *, max_chars: int, features: list[float]) -> tuple[str, str]:
    command_texts = action_command_texts(action)
    command_blob = compact("\n".join(command_texts) or action_text(action, max_chars=max_chars), max_chars=max_chars)
    paths = sorted({path.lower() for path in PATH_TOKEN_RE.findall(command_blob)})
    verbs = sorted(action_verbs(command_texts or [command_blob]))
    values = action_feature_map(features)
    action_type = action_type_from_features(features)
    broad_flags = "".join(
        flag
        for flag, name in [
            ("r", "reads_files"),
            ("w", "writes_files"),
            ("t", "runs_tests"),
            ("i", "installs_deps"),
            ("g", "git_op"),
            ("n", "network_op"),
            ("d", "destructive_op"),
            ("f", "finish_true"),
        ]
        if values[name] > 0
    ) or "none"
    verb_part = ",".join(verbs)
    path_hash = short_hash("\n".join(paths)) if paths else "none"
    arg_hash = short_hash(command_blob.lower()) if command_blob.strip() else "none"
    full = f"type={action_type}|verbs={verb_part}|flags={broad_flags}|paths={path_hash}|args={arg_hash}"
    family = f"type={action_type}|verbs={verb_part}|flags={broad_flags}"
    return full, family


def serialize_state(messages: list[dict[str, Any]], *, max_chars: int) -> str:
    parts: list[str] = []
    for message in messages:
        role = role_of(message)
        content = content_of(message)
        parts.append(f"{role.upper()}:\n{content}")
    return compact("\n\n---\n\n".join(parts), max_chars=max_chars)


def goal_prompt(row: dict[str, Any], messages: list[dict[str, Any]], *, max_chars: int) -> str:
    task = row_value(row, "instruction", "task", "prompt")
    if task:
        return compact(task, max_chars=max_chars)
    for message in messages:
        if role_of(message) == "user":
            return compact(content_of(message), max_chars=max_chars)
    return ""


def convert_rollout(row: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, str]] | None:
    messages = get_messages(row)
    assistant_turns: list[tuple[int, dict[str, Any]]] = []
    for index, message in enumerate(messages):
        if role_of(message) != "assistant":
            continue
        parsed = parse_assistant_json(content_of(message))
        if parsed is not None:
            assistant_turns.append((index, parsed))

    if not (args.min_actions <= len(assistant_turns) <= args.max_actions):
        return None

    final_message_index = assistant_turns[-1][0]
    final_state_text = serialize_state(messages[: final_message_index + 1], max_chars=args.max_state_chars)
    goal_state_hash = stable_hash(final_state_text)
    task_goal = goal_prompt(row, messages, max_chars=args.max_state_chars)
    rollout_id = row_value(row, "trial_name", "task_id", "id") or stable_hash(json.dumps(row, sort_keys=True)[:10000])

    records: list[dict[str, Any]] = []
    total_actions = len(assistant_turns)
    for action_number, (message_index, action) in enumerate(assistant_turns):
        next_index = assistant_turns[action_number + 1][0] if action_number + 1 < total_actions else message_index + 1
        state_messages = messages[:message_index]
        next_state_messages = messages[:next_index]
        state_text = serialize_state(state_messages, max_chars=args.max_state_chars)
        next_state_text = serialize_state(next_state_messages, max_chars=args.max_state_chars)
        remaining = total_actions - action_number
        next_remaining = max(0, remaining - 1)
        action_payload = action_text(action, max_chars=args.max_action_chars)
        if not action_payload.strip():
            continue
        action_features = extract_action_features(action, max_chars=args.max_action_chars)
        signature, family_signature = action_signature(
            action,
            max_chars=args.max_action_chars,
            features=action_features,
        )
        records.append(
            {
                "example_id": f"{rollout_id}:{action_number}",
                "rollout_id": rollout_id,
                "challenge_id": row_value(row, "task_id", "trial_name", "id"),
                "original_source": row_value(row, "original_source", "source", "task"),
                "original_teacher": row_value(row, "original_teacher", "teacher", "model"),
                "state_text": state_text,
                "next_state_text": next_state_text,
                "action_text": action_payload,
                "action_signature": signature,
                "action_family_signature": family_signature,
                "action_type": action_type_from_features(action_features),
                "goal_prompt": task_goal,
                "goal_policy_text": task_goal,
                "goal_state_hash": goal_state_hash,
                "goal_index": total_actions,
                "target_distance_steps": float(remaining),
                "next_distance_steps": float(next_remaining),
                "gamma": args.gamma,
                "action": {
                    "action_unit": "terminus_json",
                    "raw_json": compact(json.dumps(action, ensure_ascii=False, sort_keys=True), max_chars=args.max_action_chars),
                    "canonical_str": action_payload,
                    "commands": action.get("commands") or [],
                    "task_complete": action.get("task_complete"),
                },
                "action_features": action_features,
            }
        )

    if not records:
        return None
    state_record = {"state_hash": goal_state_hash, "serialized_state": final_state_text}
    return records, state_record


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def deterministic_pick(candidates: list[int], *, key: str, limit: int) -> list[int]:
    unique = sorted(set(candidates))
    if len(unique) <= limit:
        return unique
    rng = random.Random(int(stable_hash(key)[:15], 16))
    return sorted(rng.sample(unique, limit))


def add_transition_indices_and_negatives(rows: list[dict[str, Any]], *, max_negatives: int = 8) -> None:
    by_rollout: dict[str, list[int]] = {}
    by_family: dict[str, list[int]] = {}
    by_source: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        row["transition_index"] = index
        by_rollout.setdefault(str(row.get("rollout_id") or ""), []).append(index)
        by_family.setdefault(str(row.get("action_family_signature") or ""), []).append(index)
        source_key = "|".join(
            [
                str(row.get("original_source") or ""),
                str(row.get("challenge_id") or ""),
            ]
        )
        by_source.setdefault(source_key, []).append(index)

    for index, row in enumerate(rows):
        rollout = str(row.get("rollout_id") or "")
        family = str(row.get("action_family_signature") or "")
        source_key = "|".join(
            [
                str(row.get("original_source") or ""),
                str(row.get("challenge_id") or ""),
            ]
        )
        same_rollout = [candidate for candidate in by_rollout.get(rollout, []) if candidate != index]
        same_family = [
            candidate
            for candidate in by_family.get(family, [])
            if candidate != index and rows[candidate].get("action_text") != row.get("action_text")
        ]
        same_source = [candidate for candidate in by_source.get(source_key, []) if candidate != index]
        negative_next = [
            *deterministic_pick(same_rollout, key=f"{index}:next:rollout", limit=3),
            *deterministic_pick(same_family, key=f"{index}:next:family", limit=3),
            *deterministic_pick(same_source, key=f"{index}:next:source", limit=2),
        ]
        negative_action = [
            *deterministic_pick(same_rollout, key=f"{index}:action:rollout", limit=3),
            *deterministic_pick(same_family, key=f"{index}:action:family", limit=3),
            *deterministic_pick(same_source, key=f"{index}:action:source", limit=2),
        ]
        row["negative_next_indices"] = deterministic_pick(
            [candidate for candidate in negative_next if candidate != index],
            key=f"{index}:next:merged",
            limit=max_negatives,
        )
        row["negative_action_indices"] = deterministic_pick(
            [candidate for candidate in negative_action if candidate != index],
            key=f"{index}:action:merged",
            limit=max_negatives,
        )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rollout_limit = args.limit_rollouts if args.limit_rollouts is not None else (None if args.limit_transitions else 1000)

    rollouts: list[list[dict[str, Any]]] = []
    states: dict[str, str] = {}
    scanned = 0
    converted = 0
    transitions = 0

    for row in iter_input_rows(args):
        scanned += 1
        if row_matches(row, args):
            converted_row = convert_rollout(row, args)
            if converted_row is not None:
                records, state_record = converted_row
                rollouts.append(records)
                states.setdefault(state_record["state_hash"], state_record["serialized_state"])
                converted += 1
                transitions += len(records)
                if rollout_limit is not None and converted >= rollout_limit:
                    break
                if args.limit_transitions is not None and transitions >= args.limit_transitions:
                    break
        if scanned >= args.scan_limit:
            break

    rng.shuffle(rollouts)
    val_count = max(1, round(len(rollouts) * args.val_ratio)) if len(rollouts) > 1 else 0
    val_rollouts = rollouts[:val_count]
    train_rollouts = rollouts[val_count:]
    train_rows = [record for rollout in train_rollouts for record in rollout]
    val_rows = [record for rollout in val_rollouts for record in rollout]
    if args.limit_transitions is not None:
        overflow = max(0, len(train_rows) + len(val_rows) - args.limit_transitions)
        if overflow:
            train_rows = train_rows[:-overflow] if overflow < len(train_rows) else []
    add_transition_indices_and_negatives(train_rows)
    add_transition_indices_and_negatives(val_rows)

    write_jsonl(args.out_dir / "train.jsonl", train_rows)
    write_jsonl(args.out_dir / "val.jsonl", val_rows)
    write_jsonl(
        args.out_dir / "states.jsonl",
        (
            {"state_hash": state_hash, "serialized_state": text}
            for state_hash, text in sorted(states.items())
        ),
    )

    summary = {
        "scanned": scanned,
        "converted_rollouts": converted,
        "limit_rollouts": rollout_limit,
        "train_rollouts": len(train_rollouts),
        "val_rollouts": len(val_rollouts),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "limit_transitions": args.limit_transitions,
        "states": len(states),
        "out_dir": str(args.out_dir),
        "action_feature_dim": len(ACTION_FEATURE_NAMES),
        "action_feature_names": ACTION_FEATURE_NAMES,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
