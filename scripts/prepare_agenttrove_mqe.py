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
import random
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET = "open-thoughts/AgentTrove"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare AgentTrove rows for MQE critic training.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Local AgentTrove JSONL sample.")
    source.add_argument("--dataset", default=None, help="Hugging Face dataset id to stream.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out-dir", type=Path, default=Path("data/mqe/agenttrove"))
    parser.add_argument("--limit-rollouts", type=int, default=1000)
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
        records.append(
            {
                "example_id": f"{rollout_id}:{action_number}",
                "rollout_id": rollout_id,
                "challenge_id": row_value(row, "task_id", "trial_name", "id"),
                "original_source": row_value(row, "original_source", "source", "task"),
                "original_teacher": row_value(row, "original_teacher", "teacher", "model"),
                "state_text": state_text,
                "next_state_text": next_state_text,
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


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    rollouts: list[list[dict[str, Any]]] = []
    states: dict[str, str] = {}
    scanned = 0
    converted = 0

    for row in iter_input_rows(args):
        scanned += 1
        if row_matches(row, args):
            converted_row = convert_rollout(row, args)
            if converted_row is not None:
                records, state_record = converted_row
                rollouts.append(records)
                states.setdefault(state_record["state_hash"], state_record["serialized_state"])
                converted += 1
                if converted >= args.limit_rollouts:
                    break
        if scanned >= args.scan_limit:
            break

    rng.shuffle(rollouts)
    val_count = max(1, round(len(rollouts) * args.val_ratio)) if len(rollouts) > 1 else 0
    val_rollouts = rollouts[:val_count]
    train_rollouts = rollouts[val_count:]
    train_rows = [record for rollout in train_rollouts for record in rollout]
    val_rows = [record for rollout in val_rollouts for record in rollout]

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
        "train_rollouts": len(train_rollouts),
        "val_rollouts": len(val_rollouts),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "states": len(states),
        "out_dir": str(args.out_dir),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
