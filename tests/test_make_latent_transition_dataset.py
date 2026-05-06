from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "make_latent_transition_dataset.py"
SPEC = importlib.util.spec_from_file_location("make_latent_transition_dataset", SCRIPT_PATH)
assert SPEC is not None
make_latent_transition_dataset = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = make_latent_transition_dataset
SPEC.loader.exec_module(make_latent_transition_dataset)


class TinyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, ids: list[int], clean_up_tokenization_spaces: bool = False) -> str:
        del clean_up_tokenization_spaces
        return "".join(chr(token_id) for token_id in ids)


def step_row() -> dict[str, object]:
    return {
        "id": "run/task/assistant_0000",
        "run_id": "run",
        "task_id": "task",
        "source": "source",
        "thinking_level": "medium",
        "state_messages": [
            {"role": "user", "content": [{"type": "text", "text": "Task"}], "tool_calls": []}
        ],
        "target_assistant": {"stop_reason": "toolUse"},
        "raw_thinking": "think",
        "reward_features": {"has_tool_call": True, "tool_names": ["read"], "stop_reason": "toolUse"},
    }


def test_pair_inputs_build_positions_and_filters() -> None:
    tokenizer = TinyTokenizer()

    prefix, thinking, end, target, input_ids, raw, skip = make_latent_transition_dataset.pair_inputs(
        step_row(),
        tokenizer,
        allow_empty_thinking=False,
        min_thinking_tokens=1,
        max_prefix_tokens=None,
        max_total_tokens=None,
    )

    assert skip is None
    assert raw == "think"
    assert len(thinking) == len("think")
    assert target == thinking + end
    assert input_ids == prefix + target

    *_, skip = make_latent_transition_dataset.pair_inputs(
        step_row(),
        tokenizer,
        allow_empty_thinking=False,
        min_thinking_tokens=100,
        max_prefix_tokens=None,
        max_total_tokens=None,
    )
    assert skip == "below_min_thinking_tokens"


def test_shard_writer_writes_npz(tmp_path: Path) -> None:
    writer = make_latent_transition_dataset.ShardWriter(tmp_path, shard_size=2, compress=False)
    x = np.ones((1, 3), dtype=np.float32)
    y = np.full((1, 3), 3.0, dtype=np.float32)

    path0, row0 = writer.add(x, y)
    path1, row1 = writer.add(x + 1, y + 1)
    path2, row2 = writer.add(x + 2, y + 2)
    writer.flush()

    assert (path0, row0) == ("shards/shard_00000.npz", 0)
    assert (path1, row1) == ("shards/shard_00000.npz", 1)
    assert (path2, row2) == ("shards/shard_00001.npz", 0)
    first = np.load(tmp_path / "shards/shard_00000.npz")
    second = np.load(tmp_path / "shards/shard_00001.npz")
    assert first["x"].shape == (2, 1, 3)
    assert np.allclose(first["delta"], 2.0)
    assert second["x"].shape == (1, 1, 3)


def test_batch_token_count_and_flush_policy() -> None:
    candidate_a = make_latent_transition_dataset.PairCandidate(
        step=step_row(),
        prefix_ids=[1, 2],
        thinking_ids=[3],
        end_ids=[4],
        input_token_ids=[1, 2, 3, 4],
        raw="x",
    )
    candidate_b = make_latent_transition_dataset.PairCandidate(
        step=step_row(),
        prefix_ids=[1],
        thinking_ids=[2],
        end_ids=[3],
        input_token_ids=[1, 2, 3, 4, 5, 6],
        raw="y",
    )

    assert candidate_a.positions == [1, 3]
    assert make_latent_transition_dataset.batch_token_count([candidate_a, candidate_b]) == 12
    assert make_latent_transition_dataset.should_flush_batch(
        [candidate_a],
        candidate_b,
        batch_size=8,
        max_batch_tokens=11,
    )
    assert make_latent_transition_dataset.should_flush_batch(
        [candidate_a],
        candidate_b,
        batch_size=1,
        max_batch_tokens=None,
    )
    assert not make_latent_transition_dataset.should_flush_batch(
        [],
        candidate_b,
        batch_size=1,
        max_batch_tokens=1,
    )


def test_pair_metadata_contains_metrics() -> None:
    x = np.array([[1.0, 0.0]], dtype=np.float32)
    y = np.array([[0.0, 1.0]], dtype=np.float32)

    metadata = make_latent_transition_dataset.pair_metadata(
        step_row(),
        pair_index=0,
        shard_path="shards/shard_00000.npz",
        row_in_shard=0,
        layers=[63],
        prefix_tokens=10,
        thinking_tokens=5,
        end_tokens=2,
        total_tokens=17,
        raw_thinking_chars=5,
        x=x,
        y=y,
    )

    assert metadata["id"] == "run/task/assistant_0000"
    assert metadata["target_has_tool_call"] is True
    assert metadata["tool_names"] == ["read"]
    assert metadata["layer_metrics"][0]["layer"] == 63
    assert metadata["layer_metrics"][0]["cosine_distance"] == 1.0
