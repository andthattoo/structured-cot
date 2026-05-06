from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "train_looped_lora_pi.py"
SPEC = importlib.util.spec_from_file_location("train_looped_lora_pi", SCRIPT_PATH)
assert SPEC is not None
train_looped_lora_pi = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = train_looped_lora_pi
SPEC.loader.exec_module(train_looped_lora_pi)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]


def step_row() -> dict[str, object]:
    return {
        "id": "run/task/assistant_0000",
        "run_id": "run",
        "task_id": "task",
        "source": "test",
        "state_messages": [
            {"role": "user", "content": [{"type": "text", "text": "Task"}], "tool_calls": []},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hidden thought"},
                    {"type": "toolCall", "name": "read", "arguments": {"path": "README.md"}},
                ],
                "tool_calls": [{"name": "read", "arguments": {"path": "README.md"}}],
            },
            {"role": "tool", "content": [{"type": "text", "text": "# README"}], "tool_calls": []},
        ],
        "target_assistant": {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "do not train this"},
                {"type": "text", "text": "Done."},
            ],
            "tool_calls": [],
        },
        "reward_features": {"has_tool_call": False},
    }


def test_render_training_text_strips_thinking() -> None:
    prompt, target = train_looped_lora_pi.render_training_text(step_row())

    assert "hidden thought" not in prompt
    assert "do not train this" not in target
    assert 'TOOL_CALL read {"path": "README.md"}' in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    assert target == "Done.\n<|im_end|>"


def test_tokenize_example_masks_prompt_tokens() -> None:
    example = train_looped_lora_pi.tokenize_example(
        step_row(),
        TinyTokenizer(),
        max_total_tokens=None,
        max_target_tokens=None,
    )

    assert example is not None
    assert example["target_tokens"] == len("Done.\n<|im_end|>")
    assert example["labels"].count(-100) == example["prompt_tokens"]
    assert example["labels"][-1] == ord(">")


def test_lambda_schedule_decays_to_final_value() -> None:
    assert train_looped_lora_pi.lambda_for_step(0, total_steps=10, final_lambda=0.2) == 1.0
    assert train_looped_lora_pi.lambda_for_step(9, total_steps=10, final_lambda=0.2) == 0.2
    middle = train_looped_lora_pi.lambda_for_step(5, total_steps=10, final_lambda=0.2)
    assert 0.2 < middle < 1.0


def test_student_loop_values_support_sampled_and_all() -> None:
    class FakeRng:
        def randint(self, low: int, high: int) -> int:
            assert (low, high) == (1, 3)
            return 2

    sampled = train_looped_lora_pi.student_loop_values(
        SimpleNamespace(student_loop_mode="sampled", min_student_loops=1, max_loops=4),
        FakeRng(),
    )
    all_loops = train_looped_lora_pi.student_loop_values(
        SimpleNamespace(student_loop_mode="all", min_student_loops=1, max_loops=4),
        FakeRng(),
    )

    assert sampled == [2]
    assert all_loops == [1, 2, 3]
