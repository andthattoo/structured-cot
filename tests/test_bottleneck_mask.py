import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_dsl_sft_lora.py"
SPEC = importlib.util.spec_from_file_location("train_dsl_sft_lora", SCRIPT)
trainer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = trainer
SPEC.loader.exec_module(trainer)


def test_block_mask_blocks_response_attention_to_teacher_only():
    # Positions:
    # 0 context, 1-2 teacher_think, 3 DSL, 4 response/action, 5 padding.
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.long)
    teacher_mask = torch.tensor([[False, True, True, False, False, False]])
    response_mask = torch.tensor([[False, False, False, False, True, False]])

    mask = trainer.make_block_causal_mask(
        attention_mask,
        teacher_mask,
        response_mask,
        dtype=torch.float32,
    )

    assert mask.shape == (1, 1, 6, 6)
    assert mask[0, 0, 3, 1] == 0  # DSL can attend to teacher_think.
    assert mask[0, 0, 4, 1] < -1e20  # Response cannot attend teacher_think.
    assert mask[0, 0, 4, 2] < -1e20
    assert mask[0, 0, 4, 3] == 0  # Response can attend DSL.
    assert mask[0, 0, 1, 4] < -1e20  # Causal mask still blocks future tokens.
    assert mask[0, 0, 4, 5] < -1e20  # Padding remains masked.


def test_bottleneck_masked_loader_builds_single_teacher_dsl_action_sequence(tmp_path):
    dsl = (
        "<think>\n"
        "PLAN: seq(observe,act,verify,finish)\n"
        "STATE: need_action\n"
        "RISK: none\n"
        "NEXT: tool_call\n"
        "</think>"
    )
    action = '<tool_call>\n{"name":"terminal","arguments":{"command":"ls"}}\n</tool_call>'
    row = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "list files"},
            {"role": "assistant", "content": dsl + "\n" + action},
        ],
        "dsl_stats": {"labels": [{"_original_think": "I should inspect the directory."}]},
    }
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(row) + "\n")

    examples = trainer.load_turn_examples(
        path,
        context_messages=12,
        limit_examples=None,
        objective="bottleneck_masked",
    )

    assert len(examples) == 1
    example = examples[0]
    assert example["kind"] == "bottleneck_masked"
    text = example["text"]
    teacher_start, teacher_end = example["teacher_span"]
    target_start, target_end = example["target_span"]
    response_start, response_end = example["response_span"]

    assert text[teacher_start:teacher_end].startswith("<teacher_think>\n")
    assert "I should inspect the directory." in text[teacher_start:teacher_end]
    assert text[target_start:target_end].startswith("<think>\n")
    assert "<tool_call>" in text[target_start:target_end]
    assert text[response_start:response_end].startswith("<tool_call>")
    assert text[response_start:response_end].endswith("<|im_end|>")


def test_block_mask_collator_returns_4d_attention_mask():
    class DummyTokenizer:
        pad_token_id = 0

    collator = trainer.DataCollator(DummyTokenizer(), use_block_mask=True, mask_dtype=torch.float32)
    batch = collator(
        [
            {
                "input_ids": torch.tensor([10, 11, 12, 13]),
                "attention_mask": torch.tensor([1, 1, 1, 1]),
                "labels": torch.tensor([-100, 11, 12, 13]),
                "teacher_mask": torch.tensor([False, True, False, False]),
                "response_mask": torch.tensor([False, False, False, True]),
            }
        ]
    )

    assert batch["attention_mask"].shape == (1, 1, 4, 4)
    assert "teacher_mask" not in batch
    assert "response_mask" not in batch
    assert batch["attention_mask"][0, 0, 3, 1] < -1e20
