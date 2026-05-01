import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_mqe_critic.py"
SPEC = importlib.util.spec_from_file_location("train_mqe_critic", SCRIPT)
trainer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = trainer
SPEC.loader.exec_module(trainer)

PREPARE_SCRIPT = ROOT / "scripts" / "prepare_agenttrove_mqe.py"
PREPARE_SPEC = importlib.util.spec_from_file_location("prepare_agenttrove_mqe", PREPARE_SCRIPT)
prepare = importlib.util.module_from_spec(PREPARE_SPEC)
assert PREPARE_SPEC.loader is not None
sys.modules[PREPARE_SPEC.name] = prepare
PREPARE_SPEC.loader.exec_module(prepare)

TRANSITION_SCRIPT = ROOT / "scripts" / "train_transition_encoder.py"
TRANSITION_SPEC = importlib.util.spec_from_file_location("train_transition_encoder", TRANSITION_SCRIPT)
transition = importlib.util.module_from_spec(TRANSITION_SPEC)
assert TRANSITION_SPEC.loader is not None
sys.modules[TRANSITION_SPEC.name] = transition
TRANSITION_SPEC.loader.exec_module(transition)


def test_action_text_prefers_canonical_payload_over_verbose_raw_json():
    row = {
        "action": {
            "raw_json": '{"analysis":"verbose teacher thought","commands":[{"keystrokes":"ls\\n"}]}',
            "canonical_str": '{"commands":[{"keystrokes":"ls\\n"}]}',
        }
    }

    assert trainer._action_text(row) == '{"commands":[{"keystrokes":"ls\\n"}]}'


def test_action_text_falls_back_to_raw_json_when_needed():
    row = {
        "action": {
            "raw_json": '{"task_complete":true}',
        }
    }

    assert trainer._action_text(row) == '{"task_complete":true}'


def test_prepare_extracts_structured_action_features():
    action = {
        "commands": [
            {"keystrokes": "python3 -m pytest tests/test_api.py -q\n"},
            {"keystrokes": "curl -s http://api:8000/health\n"},
        ],
        "task_complete": False,
    }

    features = prepare.extract_action_features(action, max_chars=4000)
    names = prepare.ACTION_FEATURE_NAMES

    assert len(features) == len(names)
    assert features[names.index("has_commands")] == 1.0
    assert features[names.index("runs_tests")] == 1.0
    assert features[names.index("runs_python")] == 1.0
    assert features[names.index("network_op")] == 1.0
    assert features[names.index("has_py_path")] == 1.0
    assert features[names.index("verb_python")] == 1.0
    assert features[names.index("verb_curl")] == 1.0


def test_action_type_prefers_programmatic_effect_over_task_complete():
    command_action = {
        "commands": [{"keystrokes": "cat > main.py <<'PY'\nprint('ok')\nPY\n"}],
        "task_complete": True,
    }
    finish_only_action = {"task_complete": True}

    command_features = prepare.extract_action_features(command_action, max_chars=4000)
    finish_features = prepare.extract_action_features(finish_only_action, max_chars=4000)

    assert prepare.action_type_from_features(command_features) == "write"
    assert prepare.action_type_from_features(finish_features) == "final"


def test_prepare_adds_transition_schema_and_hard_negatives():
    args = type(
        "Args",
        (),
        {
            "min_actions": 2,
            "max_actions": 80,
            "max_state_chars": 12000,
            "max_action_chars": 4000,
            "gamma": 0.98,
        },
    )()
    row = {
        "trial_name": "toy",
        "task_id": "toy-task",
        "original_source": "unit",
        "messages": [
            {"role": "user", "content": "Inspect files and finish."},
            {"role": "assistant", "content": '{"commands":[{"keystrokes":"ls -la\\n"}]}'},
            {"role": "user", "content": "New Terminal Output:\\ntotal 4"},
            {"role": "assistant", "content": '{"commands":[{"keystrokes":"cat README.md\\n"}]}'},
        ],
    }

    converted = prepare.convert_rollout(row, args)
    assert converted is not None
    records, _ = converted
    prepare.add_transition_indices_and_negatives(records)

    assert len(records) == 2
    assert records[0]["action_text"]
    assert records[0]["action_signature"].startswith("type=")
    assert records[0]["action_family_signature"].startswith("type=")
    assert records[0]["negative_next_indices"] == [1]
    assert records[1]["negative_action_indices"] == [0]


def test_mqe_model_accepts_action_features():
    torch, _, _, _, _ = trainer.require_torch()
    model_cls = trainer.build_model_classes()
    model = model_cls(8, 16, 8, 2, 0.0, action_feature_dim=5)

    state = torch.randn(3, 8)
    next_state = torch.randn(3, 8)
    goal = torch.randn(3, 8)
    action = torch.randn(3, 8)
    action_features = torch.randn(3, 5)

    outputs = model(state, next_state, goal, action, action_features)

    assert outputs["action_distance"].shape == (3,)
    assert outputs["state_distance"].shape == (3,)


def test_transition_head_and_retrieval_loss_shapes():
    torch, _, _, _ = transition.require_torch()
    head_cls = transition.build_transition_head_class()
    head = head_cls(8, 3, 16, len(transition.ACTION_TYPES))
    state = torch.randn(4, 8)
    action = torch.randn(4, 8)
    features = torch.randn(4, 3)
    next_z = torch.nn.functional.normalize(torch.randn(4, 8), p=2, dim=-1)

    pred, aux_logits = head(state, action, features)
    loss, logits = transition.retrieval_loss(
        pred,
        next_z,
        hard_next_z=None,
        hard_next_rows=torch.empty((0,), dtype=torch.long),
        temperature=0.05,
    )

    assert pred.shape == (4, 8)
    assert aux_logits.shape == (4, len(transition.ACTION_TYPES))
    assert logits.shape == (4, 4)
    assert loss.ndim == 0


def test_mqe_transition_cache_backend_smoke(tmp_path):
    torch, _, _, _, _ = trainer.require_torch()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train_rows = []
    val_rows = []
    for split, target in [("train", train_rows), ("val", val_rows)]:
        for index in range(3 if split == "train" else 2):
            remaining = 3 - index
            target.append(
                {
                    "example_id": f"{split}:{index}",
                    "rollout_id": split,
                    "challenge_id": "toy",
                    "state_text": f"state {split} {index}",
                    "next_state_text": f"next {split} {index}",
                    "goal_prompt": "goal",
                    "goal_policy_text": "goal",
                    "goal_state_hash": f"goal-{split}",
                    "goal_index": 3,
                    "target_distance_steps": float(remaining),
                    "next_distance_steps": float(max(0, remaining - 1)),
                    "gamma": 0.98,
                    "action": {"canonical_str": f"cmd {index}"},
                    "action_features": [1.0, float(index)],
                }
            )
    (data_dir / "train.jsonl").write_text("\n".join(__import__("json").dumps(row) for row in train_rows) + "\n")
    (data_dir / "val.jsonl").write_text("\n".join(__import__("json").dumps(row) for row in val_rows) + "\n")
    (data_dir / "states.jsonl").write_text("")

    def split_cache(rows):
        count = len(rows)
        return {
            "row_ids": [row["example_id"] for row in rows],
            "z_state": torch.randn(count, 8),
            "z_action": torch.randn(count, 8),
            "z_next": torch.randn(count, 8),
            "z_goal": torch.randn(count, 8),
            "z_state_action": torch.randn(count, 8),
            "action_features": torch.randn(count, 2),
            "text_hashes": {},
        }

    cache_path = tmp_path / "transition_cache.pt"
    torch.save(
        {
            "format": "transition-cache-v1",
            "metadata": {"embedding_dim": 8, "action_feature_dim": 2},
            "splits": {"train": split_cache(train_rows), "val": split_cache(val_rows)},
        },
        cache_path,
    )
    config = trainer.MQETrainConfig(
        data_dir=str(data_dir),
        output_dir=str(tmp_path / "out"),
        encoder_backend="transition-cache",
        cache_path=str(cache_path),
        hidden_dim=8,
        latent_dim=4,
        mrn_components=2,
        batch_size=2,
        epochs=1,
        action_choice_eval_negatives=1,
        multistep_batch_size=2,
        triangle_batch_size=2,
        checkpoint_metric="val_loss",
    )

    summary = trainer.train_mqe(config)

    assert summary["precomputed_state_action"] is True
    assert summary["input_dim"] == 8
    assert "val_action_choice_acc" in summary["metrics"][-1]
