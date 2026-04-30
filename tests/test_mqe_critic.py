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
