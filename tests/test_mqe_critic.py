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
