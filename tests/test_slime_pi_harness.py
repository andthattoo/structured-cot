from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from slime_plugins import etpi_pi_harness


@dataclass
class FakeSample:
    prompt: str = "prompt"
    response: str = ""
    metadata: dict = field(default_factory=dict)


def test_parse_action_prefers_final_over_bash() -> None:
    action = etpi_pi_harness.parse_action("<bash>pytest -q</bash>\n<final>done</final>")

    assert action.kind == "final"
    assert action.body == "done"


def test_parse_action_extracts_bash() -> None:
    action = etpi_pi_harness.parse_action("Let me inspect.\n<bash>ls -la</bash>")

    assert action.kind == "bash"
    assert action.body == "ls -la"


def test_run_bash_is_disabled_by_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("ETPI_SLIME_ENABLE_SHELL", raising=False)

    result = etpi_pi_harness.run_bash(
        "echo should-not-run",
        cwd=tmp_path,
        timeout_sec=1,
        char_limit=1000,
    )

    assert result.disabled is True
    assert result.returncode == 126


def test_reward_func_uses_verify_commands(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ETPI_SLIME_ENABLE_SHELL", "1")
    sample = FakeSample(metadata={"cwd": str(tmp_path), "verify_commands": ["test -d ."]})

    reward = asyncio.run(etpi_pi_harness.reward_func(SimpleNamespace(), sample))

    assert reward == 1.0


def test_reward_func_gives_small_heuristic_reward_for_final() -> None:
    sample = FakeSample(response="<final>work complete</final>")

    reward = asyncio.run(etpi_pi_harness.reward_func(SimpleNamespace(), sample))

    assert reward == 0.2
