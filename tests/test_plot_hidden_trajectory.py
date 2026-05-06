from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "plot_hidden_trajectory.py"
SPEC = importlib.util.spec_from_file_location("plot_hidden_trajectory", SCRIPT_PATH)
assert SPEC is not None
plot_hidden_trajectory = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = plot_hidden_trajectory
SPEC.loader.exec_module(plot_hidden_trajectory)


def test_pca_2d_returns_two_columns() -> None:
    matrix = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)

    coords = plot_hidden_trajectory.pca_2d(matrix)

    assert coords.shape == (3, 2)


def test_write_and_read_csv_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]

    plot_hidden_trajectory.write_csv(path, rows)

    assert plot_hidden_trajectory.read_csv(path) == [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]
