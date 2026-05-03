from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "make_etpi_repo_tasks.py"
SPEC = importlib.util.spec_from_file_location("make_etpi_repo_tasks", SCRIPT_PATH)
assert SPEC is not None
make_etpi_repo_tasks = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = make_etpi_repo_tasks
SPEC.loader.exec_module(make_etpi_repo_tasks)


def test_task_rows_generate_public_repo_tasks_without_cuda() -> None:
    repos = [
        {"repo_id": "python_click", "domain": "python", "name": "click"},
        {"repo_id": "web_vite", "domain": "webdev", "name": "vite"},
        {"repo_id": "cpp_fmt", "domain": "cpp", "name": "fmt"},
    ]

    rows = make_etpi_repo_tasks.task_rows(repos, Path("/repos"), include_extra=True)

    assert len(rows) == 27
    assert {row["domain"] for row in rows} == {"python", "webdev", "cpp"}
    assert all(row["source"] == "etpi_public_repos_v1" for row in rows)
    assert all(row["verifiable"] is False for row in rows)
    assert any(row["task_id"] == "python_click__python_packaging" for row in rows)
    assert any(row["task_id"] == "web_vite__web_build_pipeline" for row in rows)
    assert any(row["task_id"] == "cpp_fmt__cpp_build" for row in rows)
    assert not any(row["domain"] == "cuda" for row in rows)
