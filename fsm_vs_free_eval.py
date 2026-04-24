"""
FSM vs. free thinking — zero-training comparison on a code benchmark.

For each problem, run Qwen3.6-35B-A3B (via the local llama.cpp OpenAI-compatible
server) in multiple modes:

  FREE:  standard thinking-mode generation.  Model produces its native verbose
         <think>...</think> followed by an answer.

  FSM :  grammar-constrained generation.  The same model is forced via GBNF to
         emit a compact structured plan inside <think>...</think>, then the
         grammar becomes permissive so the model can write the code freely.

  PROMPT_TERSE:
         same compact plan format requested in the prompt, but with no grammar
         constraint.  This controls for "the prompt made it terse" vs.
         "the grammar made it terse."

We measure:
  - pass@1 on the benchmark's hidden tests
  - thinking-token count (the tokens inside <think>...</think>)
  - total-completion-token count

and report a side-by-side table + compression ratio + accuracy delta.

No training.  No data pipeline.  Just constrained decoding.

Usage:
  # Make sure the server is up (see run_server.sh)
  uv run python fsm_vs_free_eval.py --n-problems 30 --dataset humaneval

  # MBPP+
  uv run python fsm_vs_free_eval.py --n-problems 50 --dataset mbpp

  # LiveCodeBench, post-cutoff (contamination-clean test)
  uv run python fsm_vs_free_eval.py --dataset livecodebench \\
      --date-cutoff 2025-12-01 --platform leetcode --n-problems 50

  # Free-only or FSM-only (for debugging)
  uv run python fsm_vs_free_eval.py --only free --n-problems 10
  uv run python fsm_vs_free_eval.py --only fsm  --n-problems 10

  # Run FREE + FSM + PROMPT_TERSE controls
  uv run python fsm_vs_free_eval.py --only all --n-problems 30
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_benchmark(name: str, n: int, args=None):
    from datasets import load_dataset
    if name == "humaneval":
        ds = load_dataset("evalplus/humanevalplus", split="test")
        rows = list(ds)
    elif name == "mbpp":
        ds = load_dataset("evalplus/mbppplus", split="test")
        rows = list(ds)
    elif name == "livecodebench":
        version = getattr(args, "lcb_version", "release_v5")
        date_cutoff = getattr(args, "date_cutoff", "")
        platform = getattr(args, "platform", "leetcode")

        print(f"  loading livecodebench/code_generation_lite (version={version})")
        try:
            ds = load_dataset(
                "livecodebench/code_generation_lite",
                split="test",
                version_tag=version,
                trust_remote_code=True,
            )
        except RuntimeError as e:
            if "Dataset scripts are no longer supported" in str(e):
                raise RuntimeError(
                    "LiveCodeBench code_generation_lite currently requires "
                    "Hugging Face datasets<4 because it uses a dataset loading "
                    "script. Run `uv sync --upgrade-package datasets` after "
                    "pulling the latest pyproject.toml, or install "
                    "`datasets>=3,<4` in this environment."
                ) from e
            raise
        rows = list(ds)
        print(f"  {len(rows)} total problems in {version}")

        if date_cutoff:
            rows = [r for r in rows if r.get("contest_date", "") >= date_cutoff]
            print(f"  {len(rows)} after contest_date >= {date_cutoff}")
        if platform:
            rows = [r for r in rows if r.get("platform", "") == platform]
            print(f"  {len(rows)} after platform == {platform}")

        # Keep only problems whose public tests are all functional.  LCB
        # has a mix of functional (LeetCode-style) and stdin (competitive-
        # programming-style) tests; we only handle functional here.
        kept = []
        for r in rows:
            raw = r.get("public_test_cases", "") or "[]"
            try:
                tests = json.loads(raw)
            except Exception:
                continue
            if not tests:
                continue
            if all(t.get("testtype") == "functional" for t in tests):
                kept.append(r)
        rows = kept
        print(f"  {len(rows)} after functional-tests-only filter")

        # Normalize a task_id for reporting — use question_id.
        for r in rows:
            r.setdefault("task_id", r.get("question_id", r.get("question_title", "unknown")))
    else:
        raise ValueError(f"unknown dataset {name}")

    return rows[:n] if n > 0 else rows


# ---------------------------------------------------------------------------
# Prompt / response helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert Python programmer.  Think carefully in your <think> "
    "block, then write correct, efficient, well-tested code.  "
    "Wrap your final code in a ```python ... ``` fenced block."
)

PROMPT_TERSE_SYSTEM_PROMPT = (
    "You are an expert Python programmer.  Think carefully but tersely.  "
    "Use exactly this thinking format before the final answer:\n"
    "<think>\n"
    "GOAL: one short line\n"
    "APPROACH: one short line\n"
    "EDGE: one short line\n"
    "</think>\n\n"
    "Then write correct, efficient, well-tested code.  "
    "Wrap your final code in a ```python ... ``` fenced block."
)

MODE_ORDER = ("free", "fsm", "prompt_terse")
MODE_LABELS = {
    "free": "FREE",
    "fsm": "FSM",
    "prompt_terse": "PROMPT_TERSE",
}


def build_user_prompt(problem: dict, dataset: str) -> str:
    if dataset == "humaneval":
        return (
            "Complete the following Python function.  Return the full function "
            "including the signature and docstring.\n\n"
            f"```python\n{problem['prompt']}```\n"
        )
    elif dataset == "mbpp":
        return (
            f"{problem['prompt']}\n\n"
            "Write the complete implementation in Python."
        )
    elif dataset == "livecodebench":
        starter = (problem.get("starter_code") or "").strip()
        starter_block = (
            f"\n\nStarter code:\n```python\n{starter}\n```" if starter else ""
        )
        return (
            f"{problem['question_content']}"
            f"{starter_block}\n\n"
            "Implement the complete solution in Python.  If a class Solution "
            "signature is provided, use it.  Return only runnable code in a "
            "```python``` block."
        )
    raise ValueError(dataset)


# Regex for extracting tokens.
THINK_OPEN_CLOSE_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
CODE_FENCED_RE = re.compile(r"```(?:python|py)?[ \t\r]*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
CODE_DEF_RE = re.compile(r"^(def\s+\w+.*?)(?=\n\S|\Z)", re.DOTALL | re.MULTILINE)
CODE_START_RE = re.compile(
    r"^(?:from\s+\S+\s+import\s+.+|import\s+.+|class\s+\w+\b.*|def\s+\w+\s*\(.*|@\w+.*)$",
    re.MULTILINE,
)
OPENING_FENCE_RE = re.compile(r"^```(?:python|py)?[ \t\r]*\n?", re.IGNORECASE)
OPENING_FENCE_ANYWHERE_RE = re.compile(r"```(?:python|py)?[ \t\r]*\n?", re.IGNORECASE)


def _strip_unmatched_fence(code: str) -> str:
    """Remove a leading opening fence when the model forgot the closing fence."""
    code = code.strip()
    code = OPENING_FENCE_RE.sub("", code, count=1).strip()
    if code.endswith("```"):
        code = code[:-3].strip()
    return code


def extract_think(text: str) -> str:
    """Pull the reasoning portion out of a response.

    Handles three formats we've seen in practice:
      1. Full tags:        "<think>...</think> code"
      2. Closing tag only: "...</think> code"          (some chat templates strip opening)
      3. No tags at all:   "raw reasoning ```python ..." (Qwen3.6 GGUF chat mode)
    """
    if "</think>" in text:
        m = THINK_OPEN_CLOSE_RE.search(text)
        if m:
            return m.group(1).strip()
        return text.split("</think>", 1)[0].strip()

    # No tags.  Thinking is whatever comes before the first fenced code block.
    m = CODE_FENCED_RE.search(text)
    if m:
        return text[: m.start()].strip()
    m = CODE_START_RE.search(text)
    if m:
        return text[: m.start()].strip()
    return ""


def extract_code_with_info(text: str) -> tuple[str, dict]:
    """Pull the Python code out of the response and record how we found it.

    Free-thinking responses often contain MULTIPLE fenced code blocks as
    the model drafts and revises its answer.  The final answer is almost
    always the LAST block, so we prefer last-match over first-match.
    Priority:
      1. Last ```python fenced block after </think>
      2. Last fenced block anywhere
      3. Unterminated ```python fenced block
      4. First code-looking block after prose
      5. Empty code if no code-like block can be found
    """
    after_think = text.split("</think>", 1)[-1] if "</think>" in text else text

    # 1. last fenced block after </think>
    matches = CODE_FENCED_RE.findall(after_think)
    if matches:
        return matches[-1], {
            "extraction_method": "last_fenced_after_think",
            "extraction_issue": "none",
        }
    # 2. last fenced block anywhere
    matches = CODE_FENCED_RE.findall(text)
    if matches:
        return matches[-1], {
            "extraction_method": "last_fenced_anywhere",
            "extraction_issue": "none",
        }
    # 3. unterminated/opening fenced block after </think>
    stripped = after_think.lstrip()
    if OPENING_FENCE_RE.match(stripped):
        return _strip_unmatched_fence(stripped), {
            "extraction_method": "opening_fence_after_think",
            "extraction_issue": "unterminated_fence",
        }

    m = OPENING_FENCE_ANYWHERE_RE.search(after_think)
    if m:
        return _strip_unmatched_fence(after_think[m.end():]), {
            "extraction_method": "opening_fence_anywhere",
            "extraction_issue": "prose_before_unterminated_fence",
        }

    # 4. code-looking block after prose/imports/classes
    m = CODE_START_RE.search(after_think)
    if m:
        issue = "no_fenced_block" if m.start() == 0 else "prose_before_code"
        return _strip_unmatched_fence(after_think[m.start():]), {
            "extraction_method": "code_start_after_think",
            "extraction_issue": issue,
        }

    # 5. def ... after </think>
    m = CODE_DEF_RE.search(after_think)
    if m:
        return m.group(1), {
            "extraction_method": "def_after_think",
            "extraction_issue": "no_fenced_block",
        }
    # 6. No runnable-looking code found.  Do not execute prose as Python.
    return "", {
        "extraction_method": "empty",
        "extraction_issue": "empty_code",
    }


def extract_code(text: str) -> str:
    """Backward-compatible code-only extractor."""
    code, _ = extract_code_with_info(text)
    return code


# ---------------------------------------------------------------------------
# Tokenizer (for thinking-token counts)
# ---------------------------------------------------------------------------

_TOK = None


def get_tokenizer(model_name: str):
    global _TOK
    if _TOK is None:
        try:
            from transformers import AutoTokenizer
            _TOK = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            print(f"  (Falling back to char-based count: {e})", file=sys.stderr)
            _TOK = "fallback"
    return _TOK


def count_tokens(text: str, model_name: str) -> int:
    tok = get_tokenizer(model_name)
    if tok == "fallback":
        # char/4 is a rough BPE approximation
        return max(1, len(text) // 4)
    return len(tok.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# Test execution (sandboxed-ish via subprocess)
# ---------------------------------------------------------------------------

def _extract_lcb_fn_name(problem: dict) -> str:
    """Find the entry-point function name for a LiveCodeBench problem."""
    meta = problem.get("metadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    for k in ("func_name", "function_name", "fn_name", "entry_point"):
        if meta.get(k):
            return meta[k]
    starter = problem.get("starter_code") or ""
    m = re.search(r"def\s+(\w+)\s*\(", starter)
    if m:
        return m.group(1)
    return ""


def _lcb_public_test_count(problem: dict) -> int:
    try:
        return len(json.loads(problem.get("public_test_cases") or "[]"))
    except Exception:
        return 0


def run_tests_livecodebench(code: str, problem: dict, timeout: int = 30) -> tuple[bool, str]:
    """Run a LCB functional-test problem via a generated harness.

    Each test is `{"input": "<repr of arg1>\\n<repr of arg2>\\n...", "output": "<repr of expected>"}`.
    We parse each line as JSON, call the entry function (`Solution().{fn}` if a
    `Solution` class is present, otherwise the top-level `{fn}`), and compare
    the return value to the JSON-parsed expected output.
    """
    import tempfile

    fn_name = _extract_lcb_fn_name(problem)
    if not fn_name:
        return False, "no_entry_point"
    try:
        tests = json.loads(problem.get("public_test_cases") or "[]")
    except Exception as e:
        return False, f"bad_tests: {e}"
    if not tests:
        return False, "no_tests"

    harness = (
        code + "\n\n"
        "import json as _json, sys as _sys\n"
        f"_tests = _json.loads({json.dumps(json.dumps(tests))})\n"
        f"_FN_NAME = {fn_name!r}\n"
        + _LCB_HARNESS_TAIL
    )

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(harness)
        script_path = tf.name
    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            timeout=timeout,
            capture_output=True,
        )
        ok = proc.returncode == 0
        err = proc.stderr.decode("utf-8", errors="ignore")[-500:] if not ok else ""
        return ok, err
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"[:200]
    finally:
        try:
            os.unlink(script_path)
        except Exception:
            pass


_LCB_HARNESS_TAIL = r"""
def _resolve_fn():
    ns = globals()
    # Prefer class Solution().<fn> — the LeetCode convention.
    if "Solution" in ns and hasattr(ns["Solution"], _FN_NAME):
        return getattr(ns["Solution"](), _FN_NAME)
    if _FN_NAME in ns:
        return ns[_FN_NAME]
    raise RuntimeError(f"entry point {_FN_NAME} not found")

def _parse(s):
    try:
        return _json.loads(s)
    except Exception:
        return s

def _run():
    fn = _resolve_fn()
    for i, t in enumerate(_tests):
        lines = [ln for ln in t["input"].splitlines() if ln.strip() != ""]
        args = [_parse(ln) for ln in lines]
        expected = _parse(t["output"])
        result = fn(*args)
        # Permit float tolerance and list-of-lists equality via JSON round-trip
        if _json.dumps(result, sort_keys=True) != _json.dumps(expected, sort_keys=True):
            raise AssertionError(
                f"test {i}: expected {expected!r}, got {result!r}"
            )

_run()
"""


def run_tests(code: str, test_code: str, entry_point: str, timeout: int = 30) -> tuple[bool, str]:
    """Run the model's code + the benchmark test suite in a fresh Python subprocess.

    Uses a tempfile (not `python -c`) because long programs + tests exceed
    Linux's argv limit (Errno 7 / E2BIG, ~128 KB).
    """
    import tempfile
    full = f"{code}\n\n{test_code}\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(full)
        script_path = tf.name
    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            timeout=timeout,
            capture_output=True,
        )
        ok = proc.returncode == 0
        err = proc.stderr.decode("utf-8", errors="ignore")[-500:] if not ok else ""
        return ok, err
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"[:200]
    finally:
        try:
            os.unlink(script_path)
        except Exception:
            pass


def _entry_point_found(code: str, dataset: str, entry_point: str, problem: dict) -> bool:
    """Best-effort static check for whether extracted code exposes the target API."""
    if not code.strip():
        return False
    if dataset == "livecodebench":
        fn_name = _extract_lcb_fn_name(problem)
        if not fn_name:
            return False
        if re.search(r"^\s*class\s+Solution\b", code, re.MULTILINE):
            return True
        return bool(re.search(rf"^\s*def\s+{re.escape(fn_name)}\s*\(", code, re.MULTILINE))
    return bool(re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", code, re.MULTILINE))


def classify_failure(result: dict) -> str:
    """Bucket non-passes into extraction, generation, syntax, and runtime causes."""
    if result.get("pass") is True:
        return "pass"
    if result.get("err", "").startswith("gen_error:"):
        return "generation_error"
    if result.get("extraction_issue") == "empty_code":
        return "extraction_empty_code"

    err = result.get("err") or ""
    if err == "TIMEOUT":
        return "timeout"
    if "SyntaxError" in err or "IndentationError" in err:
        return "syntax_error"
    if "entry point" in err and "not found" in err:
        return "missing_entry_point"
    if result.get("entry_point_found") is False:
        return "missing_entry_point"
    if "AssertionError" in err:
        return "wrong_answer"
    if "TypeError" in err:
        return "type_error"
    if "NameError" in err:
        return "runtime_name_error"
    if err:
        return "runtime_error"
    return "unknown_failure"


# ---------------------------------------------------------------------------
# Generation (via OpenAI-compatible server)
# ---------------------------------------------------------------------------

def make_client(args):
    from openai import OpenAI
    client = OpenAI(
        base_url=args.base_url,
        api_key=os.environ.get(args.api_key_env, "dummy"),
        timeout=args.request_timeout,
        max_retries=0,
    )
    # Pre-flight: verify the server is reachable before running any problems.
    try:
        _ = client.models.list()
    except Exception as e:
        print(
            f"ERROR: cannot reach the server at {args.base_url}\n"
            f"  ({type(e).__name__}: {e})\n\n"
            "Start the local llama-cpp-python server first:\n"
            "  nohup ./run_server.sh > server.log 2>&1 &\n"
            "  tail -f server.log   # wait for 'Uvicorn running on ...'\n\n"
            "Or point --base-url at whatever OpenAI-compatible endpoint you want.",
            file=sys.stderr,
        )
        sys.exit(1)
    return client


def message_text(message) -> str:
    """Return visible content, preserving separated reasoning when servers expose it."""
    content = message.content or ""
    reasoning = getattr(message, "reasoning_content", None)
    if not reasoning:
        extra = getattr(message, "model_extra", None) or {}
        reasoning = extra.get("reasoning_content") or extra.get("reasoning")
    if reasoning:
        return f"<think>\n{str(reasoning).strip()}\n</think>\n\n{content}"
    return content


def generate_free(client, model: str, user_prompt: str, max_tokens: int) -> tuple[str, int]:
    """Standard chat completion — no grammar, no constraint."""
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    text = message_text(r.choices[0].message)
    completion_tokens = r.usage.completion_tokens if r.usage else count_tokens(text, model)
    return text, completion_tokens


def generate_prompt_terse(client, model: str, user_prompt: str, max_tokens: int) -> tuple[str, int]:
    """Prompt-only terse structured thinking — no grammar constraint."""
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PROMPT_TERSE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    text = message_text(r.choices[0].message)
    completion_tokens = r.usage.completion_tokens if r.usage else count_tokens(text, model)
    return text, completion_tokens


def generate_fsm(client, model: str, user_prompt: str, grammar: str, max_tokens: int) -> tuple[str, int]:
    """Chat completion with GBNF grammar applied to the whole assistant response."""
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
        extra_body={"grammar": grammar},
    )
    text = message_text(r.choices[0].message)
    completion_tokens = r.usage.completion_tokens if r.usage else count_tokens(text, model)
    return text, completion_tokens


def modes_to_run(only: str) -> list[str]:
    if only == "both":
        return ["free", "fsm"]
    if only == "all":
        return list(MODE_ORDER)
    return [only]


def _pass(row: dict, mode: str) -> Optional[bool]:
    return row.get(mode, {}).get("pass") if mode in row else None


def _bucket(task_ids: list[str]) -> dict:
    return {"count": len(task_ids), "task_ids": task_ids}


def _ids(results: list[dict], pred) -> list[str]:
    return [r["task_id"] for r in results if pred(r)]


def build_outcome_breakdown(results: list[dict]) -> dict:
    """Count pass-set overlaps for the core pair and prompt-only control."""
    out: dict = {}

    pair_rows = [r for r in results if "free" in r and "fsm" in r]
    if pair_rows:
        out["free_vs_fsm"] = {
            "n": len(pair_rows),
            "both_pass": _bucket(_ids(pair_rows, lambda r: _pass(r, "free") is True and _pass(r, "fsm") is True)),
            "both_fail": _bucket(_ids(pair_rows, lambda r: _pass(r, "free") is False and _pass(r, "fsm") is False)),
            "fsm_only_pass": _bucket(_ids(pair_rows, lambda r: _pass(r, "free") is False and _pass(r, "fsm") is True)),
            "free_only_pass": _bucket(_ids(pair_rows, lambda r: _pass(r, "free") is True and _pass(r, "fsm") is False)),
        }

    triple_rows = [r for r in results if all(m in r for m in MODE_ORDER)]
    if triple_rows:
        out["prompt_terse_checks"] = {
            "n": len(triple_rows),
            "all_pass": _bucket(_ids(triple_rows, lambda r: all(_pass(r, m) is True for m in MODE_ORDER))),
            "all_fail": _bucket(_ids(triple_rows, lambda r: all(_pass(r, m) is False for m in MODE_ORDER))),
            "prompt_terse_only_pass": _bucket(
                _ids(triple_rows, lambda r: _pass(r, "prompt_terse") is True
                     and _pass(r, "free") is False
                     and _pass(r, "fsm") is False)
            ),
            "prompt_terse_only_fail": _bucket(
                _ids(triple_rows, lambda r: _pass(r, "prompt_terse") is False
                     and _pass(r, "free") is True
                     and _pass(r, "fsm") is True)
            ),
            "prompt_terse_matches_free_not_fsm": _bucket(
                _ids(triple_rows, lambda r: _pass(r, "prompt_terse") == _pass(r, "free")
                     and _pass(r, "prompt_terse") != _pass(r, "fsm"))
            ),
            "prompt_terse_matches_fsm_not_free": _bucket(
                _ids(triple_rows, lambda r: _pass(r, "prompt_terse") == _pass(r, "fsm")
                     and _pass(r, "prompt_terse") != _pass(r, "free"))
            ),
        }

    return out


def build_failure_accounting(results: list[dict], active_modes: list[str]) -> dict:
    out: dict = {}
    for mode in active_modes:
        rows = [r[mode] for r in results if mode in r]
        failure_types = Counter(r.get("failure_type", "unknown_failure") for r in rows if not r.get("pass"))
        extraction_methods = Counter(r.get("extraction_method", "unknown") for r in rows)
        extraction_issues = Counter(
            r.get("extraction_issue", "unknown") for r in rows
            if r.get("extraction_issue", "none") != "none"
        )
        out[mode] = {
            "n": len(rows),
            "passes": sum(1 for r in rows if r.get("pass")),
            "failures": sum(1 for r in rows if not r.get("pass")),
            "failure_types": dict(failure_types),
            "extraction_methods": dict(extraction_methods),
            "extraction_issues": dict(extraction_issues),
        }
    return out


def _format_ids(task_ids: list[str], limit: int = 16) -> str:
    if not task_ids:
        return "-"
    shown = ", ".join(str(t) for t in task_ids[:limit])
    if len(task_ids) > limit:
        shown += f", ... (+{len(task_ids) - limit})"
    return shown


def _format_counts(counts: dict) -> str:
    if not counts:
        return "-"
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def print_outcome_breakdown(breakdown: dict) -> None:
    if "free_vs_fsm" in breakdown:
        print("\n  Pass-set overlap (FREE vs FSM)")
        for key in ("both_pass", "both_fail", "fsm_only_pass", "free_only_pass"):
            b = breakdown["free_vs_fsm"][key]
            print(f"    {key:<18s} {b['count']:>3d}  {_format_ids(b['task_ids'])}")

    if "prompt_terse_checks" in breakdown:
        print("\n  PROMPT_TERSE checks")
        for key in (
            "all_pass",
            "all_fail",
            "prompt_terse_only_pass",
            "prompt_terse_only_fail",
            "prompt_terse_matches_free_not_fsm",
            "prompt_terse_matches_fsm_not_free",
        ):
            b = breakdown["prompt_terse_checks"][key]
            print(f"    {key:<34s} {b['count']:>3d}  {_format_ids(b['task_ids'])}")


def print_failure_accounting(accounting: dict) -> None:
    if not accounting:
        return
    print("\n  Failure / extraction accounting")
    for mode in MODE_ORDER:
        if mode not in accounting:
            continue
        a = accounting[mode]
        print(f"    {MODE_LABELS[mode]:<13s} failures: {a['failures']:>3d}  {_format_counts(a['failure_types'])}")
        print(f"    {'':<13s} extraction issues: {_format_counts(a['extraction_issues'])}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--api-key-env", default="DUMMY_KEY")
    p.add_argument("--model", default="qwen3.6-35b-a3b",
                   help="Model name sent in the request (local server ignores it).")
    p.add_argument("--tokenizer", default="Qwen/Qwen3.6-35B-A3B",
                   help="HF tokenizer id for counting think tokens.")

    p.add_argument("--dataset", choices=["humaneval", "mbpp", "livecodebench"],
                   default="humaneval")
    p.add_argument("--n-problems", type=int, default=30,
                   help="Problems to evaluate (0 = all).")

    # LiveCodeBench-specific filters
    p.add_argument("--lcb-version", default="release_v5",
                   help="LiveCodeBench release tag (e.g. release_v4, release_v5).")
    p.add_argument("--date-cutoff", default="",
                   help="ISO date; keep LCB problems with contest_date >= this "
                        "(e.g. 2025-12-01).  Empty = no date filter.")
    p.add_argument("--platform", default="leetcode",
                   help="LCB platform filter: leetcode / atcoder / codeforces. "
                        "Empty string = no platform filter.")
    p.add_argument("--grammar-file", default="fsm_grammar.gbnf")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--timeout", type=int, default=30,
                   help="Per-test execution timeout (seconds).")

    p.add_argument("--only", choices=["both", "all", "free", "fsm", "prompt_terse"], default="both",
                   help="'both' = FREE+FSM, 'all' = FREE+FSM+PROMPT_TERSE, or run one mode only.")
    p.add_argument("--request-timeout", type=float, default=600.0,
                   help="Per-request HTTP timeout in seconds. Hard-stops stuck calls.")
    p.add_argument("--out-dir", default="fsm_vs_free")
    args = p.parse_args()

    active_modes = modes_to_run(args.only)
    grammar = ""
    if "fsm" in active_modes:
        try:
            grammar = Path(args.grammar_file).read_text()
        except Exception as e:
            print(f"ERROR: could not read grammar file {args.grammar_file}: {e}", file=sys.stderr)
            sys.exit(1)

    client = make_client(args)

    print(f"[1/3] Loading {args.dataset} problems")
    problems = load_benchmark(args.dataset, args.n_problems, args)
    print(f"  {len(problems)} problems")
    if not problems:
        print("ERROR: no problems to evaluate after filtering.", file=sys.stderr)
        sys.exit(1)

    mode_names = ", ".join(MODE_LABELS[m] for m in active_modes)
    print(f"[2/3] Running {mode_names}")
    results = []
    t_start = time.time()

    for i, prob in enumerate(problems):
        user_prompt = build_user_prompt(prob, args.dataset)
        entry_point = prob.get("entry_point") or "candidate"
        test_code = prob.get("test", "")
        task_id = prob["task_id"]

        def _score(code: str) -> tuple[bool, str]:
            if args.dataset == "livecodebench":
                return run_tests_livecodebench(code, prob, args.timeout)
            return run_tests(code, test_code, entry_point, args.timeout)

        row = {"task_id": task_id}
        t_prob = time.time()
        print(f"  [{i+1}/{len(problems)}] {task_id:<16s} start", flush=True)

        def _generate(mode: str) -> tuple[str, int]:
            if mode == "free":
                return generate_free(client, args.model, user_prompt, args.max_tokens)
            if mode == "fsm":
                return generate_fsm(client, args.model, user_prompt, grammar, args.max_tokens)
            if mode == "prompt_terse":
                return generate_prompt_terse(client, args.model, user_prompt, args.max_tokens)
            raise ValueError(mode)

        for mode in active_modes:
            label = MODE_LABELS[mode]
            t_mode = time.time()
            try:
                print(f"    {label:<13s} generating...", flush=True)
                text, total_tokens = _generate(mode)
                gen_dt = time.time() - t_mode
                think = extract_think(text)
                code, extraction = extract_code_with_info(text)
                think_tokens = count_tokens(think, args.tokenizer)
                entry_found = _entry_point_found(code, args.dataset, entry_point, prob)
                test_detail = ""
                if args.dataset == "livecodebench":
                    test_detail = f" {_lcb_public_test_count(prob)} public cases"
                print(
                    f"    {label:<13s} generated in {gen_dt:.0f}s  "
                    f"think={think_tokens} total={int(total_tokens)}  "
                    f"extraction={extraction['extraction_issue']}  "
                    f"entry={'yes' if entry_found else 'no'}; testing{test_detail}...",
                    flush=True,
                )
                t_test = time.time()
                passed, err = _score(code)
                test_dt = time.time() - t_test
                result = {
                    "pass": passed,
                    "err": err[:200],
                    "think_tokens": think_tokens,
                    "total_tokens": int(total_tokens),
                    "raw_response": text,
                    "extracted_think": think[:500],
                    "extracted_code": code[:500],
                    **extraction,
                    "entry_point_found": entry_found,
                }
                result["failure_type"] = classify_failure(result)
                row[mode] = result
                print(
                    f"    {label:<13s} {'pass' if passed else 'fail'}  "
                    f"test={test_dt:.0f}s total={time.time() - t_mode:.0f}s  "
                    f"failure={result['failure_type']}",
                    flush=True,
                )
            except Exception as e:
                result = {
                    "pass": False,
                    "err": f"gen_error: {e}"[:300],
                    "failure_type": "generation_error",
                    "extraction_method": "not_run",
                    "extraction_issue": "generation_error",
                    "entry_point_found": False,
                }
                row[mode] = result
                print(
                    f"    {label:<13s} error after {time.time() - t_mode:.0f}s: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )

        dt = time.time() - t_prob
        results.append(row)

        def tag(d): return "✓" if d and d.get("pass") else "✗"
        def tt(d):  return d.get("think_tokens", "-") if d else "-"
        err_bits = []
        for m in active_modes:
            d = row.get(m)
            if d and not d.get("pass"):
                e = (d.get("err") or "").strip()
                if e:
                    err_bits.append(f"{m}: {e[:80]}")
        err_str = ("  |  " + " ; ".join(err_bits)) if err_bits else ""
        status_str = "   ".join(
            f"{m}={tag(row.get(m))} ({tt(row.get(m))}tt)" for m in active_modes
        )
        print(
            f"  [{i+1}/{len(problems)}] {task_id:<16s}  "
            f"{status_str}  {dt:.0f}s{err_str}",
            flush=True,
        )

    elapsed = time.time() - t_start

    # ---- Summary ----
    def mean(xs):
        return sum(xs) / max(len(xs), 1)

    mode_rows = {
        mode: [r[mode] for r in results if mode in r and r[mode].get("pass") is not None]
        for mode in active_modes
    }
    mode_summaries: dict = {}

    print(f"\n[3/3] Summary  (n={len(results)}, elapsed={elapsed:.0f}s)")
    print("  " + "-" * 70)
    for mode in active_modes:
        rows = mode_rows[mode]
        if not rows:
            continue
        pass_rate = mean([1.0 if r["pass"] else 0.0 for r in rows])
        think_mean = mean([r.get("think_tokens", 0) for r in rows])
        total_mean = mean([r.get("total_tokens", 0) for r in rows])
        mode_summaries[mode] = {
            "pass_rate": pass_rate,
            "think_tokens_mean": think_mean,
            "total_tokens_mean": total_mean,
        }
        print(f"  {MODE_LABELS[mode]:<13s}:  pass@1 = {pass_rate*100:5.1f}%   "
              f"mean think = {think_mean:6.0f} tok   "
              f"mean total = {total_mean:6.0f} tok")

    if "free" in mode_summaries and "fsm" in mode_summaries:
        acc_delta = (mode_summaries["fsm"]["pass_rate"] - mode_summaries["free"]["pass_rate"]) * 100
        compression = (
            mode_summaries["free"]["think_tokens_mean"]
            / max(mode_summaries["fsm"]["think_tokens_mean"], 1)
        )
        print("  " + "-" * 70)
        print(f"  Accuracy delta (FSM − FREE): {acc_delta:+5.1f} pp")
        print(f"  Think-token compression    : {compression:5.2f}×")

    outcome_breakdown = build_outcome_breakdown(results)
    failure_accounting = build_failure_accounting(results, active_modes)
    print_outcome_breakdown(outcome_breakdown)
    print_failure_accounting(failure_accounting)

    # ---- Save ----
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results) + "\n"
    )
    summary = {
        "args": vars(args),
        "modes": active_modes,
        "n": len(results),
        "elapsed_sec": elapsed,
    }
    for mode, mode_summary in mode_summaries.items():
        summary[mode] = mode_summary
    summary["outcome_breakdown"] = outcome_breakdown
    summary["failure_accounting"] = failure_accounting
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_per_problem_report(out / "per_problem.md", results, problems, args)
    print(f"\nSaved → {out / 'results.jsonl'}")
    print(f"Saved → {out / 'summary.json'}")
    print(f"Saved → {out / 'per_problem.md'}")


def _outcome_tag(row: dict) -> str:
    f = row.get("free", {}).get("pass") if "free" in row else None
    s = row.get("fsm",  {}).get("pass") if "fsm"  in row else None
    p = row.get("prompt_terse", {}).get("pass") if "prompt_terse" in row else None

    if p is True and f is False and s is False:
        return "🟡 PROMPT_TERSE-only pass"
    if p is False and f is True and s is True:
        return "🟠 PROMPT_TERSE-only fail"

    if f is True and s is True:
        base = "🟰 both pass"
    elif f is True and s is False:
        base = "🔻 FSM regression"
    elif f is False and s is True:
        base = "🔺 FSM wins"
    elif f is False and s is False:
        base = "❌ both fail"
    elif f is True and s is None:
        base = "free only: ✓"
    elif f is False and s is None:
        base = "free only: ✗"
    elif f is None and s is True:
        base = "fsm only: ✓"
    elif f is None and s is False:
        base = "fsm only: ✗"
    elif p is True:
        base = "prompt_terse only: ✓"
    elif p is False:
        base = "prompt_terse only: ✗"
    else:
        return "—"

    if p is not None and f is not None and s is not None:
        base += f"; prompt_terse={'✓' if p else '✗'}"
    return base


def _write_per_problem_report(path: Path, results: list, problems: list, args) -> None:
    """Write a readable markdown report with one section per problem."""
    prob_by_id = {p["task_id"]: p for p in problems}
    lines: list[str] = []

    modes_present = [m for m in MODE_ORDER if any(m in r for r in results)]
    mode_names = ", ".join(MODE_LABELS[m] for m in modes_present)
    lines.append(f"# Per-problem mode comparison — {args.dataset}, n={len(results)}\n")
    lines.append(f"**Model:** `{args.model}`  ")
    lines.append(f"**Modes:** {mode_names}  ")
    lines.append(f"**Max tokens:** {args.max_tokens}  ")
    lines.append(f"**Grammar:** `{args.grammar_file}`  \n")
    lines.append("Outcome legend: 🔺 FSM wins on a problem FREE got wrong · "
                 "🔻 FSM regresses vs FREE · 🟰 both pass · ❌ both fail · "
                 "🟡/🟠 PROMPT_TERSE-only divergence.\n")
    lines.append("---\n")

    for row in results:
        tid = row["task_id"]
        prob = prob_by_id.get(tid, {})
        lines.append(f"## {tid} — {_outcome_tag(row)}\n")

        prompt = prob.get("prompt") or prob.get("question_content", "")
        if prompt:
            snippet = prompt.strip()
            if len(snippet) > 400:
                snippet = snippet[:400].rstrip() + " …"
            lines.append("**Problem:**\n")
            lines.append("```text")
            lines.append(snippet)
            lines.append("```\n")

        def _section(name: str, d: dict) -> list[str]:
            out_ = [f"### {name}  {'✓ pass' if d.get('pass') else '✗ fail'}  "
                    f"(think: {d.get('think_tokens','-')} tok, "
                    f"total: {d.get('total_tokens','-')} tok)\n"]
            meta = []
            if d.get("failure_type") and d.get("failure_type") != "pass":
                meta.append(f"failure: `{d['failure_type']}`")
            if d.get("extraction_issue") and d.get("extraction_issue") != "none":
                meta.append(f"extraction: `{d['extraction_issue']}`")
            if d.get("extraction_method"):
                meta.append(f"method: `{d['extraction_method']}`")
            if d.get("entry_point_found") is False:
                meta.append("entry point: `not found`")
            if meta:
                out_.append("_meta:_ " + " · ".join(meta) + "\n")
            if d.get("err"):
                out_.append("_error:_")
                out_.append("```text")
                out_.append(d["err"].strip())
                out_.append("```\n")
            if d.get("extracted_think"):
                out_.append("_think:_")
                out_.append("```text")
                out_.append(d["extracted_think"].strip())
                out_.append("```\n")
            if d.get("extracted_code"):
                out_.append("_code:_")
                out_.append("```python")
                out_.append(d["extracted_code"].strip())
                out_.append("```\n")
            return out_

        for mode in MODE_ORDER:
            if mode in row:
                lines.extend(_section(MODE_LABELS[mode], row[mode]))

        # Per-problem compression (if both ran)
        f = row.get("free", {})
        s = row.get("fsm", {})
        ft, st = f.get("think_tokens"), s.get("think_tokens")
        if ft and st:
            ratio = ft / max(st, 1)
            lines.append(f"_think compression_: **{ratio:.2f}×** "
                         f"({ft} → {st} tokens)\n")

        lines.append("---\n")

    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
