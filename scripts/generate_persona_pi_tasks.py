#!/usr/bin/env python3
"""Generate persona-conditioned Pi task JSONL with an LLM.

This turns persona rows, such as nvidia/Nemotron-Personas-USA records, into
single-turn coding-agent requests. The output is directly consumable by
``scripts/generate_pi_rpc_traces.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET = "nvidia/Nemotron-Personas-USA"
DEFAULT_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
DEFAULT_PROVIDER = "openrouter"
DEFAULT_INTENTS = "build,how_to,debug,design,automation,review,data_transform"
DEFAULT_LANGUAGES = "python,javascript,typescript,cpp,shell,mixed,none"
TASK_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class PersonaRecord:
    persona_id: str
    row: dict[str, Any]


def slug(value: str, fallback: str = "task", limit: int = 120) -> str:
    cleaned = TASK_ID_RE.sub("_", value.strip()).strip("._-").lower()
    return cleaned[:limit] or fallback


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield line_no, row


def row_persona_id(row: dict[str, Any], fallback: str) -> str:
    for key in ("uuid", "persona_id", "id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return slug(value, fallback=fallback, limit=48)
    return fallback


def reservoir_sample(rows: Iterable[dict[str, Any]], *, sample_size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        if index <= sample_size:
            sample.append(row)
            continue
        replace_at = rng.randint(1, index)
        if replace_at <= sample_size:
            sample[replace_at - 1] = row
    return sample


def load_personas_from_file(path: Path, *, sample_size: int, seed: int) -> list[PersonaRecord]:
    sampled = reservoir_sample((row for _, row in read_jsonl(path)), sample_size=sample_size, seed=seed)
    return [
        PersonaRecord(persona_id=row_persona_id(row, f"persona_{idx:06d}"), row=row)
        for idx, row in enumerate(sampled, 1)
    ]


def load_personas_from_dataset(
    dataset: str,
    *,
    split: str,
    sample_size: int,
    seed: int,
    max_source_rows: int | None,
) -> list[PersonaRecord]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "datasets is required for --dataset. Run with: "
            "uv run --with datasets --with huggingface-hub python scripts/generate_persona_pi_tasks.py ..."
        ) from exc

    stream = load_dataset(dataset, split=split, streaming=True)

    def iter_rows() -> Iterable[dict[str, Any]]:
        for index, row in enumerate(stream, 1):
            yield dict(row)
            if max_source_rows is not None and index >= max_source_rows:
                break

    sampled = reservoir_sample(iter_rows(), sample_size=sample_size, seed=seed)
    return [
        PersonaRecord(persona_id=row_persona_id(row, f"persona_{idx:06d}"), row=row)
        for idx, row in enumerate(sampled, 1)
    ]


def text_field(row: dict[str, Any], key: str, max_chars: int = 700) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, list):
        text = ", ".join(str(item) for item in value[:20])
    else:
        text = str(value)
    text = " ".join(text.split())
    if not text:
        return None
    return text[:max_chars]


def persona_brief(persona: PersonaRecord) -> dict[str, Any]:
    row = persona.row
    return {
        "persona_id": persona.persona_id,
        "professional_persona": text_field(row, "professional_persona", 900),
        "persona": text_field(row, "persona", 700),
        "occupation": text_field(row, "occupation", 120),
        "skills_and_expertise": text_field(row, "skills_and_expertise", 900),
        "career_goals_and_ambitions": text_field(row, "career_goals_and_ambitions", 700),
        "hobbies_and_interests": text_field(row, "hobbies_and_interests", 500),
    }


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def generation_messages(
    persona: PersonaRecord,
    *,
    intents: list[str],
    languages: list[str],
) -> list[dict[str, str]]:
    system = (
        "You generate realistic single-turn requests for a coding agent. "
        "Given a persona, create one concrete coding, automation, debugging, design, review, or how-to task "
        "that plausibly relates to the person's work, goals, hobbies, or practical needs. "
        "Return only a JSON object. Do not include markdown. Do not mention private demographics unless directly relevant. "
        "Keep the task CPU-only, small enough for one agent episode, and free of paid APIs or external services."
    )
    user = {
        "persona": persona_brief(persona),
        "allowed_intents": intents,
        "allowed_languages": languages,
        "required_json_schema": {
            "title": "short task title",
            "intent": intents,
            "language": languages,
            "needs_workspace": "boolean; true if the agent should create/edit files in an empty directory",
            "user_request": "the exact single user message to send to the coding agent",
            "expected_artifacts": "list of filenames/directories if files should be created, otherwise []",
            "verify_commands": "list of simple local commands if applicable, otherwise []",
            "difficulty": ["easy", "medium"],
        },
        "style_rules": [
            "The user request should sound like a real user asking for help, not a benchmark prompt.",
            "If needs_workspace is true, ask for a small complete artifact with README and tests or a smoke command.",
            "If needs_workspace is false, ask for concrete advice, code snippets, a debugging plan, or a review.",
            "Avoid huge projects, network services, credentials, scraping, CUDA/GPU, mobile apps, or deployment.",
            "Prefer Python, JavaScript/TypeScript, shell, or small C++ when code is needed.",
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def openrouter_chat(
    *,
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    timeout_sec: float,
) -> str:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
    return payload["choices"][0]["message"]["content"]


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object")
    return payload


def deterministic_task(persona: PersonaRecord, *, intents: list[str], languages: list[str], index: int) -> dict[str, Any]:
    brief = persona_brief(persona)
    occupation = brief.get("occupation") or "practical user"
    skill_text = brief.get("skills_and_expertise") or brief.get("professional_persona") or "day-to-day work"
    intent = intents[index % len(intents)]
    language = languages[index % len(languages)]
    needs_workspace = intent in {"build", "automation", "data_transform"}
    if needs_workspace:
        request = (
            f"I work as a {occupation} and need a small {language} tool related to {skill_text}. "
            "Build it from this empty folder. Keep it simple, include a README, and add a smoke test or a few tests I can run locally."
        )
        artifacts = ["README.md", "tests/"]
        commands = ["python3 -m pytest -q"] if language == "python" else []
    else:
        request = (
            f"I work as a {occupation}. Give me concrete coding-agent guidance for a {language} task related to {skill_text}. "
            "Keep it practical, cite assumptions, and include a small example if helpful."
        )
        artifacts = []
        commands = []
    return {
        "title": f"{occupation} {intent}",
        "intent": intent,
        "language": language,
        "needs_workspace": needs_workspace,
        "user_request": request,
        "expected_artifacts": artifacts,
        "verify_commands": commands,
        "difficulty": "easy",
    }


def normalize_generated_task(payload: dict[str, Any], *, intents: list[str], languages: list[str]) -> dict[str, Any]:
    intent = str(payload.get("intent") or "build").strip()
    if intent not in intents:
        intent = "build" if "build" in intents else intents[0]
    language = str(payload.get("language") or "python").strip()
    if language not in languages:
        language = "python" if "python" in languages else languages[0]
    user_request = str(payload.get("user_request") or "").strip()
    if not user_request:
        raise ValueError("generated task missing user_request")
    return {
        "title": str(payload.get("title") or f"{intent} task").strip(),
        "intent": intent,
        "language": language,
        "needs_workspace": bool(payload.get("needs_workspace", intent in {"build", "automation", "data_transform"})),
        "user_request": user_request,
        "expected_artifacts": list(payload.get("expected_artifacts") or []),
        "verify_commands": list(payload.get("verify_commands") or []),
        "difficulty": str(payload.get("difficulty") or "medium"),
    }


def pi_prompt(task: dict[str, Any]) -> str:
    request = task["user_request"].strip()
    if task["needs_workspace"]:
        return (
            f"{request}\n\n"
            "You are starting in an empty directory. If you create files, keep the project small and CPU-only. "
            "Do not ask follow-up questions; make reasonable assumptions and finish with the commands I should run."
        )
    return (
        f"{request}\n\n"
        "Do not ask follow-up questions; make reasonable assumptions. If you inspect or create files, keep it minimal and CPU-only."
    )


def task_row(
    persona: PersonaRecord,
    generated: dict[str, Any],
    *,
    index: int,
    root_dir: Path,
    source: str,
    generator_model: str,
    intents: list[str] | None = None,
    languages: list[str] | None = None,
) -> dict[str, Any]:
    generated = normalize_generated_task(
        generated,
        intents=intents or split_csv(DEFAULT_INTENTS),
        languages=languages or split_csv(DEFAULT_LANGUAGES),
    )
    persona_part = slug(persona.persona_id, fallback=f"persona_{index:06d}", limit=16)
    task_id = slug(
        f"persona_{index:06d}_{persona_part}_{generated['intent']}_{generated['language']}",
        fallback=f"persona_{index:06d}",
    )
    cwd = (root_dir / task_id).resolve()
    return {
        "task_id": task_id,
        "cwd": str(cwd),
        "prompt": pi_prompt(generated),
        "source": source,
        "persona_id": persona.persona_id,
        "persona_summary": persona_brief(persona),
        "intent": generated["intent"],
        "language": generated["language"],
        "needs_workspace": generated["needs_workspace"],
        "expected_artifacts": generated["expected_artifacts"],
        "verify_commands": generated["verify_commands"],
        "difficulty": generated["difficulty"],
        "generator_model": generator_model,
        "verifiable": bool(generated["verify_commands"]),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def create_workspaces(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        cwd = Path(row["cwd"])
        cwd.mkdir(parents=True, exist_ok=True)
        count += 1
    return count


def generate_rows(args: argparse.Namespace, personas: list[PersonaRecord]) -> list[dict[str, Any]]:
    intents = split_csv(args.intents)
    languages = split_csv(args.languages)
    rows: list[dict[str, Any]] = []
    api_key = os.environ.get(args.api_key_env)
    if args.provider == "openrouter" and not args.dry_run and not api_key:
        raise SystemExit(f"{args.api_key_env} is not set")

    for index, persona in enumerate(personas, 1):
        if args.dry_run:
            generated = deterministic_task(persona, intents=intents, languages=languages, index=index)
        else:
            assert api_key is not None
            messages = generation_messages(persona, intents=intents, languages=languages)
            last_error: Exception | None = None
            for attempt in range(args.retries + 1):
                try:
                    content = openrouter_chat(
                        model=args.model,
                        messages=messages,
                        api_key=api_key,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        json_mode=not args.no_json_mode,
                        timeout_sec=args.request_timeout_sec,
                    )
                    generated = normalize_generated_task(extract_json_object(content), intents=intents, languages=languages)
                    break
                except Exception as exc:  # noqa: BLE001 - report retryable API/parsing errors.
                    last_error = exc
                    if attempt >= args.retries:
                        raise
                    time.sleep(args.retry_sleep_sec * (attempt + 1))
            else:
                raise RuntimeError(last_error)
        row = task_row(
            persona,
            generated,
            index=index + args.start_index,
            root_dir=args.root_dir.expanduser(),
            source=args.source,
            generator_model=args.model,
            intents=intents,
            languages=languages,
        )
        rows.append(row)
        print(
            json.dumps(
                {
                    "task_id": row["task_id"],
                    "intent": row["intent"],
                    "language": row["language"],
                    "needs_workspace": row["needs_workspace"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate persona-conditioned Pi task JSONL.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--personas-file", type=Path, help="Local JSONL persona rows.")
    source.add_argument("--dataset", default=None, help=f"HF dataset id, e.g. {DEFAULT_DATASET}.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-source-rows", type=int, default=None)
    parser.add_argument("--root-dir", type=Path, default=Path("/root/etpi-persona-workspaces"))
    parser.add_argument("--out", type=Path, default=Path("data/pi_tasks/persona_tasks.jsonl"))
    parser.add_argument("--create-workspaces", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--source", default="nemotron_personas_usa_llm_v1")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, choices=["openrouter"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--request-timeout-sec", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep-sec", type=float, default=2.0)
    parser.add_argument("--no-json-mode", action="store_true")
    parser.add_argument("--intents", default=DEFAULT_INTENTS)
    parser.add_argument("--languages", default=DEFAULT_LANGUAGES)
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic local task synthesis instead of an LLM.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.personas_file:
        personas = load_personas_from_file(args.personas_file, sample_size=args.sample_size, seed=args.seed)
    else:
        personas = load_personas_from_dataset(
            args.dataset or DEFAULT_DATASET,
            split=args.split,
            sample_size=args.sample_size,
            seed=args.seed,
            max_source_rows=args.max_source_rows,
        )

    rows = generate_rows(args, personas)
    write_jsonl(args.out, rows)
    created = create_workspaces(rows) if args.create_workspaces else 0
    print(
        json.dumps(
            {
                "out": str(args.out),
                "personas": len(personas),
                "tasks": len(rows),
                "workspaces_created": created,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
