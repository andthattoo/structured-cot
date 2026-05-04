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
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET = "nvidia/Nemotron-Personas-USA"
DEFAULT_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
DEFAULT_PROVIDER = "openrouter"
DEFAULT_INTENTS = "build,how_to,debug,design,automation,review,data_transform"
DEFAULT_LANGUAGES = "python,javascript,typescript,cpp,shell,mixed,none"
TASK_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")
WORKSPACE_INTENTS = {"build", "automation", "data_transform"}
INTENT_LANGUAGE_PREFERENCES = {
    "build": ("python", "javascript", "typescript", "cpp", "shell"),
    "automation": ("shell", "python", "javascript", "typescript"),
    "data_transform": ("python", "javascript", "typescript", "shell"),
    "debug": ("typescript", "javascript", "python", "cpp", "shell"),
    "design": ("cpp", "typescript", "python", "mixed", "none"),
    "review": ("mixed", "python", "typescript", "javascript", "cpp"),
    "how_to": ("none", "python", "javascript", "typescript", "shell"),
}
BAD_REQUEST_PHRASES = (
    "i work as a not_in_workforce",
    "related to ",
    "small none tool",
    "for a none task",
    "concrete coding-agent guidance for a",
    "professional_persona",
    "skills_and_expertise",
    "career_goals_and_ambitions",
)
STOPWORDS = {
    "about",
    "also",
    "analytical",
    "and",
    "both",
    "combined",
    "concrete",
    "developed",
    "education",
    "experience",
    "expertise",
    "good",
    "has",
    "have",
    "including",
    "lifelong",
    "methodical",
    "nature",
    "private",
    "professional",
    "projects",
    "public",
    "skills",
    "solid",
    "strong",
    "their",
    "with",
    "work",
    "years",
}
DOMAIN_KEYWORDS = {
    "finance": (
        "budget variance reports",
        {"finance", "financial", "budget", "forecasting", "cash", "board", "revenue", "expense", "accounting"},
    ),
    "civil engineering": (
        "construction material quantities and inspection notes",
        {"civil", "engineer", "engineering", "infrastructure", "construction", "revit", "material", "project"},
    ),
    "inventory": (
        "inventory counts and reorder levels",
        {"inventory", "retail", "cashier", "pantry", "reorder", "stock", "items"},
    ),
    "events": (
        "event schedules, volunteer shifts, and task checklists",
        {"event", "planner", "volunteer", "schedule", "festival", "community"},
    ),
    "arts": (
        "artwork inventory, exhibition planning, and artist notes",
        {"art", "arts", "artist", "watercolor", "exhibition", "gallery", "curating", "museum"},
    ),
    "research": (
        "research notes, citations, and archival records",
        {"research", "archival", "archives", "literary", "history", "humanities", "translation", "genealogy"},
    ),
    "marine science": (
        "species observations and simple ecosystem records",
        {"marine", "ecosystem", "fish", "plankton", "biology", "environmental"},
    ),
    "cooking": (
        "recipe ingredients, pantry inventory, and meal ideas",
        {"cooking", "recipe", "recipes", "culinary", "pantry", "ingredients", "spices"},
    ),
    "home maintenance": (
        "home maintenance tasks, costs, and reminders",
        {"repair", "maintenance", "plumbing", "electrical", "carpentry", "renovation", "household"},
    ),
}
GENERIC_FOCI = (
    "CSV records and summary reports",
    "local checklists and recurring reminders",
    "small project notes and status updates",
    "simple logs with dates, categories, and totals",
)


def progress(iterable, **kwargs):
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        return iterable
    return tqdm(iterable, **kwargs)


@dataclass(frozen=True)
class PersonaRecord:
    persona_id: str
    row: dict[str, Any]


@dataclass(frozen=True)
class TaskTarget:
    intent: str
    language: str
    needs_workspace: bool


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
    shuffle_buffer: int,
) -> list[PersonaRecord]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "datasets is required for --dataset. Run with: "
            "uv run --with datasets --with huggingface-hub python scripts/generate_persona_pi_tasks.py ..."
        ) from exc

    stream = load_dataset(dataset, split=split, streaming=True)

    if max_source_rows is None:
        shuffled = stream.shuffle(seed=seed, buffer_size=max(sample_size, shuffle_buffer))
        sampled = []
        for index, row in enumerate(progress(shuffled, total=sample_size, desc="sample personas"), 1):
            sampled.append(dict(row))
            if index >= sample_size:
                break
        return [
            PersonaRecord(persona_id=row_persona_id(row, f"persona_{idx:06d}"), row=row)
            for idx, row in enumerate(sampled, 1)
        ]

    def iter_rows() -> Iterable[dict[str, Any]]:
        for index, row in enumerate(stream, 1):
            yield dict(row)
            if index >= max_source_rows:
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


def compatible_languages(intent: str, languages: list[str]) -> list[str]:
    allowed = set(languages)
    preferred = [language for language in INTENT_LANGUAGE_PREFERENCES.get(intent, languages) if language in allowed]
    if intent in WORKSPACE_INTENTS:
        preferred = [language for language in preferred if language not in {"none", "mixed"}]
        if preferred:
            return preferred
        fallback = [language for language in languages if language not in {"none", "mixed"}]
        return fallback or [languages[0]]
    if intent == "debug":
        preferred = [language for language in preferred if language != "none"]
        if preferred:
            return preferred
        fallback = [language for language in languages if language != "none"]
        return fallback or [languages[0]]
    return preferred or languages


def generation_target(index: int, *, intents: list[str], languages: list[str]) -> TaskTarget:
    intent = intents[(index - 1) % len(intents)]
    options = compatible_languages(intent, languages)
    cycle = (index - 1) // len(intents)
    language = options[cycle % len(options)]
    return TaskTarget(
        intent=intent,
        language=language,
        needs_workspace=intent in WORKSPACE_INTENTS,
    )


def generation_messages(
    persona: PersonaRecord,
    *,
    intents: list[str],
    languages: list[str],
    target: TaskTarget | None = None,
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
        "target": (
            {
                "intent": target.intent,
                "language": target.language,
                "needs_workspace": target.needs_workspace,
            }
            if target
            else None
        ),
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
            "Do not paste the persona biography into the user request.",
            "Do not use placeholder phrasing like 'related to', occupation codes like 'not_in_workforce', or 'none tool'.",
            "If language is none, needs_workspace must be false and the request must be advice-only.",
            "If needs_workspace is true, choose a real language such as python, javascript, typescript, cpp, or shell.",
            "Make the request concrete: name the input, output, artifact, bug, review target, or decision the user needs.",
        ],
    }
    if target:
        user["style_rules"].extend(
            [
                f"Set intent exactly to {target.intent}.",
                f"Set language exactly to {target.language}.",
                f"Set needs_workspace exactly to {str(target.needs_workspace).lower()}.",
                "Do not change the target fields even if another task type seems easier.",
            ]
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def task_response_schema(*, intents: list[str], languages: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "title",
            "intent",
            "language",
            "needs_workspace",
            "user_request",
            "expected_artifacts",
            "verify_commands",
            "difficulty",
        ],
        "properties": {
            "title": {
                "type": "string",
                "description": "A short, human-readable title for the generated task.",
            },
            "intent": {
                "type": "string",
                "enum": intents,
                "description": "The task category.",
            },
            "language": {
                "type": "string",
                "enum": languages,
                "description": "The primary programming language or none for advice-only tasks.",
            },
            "needs_workspace": {
                "type": "boolean",
                "description": "True when the coding agent should create or edit files in an empty workspace.",
            },
            "user_request": {
                "type": "string",
                "description": "The exact single user message to send to the coding agent.",
            },
            "expected_artifacts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filenames or directories expected from build/edit tasks, otherwise an empty list.",
            },
            "verify_commands": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Simple local commands that can smoke-test the result, otherwise an empty list.",
            },
            "difficulty": {
                "type": "string",
                "enum": ["easy", "medium"],
                "description": "Keep generated tasks modest enough for one Pi episode.",
            },
        },
    }


def openrouter_chat(
    *,
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    structured_schema: dict[str, Any] | None,
    timeout_sec: float,
) -> str:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if structured_schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "persona_coding_task",
                "strict": True,
                "schema": structured_schema,
            },
        }
    elif json_mode:
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
    message = payload["choices"][0].get("message") or {}
    content = message.get("content")
    if content is None:
        content = message.get("reasoning")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"OpenRouter returned empty message content: {json.dumps(payload)[:1000]}")
    return content


def extract_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("LLM response text must be a string")
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


def deterministic_task(
    persona: PersonaRecord,
    *,
    intents: list[str],
    languages: list[str],
    index: int,
    target: TaskTarget | None = None,
) -> dict[str, Any]:
    brief = persona_brief(persona)
    occupation = human_occupation(brief.get("occupation"))
    focus = persona_focus(brief, occupation=occupation, index=index)
    if target is None:
        intent = intents[index % len(intents)]
        language = languages[index % len(languages)]
        needs_workspace = intent in WORKSPACE_INTENTS
    else:
        intent = target.intent
        language = target.language
        needs_workspace = target.needs_workspace
    if needs_workspace and language == "none":
        language = "python" if "python" in languages else next((item for item in languages if item != "none"), "python")
    if needs_workspace:
        subject = f" for my work as a {occupation}" if occupation else ""
        if intent == "data_transform":
            request = (
                f"Please build a small {language} tool{subject} that reads a CSV of {focus}, "
                "prints a cleaned summary, and writes a normalized output file. Include a README, sample CSV, and a smoke test."
            )
        elif intent == "automation" or language == "shell":
            request = (
                f"Please build a small {language} utility{subject} that automates a local checklist for {focus}. "
                "It should run locally, include a README, sample input, and a simple smoke-test command."
            )
        else:
            request = (
                f"Please build a small {language} project{subject} for tracking {focus}. "
                "Include a README, a tiny sample input if useful, and a smoke test or unit test I can run locally."
            )
        artifacts = ["README.md", "tests/"]
        commands = ["python3 -m pytest -q"] if language == "python" else []
    else:
        subject = f" as a {occupation}" if occupation else ""
        language_part = "" if language in {"none", "mixed"} else f" in {language}"
        script_phrase = f" {language}" if language not in {"none", "mixed"} else ""
        if intent == "debug":
            request = (
                f"I need help debugging a small{script_phrase} script{subject} that processes {focus}. "
                "Give me a practical debugging plan, likely failure points, and a tiny example of the kind of test I should add."
            )
        elif intent == "review":
            request = (
                f"I need a code review checklist{language_part}{subject} for a small tool that handles {focus}. "
                "Focus on correctness, edge cases, maintainability, and tests."
            )
        elif intent == "design":
            request = (
                f"I need a lightweight design plan{language_part}{subject} for a local tool around {focus}. "
                "Describe the data model, modules, edge cases, and a minimal test plan."
            )
        else:
            request = (
                f"I need practical coding-agent guidance{subject} for a small local workflow around {focus}. "
                f"Give me a concrete plan{language_part}, include assumptions, and show a small example or checklist I can use."
            )
        artifacts = []
        commands = []
    return {
        "title": f"{occupation or 'persona'} {intent}",
        "intent": intent,
        "language": language,
        "needs_workspace": needs_workspace,
        "user_request": request,
        "expected_artifacts": artifacts,
        "verify_commands": commands,
        "difficulty": "easy",
    }


def human_occupation(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip().replace("_", " ")
    if not text or text.lower() in {"not in workforce", "not_in_workforce", "none", "unknown"}:
        return None
    return text[:80]


def first_sentence(text: str, *, limit: int) -> str:
    text = " ".join(str(text).split())
    if not text:
        return ""
    match = re.search(r"(?<=[.!?])\s+", text)
    if match and match.start() <= limit:
        text = text[: match.start()].strip()
    return text[:limit].rstrip(" ,.;:")


def persona_text_blob(brief: dict[str, Any], *, occupation: str | None = None) -> str:
    parts = [occupation or ""]
    for key in (
        "skills_and_expertise",
        "career_goals_and_ambitions",
        "hobbies_and_interests",
        "professional_persona",
        "persona",
    ):
        value = brief.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(first_sentence(value, limit=500))
    return " ".join(parts).lower().replace("_", " ")


def persona_focus(brief: dict[str, Any], *, occupation: str | None = None, index: int = 0) -> str:
    blob = persona_text_blob(brief, occupation=occupation)
    matched: list[tuple[str, str, int]] = []
    for domain, (focus, keywords) in DOMAIN_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in blob)
        if score:
            matched.append((domain, focus, score))
    if matched:
        matched.sort(key=lambda item: (-item[2], item[0]))
        return matched[0][1]
    return GENERIC_FOCI[index % len(GENERIC_FOCI)]


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
    normalized = {
        "title": str(payload.get("title") or f"{intent} task").strip(),
        "intent": intent,
        "language": language,
        "needs_workspace": bool(payload.get("needs_workspace", intent in WORKSPACE_INTENTS)),
        "user_request": user_request,
        "expected_artifacts": list(payload.get("expected_artifacts") or []),
        "verify_commands": list(payload.get("verify_commands") or []),
        "difficulty": str(payload.get("difficulty") or "medium"),
    }
    if payload.get("generation_error"):
        normalized["generation_error"] = str(payload["generation_error"])
    return normalized


def generated_task_quality_errors(task: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    request = str(task.get("user_request") or "").strip()
    lowered = request.lower()
    if len(request) < 40:
        errors.append("user_request is too short")
    if len(request) > 1200:
        errors.append("user_request is too long")
    if len(request) > 450 and task.get("generation_error"):
        errors.append("fallback user_request is too long")
    for phrase in BAD_REQUEST_PHRASES:
        if phrase in lowered:
            errors.append(f"user_request contains low-quality phrase: {phrase}")
            break
    if re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+\b", request):
        errors.append("user_request appears to include a full persona name")
    language = str(task.get("language") or "")
    needs_workspace = bool(task.get("needs_workspace"))
    if needs_workspace and language == "none":
        errors.append("workspace tasks cannot use language=none")
    if language == "none" and task.get("verify_commands"):
        errors.append("language=none tasks cannot have verify_commands")
    if needs_workspace and not task.get("expected_artifacts"):
        errors.append("workspace tasks should name expected_artifacts")
    if re.search(r"\b(processes|handles|around|tracking) [a-z]+, [a-z]+, and [a-z]+\b", lowered):
        errors.append("user_request has low-signal keyword triple")
    return errors


def target_mismatch_errors(task: dict[str, Any], target: TaskTarget | None) -> list[str]:
    if target is None:
        return []
    errors: list[str] = []
    if task.get("intent") != target.intent:
        errors.append(f"intent does not match target {target.intent}")
    if task.get("language") != target.language:
        errors.append(f"language does not match target {target.language}")
    if bool(task.get("needs_workspace")) != target.needs_workspace:
        errors.append(f"needs_workspace does not match target {target.needs_workspace}")
    return errors


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
        "task_generation_fallback": bool(generated.get("generation_error")),
        "task_generation_error": generated.get("generation_error"),
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


def fallback_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("task_generation_fallback") is True)


def fallback_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return fallback_count(rows) / len(rows)


def enforce_fallback_rate(rows: list[dict[str, Any]], *, max_fallback_rate: float | None) -> None:
    if max_fallback_rate is None:
        return
    rate = fallback_rate(rows)
    if rate > max_fallback_rate:
        raise RuntimeError(
            f"fallback rate {rate:.3f} exceeded --max-fallback-rate {max_fallback_rate:.3f} "
            f"({fallback_count(rows)}/{len(rows)} rows)"
        )


def generate_one_row(
    *,
    args: argparse.Namespace,
    persona: PersonaRecord,
    index: int,
    intents: list[str],
    languages: list[str],
    api_key: str | None,
) -> dict[str, Any]:
    target = generation_target(index, intents=intents, languages=languages) if args.balance_targets else None
    schema_intents = [target.intent] if target else intents
    schema_languages = [target.language] if target else languages
    structured_schema = None
    if not args.no_json_mode and args.structured_output:
        structured_schema = task_response_schema(intents=schema_intents, languages=schema_languages)
    if args.dry_run:
        generated = deterministic_task(persona, intents=intents, languages=languages, index=index, target=target)
    else:
        assert api_key is not None
        messages = generation_messages(persona, intents=schema_intents, languages=schema_languages, target=target)
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
                    structured_schema=structured_schema,
                    timeout_sec=args.request_timeout_sec,
                )
                generated = normalize_generated_task(extract_json_object(content), intents=intents, languages=languages)
                quality_errors = generated_task_quality_errors(generated) + target_mismatch_errors(generated, target)
                if quality_errors:
                    raise ValueError("generated task failed quality gate: " + "; ".join(quality_errors))
                break
            except Exception as exc:  # noqa: BLE001 - report retryable API/parsing errors.
                last_error = exc
                if attempt >= args.retries:
                    if args.fallback_on_error:
                        generated = deterministic_task(
                            persona,
                            intents=intents,
                            languages=languages,
                            index=index,
                            target=target,
                        )
                        generated["generation_error"] = repr(last_error)
                        break
                    raise
                time.sleep(args.retry_sleep_sec * (attempt + 1))
        else:
            raise RuntimeError(last_error)
    return task_row(
        persona,
        generated,
        index=index + args.start_index,
        root_dir=args.root_dir.expanduser(),
        source=args.source,
        generator_model=args.model,
        intents=intents,
        languages=languages,
    )


def print_row_progress(row: dict[str, Any]) -> None:
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


def generate_rows(args: argparse.Namespace, personas: list[PersonaRecord]) -> list[dict[str, Any]]:
    intents = split_csv(args.intents)
    languages = split_csv(args.languages)
    if not intents:
        raise ValueError("--intents must include at least one value")
    if not languages:
        raise ValueError("--languages must include at least one value")
    rows: list[dict[str, Any]] = []
    api_key = os.environ.get(args.api_key_env)
    if args.provider == "openrouter" and not args.dry_run and not api_key:
        raise SystemExit(f"{args.api_key_env} is not set")

    concurrency = max(1, args.concurrency)
    if concurrency == 1:
        for index, persona in enumerate(progress(personas, total=len(personas), desc="generate tasks"), 1):
            row = generate_one_row(
                args=args,
                persona=persona,
                index=index,
                intents=intents,
                languages=languages,
                api_key=api_key,
            )
            rows.append(row)
            print_row_progress(row)
        return rows

    rows_by_index: list[dict[str, Any] | None] = [None] * len(personas)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                generate_one_row,
                args=args,
                persona=persona,
                index=index,
                intents=intents,
                languages=languages,
                api_key=api_key,
            ): index
            for index, persona in enumerate(personas, 1)
        }
        for future in progress(as_completed(futures), total=len(futures), desc="generate tasks"):
            index = futures[future]
            row = future.result()
            rows_by_index[index - 1] = row
            print_row_progress(row)
    return [row for row in rows_by_index if row is not None]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate persona-conditioned Pi task JSONL.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--personas-file", type=Path, help="Local JSONL persona rows.")
    source.add_argument("--dataset", default=None, help=f"HF dataset id, e.g. {DEFAULT_DATASET}.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-source-rows", type=int, default=None)
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=10000,
        help="Streaming shuffle buffer for HF dataset sampling when --max-source-rows is not set.",
    )
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
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent task-generation requests. Keep modest to avoid provider rate limits.",
    )
    parser.add_argument(
        "--max-fallback-rate",
        type=float,
        default=1.0,
        help="Fail after generation if the fraction of deterministic fallback rows exceeds this value.",
    )
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep-sec", type=float, default=2.0)
    parser.set_defaults(structured_output=True)
    parser.add_argument(
        "--structured-output",
        dest="structured_output",
        action="store_true",
        help="Use OpenRouter response_format=json_schema strict structured output. This is the default.",
    )
    parser.add_argument(
        "--no-structured-output",
        dest="structured_output",
        action="store_false",
        help="Use plain JSON mode instead of strict JSON Schema mode.",
    )
    parser.set_defaults(balance_targets=True)
    parser.add_argument(
        "--balance-targets",
        dest="balance_targets",
        action="store_true",
        help="Assign balanced intent/language targets to each persona. This is the default.",
    )
    parser.add_argument(
        "--no-balance-targets",
        dest="balance_targets",
        action="store_false",
        help="Let the task generator choose intent/language freely.",
    )
    parser.set_defaults(fallback_on_error=True)
    parser.add_argument(
        "--fallback-on-error",
        dest="fallback_on_error",
        action="store_true",
        help="Emit deterministic local tasks when LLM generation fails after retries. This is the default.",
    )
    parser.add_argument(
        "--no-fallback-on-error",
        dest="fallback_on_error",
        action="store_false",
        help="Fail the whole run if LLM generation fails after retries.",
    )
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
            shuffle_buffer=args.shuffle_buffer,
        )

    rows = generate_rows(args, personas)
    enforce_fallback_rate(rows, max_fallback_rate=args.max_fallback_rate)
    write_jsonl(args.out, rows)
    created = create_workspaces(rows) if args.create_workspaces else 0
    print(
        json.dumps(
            {
                "out": str(args.out),
                "personas": len(personas),
                "tasks": len(rows),
                "fallback": fallback_count(rows),
                "fallback_rate": fallback_rate(rows),
                "workspaces_created": created,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
