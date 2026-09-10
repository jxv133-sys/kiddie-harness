"""Atomic planning step: goal -> ordered list of files to build.

This is the one step whose output is structured data rather than code, so
every attempt uses Ollama's JSON-schema constrained decoding: the
response is always parseable JSON, never prose. The failure mode that
still exists -- a reasoning model, unable to think first inside the
grammar, fills it with the minimal legal value `{"files": []}` -- is met
with a bounded retry at a rising temperature and a blunter prompt, the
same shape of retry every other step already has.
"""

from __future__ import annotations

import dataclasses
import json
import posixpath
from pathlib import Path

from ..llm_client import OllamaClient, OllamaError

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_PLAN_TEMPLATE = (_PROMPTS_DIR / "plan.md").read_text()

_RETRY_SUFFIX = (
    "\n\nYour previous answer had an empty list. You MUST return at least "
    "one .py file. An empty list is not an acceptable answer."
)
_RETRY_TEMPERATURE_STEP = 0.2
_RETRY_TEMPERATURE_MAX = 0.9

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "purpose": {"type": "string"},
                },
                "required": ["path", "purpose"],
            },
        },
    },
    "required": ["files"],
}


class PlanError(RuntimeError):
    """Raised when the planner's response can't be turned into file tasks."""


@dataclasses.dataclass
class FileTask:
    path: str
    purpose: str


def _tasks_from_plan(data: dict) -> list[FileTask]:
    """Validate and normalise a parsed plan into file tasks.

    Normalisation handles what a weak planner hands back:
     - drop non-Python entries (a README, a requirements.txt) -- this
       harness only generates and verifies Python;
     - flatten any subdirectory path to a bare filename -- every file
       lives in one flat run directory, and a `pkg/core.py` would break
       both its import-check (wrong cwd) and its companion test (wrong
       import path);
     - keep only the first mention of each name -- a repeat would just
       have the second generation overwrite the first and burn budget.
    """
    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise PlanError(f"Planner returned no files: {data}")

    tasks: list[FileTask] = []
    seen: set[str] = set()
    for f in files:
        try:
            name = posixpath.basename(str(f["path"]).strip().replace("\\", "/"))
            purpose = f["purpose"]
        except (TypeError, KeyError) as exc:
            raise PlanError(f"Malformed file entry in plan: {f!r}") from exc
        if not name.endswith(".py") or name in seen:
            continue
        seen.add(name)
        tasks.append(FileTask(path=name, purpose=purpose))

    if not tasks:
        raise PlanError(f"Planner returned no Python files: {data}")
    return tasks


def plan_files(
    client: OllamaClient,
    goal: str,
    *,
    temperature: float,
    max_tokens: int,
    max_attempts: int = 3,
) -> list[FileTask]:
    """Goal -> ordered list of (path, purpose).

    Schema-constrained every attempt; a retry escalates the temperature
    and appends a blunter instruction to get past a reasoning model that
    filled the grammar with an empty list.
    """
    base_prompt = _PLAN_TEMPLATE.format(goal=goal)
    last_error = "no attempts made"

    for attempt in range(max_attempts):
        prompt = base_prompt
        temp = temperature
        if attempt:
            prompt = base_prompt + _RETRY_SUFFIX
            temp = round(
                min(temperature + _RETRY_TEMPERATURE_STEP * attempt, _RETRY_TEMPERATURE_MAX), 3
            )
        try:
            response = client.generate(
                prompt,
                json_schema=PLAN_SCHEMA,
                temperature=temp,
                max_tokens=max_tokens,
            )
            return _tasks_from_plan(json.loads(response.text))
        except (OllamaError, PlanError, json.JSONDecodeError) as exc:
            last_error = str(exc)

    raise PlanError(
        f"Planner produced no usable file list in {max_attempts} attempt(s): {last_error}"
    )
