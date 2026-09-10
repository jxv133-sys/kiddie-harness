"""Atomic planning step: goal -> ordered list of files to build.

This is the one step whose output is structured data rather than code.
Attempt 0 uses Ollama's JSON-schema constrained decoding, so the response
is guaranteed to parse. A reasoning model, unable to think first inside
the grammar, sometimes fills it with `{"files": []}`; the retries escalate
temperature with a blunter prompt, and the final attempt drops the schema
entirely so the model can think and answer in its own format (JSON or a
markdown list), which is then parsed here.
"""

from __future__ import annotations

import dataclasses
import json
import posixpath
import re
from pathlib import Path

from ..llm_client import OllamaClient, OllamaError
from ..postprocess import strip_code_fences

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


_PY_NAME_RE = re.compile(r"([A-Za-z_][\w-]*\.py)")


def _parse_free_form(text: str) -> dict:
    """Turn an unconstrained planner response into `{"files": [...]}`.

    Tries, in order: the whole thing as JSON, a `{...}` object embedded in
    prose, then a markdown/numbered list of `*.py` filenames (a reasoning
    model's natural format when it isn't grammar-constrained).
    """
    candidate = strip_code_fences(text)  # also drops a leading <think> block

    for blob in (candidate, _first_brace_object(candidate)):
        if blob is None:
            continue
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "files" in data:
            return data

    lines = candidate.splitlines()
    files: list[dict] = []
    seen: set[str] = set()
    for i, line in enumerate(lines):
        match = _PY_NAME_RE.search(line)
        if not match:
            continue
        name = match.group(1)
        if name in seen:
            continue
        seen.add(name)
        purpose = line[match.end() :].lstrip(" \t:-*").strip(" *`")
        if not purpose:
            # a common layout is `1. **name.py**` then `- Purpose: ...`
            # on one of the next couple of lines
            for follow in lines[i + 1 : i + 3]:
                if _PY_NAME_RE.search(follow):
                    break
                stripped = follow.strip(" \t-*`")
                if ":" in stripped:
                    stripped = stripped.split(":", 1)[1].strip()
                if stripped:
                    purpose = stripped
                    break
        files.append({"path": name, "purpose": purpose or "(purpose not specified by the planner)"})
    if files:
        return {"files": files}

    raise PlanError(f"No file list found in planner response: {text[:200]}")


def _first_brace_object(text: str) -> str | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None


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
        # The last attempt drops the schema so a reasoning model can think
        # first and answer in its own format; earlier attempts stay
        # grammar-constrained and just escalate.
        free_form = attempt == max_attempts - 1 and max_attempts > 1
        prompt = base_prompt if attempt == 0 else base_prompt + _RETRY_SUFFIX
        temp = temperature
        if attempt:
            temp = round(
                min(temperature + _RETRY_TEMPERATURE_STEP * attempt, _RETRY_TEMPERATURE_MAX), 3
            )
        try:
            response = client.generate(
                prompt,
                json_schema=None if free_form else PLAN_SCHEMA,
                temperature=temp,
                max_tokens=max_tokens,
            )
            data = _parse_free_form(response.text) if free_form else json.loads(response.text)
            return _tasks_from_plan(data)
        except (OllamaError, PlanError, json.JSONDecodeError) as exc:
            last_error = str(exc)

    raise PlanError(
        f"Planner produced no usable file list in {max_attempts} attempt(s): {last_error}"
    )
