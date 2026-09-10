"""Atomic planning step: goal -> ordered list of files to build.

This is the one step whose output is structured data rather than code, so
it uses Ollama's JSON-schema constrained decoding: the response is
guaranteed to parse, not just requested to.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from ..llm_client import OllamaClient

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_PLAN_TEMPLATE = (_PROMPTS_DIR / "plan.md").read_text()

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


def plan_files(
    client: OllamaClient, goal: str, *, temperature: float, max_tokens: int
) -> list[FileTask]:
    """Goal -> ordered list of (path, purpose). One-shot, schema-constrained."""
    prompt = _PLAN_TEMPLATE.format(goal=goal)
    response = client.generate(
        prompt,
        json_schema=PLAN_SCHEMA,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    try:
        data = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise PlanError(f"Planner returned invalid JSON despite schema: {response.text[:200]}") from exc

    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise PlanError(f"Planner returned no files: {data}")

    # Weak planners slip in a README, a requirements.txt, a setup.cfg --
    # this harness only generates and verifies Python, so drop anything
    # else here rather than letting codegen try to write Python into it.
    # Repeated paths (also common) would just have the second generation
    # overwrite the first and waste the iteration budget, so keep the
    # first mention of each.
    tasks: list[FileTask] = []
    seen: set[str] = set()
    for f in files:
        path = str(f["path"]).strip()
        if not path.endswith(".py"):
            continue
        if path in seen:
            continue
        seen.add(path)
        tasks.append(FileTask(path=path, purpose=f["purpose"]))

    if not tasks:
        raise PlanError(f"Planner returned no Python files: {data}")

    return tasks
