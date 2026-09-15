"""Whole-project review step: does the finished project actually hold
together, not just each file on its own?

`critic.py` already judges one file against its own spec, but it never
sees any other file -- it can't catch a mismatch between what one file
provides and what another expects (a server that doesn't actually serve
the page it was given, a shared name used two different ways). This is
the same kind of check, one level up: all the files, once, looking for
problems only visible when they're considered together.

Two atomic calls, same shape as critic.py: `find_issues` (one model's
opinion of what's wrong) and `confirm_issue` (a second, independent
model's opinion of whether a specific claimed problem is real). Neither
loops or retries -- that's the orchestrator's job, same rule as every
other step here. Both fail open: an unreliable opinion must never
manufacture a finding -- an error or unparseable response means "found
nothing" / "not confirmed", never the reverse.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path

from ..llm_client import OllamaClient, OllamaError

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_REVIEW_TEMPLATE = (_PROMPTS_DIR / "super_review.md").read_text()
_CONFIRM_TEMPLATE = (_PROMPTS_DIR / "super_review_confirm.md").read_text()

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["file", "description"],
            },
        },
    },
    "required": ["issues"],
}

CONFIRM_SCHEMA = {
    "type": "object",
    "properties": {"confirmed": {"type": "boolean"}},
    "required": ["confirmed"],
}

# All files together can be a lot of text -- capped the same way
# orchestrator._sibling_context caps a single file's dependency context,
# just with more room since this genuinely needs every file, not just a
# few declared dependencies. Files beyond the cap are left out rather
# than truncated mid-file, which would just confuse the model about
# where one file's content ends and the next begins.
_FILES_CHAR_CAP = 20_000


@dataclasses.dataclass
class Issue:
    file: str
    description: str


def _render_files(files: list[tuple[str, str]]) -> str:
    blocks = []
    used = 0
    for path, content in files:
        block = f"### {path}\n{content}"
        if used + len(block) > _FILES_CHAR_CAP:
            break
        used += len(block)
        blocks.append(block)
    return "\n\n".join(blocks)


def find_issues(
    client: OllamaClient,
    goal: str,
    files: list[tuple[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    on_chunk: Callable[[str], None] | None = None,
) -> list[Issue]:
    """All the finished files, once -> a list of whole-project problems.
    Fails open: a call that errors or comes back unparseable is treated
    as "found nothing" -- an unreliable opinion must never invent a
    finding."""
    prompt = _REVIEW_TEMPLATE.format(goal=goal, files=_render_files(files))
    try:
        response = client.generate(
            prompt,
            json_schema=REVIEW_SCHEMA,
            temperature=temperature,
            max_tokens=max_tokens,
            on_chunk=on_chunk,
        )
        data = json.loads(response.text)
        return [
            Issue(file=str(item["file"]), description=str(item["description"]))
            for item in data["issues"]
            if isinstance(item, dict) and "file" in item and "description" in item
        ]
    except (OllamaError, json.JSONDecodeError, KeyError, TypeError):
        return []


def confirm_issue(
    client: OllamaClient,
    files: list[tuple[str, str]],
    issue: Issue,
    *,
    temperature: float,
    max_tokens: int,
    on_chunk: Callable[[str], None] | None = None,
) -> bool:
    """A second, independent opinion on one claimed issue. Fails open
    toward "not confirmed": an unreliable confirm call must never
    manufacture a false positive."""
    prompt = _CONFIRM_TEMPLATE.format(
        file=issue.file, description=issue.description, files=_render_files(files)
    )
    try:
        response = client.generate(
            prompt,
            json_schema=CONFIRM_SCHEMA,
            temperature=temperature,
            max_tokens=max_tokens,
            on_chunk=on_chunk,
        )
        data = json.loads(response.text)
        return bool(data["confirmed"])
    except (OllamaError, json.JSONDecodeError, KeyError, TypeError):
        return False
