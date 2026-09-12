"""Atomic critic step: one file's spec + contents -> does it hold up?

This is the one verification step that is *not* real tooling -- everything
in `harness.steps.verify` is a compiler/linter/interpreter result; this is
another LLM's opinion. That is a deliberate, narrow exception to this
project's own rule (see CLAUDE.md), made because "does this file actually
do what its spec asked for" has no deterministic check -- only a human, or
a model, can judge it. Because it's an opinion and can be wrong, the
orchestrator never lets it hard-fail a file that already compiles, lints,
and imports clean: see `FileRunResult.spec_flagged` in orchestrator.py.

Kept atomic like every other step here: one file's spec and contents in,
one verdict out. Never sees sibling files, the overall goal, or anything
else -- the same narrow-context discipline as codegen and the fix loop.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from ..llm_client import OllamaClient, OllamaError

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_CRITIC_TEMPLATE = (_PROMPTS_DIR / "critic.md").read_text()

CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "follows_spec": {"type": "boolean"},
        "issues": {"type": "string"},
    },
    "required": ["follows_spec", "issues"],
}


@dataclasses.dataclass
class CriticResult:
    follows_spec: bool
    issues: str


def critique_file(
    client: OllamaClient,
    spec: str,
    code: str,
    path: str,
    *,
    temperature: float,
    max_tokens: int,
) -> CriticResult:
    """Judge `code` against `spec`. Fails open: a call that errors or
    comes back unparseable is treated as "follows the spec" -- an
    unreliable opinion should never be the thing that sinks an otherwise
    working file, only a clear, parsed "no" should."""
    prompt = _CRITIC_TEMPLATE.format(path=path, spec=spec, code=code)
    try:
        response = client.generate(
            prompt, json_schema=CRITIC_SCHEMA, temperature=temperature, max_tokens=max_tokens
        )
        data = json.loads(response.text)
        return CriticResult(
            follows_spec=bool(data["follows_spec"]), issues=str(data.get("issues") or "")
        )
    except (OllamaError, json.JSONDecodeError, KeyError, TypeError):
        return CriticResult(follows_spec=True, issues="")
