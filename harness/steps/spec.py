"""Atomic per-file spec step: one file's purpose -> a short bullet spec.

Kept separate from code generation on purpose: asking a small model to
plan what a file should contain *and* write the code in the same call is
exactly the kind of compound instruction that causes unreliable output.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..llm_client import OllamaClient
from ..postprocess import strip_reasoning
from .plan import FileTask

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_SPEC_TEMPLATE = (_PROMPTS_DIR / "spec.md").read_text()


def looks_like_a_spec(text: str) -> bool:
    """True if at least one line is a real bullet point, matching the
    prompt's own explicit contract ("each starting with '-'"). Guards
    against the same "vacuous pass" failure mode verify.py's
    html_check/css_check/js_check catch downstream: a model that
    responds with disclaimer or commentary prose instead of a spec
    produces text no deterministic check would ever flag once it's
    baked into the codegen instruction -- this catches it here instead,
    before it can quietly corrupt everything built from it."""
    return any(line.strip().startswith("-") for line in text.splitlines())


def write_spec(
    client: OllamaClient,
    goal: str,
    file_task: FileTask,
    *,
    temperature: float,
    max_tokens: int,
    on_chunk: Callable[[str], None] | None = None,
) -> str:
    """Goal + one file's purpose -> a short bullet-point spec for that file only."""
    prompt = _SPEC_TEMPLATE.format(goal=goal, path=file_task.path, purpose=file_task.purpose)
    response = client.generate(
        prompt, temperature=temperature, max_tokens=max_tokens, on_chunk=on_chunk
    )
    return strip_reasoning(response.text)
