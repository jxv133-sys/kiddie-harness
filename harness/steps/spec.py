"""Atomic per-file spec step: one file's purpose -> a short bullet spec.

Kept separate from code generation on purpose: asking a small model to
plan what a file should contain *and* write the code in the same call is
exactly the kind of compound instruction that causes unreliable output.
"""

from __future__ import annotations

from pathlib import Path

from ..llm_client import OllamaClient
from .plan import FileTask

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_SPEC_TEMPLATE = (_PROMPTS_DIR / "spec.md").read_text()


def write_spec(
    client: OllamaClient,
    goal: str,
    file_task: FileTask,
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    """Goal + one file's purpose -> a short bullet-point spec for that file only."""
    prompt = _SPEC_TEMPLATE.format(goal=goal, path=file_task.path, purpose=file_task.purpose)
    response = client.generate(prompt, temperature=temperature, max_tokens=max_tokens)
    return response.text.strip()
