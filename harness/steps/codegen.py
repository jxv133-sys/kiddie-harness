"""Atomic code-generation and code-fix LLM calls.

Each function here does exactly one thing and returns exactly one output
contract (a raw file body). Neither function loops, retries, or decides
what happens next -- that's the orchestrator's job.
"""

from __future__ import annotations

from pathlib import Path

from ..llm_client import OllamaClient
from ..postprocess import strip_code_fences

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_CODEGEN_TEMPLATE = (_PROMPTS_DIR / "codegen.md").read_text()
_FIX_TEMPLATE = (_PROMPTS_DIR / "fix.md").read_text()


def generate_file(client: OllamaClient, goal: str, *, temperature: float, max_tokens: int) -> str:
    """Goal -> full file content. One-shot, no prior code, no error context."""
    prompt = _CODEGEN_TEMPLATE.format(goal=goal)
    response = client.generate(prompt, temperature=temperature, max_tokens=max_tokens)
    return strip_code_fences(response.text)


def fix_file(
    client: OllamaClient,
    *,
    code: str,
    error: str,
    stage: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Current file + exact error -> corrected full file. Nothing else in context."""
    prompt = _FIX_TEMPLATE.format(code=code, error=error, stage=stage)
    response = client.generate(prompt, temperature=temperature, max_tokens=max_tokens)
    return strip_code_fences(response.text)
