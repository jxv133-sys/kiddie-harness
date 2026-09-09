"""Atomic test-writing step: one file's spec -> a pytest test file for it.

Kept as its own step and its own prompt, never combined with the
implementation call -- asking a small model to write a file and its own
test in one shot is exactly the kind of compound instruction that causes
unreliable output.
"""

from __future__ import annotations

from pathlib import Path

from ..llm_client import OllamaClient
from ..postprocess import strip_code_fences

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_TESTGEN_TEMPLATE = (_PROMPTS_DIR / "testgen.md").read_text()


def generate_test_file(client: OllamaClient, instruction: str, *, temperature: float, max_tokens: int) -> str:
    """Instruction describing the module under test -> full test file content.

    Mirrors codegen.generate_file's signature exactly so the same bounded
    generate/verify/fix loop can drive either one.
    """
    prompt = _TESTGEN_TEMPLATE.format(instruction=instruction)
    response = client.generate(prompt, temperature=temperature, max_tokens=max_tokens)
    return strip_code_fences(response.text)
