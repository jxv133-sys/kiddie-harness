"""Deterministic cleanup of raw LLM code output.

Small models are asked to return *only* source code, but they occasionally
wrap the answer in a markdown fence or add a stray sentence before/after it
anyway. Rather than trusting the prompt to prevent that, we strip it in code
so a cosmetic slip never counts as a failure on its own -- if the stripped
result still doesn't compile, that's a normal verifier failure and goes
through the regular fix loop.
"""

from __future__ import annotations

import re

_FENCE_RE = re.compile(
    r"^\s*```[a-zA-Z0-9_+-]*\n(.*?)\n?```\s*$",
    re.DOTALL,
)


def strip_code_fences(text: str) -> str:
    """Extract the code body if the whole response is one fenced block.

    If there's no single wrapping fence (e.g. no fence at all, or the model
    added prose around it), return the text unchanged/trimmed and let the
    verifier be the judge -- we do not try to guess at partial extraction.
    """
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    if match:
        return match.group(1)
    return stripped
