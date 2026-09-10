"""Deterministic cleanup of raw LLM code output.

Small models are asked to return *only* source code, but they routinely
ignore that: they wrap the answer in a markdown fence, add a stray
sentence before or after it, or -- for reasoning models -- emit a whole
chain-of-thought before the code. Rather than trusting the prompt to
prevent that, we strip it in code so a cosmetic slip never counts as a
failure on its own. If the cleaned result still doesn't compile, that's a
normal verifier failure and goes through the regular fix loop.

The one thing we will not do is invent code: every transformation here
either drops a delimited wrapper (a ``<think>`` block, a markdown fence)
or picks one already-delimited fenced block out of surrounding prose.
"""

from __future__ import annotations

import re

# A whole response that is exactly one fenced block (optionally with a
# language tag), and nothing else.
_WHOLE_FENCE_RE = re.compile(
    r"^\s*```[a-zA-Z0-9_+-]*\n(.*?)\n?```\s*$",
    re.DOTALL,
)

# Any fenced block appearing somewhere inside a larger response.
_ANY_FENCE_RE = re.compile(
    r"```[a-zA-Z0-9_+-]*\n(.*?)\n?```",
    re.DOTALL,
)

# Reasoning models emit chain-of-thought wrapped in <think>...</think>.
# Depending on the model/template, the opening tag is sometimes consumed
# by Ollama and only a bare closing </think> reaches us, so match either
# a full pair or an orphan close and drop everything up to the last one.
_THINK_RE = re.compile(r"(?is)^.*?</think\s*>")


def strip_code_fences(text: str) -> str:
    """Reduce a raw model response to just the file body it was meant to be.

    Order matters:
    1. Drop a leading reasoning block (``<think>...</think>`` or an orphan
       trailing ``</think>``).
    2. If what remains is exactly one fenced block, return its body.
    3. If fenced blocks appear amid prose, return the last block's body
       (a model that "reconsiders" puts its final answer last).
    4. Otherwise return the trimmed text and let the verifier judge it.
    """
    stripped = text.strip()

    think_match = _THINK_RE.match(stripped)
    if think_match:
        stripped = stripped[think_match.end() :].strip()

    whole = _WHOLE_FENCE_RE.match(stripped)
    if whole:
        return whole.group(1)

    blocks = _ANY_FENCE_RE.findall(stripped)
    if blocks:
        return blocks[-1].strip("\n")

    return stripped
