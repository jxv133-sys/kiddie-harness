"""Shared test double for OllamaClient.

No real network/model calls -- keeps tests fast and deterministic while
exercising the exact same orchestrator/step code paths a real
Ollama-backed run would use.
"""

from __future__ import annotations

from pathlib import Path

from harness.config import Config


class FakeResponse:
    def __init__(self, text: str, done_reason: str | None = None):
        self.text = text
        self.done_reason = done_reason
        self.raw: dict = {"done_reason": done_reason} if done_reason else {}

    @property
    def truncated(self) -> bool:
        return self.done_reason == "length"


class FakeClient:
    """Returns queued responses in order, one per `generate` call.

    Each queued item is either a plain `str` (done_reason=None) or a
    `(text, done_reason)` tuple, so tests can simulate a truncated
    response (`done_reason="length"`) without touching real network code.
    """

    def __init__(self, responses: list[str | tuple[str, str]]):
        self._responses = list(responses)
        self.calls: list[str] = []
        self.max_tokens_calls: list[int] = []

    def generate(self, prompt: str, *, system=None, json_schema=None, temperature=0.2, max_tokens=2048):
        self.calls.append(prompt)
        self.max_tokens_calls.append(max_tokens)
        if not self._responses:
            raise AssertionError("FakeClient ran out of queued responses")
        item = self._responses.pop(0)
        if isinstance(item, tuple):
            text, done_reason = item
            return FakeResponse(text, done_reason=done_reason)
        return FakeResponse(item)


def make_config(
    tmp_path: Path,
    max_fix_attempts: int = 3,
    max_total_iterations: int = 25,
    max_tokens: int = 512,
    max_tokens_ceiling: int = 4096,
) -> Config:
    return Config(
        ollama_host="http://unused",
        model="fake-model",
        timeout_seconds=1,
        temperature=0.2,
        max_tokens=max_tokens,
        max_tokens_ceiling=max_tokens_ceiling,
        max_fix_attempts=max_fix_attempts,
        max_total_iterations=max_total_iterations,
        workspace_root=tmp_path / "workspace",
    )
