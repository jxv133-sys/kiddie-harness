"""Shared test double for OllamaClient.

No real network/model calls -- keeps tests fast and deterministic while
exercising the exact same orchestrator/step code paths a real
Ollama-backed run would use.
"""

from __future__ import annotations

from pathlib import Path

from harness.config import Config


class FakeResponse:
    def __init__(self, text: str):
        self.text = text
        self.raw: dict = {}


class FakeClient:
    """Returns queued responses in order, one per `generate` call."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[str] = []

    def generate(self, prompt: str, *, system=None, json_schema=None, temperature=0.2, max_tokens=2048):
        self.calls.append(prompt)
        if not self._responses:
            raise AssertionError("FakeClient ran out of queued responses")
        return FakeResponse(self._responses.pop(0))


def make_config(
    tmp_path: Path, max_fix_attempts: int = 3, max_total_iterations: int = 25
) -> Config:
    return Config(
        ollama_host="http://unused",
        model="fake-model",
        timeout_seconds=1,
        temperature=0.2,
        max_tokens=512,
        max_fix_attempts=max_fix_attempts,
        max_total_iterations=max_total_iterations,
        workspace_root=tmp_path / "workspace",
    )
