"""Shared test double for OllamaClient.

No real network/model calls -- keeps tests fast and deterministic while
exercising the exact same orchestrator/step code paths a real
Ollama-backed run would use.
"""

from __future__ import annotations

import threading
import time
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

    Each queued item is either a plain `str` (done_reason=None), a
    `(text, done_reason)` tuple (to simulate a truncated response), or an
    `Exception` instance, which is raised on that call -- lets a test
    simulate the model becoming unreachable partway through a run.
    """

    def __init__(
        self,
        responses: list[str | tuple[str, str]],
        *,
        delay: float = 0.0,
        first_call_barrier: threading.Barrier | None = None,
        host: str = "http://fake",
    ):
        self._responses = list(responses)
        self._delay = delay
        self._barrier = first_call_barrier
        self._seen_first = False
        self.host = host
        self.calls: list[str] = []
        self.max_tokens_calls: list[int] = []
        self.temperature_calls: list[float] = []

    def generate(self, prompt: str, *, system=None, json_schema=None, temperature=0.2, max_tokens=2048):
        if self._barrier is not None and not self._seen_first:
            self._seen_first = True
            self._barrier.wait()
        if self._delay:
            time.sleep(self._delay)
        self.calls.append(prompt)
        self.max_tokens_calls.append(max_tokens)
        self.temperature_calls.append(temperature)
        if not self._responses:
            raise AssertionError("FakeClient ran out of queued responses")
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
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
    critic_enabled: bool = False,
) -> Config:
    # critic_enabled defaults off here (unlike config/default.yaml, where
    # it's on): the critic is its own concern with its own tests
    # (test_critic.py); leaving it off by default keeps every other
    # FakeClient-based test's queued response count exactly what it was
    # before the critic existed, instead of forcing every test file in
    # the suite to account for its extra call.
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
        critic_enabled=critic_enabled,
    )
