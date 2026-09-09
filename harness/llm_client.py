"""Thin wrapper around the Ollama HTTP API.

Every call here is deliberately single-purpose: one prompt in, one string
(or one schema-constrained JSON object) out. The orchestrator is the only
place that decides what happens next -- this module never loops or retries
on its own, so failures are always visible to the caller.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import requests


class OllamaError(RuntimeError):
    """Raised when the Ollama server can't be reached or returns bad data."""


@dataclasses.dataclass
class LLMResponse:
    text: str
    raw: dict[str, Any]
    done_reason: str | None = None

    @property
    def truncated(self) -> bool:
        """True if Ollama stopped because it ran out of max_tokens, not
        because the model finished -- a distinct failure mode from an
        ordinary syntax error."""
        return self.done_reason == "length"


class OllamaClient:
    def __init__(self, host: str, model: str, timeout_seconds: int = 120):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Run one single-purpose generation call.

        If json_schema is given, Ollama constrains decoding to that schema
        (format compliance guaranteed by the inference layer, not by asking
        nicely in the prompt).
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system
        if json_schema is not None:
            payload["format"] = json_schema

        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout_seconds,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(f"Could not reach Ollama at {self.host}: {exc}") from exc

        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise OllamaError(f"Ollama returned non-JSON response: {resp.text[:200]}") from exc

        text = data.get("response")
        if text is None:
            raise OllamaError(f"Ollama response missing 'response' field: {data}")

        return LLMResponse(text=text, raw=data, done_reason=data.get("done_reason"))

    def generate_json(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """Convenience wrapper for schema-constrained structured steps."""
        response = self.generate(
            prompt,
            system=system,
            json_schema=json_schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            return json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise OllamaError(
                f"Model returned invalid JSON despite schema constraint: {response.text[:200]}"
            ) from exc
