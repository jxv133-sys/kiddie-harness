"""Loads and resolves harness configuration from config/default.yaml."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


@dataclasses.dataclass(frozen=True)
class Endpoint:
    """One Ollama backend the multi-file loop can dispatch a file to."""

    host: str
    model: str
    timeout_seconds: int


@dataclasses.dataclass
class Config:
    ollama_host: str
    model: str
    timeout_seconds: int
    temperature: float
    max_tokens: int
    max_tokens_ceiling: int
    max_fix_attempts: int
    max_total_iterations: int
    workspace_root: Path
    # Extra endpoints for parallel multi-file generation. Empty -> the
    # single `ollama_host`/`model` above is the only backend.
    endpoints: tuple[Endpoint, ...] = ()

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        path = path or DEFAULT_CONFIG_PATH
        raw = yaml.safe_load(path.read_text())
        ollama = raw["ollama"]
        generation = raw["generation"]
        retries = raw["retries"]
        workspace = raw["workspace"]
        endpoints = tuple(
            Endpoint(
                host=e["host"],
                model=e.get("model", ollama["model"]),
                timeout_seconds=e.get("timeout_seconds", ollama["timeout_seconds"]),
            )
            for e in (raw.get("endpoints") or [])
        )
        return cls(
            ollama_host=ollama["host"],
            model=ollama["model"],
            timeout_seconds=ollama["timeout_seconds"],
            temperature=generation["temperature"],
            max_tokens=generation["max_tokens"],
            max_tokens_ceiling=generation["max_tokens_ceiling"],
            max_fix_attempts=retries["max_fix_attempts"],
            max_total_iterations=retries["max_total_iterations"],
            workspace_root=Path(workspace["root"]),
            endpoints=endpoints,
        )

    def resolved_endpoints(self) -> list[Endpoint]:
        """The backends to dispatch files to: the explicit `endpoints`
        list, or the single `ollama_host`/`model` when none are given."""
        return list(self.endpoints) or [
            Endpoint(self.ollama_host, self.model, self.timeout_seconds)
        ]

    def with_overrides(
        self,
        *,
        model: str | None = None,
        host: str | None = None,
        max_fix_attempts: int | None = None,
        timeout_seconds: int | None = None,
        endpoints: tuple[Endpoint, ...] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_tokens_ceiling: int | None = None,
        max_total_iterations: int | None = None,
    ) -> Config:
        return dataclasses.replace(
            self,
            model=model or self.model,
            ollama_host=host or self.ollama_host,
            max_fix_attempts=max_fix_attempts or self.max_fix_attempts,
            timeout_seconds=timeout_seconds or self.timeout_seconds,
            endpoints=self.endpoints if endpoints is None else endpoints,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens or self.max_tokens,
            max_tokens_ceiling=max_tokens_ceiling or self.max_tokens_ceiling,
            max_total_iterations=max_total_iterations or self.max_total_iterations,
        )
