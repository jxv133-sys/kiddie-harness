"""Loads and resolves harness configuration from config/default.yaml."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


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

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        path = path or DEFAULT_CONFIG_PATH
        raw = yaml.safe_load(path.read_text())
        ollama = raw["ollama"]
        generation = raw["generation"]
        retries = raw["retries"]
        workspace = raw["workspace"]
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
        )

    def with_overrides(
        self,
        *,
        model: str | None = None,
        host: str | None = None,
        max_fix_attempts: int | None = None,
        timeout_seconds: int | None = None,
    ) -> Config:
        return dataclasses.replace(
            self,
            model=model or self.model,
            ollama_host=host or self.ollama_host,
            max_fix_attempts=max_fix_attempts or self.max_fix_attempts,
            timeout_seconds=timeout_seconds or self.timeout_seconds,
        )
