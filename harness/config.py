"""Loads and resolves harness configuration from config/default.yaml."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


#: What an endpoint is for, see orchestrator.partition_clients_by_role.
#: "balanced" (the default) means "no opinion" -- an endpoint tagged this
#: way (or every endpoint, if none are tagged at all) does everything,
#: exactly like before this existed. "overflow" is for a slower/weaker
#: endpoint that should sit out unless it can genuinely help: it never
#: becomes the plan/critic client, and in the per-file dispatch loop it
#: only claims a file once every non-"overflow" endpoint is already busy
#: building something else -- a solo goal (or the first file of any
#: goal) always goes to the faster/preferred endpoint(s) alone.
ROLES = ("smart", "quick", "balanced", "overflow")


@dataclasses.dataclass(frozen=True)
class Endpoint:
    """One Ollama backend the multi-file loop can dispatch a file to."""

    host: str
    model: str
    timeout_seconds: int
    # "smart" (bigger/slower/more careful -- plan, critic, integration
    # fixes), "quick" (smaller/faster -- the high-volume per-file spec/
    # codegen/fix grind), "balanced" (does either), or "overflow" (only
    # the per-file grind, and only once every other endpoint is busy --
    # see ROLES above).
    role: str = "balanced"


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
    # Whether a generated file's own critic check (does it hold up against
    # its spec?) runs at all. On by default; the fastest way to turn it
    # off is per-run (--no-critic / the GUI's settings screen), not
    # editing the yaml.
    critic_enabled: bool = True
    # Whether the whole-project review (steps/super_review.py) runs once
    # every file is built -- a second model's opinion confirms a finding
    # before it's reported, same fails-open, advisory-only contract as
    # critic_enabled. Off by default: unlike the per-file critic, this is
    # a new, heavier check (up to two extra calls per finding), opt-in
    # until proven.
    super_review_enabled: bool = False
    # Once a file's fix loop reaches this many attempts, an idle endpoint
    # tagged "smart" or "balanced" may start its own independent attempt
    # at the same file in parallel (see orchestrator._generate_files) --
    # whichever finishes first wins. 0 disables this entirely.
    branch_after_fixes: int = 0

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
                role=e.get("role") or "balanced",
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
            critic_enabled=bool((raw.get("critic") or {}).get("enabled", True)),
            super_review_enabled=bool((raw.get("super_review") or {}).get("enabled", False)),
            branch_after_fixes=int(retries.get("branch_after_fixes", 0)),
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
        critic_enabled: bool | None = None,
        super_review_enabled: bool | None = None,
        branch_after_fixes: int | None = None,
    ) -> Config:
        # `is None` throughout, not `or` -- an explicit 0 (e.g. "no fix
        # attempts, just report the first failure") is a real, meaningful
        # override, not "unset"; `or` would silently discard it and keep
        # the old value instead.
        return dataclasses.replace(
            self,
            model=model or self.model,
            ollama_host=host or self.ollama_host,
            max_fix_attempts=self.max_fix_attempts if max_fix_attempts is None else max_fix_attempts,
            timeout_seconds=self.timeout_seconds if timeout_seconds is None else timeout_seconds,
            endpoints=self.endpoints if endpoints is None else endpoints,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            max_tokens_ceiling=(
                self.max_tokens_ceiling if max_tokens_ceiling is None else max_tokens_ceiling
            ),
            max_total_iterations=(
                self.max_total_iterations if max_total_iterations is None else max_total_iterations
            ),
            critic_enabled=self.critic_enabled if critic_enabled is None else critic_enabled,
            super_review_enabled=(
                self.super_review_enabled if super_review_enabled is None else super_review_enabled
            ),
            branch_after_fixes=(
                self.branch_after_fixes if branch_after_fixes is None else branch_after_fixes
            ),
        )

    def apply_overrides(self, **overrides) -> None:
        """Same fields and semantics as `with_overrides`, but mutates
        `self` in place instead of returning a new instance.

        `with_overrides` is what every *new* run gets: a fresh, separate
        Config. This is for a run already in progress -- the orchestrator
        loop holds this exact object and reads `config.<field>` fresh at
        every use site (never a snapshot taken once at the start), so
        mutating it here is picked up on the loop's very next read, with
        no extra plumbing to push a replacement Config through a running
        thread. Used by the GUI's pause/resume: settings changed while
        paused take effect the moment the run resumes."""
        updated = self.with_overrides(**overrides)
        for field in dataclasses.fields(self):
            setattr(self, field.name, getattr(updated, field.name))
