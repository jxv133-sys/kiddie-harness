"""Turns a run's log.jsonl into something scannable.

Pure log parsing -- no LLM calls, no filesystem writes. Used both by the
`harness inspect` CLI subcommand (after the fact) and at the end of every
`harness run` (live), so both paths render the exact same information the
same way.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

_CALL_EVENTS = {"plan", "spec", "codegen", "fix", "integration_fix"}


@dataclasses.dataclass
class FileSummary:
    path: str
    success: bool
    attempts: int
    truncated: bool
    advisory: bool = False


@dataclasses.dataclass
class IntegrationSummary:
    stage: str
    success: bool


@dataclasses.dataclass
class RunSummary:
    run_id: str
    files: list[FileSummary]
    integration: IntegrationSummary | None
    total_llm_calls: int
    stopped_early: bool
    succeeded: bool
    # False when the log has no terminal event -- the process was killed
    # (Ctrl-C, OOM, timeout) before the run reached a verdict.
    finished: bool = True


def load_run_summary(log_path: Path) -> RunSummary:
    """Reconstruct a RunSummary from a run's log.jsonl."""
    files: dict[str, dict] = {}
    advisory_paths: set[str] = set()
    integration: IntegrationSummary | None = None
    total_llm_calls = 0
    stopped_early = False
    saw_giving_up = False
    run_result: bool | None = None
    finished = False

    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        event = record.get("event")

        if event in _CALL_EVENTS:
            total_llm_calls += 1

        if event in ("codegen", "fix"):
            entry = files.setdefault(record["path"], {"attempts": 0, "truncated": False, "success": False})
            if event == "fix":
                entry["attempts"] += 1
            if record.get("truncated"):
                entry["truncated"] = True
        elif event == "verify":
            entry = files.setdefault(record["path"], {"attempts": 0, "truncated": False, "success": False})
            entry["success"] = record["success"]
        elif event == "advisory_test":
            advisory_paths.add(record["path"])
        elif event == "integration_verify":
            integration = IntegrationSummary(stage=record["stage"], success=record["success"])
            finished = True
        elif event == "budget_exhausted":
            stopped_early = True
            finished = True
        elif event == "giving_up":
            saw_giving_up = True
            finished = True
        elif event == "run_result":
            run_result = record["success"]
            finished = True

    # A run_result event is authoritative; the giving_up heuristic is the
    # fallback for logs written before that event existed.
    succeeded = run_result if run_result is not None else not saw_giving_up

    file_summaries = [
        FileSummary(
            path=path,
            success=data["success"],
            attempts=data["attempts"],
            truncated=data["truncated"],
            advisory=path in advisory_paths,
        )
        for path, data in files.items()
    ]

    return RunSummary(
        run_id=log_path.parent.name,
        files=file_summaries,
        integration=integration,
        total_llm_calls=total_llm_calls,
        stopped_early=stopped_early,
        succeeded=succeeded,
        finished=finished,
    )


def render_table(summary: RunSummary) -> str:
    """Render a RunSummary as a compact, human-readable text table."""
    lines = [f"Run {summary.run_id}"]

    for f in summary.files:
        if f.success:
            status = "ok"
        elif f.advisory:
            status = "advisory"
        else:
            status = "FAILED"
        note = " (truncated at least once)" if f.truncated else ""
        if f.advisory:
            note += " -- generated test never passed; not blocking the run"
        lines.append(f"  [{status}] {f.path} ({f.attempts} fix attempt(s)){note}")

    if summary.integration is not None:
        status = "ok" if summary.integration.success else "FAILED"
        lines.append(f"  [{status}] integration check ({summary.integration.stage})")

    if not summary.finished:
        result = "INCOMPLETE: the run did not finish (log has no terminal event)"
    elif summary.succeeded:
        result = "SUCCESS"
    elif summary.stopped_early:
        result = "STOPPED: iteration budget exhausted"
    else:
        result = "FAILED"

    advisory_count = sum(1 for f in summary.files if f.advisory)
    advisory_note = ""
    if advisory_count:
        advisory_note = f", {advisory_count} advisory test(s) not passing"
    lines.append(
        f"Result: {result} ({summary.total_llm_calls} LLM call(s) total{advisory_note})"
    )

    return "\n".join(lines)
