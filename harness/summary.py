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

_CALL_EVENTS = {
    "plan",
    "spec",
    "codegen",
    "fix",
    "integration_fix",
    "critic_check",
    "super_review",
    "super_review_confirm",
}


@dataclasses.dataclass
class FileSummary:
    path: str
    success: bool
    attempts: int
    truncated: bool
    advisory: bool = False
    # True when the file passed every real check (compile/lint/import)
    # but the critic step (an LLM's opinion, not real tooling) disagreed
    # with it against its own spec, even after normal fix attempts.
    # `success` stays False (an honest record of the critic's verdict);
    # this flag is what keeps it from blocking the run -- see
    # orchestrator.FileRunResult.spec_flagged.
    spec_flagged: bool = False
    # The last verifier output for this file (the error, when it failed;
    # or "skipped: ..." when a dependency didn't build).
    last_error: str = ""
    # True when this file's winning result came from a branch -- a
    # second, independent attempt an idle endpoint started after this
    # file's original attempt got stuck (see
    # orchestrator._generate_files / Config.branch_after_fixes).
    branched: bool = False


@dataclasses.dataclass
class IntegrationSummary:
    stage: str
    success: bool
    output: str = ""


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
    # True when the run stopped because the model became unreachable
    # partway through, rather than on the code.
    aborted: bool = False
    abort_reason: str = ""
    # Whole-project findings from steps/super_review.py, in the order
    # they were reported: {"file", "description", "confirmed"}. Empty
    # when super_review never ran or found nothing.
    cross_file_issues: list[dict] = dataclasses.field(default_factory=list)
    # True the moment a super_review event lands in the log -- separate
    # from `cross_file_issues` being non-empty, since "reviewed and found
    # nothing" and "hasn't started reviewing yet" must read differently
    # (the GUI's phase stepper needs to tell them apart).
    super_review_started: bool = False
    # The original goal text, straight from the run's own "goal" event --
    # empty for a log written before this field existed. Lets a viewer
    # (the GUI's live header, `harness inspect`) show what a run is
    # actually building without relying on the caller to have kept the
    # text around itself, which the GUI can't when a run was started
    # through a raw API call rather than its own form.
    goal: str = ""


def _new_file_entry() -> dict:
    return {"attempts": 0, "truncated": False, "success": False, "last_error": "", "branched": False}


def load_run_summary(log_path: Path) -> RunSummary:
    """Reconstruct a RunSummary from a run's log.jsonl."""
    files: dict[str, dict] = {}
    advisory_paths: set[str] = set()
    spec_flagged_paths: set[str] = set()
    integration: IntegrationSummary | None = None
    total_llm_calls = 0
    stopped_early = False
    saw_giving_up = False
    run_result: bool | None = None
    finished = False
    aborted = False
    abort_reason = ""
    found_issues: list[dict] = []
    confirmed_by_key: dict[tuple[str, str], bool] = {}
    super_review_started = False
    goal = ""

    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        event = record.get("event")

        if event in _CALL_EVENTS:
            total_llm_calls += 1

        if event == "goal":
            goal = record.get("goal", "")
        elif event in ("codegen", "fix"):
            entry = files.setdefault(record["path"], _new_file_entry())
            if event == "fix":
                entry["attempts"] += 1
            if record.get("truncated"):
                entry["truncated"] = True
        elif event == "verify":
            entry = files.setdefault(record["path"], _new_file_entry())
            entry["success"] = record["success"]
            entry["last_error"] = "" if record["success"] else record.get("output", "")
        elif event == "skipped":
            entry = files.setdefault(record["path"], _new_file_entry())
            entry["last_error"] = f"skipped: {record.get('reason', 'a dependency did not build')}"
        elif event == "file_result":
            # The one authoritative word on a file's final success/
            # branched status -- codegen/fix/verify events for a file a
            # branch won were logged under that branch's own scratch
            # path (see orchestrator._generate_files), so they can't be
            # trusted for the file's real, final outcome the way this
            # can. Never skipped: every path that reaches record() gets
            # exactly one of these.
            entry = files.setdefault(record["path"], _new_file_entry())
            entry["success"] = record["success"]
            entry["branched"] = record.get("branched", False)
            if record["success"]:
                entry["last_error"] = ""
        elif event == "advisory_test":
            advisory_paths.add(record["path"])
        elif event == "spec_flagged":
            spec_flagged_paths.add(record["path"])
        elif event == "integration_verify":
            integration = IntegrationSummary(
                stage=record["stage"],
                success=record["success"],
                output="" if record["success"] else record.get("output", ""),
            )
            finished = True
        elif event == "super_review":
            super_review_started = True
            found_issues = record.get("issues") or []
        elif event == "super_review_confirm":
            confirmed_by_key[(record["file"], record["description"])] = record["confirmed"]
        elif event == "budget_exhausted":
            stopped_early = True
            finished = True
        elif event == "giving_up":
            saw_giving_up = True
            finished = True
        elif event == "run_aborted":
            aborted = True
            abort_reason = record.get("reason", "")
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
            spec_flagged=path in spec_flagged_paths,
            last_error=data.get("last_error", ""),
            branched=data.get("branched", False),
        )
        for path, data in files.items()
        # A losing branch's own codegen/fix/verify events were logged
        # under its scratch path (".branch-<real name>") and it never
        # gets a file_result of its own (see orchestrator._generate_
        # files) -- exclude that internal bookkeeping from the report
        # entirely rather than showing it as if it were a real file.
        if not Path(path).name.startswith(".branch-")
    ]

    cross_file_issues = [
        {
            "file": item["file"],
            "description": item["description"],
            "confirmed": confirmed_by_key.get((item["file"], item["description"]), False),
        }
        for item in found_issues
    ]

    return RunSummary(
        run_id=log_path.parent.name,
        goal=goal,
        files=file_summaries,
        integration=integration,
        total_llm_calls=total_llm_calls,
        stopped_early=stopped_early,
        succeeded=succeeded,
        finished=finished,
        aborted=aborted,
        abort_reason=abort_reason,
        cross_file_issues=cross_file_issues,
        super_review_started=super_review_started,
    )


def render_table(summary: RunSummary) -> str:
    """Render a RunSummary as a compact, human-readable text table."""
    lines = [f"Run {summary.run_id}"]

    for f in summary.files:
        if f.success:
            status = "ok"
        elif f.spec_flagged:
            status = "flagged"
        elif f.advisory:
            status = "advisory"
        else:
            status = "FAILED"
        note = " (truncated at least once)" if f.truncated else ""
        if f.advisory:
            note += " -- generated test never passed; not blocking the run"
        elif f.spec_flagged:
            note += " -- passes every real check, but the critic disagrees; not blocking the run"
        if f.branched:
            note += " -- won by a branch to another endpoint"
        lines.append(f"  [{status}] {f.path} ({f.attempts} fix attempt(s)){note}")

    if summary.integration is not None:
        status = "ok" if summary.integration.success else "FAILED"
        lines.append(f"  [{status}] integration check ({summary.integration.stage})")

    for issue in summary.cross_file_issues:
        tag = "confirmed" if issue["confirmed"] else "unconfirmed"
        lines.append(f"  [review:{tag}] {issue['file']} -- {issue['description']}")

    if summary.aborted:
        detail = f" ({summary.abort_reason})" if summary.abort_reason else ""
        result = f"ABORTED: the model became unreachable mid-run{detail}"
    elif not summary.finished:
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
    flagged_count = sum(1 for f in summary.files if f.spec_flagged)
    flagged_note = ""
    if flagged_count:
        flagged_note = f", {flagged_count} file(s) flagged by the critic"
    lines.append(
        f"Result: {result} ({summary.total_llm_calls} LLM call(s) total"
        f"{advisory_note}{flagged_note})"
    )

    return "\n".join(lines)
