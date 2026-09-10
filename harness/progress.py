"""Turns session events into live, human-readable progress lines.

A multi-file run can sit silent for minutes otherwise (each LLM call is
tens of seconds on modest local hardware). This taps the same event stream
`summary.py` reads after the fact, live, without touching the
orchestrator's control flow -- it's a pure formatter plus a thin adapter
to `Session.on_event`'s callback shape.
"""

from __future__ import annotations

from collections.abc import Callable


def format_event(event: str, fields: dict) -> str | None:
    """One line per event, or None to stay silent.

    Silent: "goal" (redundant with the "Run <id>: goal = ..." preamble
    already printed before the loop starts) and "giving_up" (redundant
    with the final summary table's Result line). Unrecognized events are
    also silent, so a future new event type doesn't break this formatter.
    """
    if event == "plan":
        return f"[plan] {len(fields['files'])} file(s) planned"

    if event == "spec":
        return f"[spec] {fields['path']}"

    if event == "codegen":
        note = " (truncated)" if fields.get("truncated") else ""
        return f"[codegen] {fields['path']}{note}"

    if event == "verify":
        status = "ok" if fields["success"] else "FAILED"
        return f"[verify:{fields['stage']}] {fields['path']} -> {status}"

    if event == "fix":
        note = " (truncated)" if fields.get("truncated") else ""
        return f"[fix] {fields['path']} -> attempt {fields['attempt']}{note}"

    if event == "fix_noop":
        return (
            f"[fix] {fields['path']} -> attempt {fields['attempt']} "
            f"repeated its previous output (raising temperature)"
        )

    if event == "advisory_test":
        return (
            f"[advisory] {fields['path']} -> generated test never passed "
            f"(not blocking the run)"
        )

    if event == "integration_verify":
        status = "ok" if fields["success"] else "FAILED"
        return f"[integration:{fields['stage']}] -> {status}"

    if event == "integration_fix":
        return f"[integration-fix] {fields['path']} -> round {fields['round']}"

    if event == "budget_exhausted":
        return f"[budget] exhausted before {fields['before']} ({fields['iterations']} iteration(s) used)"

    return None


def _default_print(line: str) -> None:
    print(line, flush=True)


def console_reporter(print_fn: Callable[[str], None] = _default_print) -> Callable[[str, dict], None]:
    """Adapts format_event into the (event, fields) -> None shape Session.on_event expects."""

    def reporter(event: str, fields: dict) -> None:
        line = format_event(event, fields)
        if line is not None:
            print_fn(line)

    return reporter
