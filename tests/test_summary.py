import json
from pathlib import Path

from harness.summary import (
    FileSummary,
    IntegrationSummary,
    RunSummary,
    load_run_summary,
    render_table,
)


def _write_log(tmp_path: Path, records: list[dict]) -> Path:
    run_dir = tmp_path / "20260101-000000-abcd1234"
    run_dir.mkdir()
    log_path = run_dir / "log.jsonl"
    with log_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return log_path


def test_load_run_summary_success_case(tmp_path: Path):
    log_path = _write_log(
        tmp_path,
        [
            {"event": "goal", "goal": "do a thing"},
            {"event": "plan", "files": [{"path": "helper.py", "purpose": "x"}]},
            {"event": "spec", "path": "helper.py", "spec": "- x"},
            {"event": "codegen", "path": "helper.py", "code": "x = 1", "truncated": False},
            {
                "event": "verify",
                "path": "helper.py",
                "attempt": 0,
                "stage": "compile",
                "success": True,
                "output": "",
            },
            {"event": "integration_verify", "stage": "run", "success": True, "output": ""},
        ],
    )

    summary = load_run_summary(log_path)

    assert summary.run_id == "20260101-000000-abcd1234"
    assert summary.goal == "do a thing"
    assert summary.succeeded
    assert not summary.stopped_early
    assert summary.total_llm_calls == 3  # plan + spec + codegen
    assert summary.files == [FileSummary(path="helper.py", success=True, attempts=0, truncated=False)]
    assert summary.integration == IntegrationSummary(stage="run", success=True)


def test_load_run_summary_with_fix_and_truncation(tmp_path: Path):
    log_path = _write_log(
        tmp_path,
        [
            {"event": "codegen", "path": "main.py", "code": "incomplete", "truncated": True},
            {
                "event": "verify",
                "path": "main.py",
                "attempt": 0,
                "stage": "compile",
                "success": False,
                "output": "err",
            },
            {"event": "fix", "path": "main.py", "attempt": 1, "code": "print(1)", "truncated": False},
            {
                "event": "verify",
                "path": "main.py",
                "attempt": 1,
                "stage": "run",
                "success": True,
                "output": "",
            },
            {"event": "run_result", "success": True},
        ],
    )

    summary = load_run_summary(log_path)

    assert summary.files == [FileSummary(path="main.py", success=True, attempts=1, truncated=True)]
    assert summary.total_llm_calls == 2  # codegen + fix
    assert summary.succeeded
    assert summary.finished
    assert summary.goal == ""  # no "goal" event in this fixture


def test_load_run_summary_flags_a_run_that_never_finished(tmp_path: Path):
    log_path = _write_log(
        tmp_path,
        [
            {"event": "codegen", "path": "main.py", "code": "x = 1", "truncated": False},
            {
                "event": "verify",
                "path": "main.py",
                "attempt": 0,
                "stage": "compile",
                "success": True,
                "output": "",
            },
            # process was killed here -- no run_result, no giving_up
        ],
    )

    summary = load_run_summary(log_path)

    assert not summary.finished
    assert "INCOMPLETE" in render_table(summary)


def test_load_run_summary_reports_an_aborted_run(tmp_path: Path):
    log_path = _write_log(
        tmp_path,
        [
            {"event": "codegen", "path": "a.py", "code": "x=1", "truncated": False},
            {
                "event": "verify",
                "path": "a.py",
                "attempt": 0,
                "stage": "import",
                "success": True,
                "output": "",
            },
            {"event": "run_aborted", "reason": "Could not reach Ollama: Read timed out"},
            {"event": "run_result", "success": False},
        ],
    )

    summary = load_run_summary(log_path)

    assert summary.finished
    assert summary.aborted
    assert not summary.succeeded
    table = render_table(summary)
    assert "ABORTED" in table
    assert "timed out" in table
    # the file that completed before the outage is still listed
    assert [f.path for f in summary.files] == ["a.py"]


def test_run_result_event_is_authoritative_over_the_giving_up_heuristic(tmp_path: Path):
    log_path = _write_log(
        tmp_path,
        [
            {"event": "codegen", "path": "a.py", "code": "x=1", "truncated": False},
            {
                "event": "verify",
                "path": "a.py",
                "attempt": 0,
                "stage": "import",
                "success": True,
                "output": "",
            },
            {"event": "integration_verify", "stage": "pytest", "success": True, "output": ""},
            {"event": "run_result", "success": True},
        ],
    )

    summary = load_run_summary(log_path)

    assert summary.succeeded
    assert summary.finished


def test_load_run_summary_gives_up_and_stopped_early(tmp_path: Path):
    log_path = _write_log(
        tmp_path,
        [
            {"event": "codegen", "path": "a.py", "code": "bad(", "truncated": False},
            {
                "event": "verify",
                "path": "a.py",
                "attempt": 0,
                "stage": "compile",
                "success": False,
                "output": "err",
            },
            {"event": "budget_exhausted", "before": "b.py", "iterations": 2},
            {"event": "giving_up", "iterations": 2, "stopped_early": True},
        ],
    )

    summary = load_run_summary(log_path)

    assert not summary.succeeded
    assert summary.stopped_early


def test_render_table_formats_success():
    summary = RunSummary(
        run_id="r1",
        files=[FileSummary(path="a.py", success=True, attempts=0, truncated=False)],
        integration=IntegrationSummary(stage="run", success=True),
        total_llm_calls=3,
        stopped_early=False,
        succeeded=True,
    )

    table = render_table(summary)

    assert "Run r1" in table
    assert "[ok] a.py (0 fix attempt(s))" in table
    assert "[ok] integration check (run)" in table
    assert "Result: SUCCESS (3 LLM call(s) total)" in table


def test_render_table_formats_failure_with_truncation_note():
    summary = RunSummary(
        run_id="r2",
        files=[FileSummary(path="b.py", success=False, attempts=2, truncated=True)],
        integration=None,
        total_llm_calls=5,
        stopped_early=False,
        succeeded=False,
    )

    table = render_table(summary)

    assert "[FAILED] b.py (2 fix attempt(s)) (truncated at least once)" in table
    assert "Result: FAILED (5 LLM call(s) total)" in table


def test_load_run_summary_marks_an_advisory_test_and_still_succeeds(tmp_path: Path):
    log_path = _write_log(
        tmp_path,
        [
            {"event": "codegen", "path": "main.py", "code": "x = 1", "truncated": False},
            {
                "event": "verify",
                "path": "main.py",
                "attempt": 0,
                "stage": "import",
                "success": True,
                "output": "",
            },
            {"event": "codegen", "path": "test_main.py", "code": "...", "truncated": False},
            {
                "event": "verify",
                "path": "test_main.py",
                "attempt": 0,
                "stage": "pytest",
                "success": False,
                "output": "boom",
            },
            {"event": "advisory_test", "path": "test_main.py", "last_error": "boom"},
            {"event": "integration_verify", "stage": "run", "success": True, "output": ""},
        ],
    )

    summary = load_run_summary(log_path)

    assert summary.succeeded
    test_file = next(f for f in summary.files if f.path == "test_main.py")
    assert test_file.advisory
    assert not test_file.success


def test_render_table_shows_an_advisory_test_without_failing_the_run():
    summary = RunSummary(
        run_id="r",
        files=[
            FileSummary(path="main.py", success=True, attempts=0, truncated=False),
            FileSummary(
                path="test_main.py", success=False, attempts=5, truncated=False, advisory=True
            ),
        ],
        integration=IntegrationSummary(stage="run", success=True),
        total_llm_calls=9,
        stopped_early=False,
        succeeded=True,
    )

    table = render_table(summary)

    assert "[advisory] test_main.py" in table
    assert "[FAILED]" not in table
    assert "Result: SUCCESS" in table
    assert "1 advisory test" in table


def test_load_run_summary_keeps_the_failing_verifier_output(tmp_path: Path):
    log_path = _write_log(
        tmp_path,
        [
            {"event": "codegen", "path": "main.py", "code": "def broken(:", "truncated": False},
            {
                "event": "verify",
                "path": "main.py",
                "attempt": 0,
                "stage": "compile",
                "success": False,
                "output": "SyntaxError: invalid syntax",
            },
            {"event": "giving_up", "attempts": 0},
            {"event": "run_result", "success": False},
        ],
    )

    summary = load_run_summary(log_path)

    assert summary.files[0].last_error == "SyntaxError: invalid syntax"


def test_load_run_summary_keeps_the_integration_error_and_skips(tmp_path: Path):
    log_path = _write_log(
        tmp_path,
        [
            {"event": "codegen", "path": "core.py", "code": "x=1", "truncated": False},
            {
                "event": "verify",
                "path": "core.py",
                "attempt": 0,
                "stage": "import",
                "success": True,
                "output": "",
            },
            {"event": "skipped", "path": "main.py", "reason": "a dependency did not build"},
            {"event": "integration_verify", "stage": "pytest", "success": False, "output": "1 failed"},
            {"event": "giving_up", "attempts": 0},
            {"event": "run_result", "success": False},
        ],
    )

    summary = load_run_summary(log_path)

    assert summary.integration is not None and summary.integration.output == "1 failed"
    skipped = next(f for f in summary.files if f.path == "main.py")
    assert skipped.last_error.startswith("skipped:")
    assert not skipped.success


def test_render_table_formats_stopped_early():
    summary = RunSummary(
        run_id="r3",
        files=[FileSummary(path="a.py", success=True, attempts=0, truncated=False)],
        integration=None,
        total_llm_calls=2,
        stopped_early=True,
        succeeded=False,
    )

    table = render_table(summary)

    assert "Result: STOPPED: iteration budget exhausted (2 LLM call(s) total)" in table


def test_load_run_summary_pairs_a_finding_with_its_confirmation(tmp_path: Path):
    log_path = _write_log(
        tmp_path,
        [
            {"event": "plan", "files": [{"path": "a.py", "purpose": "x"}]},
            {"event": "integration_verify", "stage": "run", "success": True, "output": ""},
            {
                "event": "super_review",
                "issues": [{"file": "a.py", "description": "never imports b.py"}],
                "endpoint": "http://smart",
            },
            {
                "event": "super_review_confirm",
                "file": "a.py",
                "description": "never imports b.py",
                "confirmed": True,
                "endpoint": "http://quick",
            },
            {"event": "run_result", "success": True},
        ],
    )

    summary = load_run_summary(log_path)

    assert summary.super_review_started
    assert summary.cross_file_issues == [
        {"file": "a.py", "description": "never imports b.py", "confirmed": True}
    ]
    assert summary.total_llm_calls == 3  # plan + super_review + super_review_confirm


def test_load_run_summary_keeps_an_unconfirmed_finding_visible(tmp_path: Path):
    # No super_review_confirm event at all (single-endpoint run, no second
    # client to ask) -- the finding must still show up, just unconfirmed.
    log_path = _write_log(
        tmp_path,
        [
            {"event": "plan", "files": [{"path": "a.py", "purpose": "x"}]},
            {
                "event": "super_review",
                "issues": [{"file": "a.py", "description": "maybe an issue"}],
                "endpoint": "http://smart",
            },
            {"event": "run_result", "success": True},
        ],
    )

    summary = load_run_summary(log_path)

    assert summary.cross_file_issues == [
        {"file": "a.py", "description": "maybe an issue", "confirmed": False}
    ]


def test_load_run_summary_super_review_not_started_when_absent(tmp_path: Path):
    log_path = _write_log(
        tmp_path,
        [
            {"event": "plan", "files": [{"path": "a.py", "purpose": "x"}]},
            {"event": "run_result", "success": True},
        ],
    )

    summary = load_run_summary(log_path)

    assert not summary.super_review_started
    assert summary.cross_file_issues == []


def test_render_table_shows_confirmed_and_unconfirmed_findings():
    summary = RunSummary(
        run_id="r4",
        files=[],
        integration=IntegrationSummary(stage="run", success=True),
        total_llm_calls=4,
        stopped_early=False,
        succeeded=True,
        cross_file_issues=[
            {"file": "a.py", "description": "real problem", "confirmed": True},
            {"file": "b.py", "description": "maybe a problem", "confirmed": False},
        ],
    )

    table = render_table(summary)

    assert "[review:confirmed] a.py -- real problem" in table
    assert "[review:unconfirmed] b.py -- maybe a problem" in table


def test_load_run_summary_marks_a_branched_file_and_drops_its_scratch_path(tmp_path: Path):
    # A branch's own codegen/fix/verify events are logged under a scratch
    # path (".branch-<real name>") so they can never collide on disk with
    # the original attempt -- those must never show up as their own fake
    # "file" in the summary, and file_result (not the original's own,
    # possibly-failing verify events) is the authoritative word on the
    # real path's final outcome.
    log_path = _write_log(
        tmp_path,
        [
            {"event": "plan", "files": [{"path": "main.py", "purpose": "x"}]},
            # the original's own (failing) progress on the real path
            {"event": "codegen", "path": "main.py", "code": "bad(", "truncated": False},
            {"event": "verify", "path": "main.py", "attempt": 0, "stage": "compile",
             "success": False, "output": "SyntaxError"},
            # a branch's progress under the scratch path -- it wins
            {"event": "codegen", "path": ".branch-main.py", "code": "x = 1", "truncated": False},
            {"event": "verify", "path": ".branch-main.py", "attempt": 0, "stage": "compile",
             "success": True, "output": ""},
            {"event": "file_result", "path": "main.py", "success": True, "branched": True},
            {"event": "run_result", "success": True},
        ],
    )

    summary = load_run_summary(log_path)

    assert len(summary.files) == 1  # the scratch-path entry never appears
    main = summary.files[0]
    assert main.path == "main.py"
    assert main.success
    assert main.branched
    assert main.last_error == ""  # not the original's stale SyntaxError


def test_load_run_summary_leaves_branched_false_for_an_ordinary_file(tmp_path: Path):
    log_path = _write_log(
        tmp_path,
        [
            {"event": "plan", "files": [{"path": "main.py", "purpose": "x"}]},
            {"event": "codegen", "path": "main.py", "code": "x = 1", "truncated": False},
            {"event": "verify", "path": "main.py", "attempt": 0, "stage": "compile",
             "success": True, "output": ""},
            {"event": "file_result", "path": "main.py", "success": True, "branched": False},
            {"event": "run_result", "success": True},
        ],
    )

    summary = load_run_summary(log_path)

    assert summary.files == [
        FileSummary(path="main.py", success=True, attempts=0, truncated=False)
    ]


def test_render_table_notes_a_branched_file():
    summary = RunSummary(
        run_id="r5",
        files=[FileSummary(path="main.py", success=True, attempts=1, truncated=False, branched=True)],
        integration=None,
        total_llm_calls=5,
        stopped_early=False,
        succeeded=True,
    )

    table = render_table(summary)

    assert "won by a branch to another endpoint" in table
