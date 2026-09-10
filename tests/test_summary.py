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
        ],
    )

    summary = load_run_summary(log_path)

    assert summary.files == [FileSummary(path="main.py", success=True, attempts=1, truncated=True)]
    assert summary.total_llm_calls == 2  # codegen + fix
    assert summary.succeeded


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
