from harness.progress import console_reporter, format_event


def test_format_event_returns_none_for_goal_and_giving_up():
    assert format_event("goal", {"goal": "x"}) is None
    assert format_event("giving_up", {"attempts": 3}) is None


def test_format_event_critic_check_agrees():
    line = format_event("critic_check", {"path": "main.py", "follows_spec": True, "issues": ""})
    assert line == "[critic] main.py -> ok"


def test_format_event_critic_check_agrees_shows_the_endpoint():
    line = format_event(
        "critic_check",
        {"path": "main.py", "follows_spec": True, "issues": "", "endpoint": "http://smart:11434"},
    )
    assert line == "[critic] main.py -> ok @ http://smart:11434"


def test_format_event_critic_check_disagreement_is_silent():
    # Already surfaced as its own [verify:critic] -> FAILED line with the
    # same issues text -- a second line here would just be noise.
    line = format_event(
        "critic_check", {"path": "main.py", "follows_spec": False, "issues": "wrong output"}
    )
    assert line is None


def test_format_event_endpoint_retired():
    line = format_event(
        "endpoint_retired",
        {"path": "b.py", "endpoint": "http://h2:11434", "reason": "Read timed out"},
    )
    assert line == "[endpoint] http://h2:11434 failed on b.py (Read timed out) -- requeued for another endpoint"


def test_format_event_returns_none_for_unknown_event():
    assert format_event("something_future", {}) is None


def test_format_event_plan():
    line = format_event("plan", {"files": [{"path": "a.py", "purpose": "x"}, {"path": "b.py", "purpose": "y"}]})
    assert line == "[plan] 2 file(s) planned"


def test_format_event_plan_shows_the_endpoint():
    line = format_event(
        "plan", {"files": [{"path": "a.py", "purpose": "x"}], "endpoint": "http://smart:11434"}
    )
    assert line == "[plan] 1 file(s) planned @ http://smart:11434"


def test_format_event_spec():
    assert format_event("spec", {"path": "main.py", "spec": "- x"}) == "[spec] main.py"


def test_format_event_spec_notes_which_endpoint_wrote_it():
    line = format_event("spec", {"path": "main.py", "spec": "- x", "endpoint": "http://h2:11434"})
    assert line == "[spec] main.py @ http://h2:11434"


def test_format_event_codegen_and_codegen_truncated():
    assert format_event("codegen", {"path": "main.py", "code": "x", "truncated": False}) == "[codegen] main.py"
    assert (
        format_event("codegen", {"path": "main.py", "code": "x", "truncated": True})
        == "[codegen] main.py (truncated)"
    )


def test_format_event_codegen_notes_which_endpoint_built_it():
    line = format_event(
        "codegen", {"path": "b.py", "code": "x", "truncated": False, "endpoint": "http://h2:11434"}
    )
    assert line == "[codegen] b.py @ http://h2:11434"


def test_format_event_verify_success_and_failure():
    ok = format_event(
        "verify", {"path": "main.py", "attempt": 0, "stage": "run", "success": True, "output": ""}
    )
    failed = format_event(
        "verify", {"path": "main.py", "attempt": 1, "stage": "compile", "success": False, "output": "err"}
    )
    assert ok == "[verify:run] main.py -> ok"
    assert failed == "[verify:compile] main.py -> FAILED"


def test_format_event_fix():
    assert (
        format_event("fix", {"path": "main.py", "attempt": 1, "code": "x", "truncated": False})
        == "[fix] main.py -> attempt 1"
    )
    assert (
        format_event("fix", {"path": "main.py", "attempt": 2, "code": "x", "truncated": True})
        == "[fix] main.py -> attempt 2 (truncated)"
    )


def test_format_event_fix_noop():
    line = format_event("fix_noop", {"path": "main.py", "attempt": 2, "stage": "compile"})
    assert line == "[fix] main.py -> attempt 2 repeated its previous output (raising temperature)"


def test_format_event_spec_rejected():
    line = format_event(
        "spec_rejected", {"path": "main.py", "spec": "Sure, here you go.", "attempt": 1}
    )
    assert line == "[spec] main.py -> attempt 1 didn't look like a real spec (retrying, hotter)"


def test_format_event_advisory_test():
    line = format_event("advisory_test", {"path": "test_main.py", "last_error": "boom"})
    assert line == "[advisory] test_main.py -> generated test never passed (not blocking the run)"


def test_format_event_integration_verify():
    ok = format_event("integration_verify", {"stage": "pytest", "success": True, "output": ""})
    failed = format_event("integration_verify", {"stage": "run", "success": False, "output": "err"})
    assert ok == "[integration:pytest] -> ok"
    assert failed == "[integration:run] -> FAILED"


def test_format_event_integration_fix():
    line = format_event("integration_fix", {"path": "main.py", "round": 1, "code": "x"})
    assert line == "[integration-fix] main.py -> round 1"


def test_format_event_budget_exhausted():
    line = format_event("budget_exhausted", {"before": "main.py", "iterations": 25})
    assert line == "[budget] exhausted before main.py (25 iteration(s) used)"


def test_format_event_run_paused_and_resumed():
    assert format_event("run_paused", {}) == "[paused] waiting to resume…"
    assert format_event("run_resumed", {}) == "[resumed]"


def test_format_event_super_review_no_issues():
    line = format_event("super_review", {"issues": [], "endpoint": "http://smart:11434"})
    assert line == "[review] no whole-project issues found @ http://smart:11434"


def test_format_event_super_review_found_issues():
    line = format_event(
        "super_review",
        {"issues": [{"file": "a.py", "description": "x"}], "endpoint": "http://smart:11434"},
    )
    assert line == "[review] 1 possible issue(s) found @ http://smart:11434 -- confirming…"


def test_format_event_super_review_confirm():
    line = format_event(
        "super_review_confirm",
        {"file": "main.py", "description": "never imports helper", "confirmed": True},
    )
    assert line == "[review] main.py -> confirmed: never imports helper"

    line = format_event(
        "super_review_confirm",
        {"file": "main.py", "description": "never imports helper", "confirmed": False},
    )
    assert line == "[review] main.py -> not confirmed: never imports helper"


def test_console_reporter_prints_only_non_silent_events():
    printed = []
    reporter = console_reporter(print_fn=printed.append)

    reporter("goal", {"goal": "x"})
    reporter("codegen", {"path": "main.py", "code": "x", "truncated": False})

    assert printed == ["[codegen] main.py"]


def test_console_reporter_default_print_fn_flushes(capsys):
    reporter = console_reporter()

    reporter("codegen", {"path": "main.py", "code": "x", "truncated": False})

    assert "[codegen] main.py" in capsys.readouterr().out
