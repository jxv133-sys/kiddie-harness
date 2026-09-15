import json
import threading
import time
from pathlib import Path

from harness.llm_client import OllamaError
from harness.orchestrator import MultiFileLoop
from harness.session import Session
from harness.steps.plan import FileTask

from .fakes import FakeClient, make_config

_LEAF_PAIR_PLAN = json.dumps(
    {
        "files": [
            {"path": "a.py", "purpose": "leaf a", "depends_on": []},
            {"path": "b.py", "purpose": "leaf b", "depends_on": []},
        ]
    }
)
_GENERIC = ["- do a thing", "def thing():\n    return 1\n"]

_PLAN_TWO_FILES = json.dumps(
    {
        "files": [
            {"path": "helper.py", "purpose": "add two numbers"},
            {"path": "main.py", "purpose": "entry point"},
        ]
    }
)


def test_succeeds_across_files_with_integration_check(tmp_path: Path):
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    client = FakeClient(
        [
            _PLAN_TWO_FILES,
            "- add two numbers",  # spec for helper.py
            "def add(a, b):\n    return a + b\n",  # codegen for helper.py
            "- expose a main() that adds two numbers and print it when run",  # spec for main.py
            (
                "from helper import add\n\n\n"
                "def main():\n    print(add(2, 3))\n\n\n"
                'if __name__ == "__main__":\n    main()\n'
            ),  # codegen for main.py
        ]
    )

    result = MultiFileLoop(client, config, session).run("a script that adds two numbers")

    assert result.success
    assert not result.stopped_early
    assert [f.path for f in result.files] == [
        str(session.run_dir / "helper.py"),
        str(session.run_dir / "main.py"),
    ]
    assert all(f.success for f in result.files)
    assert result.integration is not None
    assert result.integration.success
    assert (session.run_dir / "helper.py").read_text() == "def add(a, b):\n    return a + b"


def test_a_file_the_critic_disagrees_with_does_not_sink_the_run(tmp_path: Path):
    config = make_config(tmp_path, critic_enabled=True, max_fix_attempts=1)
    session = Session.create(config.workspace_root)
    disagree = '{"follows_spec": false, "issues": "does not handle negative numbers"}'
    agree = '{"follows_spec": true, "issues": ""}'
    client = FakeClient(
        [
            _PLAN_TWO_FILES,
            "- add two numbers",  # spec for helper.py
            "def add(a, b):\n    return a + b\n",  # codegen for helper.py
            disagree,  # critic: no
            "def add(a, b):\n    return a + b\n",  # fix attempt 1 (unchanged -- still correct)
            disagree,  # critic: still no -- fix attempts exhausted
            "- call add and print it",  # spec for main.py
            (
                "from helper import add\n\n\n"
                "def main():\n    print(add(2, 3))\n\n\n"
                'if __name__ == "__main__":\n    main()\n'
            ),  # codegen for main.py
            agree,  # critic: main.py is fine
        ]
    )

    result = MultiFileLoop(client, config, session).run("a script that adds two numbers")

    # The critic's opinion never gets the final word over real tooling --
    # helper.py compiles, lints, and imports clean, so the run succeeds.
    assert result.success
    assert result.integration is not None and result.integration.success
    helper = next(f for f in result.files if f.path.endswith("helper.py"))
    assert not helper.success  # honest: the critic's verdict was "no"
    assert helper.spec_flagged
    assert "negative numbers" in helper.last_output
    # main.py still saw helper.py's real source as sibling context, even
    # though helper.py is only spec_flagged, not plain "success".
    main_codegen_call = client.calls[7]
    assert "def add(a, b):" in main_codegen_call


def test_codegen_instruction_carries_the_sibling_modules_already_built(tmp_path: Path):
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    client = FakeClient(
        [
            _PLAN_TWO_FILES,
            "- add two numbers",  # spec for helper.py
            "def add(a, b):\n    return a + b\n",  # codegen for helper.py
            "- call add and print it",  # spec for main.py
            (
                "from helper import add\n\n"
                "def main():\n    print(add(2, 3))\n\n"
                'if __name__ == "__main__":\n    main()\n'
            ),  # codegen for main.py
        ]
    )

    MultiFileLoop(client, config, session).run("a script that adds two numbers")

    # calls: plan(0), spec-helper(1), codegen-helper(2), spec-main(3), codegen-main(4)
    assert "def add(a, b):" in client.calls[4]
    assert "helper.py" in client.calls[4]
    # the first file's codegen had no siblings yet
    assert "def add(a, b):" not in client.calls[2]


def test_a_python_files_html_dependency_is_never_framed_as_importable(tmp_path: Path):
    # Regression: _sibling_context used to fence *every* dependency's
    # source as ```python and tell the model to "import ... by module
    # name" regardless of the dependency's real language. A Python file
    # (e.g. server.py) depending on an .html file it serves would be
    # shown that HTML labeled as Python and told to `from login_page
    # import ...` it -- which Python cannot do at all, since .html isn't
    # an importable module. A real, observed failure mode, not a
    # hypothetical one.
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    html_source = "<html><body><h1>Log in</h1></body></html>"
    client = FakeClient(
        [
            json.dumps(
                {
                    "files": [
                        {"path": "login_page.html", "purpose": "the login page", "depends_on": []},
                        {
                            "path": "server.py",
                            "purpose": "serves the login page",
                            "depends_on": ["login_page.html"],
                        },
                    ]
                }
            ),
            "- a login form",  # spec for login_page.html
            html_source,  # codegen for login_page.html
            "- serve login_page.html over HTTP",  # spec for server.py
            (
                "import http.server\n\n\n"
                'if __name__ == "__main__":\n'
                '    server = http.server.HTTPServer(("", 8000), '
                "http.server.SimpleHTTPRequestHandler)\n"
                "    server.serve_forever()\n"
            ),  # codegen for server.py
        ]
    )

    MultiFileLoop(client, config, session).run("a login page with a server")

    # calls: plan(0), spec-html(1), codegen-html(2), spec-server(3), codegen-server(4)
    server_instruction = client.calls[4]
    assert html_source in server_instruction
    assert "```html" in server_instruction
    assert "```python\n" + html_source not in server_instruction
    assert "must never be `import`ed" in server_instruction
    # no Python sibling exists, so the Python-only "import by module
    # name" framing must not appear at all
    assert "import what you need" not in server_instruction


def test_catches_and_fixes_a_bad_cross_file_import_during_its_own_generation(tmp_path: Path):
    # Mirrors a real failure: a file imports a sibling module under the
    # wrong name. The bug must be caught (and fixed) during that file's
    # own generation.
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    client = FakeClient(
        [
            _PLAN_TWO_FILES,
            "- add two numbers",  # spec for helper.py
            "def add(a, b):\n    return a + b\n",  # codegen for helper.py
            "- expose a main() that adds two numbers and print it when run",  # spec for main.py
            "from helpr import add\n\ndef main():\n    return add(2, 3)\n",  # codegen: wrong module name
            "from helper import add\n\ndef main():\n    return add(2, 3)\n",  # fix: corrected import
        ]
    )

    result = MultiFileLoop(client, config, session).run("a script that adds two numbers")

    assert result.success
    main_file = next(f for f in result.files if Path(f.path).name == "main.py")
    assert main_file.attempts == 1


def test_a_disclaimer_prose_spec_response_is_rejected_and_retried(tmp_path: Path):
    # write_spec itself has no validation -- a model that responds with
    # commentary instead of real bullet points used to flow straight into
    # the codegen instruction unflagged, corrupting everything built from
    # it without ever surfacing the real cause.
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    client = FakeClient(
        [
            json.dumps({"files": [{"path": "main.py", "purpose": "print hello", "depends_on": []}]}),
            "Sure, here's a specification for main.py based on your request.",  # rejected
            "- print hello",  # accepted retry
            "print('hello')\n",  # codegen
        ]
    )

    result = MultiFileLoop(client, config, session).run("print hello")

    assert result.success
    assert len(client.calls) == 4  # plan + 2 spec attempts + codegen -- both spec calls were real
    events = [
        json.loads(line)["event"] for line in (session.run_dir / "log.jsonl").read_text().splitlines()
    ]
    assert events.count("spec_rejected") == 1
    assert events.count("spec") == 1  # only the accepted attempt is logged as "spec"


def test_a_persistently_bad_spec_still_proceeds_once_retries_are_exhausted(tmp_path: Path):
    # Spec retries are a small, bounded nudge, not a guarantee -- codegen
    # and verify remain the real backstop, so a file must still get built
    # (and judged by real tooling) even if every spec attempt is bad.
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    client = FakeClient(
        [
            json.dumps({"files": [{"path": "main.py", "purpose": "print hello", "depends_on": []}]}),
            "I cannot help with that request.",  # attempt 1 -- rejected
            "I'm not able to generate this.",  # attempt 2 -- rejected
            "Sorry, I can't complete this task.",  # attempt 3 -- exhausted, used anyway
            "print('hello')\n",  # codegen still happens
        ]
    )

    result = MultiFileLoop(client, config, session).run("print hello")

    assert result.success
    assert len(client.calls) == 5  # plan + 3 spec attempts + codegen
    events = [
        json.loads(line)["event"] for line in (session.run_dir / "log.jsonl").read_text().splitlines()
    ]
    assert events.count("spec_rejected") == 2  # attempts 1 and 2; the 3rd is used regardless
    assert events.count("spec") == 1


def test_auto_fixes_a_per_file_lint_issue_without_calling_the_llm(tmp_path: Path):
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    client = FakeClient(
        [
            json.dumps({"files": [{"path": "main.py", "purpose": "entry point"}]}),
            "- print hello",  # spec
            "import os\nprint('hello')\n",  # codegen: unused import -- ruff auto-fixes this itself
        ]
    )

    result = MultiFileLoop(client, config, session).run("print hello")

    assert result.success
    assert len(result.files) == 1
    assert result.files[0].attempts == 0
    assert len(client.calls) == 3  # plan, spec, codegen -- no LLM fix call was needed


def test_recovers_from_an_unfixable_per_file_lint_issue(tmp_path: Path):
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    client = FakeClient(
        [
            json.dumps({"files": [{"path": "main.py", "purpose": "entry point"}]}),
            "- print hello",  # spec
            "print(undefined_name)\n",  # codegen: undefined name -- ruff can't fix this itself
            "print('hello')\n",  # fix: clean
        ]
    )

    result = MultiFileLoop(client, config, session).run("print hello")

    assert result.success
    assert len(result.files) == 1
    assert result.files[0].attempts == 1


def test_gives_up_when_a_file_never_passes(tmp_path: Path):
    config = make_config(tmp_path, max_fix_attempts=1)
    session = Session.create(config.workspace_root)
    broken = "def broken(:\n"
    client = FakeClient(
        [
            json.dumps({"files": [{"path": "bad.py", "purpose": "broken code"}]}),
            "- do something impossible",
            broken,
            broken,  # one fix attempt, still broken
            # no test is ever generated for a file that never compiles
        ]
    )

    result = MultiFileLoop(client, config, session).run("do something impossible")

    assert not result.success
    assert len(result.files) == 1
    assert not result.files[0].success
    assert result.integration is None  # never reached: the file itself failed


def test_stops_early_when_iteration_budget_is_exhausted(tmp_path: Path):
    config = make_config(tmp_path, max_total_iterations=2)
    session = Session.create(config.workspace_root)
    client = FakeClient(
        [
            _PLAN_TWO_FILES,
            "- add two numbers",
            "def add(a, b):\n    return a + b\n",
            # budget is exhausted right after this: no test for helper.py,
            # and the second file's spec/codegen should never be requested
        ]
    )

    result = MultiFileLoop(client, config, session).run("a script that adds two numbers")

    assert not result.success
    assert result.stopped_early
    assert len(result.files) == 1


def test_sanitizes_path_traversal_in_planned_file_path(tmp_path: Path):
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    client = FakeClient(
        [
            json.dumps({"files": [{"path": "../../etc/evil.py", "purpose": "escape attempt"}]}),
            "- do nothing bad",
            "print('hi')\n",
            "def test_placeholder():\n    assert True\n",  # test for evil.py
        ]
    )

    result = MultiFileLoop(client, config, session).run("try to escape the workspace")

    written_path = Path(result.files[0].path)
    # plan_files flattens to a bare filename, so the traversal collapses
    # to a plain file directly in the run directory.
    assert written_path == session.run_dir / "evil.py"


def test_skips_test_generation_for_a_planner_provided_test_file(tmp_path: Path):
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    client = FakeClient(
        [
            json.dumps({"files": [{"path": "test_already.py", "purpose": "a test file"}]}),
            "- assert true",  # spec
            "def test_ok():\n    assert True\n",  # codegen -- no companion test generated for this
        ]
    )

    result = MultiFileLoop(client, config, session).run("just a bare test file")

    assert result.success
    assert len(result.files) == 1
    assert result.integration is not None
    assert result.integration.stage == "pytest"


def test_fix_prompt_reflects_ruffs_autofix_when_a_second_issue_remains(tmp_path: Path):
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    client = FakeClient(
        [
            json.dumps({"files": [{"path": "main.py", "purpose": "entry point"}]}),
            "- print hello",  # spec
            # unused import (auto-fixed by ruff) + undefined name (not auto-fixable)
            "import os\nprint(undefined_name)\n",
            "print('hello')\n",  # fix: clean
            "def test_placeholder():\n    assert True\n",  # test for main.py
        ]
    )

    MultiFileLoop(client, config, session).run("print hello")

    fix_prompt = client.calls[3]  # plan, spec, codegen, fix
    assert "import os" not in fix_prompt


def test_a_planner_listed_test_that_never_passes_is_advisory_not_fatal(tmp_path: Path):
    config = make_config(tmp_path, max_fix_attempts=3)
    session = Session.create(config.workspace_root)
    bad_test = 'from greeter import greet\n\ndef test_greet():\n    assert greet() == "bye"\n'
    client = FakeClient(
        [
            json.dumps(
                {
                    "files": [
                        {"path": "greeter.py", "purpose": "greet"},
                        {"path": "test_greeter.py", "purpose": "tests for greeter"},
                    ]
                }
            ),
            "- expose greet()",  # spec for greeter.py
            'def greet():\n    return "hi"\n',  # codegen -- imports clean
            "- test greet()",  # spec for test_greeter.py
            bad_test,  # test_greeter.py: fails
            bad_test,  # fix 1
            bad_test,  # fix 2
            bad_test,  # fix 3
        ]
    )

    result = MultiFileLoop(client, config, session).run("a greeter with tests")

    assert result.success  # greeter.py is fine, so the run succeeds
    test_result = next(f for f in result.files if Path(f.path).name == "test_greeter.py")
    assert not test_result.success
    assert test_result.advisory
    assert result.integration is not None and result.integration.success


def test_entry_script_that_needs_argv_still_passes_integration_via_import_check(tmp_path: Path):
    config = make_config(tmp_path, max_fix_attempts=1)
    session = Session.create(config.workspace_root)
    argv_cli = (
        "import sys\n\n"
        "def run(path):\n    return path.upper()\n\n"
        "def main():\n"
        "    if len(sys.argv) != 2:\n"
        '        print("usage: main.py <path>")\n'
        "        sys.exit(1)\n"
        "    print(run(sys.argv[1]))\n\n"
        'if __name__ == "__main__":\n    main()\n'
    )
    bad_test = "from main import run\n\ndef test_run():\n    assert run('a') == 'B'\n"
    client = FakeClient(
        [
            json.dumps({"files": [{"path": "main.py", "purpose": "cli"}]}),
            "- parse argv",
            argv_cli,
            bad_test,
            bad_test,  # fix: still wrong -> advisory
        ]
    )

    result = MultiFileLoop(client, config, session).run("a cli")

    # run_script exits non-zero (no argv), but the module and its imports
    # load fine, so integration passes and the run succeeds.
    assert result.success
    assert result.integration is not None and result.integration.success


def test_advisory_test_is_left_out_of_the_integration_pytest_run(tmp_path: Path):
    config = make_config(tmp_path, max_fix_attempts=1)
    session = Session.create(config.workspace_root)
    wrong = "from calc import add\n\ndef test_add():\n    assert add(1, 1) == 3\n"
    client = FakeClient(
        [
            json.dumps(
                {
                    "files": [
                        {"path": "calc.py", "purpose": "adder"},
                        {"path": "test_calc.py", "purpose": "tests"},
                    ]
                }
            ),
            "- expose add()",
            "def add(a, b):\n    return a + b\n",
            "- test add()",
            wrong,
            wrong,  # fix: still wrong -> advisory
        ]
    )

    result = MultiFileLoop(client, config, session).run("an adder with tests")

    # test_calc.py is advisory-failed and left out of the pytest run;
    # calc.py has no runnable entry, so integration has nothing to fail
    # on and the run still succeeds.
    assert result.success
    assert [Path(f.path).name for f in result.files if f.advisory] == ["test_calc.py"]


def test_two_endpoints_build_independent_files_in_parallel(tmp_path: Path):
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    barrier = threading.Barrier(2, timeout=5)
    x = FakeClient(list(_GENERIC), first_call_barrier=barrier, host="http://endpoint-x")
    y = FakeClient(list(_GENERIC), first_call_barrier=barrier, host="http://endpoint-y")

    result = MultiFileLoop(
        FakeClient([_LEAF_PAIR_PLAN]), config, session, pool_clients=[x, y]
    ).run("two leaves")

    assert result.success
    assert {Path(f.path).name for f in result.files} == {"a.py", "b.py"}
    # the barrier only releases once BOTH endpoints have started a file
    assert x.calls and y.calls
    # ...and which endpoint built which file is visible in the log, not
    # just inferable from the fact that both clients got called. Which
    # worker claims which file is a race, so only assert both endpoints
    # were used and each file is attributed to exactly one of them.
    events = [json.loads(line) for line in session.log_path.read_text().splitlines()]
    codegen_endpoints = {e["path"].split("/")[-1]: e["endpoint"] for e in events if e["event"] == "codegen"}
    assert set(codegen_endpoints) == {"a.py", "b.py"}
    assert set(codegen_endpoints.values()) == {"http://endpoint-x", "http://endpoint-y"}


def test_a_dependent_is_built_after_its_dependency_and_sees_its_source(tmp_path: Path):
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    plan = json.dumps(
        {
            "files": [
                {"path": "core.py", "purpose": "logic", "depends_on": []},
                {"path": "main.py", "purpose": "entry", "depends_on": ["core.py"]},
            ]
        }
    )
    x = FakeClient(["- spec", "def helper():\n    return 1\n"] * 2)
    y = FakeClient(["- spec", "def helper():\n    return 1\n"] * 2)

    result = MultiFileLoop(FakeClient([plan]), config, session, pool_clients=[x, y]).run("c+m")

    assert result.success
    events = [json.loads(line) for line in session.log_path.read_text().splitlines()]
    core_ok = next(
        i
        for i, e in enumerate(events)
        if e["event"] == "verify" and e["path"].endswith("core.py") and e["success"]
    )
    main_gen = next(
        i for i, e in enumerate(events) if e["event"] == "codegen" and e["path"].endswith("main.py")
    )
    assert core_ok < main_gen
    main_prompt = next(c for c in (x.calls + y.calls) if "Create the file `main.py`" in c)
    assert "def helper():" in main_prompt  # core.py's source was in the prompt


def test_a_file_whose_dependency_fails_is_skipped_and_the_run_fails(tmp_path: Path):
    config = make_config(tmp_path, max_fix_attempts=1)
    session = Session.create(config.workspace_root)
    plan = json.dumps(
        {
            "files": [
                {"path": "core.py", "purpose": "x", "depends_on": []},
                {"path": "main.py", "purpose": "y", "depends_on": ["core.py"]},
            ]
        }
    )
    broken = "def broken(:\n"
    client = FakeClient([plan, "- spec", broken, broken])

    result = MultiFileLoop(client, config, session).run("x")

    assert not result.success
    assert {Path(f.path).name for f in result.files} == {"core.py", "main.py"}
    main_res = next(f for f in result.files if Path(f.path).name == "main.py")
    assert not main_res.success and not main_res.advisory
    events = [json.loads(line)["event"] for line in session.log_path.read_text().splitlines()]
    assert "skipped" in events


def test_a_dead_endpoint_does_not_sink_a_run_another_endpoint_can_finish(tmp_path: Path):
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    good = FakeClient(["- spec", "def thing():\n    return 1\n"] * 2, delay=0.05)
    # Exactly one queued OllamaError, no more -- proves the dead endpoint
    # is retired on the *first* failure, not retried in place, since a
    # second attempt would find its queue empty and crash the worker
    # thread with an AssertionError instead of cleanly failing over. A
    # good endpoint is still active, so there's no reason to wait out a
    # retry here rather than handing the file off immediately.
    dead = FakeClient([OllamaError("endpoint B is down")])

    result = MultiFileLoop(
        FakeClient([_LEAF_PAIR_PLAN]), config, session, pool_clients=[good, dead]
    ).run("two leaves")

    assert result.success
    assert not result.aborted
    assert {Path(f.path).name for f in result.files} == {"a.py", "b.py"}
    # the dropped endpoint is visible in the log, not just inferable from
    # the file it was assigned quietly reappearing on another worker
    events = [json.loads(line) for line in session.log_path.read_text().splitlines()]
    retired = [e for e in events if e["event"] == "endpoint_retired"]
    assert len(retired) == 1
    assert retired[0]["reason"] == "endpoint B is down"


def test_a_transient_blip_recovers_via_retry_when_no_other_endpoint_exists(tmp_path: Path):
    # The flip side of the "dead endpoint hands off immediately" test
    # above: with only one endpoint, there's no one to hand off to, so a
    # transient failure should be retried in place instead of aborting
    # the whole run over what clears up a moment later.
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    plan = json.dumps({"files": [{"path": "a.py", "purpose": "x", "depends_on": []}]})
    client = FakeClient(
        [
            plan,
            OllamaError("connection reset"),  # transient -- spec for a.py, first try
            "- spec",  # succeeds on retry
            "def thing():\n    return 1\n",
        ]
    )

    result = MultiFileLoop(client, config, session).run("goal")

    assert result.success
    assert not result.aborted
    assert len(client.calls) == 4


def test_a_transient_blip_during_planning_recovers_via_retry(tmp_path: Path):
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    plan = json.dumps({"files": [{"path": "a.py", "purpose": "x", "depends_on": []}]})
    client = FakeClient(
        [
            OllamaError("connection reset"),  # transient -- plan, first try
            plan,  # succeeds on retry
            "- spec",
            "def thing():\n    return 1\n",
        ]
    )

    result = MultiFileLoop(client, config, session).run("goal")

    assert result.success
    assert not result.aborted


def test_an_idle_worker_does_not_retire_just_because_the_only_file_is_already_claimed(
    tmp_path: Path,
):
    # Regression: with a single file, the second worker finds nothing to
    # claim from its very first check and (before the fix) retired for
    # good right then -- so when the worker actually holding that file
    # hit OllamaError and requeued it, nobody was left to pick it up and
    # the whole run aborted, even though a perfectly good second endpoint
    # was sitting right there. `dying` is first in the pool -- in CPython
    # the first-started thread reliably wins an uncontended lock this
    # short-lived, so it claims the only file; `good` finds pending empty
    # immediately and must not treat that as "no more work, ever".
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    plan = json.dumps({"files": [{"path": "a.py", "purpose": "x", "depends_on": []}]})
    # The delay gives the second worker time to observe "nothing to
    # claim" *before* the failure requeues the file -- without it, the
    # requeue can (depending on scheduling) land before the second worker
    # ever checks, which would pass by accident and prove nothing.
    dying = FakeClient([OllamaError("connection reset")], delay=0.1)
    good = FakeClient(["- spec", "def thing():\n    return 1\n"])

    result = MultiFileLoop(
        FakeClient([plan]), config, session, pool_clients=[dying, good]
    ).run("one leaf")

    assert result.success
    assert not result.aborted
    assert [Path(f.path).name for f in result.files] == ["a.py"]


def test_a_file_exceeding_the_fix_threshold_gets_branched_to_an_idle_endpoint(tmp_path: Path):
    # A file stuck deep in its fix loop can borrow an idle, branch-
    # eligible endpoint instead of leaving it sitting around -- see
    # _generate_files' claim_branch/settle. Both pool clients get an
    # identical, interchangeable queue since which one claims the file
    # first (and so plays "original" vs "branch") is a genuine race;
    # the assertions only check aggregate outcomes that must hold
    # regardless of who wins it.
    config = make_config(tmp_path).with_overrides(branch_after_fixes=1)
    session = Session.create(config.workspace_root)
    plan = json.dumps({"files": [{"path": "main.py", "purpose": "x", "depends_on": []}]})
    queue = ["- spec", "bad(", "bad(", "x = 1\n"]
    a = FakeClient([plan, *queue], host="http://a")
    b = FakeClient(list(queue), host="http://b")

    result = MultiFileLoop(
        a, config, session, pool_clients=[a, b], branch_pool=[a, b]
    ).run("goal")

    assert result.success
    assert (session.run_dir / "main.py").read_text() == "x = 1"
    assert not (session.run_dir / ".branch-main.py").exists()  # scratch file cleaned up

    events = [json.loads(line) for line in session.log_path.read_text().splitlines()]
    assert any(e["event"] == "branch_race_won" for e in events)
    assert sum(1 for e in events if e["event"] == "file_result") == 1  # never double-recorded


def test_an_endpoint_outside_the_branch_pool_never_branches(tmp_path: Path):
    # Only "a" is branch-eligible (imagine a real run: "b" tagged
    # "quick") -- "b" sitting idle must never pick up a's struggling
    # file even though branching is otherwise on.
    config = make_config(tmp_path, max_fix_attempts=2).with_overrides(branch_after_fixes=1)
    session = Session.create(config.workspace_root)
    plan = json.dumps({"files": [{"path": "main.py", "purpose": "x", "depends_on": []}]})
    a = FakeClient([plan, "- spec", "bad(", "bad(", "x = 1\n"], host="http://a")
    b = FakeClient([], host="http://b")  # would raise "ran out" if ever asked to do anything

    result = MultiFileLoop(
        a, config, session, pool_clients=[a, b], branch_pool=[a]
    ).run("goal")

    assert result.success
    assert b.calls == []


def test_branch_after_fixes_zero_disables_branching_entirely(tmp_path: Path):
    # The default -- must be a true no-op even with a fully eligible
    # branch pool and a file that genuinely struggles.
    config = make_config(tmp_path, max_fix_attempts=2)  # branch_after_fixes defaults to 0
    session = Session.create(config.workspace_root)
    plan = json.dumps({"files": [{"path": "main.py", "purpose": "x", "depends_on": []}]})
    a = FakeClient([plan, "- spec", "bad(", "bad(", "x = 1\n"], host="http://a")
    b = FakeClient([], host="http://b")

    result = MultiFileLoop(
        a, config, session, pool_clients=[a, b], branch_pool=[a, b]
    ).run("goal")

    assert result.success
    assert b.calls == []


def test_branching_never_kicks_in_before_the_threshold_is_crossed(tmp_path: Path):
    config = make_config(tmp_path).with_overrides(branch_after_fixes=10)
    session = Session.create(config.workspace_root)
    plan = json.dumps({"files": [{"path": "main.py", "purpose": "x", "depends_on": []}]})
    a = FakeClient([plan, "- spec", "x = 1\n"], host="http://a")  # succeeds first try, no fixes
    b = FakeClient([], host="http://b")

    result = MultiFileLoop(
        a, config, session, pool_clients=[a, b], branch_pool=[a, b]
    ).run("goal")

    assert result.success
    assert b.calls == []


def test_run_aborts_gracefully_when_the_model_becomes_unreachable(tmp_path: Path):
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    client = FakeClient(
        [
            _PLAN_TWO_FILES,
            "- add two numbers",  # spec for helper.py
            "def add(a, b):\n    return a + b\n",  # helper.py codegen
            # spec for main.py -- host gone. The sole worker retries its
            # own endpoint (there's no other to fall back to) up to
            # _ENDPOINT_RETRY_ATTEMPTS times before the run truly aborts.
            OllamaError("Read timed out"),
            OllamaError("Read timed out"),
            OllamaError("Read timed out"),
        ]
    )

    result = MultiFileLoop(client, config, session).run("a script that adds two numbers")

    assert not result.success
    assert result.aborted
    assert "timed out" in result.abort_reason
    # the work that completed before the outage is kept
    assert [Path(f.path).name for f in result.files] == ["helper.py"]
    events = [json.loads(line)["event"] for line in session.log_path.read_text().splitlines()]
    assert "run_aborted" in events
    assert events[-1] == "run_result"


def test_find_implicated_file_does_not_match_a_name_inside_another_name(tmp_path: Path):
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    loop = MultiFileLoop(FakeClient([]), config, session)

    domain = session.run_dir / "domain.py"
    main = session.run_dir / "main.py"
    error = f'File "{domain}", line 3, in <module>\n    boom\nNameError: boom'

    implicated = loop._find_implicated_file(error, [main, domain])

    assert implicated == domain


def test_integration_fix_writes_corrected_code_to_the_implicated_file(tmp_path: Path):
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    loop = MultiFileLoop(FakeClient(["def test_thing():\n    assert True\n"]), config, session)

    test_path = session.run_dir / "test_thing.py"
    test_path.write_text("def test_thing():\n    assert False\n")

    tasks = [FileTask(path="thing.py", purpose="x")]
    result, _iterations = loop._run_integration_with_fixes(
        tasks, [test_path], has_tests=True, iterations=1
    )

    assert test_path.read_text() == "def test_thing():\n    assert True"
    assert result.success


def test_integration_fix_recovers_from_a_transient_connection_blip(tmp_path: Path):
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    loop = MultiFileLoop(
        FakeClient(
            [
                OllamaError("connection reset"),  # transient -- first attempt
                "def test_thing():\n    assert True\n",  # succeeds on retry
            ]
        ),
        config,
        session,
    )

    test_path = session.run_dir / "test_thing.py"
    test_path.write_text("def test_thing():\n    assert False\n")

    tasks = [FileTask(path="thing.py", purpose="x")]
    result, _iterations = loop._run_integration_with_fixes(
        tasks, [test_path], has_tests=True, iterations=1
    )

    assert test_path.read_text() == "def test_thing():\n    assert True"
    assert result.success


def test_integration_fix_rounds_sample_at_a_rising_temperature(tmp_path: Path):
    config = make_config(tmp_path, max_fix_attempts=3)
    session = Session.create(config.workspace_root)
    client = FakeClient(["def test_thing():\n    assert False\n"] * 4)
    loop = MultiFileLoop(client, config, session)

    test_path = session.run_dir / "test_thing.py"
    test_path.write_text("def test_thing():\n    assert False\n")

    loop._run_integration_with_fixes(
        [FileTask(path="thing.py", purpose="x")], [test_path], has_tests=True, iterations=1
    )

    assert len(client.temperature_calls) == 3
    assert client.temperature_calls == sorted(client.temperature_calls)
    assert client.temperature_calls[0] > config.temperature


def test_a_cancelled_run_aborts_before_claiming_any_file(tmp_path: Path):
    config = make_config(tmp_path, max_fix_attempts=5)
    session = Session.create(config.workspace_root)
    plan = json.dumps({"files": [{"path": "a.py", "purpose": "x", "depends_on": []}]})
    # Never actually reached: a cancel already set before dispatch starts
    # is noticed at the top of the worker loop, before it claims a file.
    client = FakeClient([plan, "should not be requested"])
    cancel_event = threading.Event()
    cancel_event.set()

    result = MultiFileLoop(client, config, session, cancel_event=cancel_event).run("goal")

    assert not result.success
    assert result.aborted
    assert result.abort_reason == "cancelled by user"
    assert len(client.calls) == 1  # just the plan call


def test_a_paused_run_does_not_claim_a_file_until_resumed(tmp_path: Path):
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    plan = json.dumps({"files": [{"path": "a.py", "purpose": "x", "depends_on": []}]})
    client = FakeClient([plan, *_GENERIC])
    pause_event = threading.Event()
    pause_event.set()

    loop = MultiFileLoop(client, config, session, pause_event=pause_event)
    outcome: dict = {}
    t = threading.Thread(target=lambda: outcome.update(result=loop.run("goal")))
    t.start()
    try:
        # Give the worker loop plenty of chances to (wrongly) claim a.py
        # while paused -- only the plan call should have happened.
        time.sleep(0.6)
        assert len(client.calls) == 1
    finally:
        pause_event.clear()
        t.join(timeout=5)

    assert outcome["result"].success
    assert len(client.calls) == 3  # plan, spec, codegen


def test_settings_changed_while_paused_take_effect_on_the_next_fix_attempt(tmp_path: Path):
    # max_fix_attempts=1 alone would stop after exactly one fix attempt --
    # proves the bump to 3, applied while paused, is what lets a second
    # attempt (and the eventual success) happen at all. pause_event isn't
    # set until the initial codegen call has already landed: setting it
    # any earlier would trip the *dispatcher's* own pause checkpoint and
    # stop the file from ever being claimed, never reaching a fix attempt
    # at all (see test_a_paused_run_does_not_claim_a_file_until_resumed).
    config = make_config(tmp_path, max_fix_attempts=1)
    session = Session.create(config.workspace_root)
    plan = json.dumps({"files": [{"path": "a.py", "purpose": "x", "depends_on": []}]})
    client = FakeClient(
        [
            plan,
            "- do a thing",  # spec
            "not python(",  # codegen: invalid syntax
            "still not python(",  # fix attempt 1: also invalid
            "def thing():\n    return 1\n",  # fix attempt 2: valid
        ],
        delay=0.05,
    )
    pause_event = threading.Event()

    loop = MultiFileLoop(client, config, session, pause_event=pause_event)
    outcome: dict = {}
    t = threading.Thread(target=lambda: outcome.update(result=loop.run("goal")))
    t.start()
    try:
        deadline = time.monotonic() + 5
        while len(client.calls) < 3 and time.monotonic() < deadline:
            time.sleep(0.002)
        pause_event.set()
        assert len(client.calls) == 3  # plan, spec, codegen -- no fix call yet

        # Give the (paused) worker every chance to wrongly sneak a fix
        # call through before trusting that it hasn't.
        time.sleep(0.3)
        assert len(client.calls) == 3

        config.apply_overrides(max_fix_attempts=3)
    finally:
        pause_event.clear()
        t.join(timeout=5)

    assert outcome["result"].success
    assert len(client.calls) == 5  # plan, spec, codegen, fix 1, fix 2


def test_multi_file_loop_tracks_each_llm_call_with_its_kind(tmp_path: Path):
    config = make_config(tmp_path, critic_enabled=True)
    session = Session.create(config.workspace_root)
    seen: list[tuple[str, str, str]] = []
    real_track_call = session.track_call

    def spy(kind, path, endpoint):
        seen.append((kind, path, endpoint))
        return real_track_call(kind, path, endpoint)

    session.track_call = spy
    agree = '{"follows_spec": true, "issues": ""}'
    client = FakeClient(
        [
            _PLAN_TWO_FILES,
            "- add two numbers",  # spec for helper.py
            "def add(a, b):\n    return a + b\n",  # codegen for helper.py
            agree,  # critic for helper.py
            "- call add and print it",  # spec for main.py
            (
                "from helper import add\n\n\n"
                "def main():\n    print(add(2, 3))\n\n\n"
                'if __name__ == "__main__":\n    main()\n'
            ),  # codegen for main.py
            agree,  # critic for main.py
        ]
    )

    result = MultiFileLoop(client, config, session).run("a script that adds two numbers")

    assert result.success
    kinds = [kind for kind, _path, _ep in seen]
    assert kinds == ["plan", "spec", "codegen", "critic", "spec", "codegen", "critic"]
    assert all(ep == "http://fake" for _k, _p, ep in seen)
    # spec logs the bare task path; codegen/critic log the full file path
    assert seen[4] == ("spec", "main.py", "http://fake")
    assert seen[5][1].endswith("main.py")
    assert seen[6][1].endswith("main.py")


def test_integration_fix_is_tracked_as_its_own_kind(tmp_path: Path):
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    seen: list[str] = []
    real_track_call = session.track_call

    def spy(kind, path, endpoint):
        seen.append(kind)
        return real_track_call(kind, path, endpoint)

    session.track_call = spy
    loop = MultiFileLoop(FakeClient(["def test_thing():\n    assert True\n"]), config, session)

    test_path = session.run_dir / "test_thing.py"
    test_path.write_text("def test_thing():\n    assert False\n")

    loop._run_integration_with_fixes(
        [FileTask(path="thing.py", purpose="x")], [test_path], has_tests=True, iterations=1
    )

    assert seen == ["integration_fix"]


def test_super_review_finds_and_confirms_a_cross_file_issue(tmp_path: Path):
    config = make_config(tmp_path).with_overrides(super_review_enabled=True)
    session = Session.create(config.workspace_root)
    # `client` (plan + reviewer) and the sole pool client are deliberately
    # different objects, so confirm_client (self._pool minus self.client)
    # resolves to the pool client unambiguously.
    plan_and_review_client = FakeClient(
        [
            json.dumps({"files": [{"path": "main.py", "purpose": "x", "depends_on": []}]}),
            json.dumps(
                {"issues": [{"file": "main.py", "description": "doesn't do what the goal asked"}]}
            ),
        ]
    )
    builder_client = FakeClient(
        [
            "- print hello",
            "print('hello')\n",
            json.dumps({"confirmed": True}),
        ]
    )

    result = MultiFileLoop(
        plan_and_review_client, config, session, pool_clients=[builder_client]
    ).run("print hello")

    assert result.success
    assert result.cross_file_issues == [
        {"file": "main.py", "description": "doesn't do what the goal asked", "confirmed": True}
    ]


def test_super_review_shows_an_unconfirmed_issue_rather_than_dropping_it(tmp_path: Path):
    # A finding the second reviewer disagrees with is still reported --
    # silently dropping it would risk losing a real issue just because
    # two small models didn't happen to agree.
    config = make_config(tmp_path).with_overrides(super_review_enabled=True)
    session = Session.create(config.workspace_root)
    plan_and_review_client = FakeClient(
        [
            json.dumps({"files": [{"path": "main.py", "purpose": "x", "depends_on": []}]}),
            json.dumps({"issues": [{"file": "main.py", "description": "maybe an issue"}]}),
        ]
    )
    builder_client = FakeClient(
        [
            "- print hello",
            "print('hello')\n",
            json.dumps({"confirmed": False}),
        ]
    )

    result = MultiFileLoop(
        plan_and_review_client, config, session, pool_clients=[builder_client]
    ).run("print hello")

    assert result.success
    assert result.cross_file_issues == [
        {"file": "main.py", "description": "maybe an issue", "confirmed": False}
    ]


def test_super_review_skips_confirmation_with_only_one_endpoint(tmp_path: Path):
    # No second, distinct client to ask -- every finding comes back
    # unconfirmed rather than the feature crashing or inventing a verdict.
    config = make_config(tmp_path).with_overrides(super_review_enabled=True)
    session = Session.create(config.workspace_root)
    client = FakeClient(
        [
            json.dumps({"files": [{"path": "main.py", "purpose": "x", "depends_on": []}]}),
            "- print hello",
            "print('hello')\n",
            json.dumps({"issues": [{"file": "main.py", "description": "an issue"}]}),
        ]
    )

    result = MultiFileLoop(client, config, session).run("print hello")

    assert result.success
    assert result.cross_file_issues == [
        {"file": "main.py", "description": "an issue", "confirmed": False}
    ]


def test_super_review_is_off_by_default(tmp_path: Path):
    config = make_config(tmp_path)  # super_review_enabled defaults False
    session = Session.create(config.workspace_root)
    client = FakeClient(
        [
            json.dumps({"files": [{"path": "main.py", "purpose": "x", "depends_on": []}]}),
            "- print hello",
            "print('hello')\n",
        ]
    )

    result = MultiFileLoop(client, config, session).run("print hello")

    assert result.success
    assert result.cross_file_issues == []
    assert len(client.calls) == 3  # no extra review call made
