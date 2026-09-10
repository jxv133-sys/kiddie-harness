import json
from pathlib import Path

from harness.llm_client import OllamaError
from harness.orchestrator import MultiFileLoop
from harness.session import Session
from harness.steps.plan import FileTask

from .fakes import FakeClient, make_config

_PLAN_TWO_FILES = json.dumps(
    {
        "files": [
            {"path": "helper.py", "purpose": "add two numbers"},
            {"path": "main.py", "purpose": "entry point"},
        ]
    }
)


def test_succeeds_across_files_with_tests_and_integration_check(tmp_path: Path):
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    client = FakeClient(
        [
            _PLAN_TWO_FILES,
            "- add two numbers",  # spec for helper.py
            "def add(a, b):\n    return a + b\n",  # codegen for helper.py
            "from helper import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",  # test for helper.py
            "- expose a main() that adds two numbers and print it when run",  # spec for main.py
            (
                "from helper import add\n\n\n"
                "def main():\n    return add(2, 3)\n\n\n"
                'if __name__ == "__main__":\n    print(main())\n'
            ),  # codegen for main.py
            "from main import main\n\ndef test_main():\n    assert main() == 5\n",  # test for main.py
        ]
    )

    result = MultiFileLoop(client, config, session).run("a script that adds two numbers")

    assert result.success
    assert not result.stopped_early
    assert [f.path for f in result.files] == [
        str(session.run_dir / "helper.py"),
        str(session.run_dir / "test_helper.py"),
        str(session.run_dir / "main.py"),
        str(session.run_dir / "test_main.py"),
    ]
    assert all(f.success for f in result.files)
    assert result.integration is not None
    assert result.integration.success
    assert result.integration.stage == "pytest"
    assert (session.run_dir / "helper.py").read_text() == "def add(a, b):\n    return a + b"


def test_catches_and_fixes_a_bad_cross_file_import_during_its_own_generation(tmp_path: Path):
    # Mirrors a real failure: a file imports a sibling module under the
    # wrong name. The bug must be caught (and fixed) during that file's
    # own generation, not misattributed to its later companion test file.
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    client = FakeClient(
        [
            _PLAN_TWO_FILES,
            "- add two numbers",  # spec for helper.py
            "def add(a, b):\n    return a + b\n",  # codegen for helper.py
            "from helper import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",  # test for helper.py
            "- expose a main() that adds two numbers and print it when run",  # spec for main.py
            "from helpr import add\n\ndef main():\n    return add(2, 3)\n",  # codegen: wrong module name
            "from helper import add\n\ndef main():\n    return add(2, 3)\n",  # fix: corrected import
            "from main import main\n\ndef test_main():\n    assert main() == 5\n",  # test for main.py
        ]
    )

    result = MultiFileLoop(client, config, session).run("a script that adds two numbers")

    assert result.success
    main_file = next(f for f in result.files if Path(f.path).name == "main.py")
    assert main_file.attempts == 1
    test_files = [f for f in result.files if Path(f.path).name.startswith("test_")]
    assert len(test_files) == 2  # test generation for main.py still happened after the fix


def test_auto_fixes_a_per_file_lint_issue_without_calling_the_llm(tmp_path: Path):
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    client = FakeClient(
        [
            json.dumps({"files": [{"path": "main.py", "purpose": "entry point"}]}),
            "- print hello",  # spec
            "import os\nprint('hello')\n",  # codegen: unused import -- ruff auto-fixes this itself
            "def test_placeholder():\n    assert True\n",  # test for main.py
        ]
    )

    result = MultiFileLoop(client, config, session).run("print hello")

    assert result.success
    assert len(result.files) == 2
    assert result.files[0].attempts == 0
    assert len(client.calls) == 4  # no LLM fix call was needed


def test_recovers_from_an_unfixable_per_file_lint_issue(tmp_path: Path):
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    client = FakeClient(
        [
            json.dumps({"files": [{"path": "main.py", "purpose": "entry point"}]}),
            "- print hello",  # spec
            "print(undefined_name)\n",  # codegen: undefined name -- ruff can't fix this itself
            "print('hello')\n",  # fix: clean
            "def test_placeholder():\n    assert True\n",  # test for main.py
        ]
    )

    result = MultiFileLoop(client, config, session).run("print hello")

    assert result.success
    assert len(result.files) == 2
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


def test_test_generation_prompt_includes_the_real_module_source(tmp_path: Path):
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    client = FakeClient(
        [
            json.dumps({"files": [{"path": "main.py", "purpose": "entry point"}]}),
            "- return None on bad input",  # spec
            "def handle(x):\n    return None\n",  # codegen
            "from main import handle\n\ndef test_handle():\n    assert handle(1) is None\n",
        ]
    )

    MultiFileLoop(client, config, session).run("do a thing")

    testgen_prompt = client.calls[3]  # plan, spec, codegen, testgen
    assert "def handle(x):" in testgen_prompt
    assert "return None" in testgen_prompt


def test_test_fix_prompt_includes_the_real_module_source(tmp_path: Path):
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    client = FakeClient(
        [
            json.dumps({"files": [{"path": "main.py", "purpose": "entry point"}]}),
            "- a function f",  # spec
            "def f():\n    return 1\n",  # codegen -- compiles, lints, imports
            "from main import f\n\ndef test_f():\n    assert f() == 2\n",  # test: fails pytest
            "from main import f\n\ndef test_f():\n    assert f() == 1\n",  # test fix: passes
        ]
    )

    result = MultiFileLoop(client, config, session).run("a script")

    assert result.success
    test_fix_prompt = client.calls[4]  # plan, spec, codegen, testgen, test-fix
    assert "def f():\n    return 1" in test_fix_prompt


def test_a_companion_test_that_never_passes_is_advisory_not_fatal(tmp_path: Path):
    config = make_config(tmp_path, max_fix_attempts=3)
    session = Session.create(config.workspace_root)
    bad_test = 'from main import greet\n\ndef test_greet():\n    assert greet() == "bye"\n'
    client = FakeClient(
        [
            json.dumps({"files": [{"path": "main.py", "purpose": "entry point"}]}),
            "- expose greet()",  # spec
            (
                'def greet():\n    return "hi"\n\n\n'
                'if __name__ == "__main__":\n    print(greet())\n'
            ),  # codegen -- runs clean, imports clean
            bad_test,  # companion test: fails
            bad_test,  # fix 1: still fails
            bad_test,  # fix 2
            bad_test,  # fix 3
        ]
    )

    result = MultiFileLoop(client, config, session).run("a greeter")

    assert result.success  # the deliverable works, so the run succeeds
    test_result = next(f for f in result.files if Path(f.path).name == "test_main.py")
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
    client = FakeClient(
        [
            json.dumps({"files": [{"path": "calc.py", "purpose": "adder"}]}),
            "- expose add()",
            "def add(a, b):\n    return a + b\n",
            "from calc import add\n\ndef test_add():\n    assert add(1, 1) == 3\n",  # wrong
            "from calc import add\n\ndef test_add():\n    assert add(1, 1) == 3\n",  # fix: still wrong
        ]
    )

    result = MultiFileLoop(client, config, session).run("an adder")

    # calc.py has no runnable entry and its only test is advisory-failed;
    # integration has nothing left to fail on, so the run still succeeds.
    assert result.success
    assert [Path(f.path).name for f in result.files if f.advisory] == ["test_calc.py"]


def test_run_aborts_gracefully_when_the_model_becomes_unreachable(tmp_path: Path):
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    client = FakeClient(
        [
            _PLAN_TWO_FILES,
            "- add two numbers",  # spec for helper.py
            "def add(a, b):\n    return a + b\n",  # helper.py codegen
            "from helper import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",  # test
            OllamaError("Read timed out"),  # spec for main.py -- host gone
        ]
    )

    result = MultiFileLoop(client, config, session).run("a script that adds two numbers")

    assert not result.success
    assert result.aborted
    assert "timed out" in result.abort_reason
    # the work that completed before the outage is kept
    assert [Path(f.path).name for f in result.files] == ["helper.py", "test_helper.py"]
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
