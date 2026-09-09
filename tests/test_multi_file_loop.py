import json
from pathlib import Path

from harness.orchestrator import MultiFileLoop
from harness.session import Session

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


def test_recovers_from_a_per_file_lint_failure(tmp_path: Path):
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    client = FakeClient(
        [
            json.dumps({"files": [{"path": "main.py", "purpose": "entry point"}]}),
            "- print hello",  # spec
            "import os\nprint('hello')\n",  # codegen: unused import -> lint failure
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
    assert session.run_dir in written_path.parents
    assert written_path == session.run_dir / "etc" / "evil.py"


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
