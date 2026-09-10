import os
from pathlib import Path

from harness.steps.verify import (
    compile_check,
    import_check,
    lint_check,
    main_guard_check,
    run_pytest,
    run_script,
    verify_python_file,
    verify_python_file_static,
)


def test_compile_check_passes_on_valid_syntax(tmp_path: Path):
    f = tmp_path / "ok.py"
    f.write_text("x = 1\n")
    result = compile_check(f)
    assert result.success
    assert result.stage == "compile"


def test_compile_check_fails_on_syntax_error(tmp_path: Path):
    f = tmp_path / "bad.py"
    f.write_text("def broken(:\n")
    result = compile_check(f)
    assert not result.success
    assert result.stage == "compile"
    assert result.output


def test_run_script_reports_success(tmp_path: Path):
    f = tmp_path / "ok.py"
    f.write_text("print('hello')\n")
    result = run_script(f)
    assert result.success
    assert result.stage == "run"


def test_run_script_reports_runtime_error(tmp_path: Path):
    f = tmp_path / "raises.py"
    f.write_text("raise ValueError('boom')\n")
    result = run_script(f)
    assert not result.success
    assert "ValueError" in result.output


def test_verify_python_file_stops_at_compile_stage_on_syntax_error(tmp_path: Path):
    f = tmp_path / "bad.py"
    f.write_text("def broken(:\n")
    result = verify_python_file(f)
    assert not result.success
    assert result.stage == "compile"


def test_verify_python_file_runs_after_successful_compile(tmp_path: Path):
    f = tmp_path / "ok.py"
    f.write_text("print('hello')\n")
    result = verify_python_file(f)
    assert result.success
    assert result.stage == "run"


def test_lint_check_passes_on_clean_file(tmp_path: Path):
    f = tmp_path / "ok.py"
    f.write_text("x = 1\nprint(x)\n")
    result = lint_check(f)
    assert result.success
    assert result.stage == "lint"


def test_lint_check_auto_fixes_an_unused_import(tmp_path: Path):
    # Unused imports are safe for ruff to fix itself -- this must succeed
    # without needing an LLM fix call.
    f = tmp_path / "unused_import.py"
    f.write_text("import os\nx = 1\n")
    result = lint_check(f)
    assert result.success
    assert result.stage == "lint"
    assert "os" not in f.read_text()


def test_lint_check_fails_on_a_violation_ruff_cannot_fix(tmp_path: Path):
    # An undefined name isn't something ruff can guess a fix for, so it
    # must still be reported as a real failure.
    f = tmp_path / "undefined_name.py"
    f.write_text("print(undefined_name)\n")
    result = lint_check(f)
    assert not result.success
    assert result.stage == "lint"
    assert result.output


def test_verify_python_file_static_stops_at_compile_on_syntax_error(tmp_path: Path):
    f = tmp_path / "bad.py"
    f.write_text("def broken(:\n")
    result = verify_python_file_static(f)
    assert not result.success
    assert result.stage == "compile"


def test_import_check_succeeds_when_a_sibling_module_exists(tmp_path: Path):
    (tmp_path / "helper.py").write_text("def add(a, b):\n    return a + b\n")
    f = tmp_path / "main.py"
    f.write_text("from helper import add\n\nprint(add(2, 3))\n")
    result = import_check(f)
    assert result.success
    assert result.stage == "import"


def test_import_check_fails_on_an_unresolvable_import(tmp_path: Path):
    f = tmp_path / "main.py"
    f.write_text("import definitely_not_a_real_module_xyz\n")
    result = import_check(f)
    assert not result.success
    assert result.stage == "import"
    assert "definitely_not_a_real_module_xyz" in result.output


def test_import_check_succeeds_with_a_relative_path_from_a_different_cwd(tmp_path: Path, monkeypatch):
    # Regression: config/default.yaml's workspace root is relative, so in
    # real CLI use `path` is a relative path like "workspace/<run-id>/x.py"
    # -- and the caller's cwd need not match the file's directory at all.
    # A prior version of import_check set cwd to the file's own directory
    # but still passed that same relative string as the argument, which
    # got re-resolved against the new cwd and doubled the path.
    project_dir = tmp_path / "project"
    run_dir = project_dir / "workspace" / "run123"
    run_dir.mkdir(parents=True)
    (run_dir / "helper.py").write_text("def add(a, b):\n    return a + b\n")
    (run_dir / "main.py").write_text("from helper import add\n\nprint(add(2, 3))\n")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    relative_path = Path("..") / "project" / "workspace" / "run123" / "main.py"
    result = import_check(relative_path)

    assert result.success
    assert result.stage == "import"


def test_import_check_does_not_execute_the_main_block(tmp_path: Path):
    # A __main__ block that would raise if actually run must not fire --
    # only import-time resolution is being checked.
    f = tmp_path / "main.py"
    f.write_text('if __name__ == "__main__":\n    raise RuntimeError("should not run")\n')
    result = import_check(f)
    assert result.success


def test_verify_python_file_static_stops_at_lint_before_import_check(tmp_path: Path):
    f = tmp_path / "undefined_name.py"
    f.write_text("print(undefined_name)\n")
    result = verify_python_file_static(f)
    assert not result.success
    assert result.stage == "lint"


def test_verify_python_file_static_runs_import_check_after_compile_and_lint(tmp_path: Path):
    # Compiles fine, lints clean (the import is used, so ruff has nothing
    # to flag or auto-fix), no unguarded top-level code, but the import
    # doesn't resolve -- proves import_check runs as the last stage, not
    # instead of the earlier ones.
    f = tmp_path / "main.py"
    f.write_text(
        "import definitely_not_a_real_module_xyz\n\n"
        "VALUE = definitely_not_a_real_module_xyz.thing\n"
    )
    result = verify_python_file_static(f)
    assert not result.success
    assert result.stage == "import"


def test_main_guard_check_passes_on_a_pure_module(tmp_path: Path):
    f = tmp_path / "lib.py"
    f.write_text('"""docs."""\nimport sys\n\nMAX = 10\n\ndef go(x):\n    return x + 1\n')
    result = main_guard_check(f)
    assert result.success
    assert result.stage == "guard"


def test_main_guard_check_passes_when_top_level_code_is_guarded(tmp_path: Path):
    f = tmp_path / "main.py"
    f.write_text(
        "import sys\n\n"
        "def main():\n    print(sys.argv)\n\n"
        'if __name__ == "__main__":\n    main()\n'
    )
    assert main_guard_check(f).success


def test_main_guard_check_flags_an_unguarded_entry_script(tmp_path: Path):
    f = tmp_path / "main.py"
    f.write_text(
        "import sys\n\n"
        "def add(a, b):\n    return a + b\n\n"
        'if len(sys.argv) != 3:\n    print("usage")\n    sys.exit(1)\n\n'
        "print(add(float(sys.argv[1]), float(sys.argv[2])))\n"
    )
    result = main_guard_check(f)
    assert not result.success
    assert result.stage == "guard"
    assert "__main__" in result.output


def test_verify_python_file_static_stops_at_guard_before_import_check(tmp_path: Path):
    helper = tmp_path / "helper.py"
    helper.write_text("def add(a, b):\n    return a + b\n")
    f = tmp_path / "main.py"
    f.write_text("from helper import add\n\ndef run():\n    return add(1, 2)\n\nprint(run())\n")
    result = verify_python_file_static(f)
    assert not result.success
    assert result.stage == "guard"


def test_main_guard_check_ignores_a_plain_script_with_no_defs(tmp_path: Path):
    f = tmp_path / "script.py"
    f.write_text("import sys\n\nif len(sys.argv) > 1:\n    print(sys.argv[1])\n")
    assert main_guard_check(f).success


def test_run_pytest_passes_on_passing_test(tmp_path: Path):
    (tmp_path / "test_sample.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n")
    result = run_pytest(tmp_path)
    assert result.success
    assert result.stage == "pytest"


def test_run_pytest_fails_on_failing_test(tmp_path: Path):
    (tmp_path / "test_sample.py").write_text("def test_bad():\n    assert 1 + 1 == 3\n")
    result = run_pytest(tmp_path)
    assert not result.success
    assert result.stage == "pytest"


def test_run_pytest_ignores_the_paths_it_is_told_to(tmp_path: Path):
    (tmp_path / "test_good.py").write_text("def test_ok():\n    assert True\n")
    bad = tmp_path / "test_bad.py"
    bad.write_text("def test_bad():\n    assert False\n")

    assert not run_pytest(tmp_path).success
    assert run_pytest(tmp_path, ignore=[bad]).success


def test_run_pytest_sees_an_in_place_rewrite_of_the_same_size(tmp_path: Path, monkeypatch):
    # The fix loop rewrites a test file and re-runs pytest. If the new
    # version has the same byte length and a near-identical mtime, pytest's
    # assertion-rewrite .pyc cache can serve the old bytecode. Force the
    # mtime to be identical across both writes to make the collision
    # deterministic, then prove run_pytest still reports the new result.
    test_file = tmp_path / "test_sample.py"
    module = tmp_path / "sample.py"
    module.write_text("def value():\n    return 1\n")

    original_write = Path.write_text

    def frozen_mtime_write(self, data, *args, **kwargs):
        result = original_write(self, data, *args, **kwargs)
        os.utime(self, (1_000_000_000, 1_000_000_000))
        return result

    monkeypatch.setattr(Path, "write_text", frozen_mtime_write)

    test_file.write_text("from sample import value\n\ndef test_v():\n    assert value() == 2\n")
    assert not run_pytest(test_file).success

    test_file.write_text("from sample import value\n\ndef test_v():\n    assert value() == 1\n")
    assert run_pytest(test_file).success
