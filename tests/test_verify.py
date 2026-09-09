from pathlib import Path

from harness.steps.verify import (
    compile_check,
    import_check,
    lint_check,
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
    # to flag or auto-fix), but the import doesn't resolve -- proves
    # import_check runs as a third stage, not instead of compile/lint.
    f = tmp_path / "main.py"
    f.write_text("import definitely_not_a_real_module_xyz\n\nprint(definitely_not_a_real_module_xyz)\n")
    result = verify_python_file_static(f)
    assert not result.success
    assert result.stage == "import"


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
