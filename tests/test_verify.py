from pathlib import Path

from harness.steps.verify import (
    compile_check,
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


def test_lint_check_fails_on_unused_import(tmp_path: Path):
    f = tmp_path / "unused_import.py"
    f.write_text("import os\nx = 1\n")
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


def test_verify_python_file_static_never_executes_the_file(tmp_path: Path):
    # A file that would fail if actually run (the module doesn't exist) but
    # is syntactically valid and doesn't trip any lint rule must still pass
    # the static check -- neither compiling nor linting executes imports.
    f = tmp_path / "ok.py"
    f.write_text("import definitely_not_a_real_module_xyz\n\nprint(definitely_not_a_real_module_xyz)\n")
    result = verify_python_file_static(f)
    assert result.success


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
