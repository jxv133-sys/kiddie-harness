from pathlib import Path

from harness.steps.verify import compile_check, run_script, verify_python_file


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
