from pathlib import Path

from harness.steps.verify import batch_check, powershell_check, verify_generated_file


def test_batch_check_passes_on_a_well_formed_script(tmp_path: Path):
    f = tmp_path / "run.bat"
    f.write_text('@echo off\nset NAME=world\necho hello %NAME%\nif exist "out.txt" (\n  echo found\n)\n')
    result = batch_check(f)
    assert result.success
    assert result.stage == "compile"


def test_batch_check_fails_on_an_unclosed_paren(tmp_path: Path):
    f = tmp_path / "run.bat"
    f.write_text("@echo off\nif exist out.txt (\n  echo found\n")
    result = batch_check(f)
    assert not result.success


def test_batch_check_fails_on_an_unterminated_string(tmp_path: Path):
    f = tmp_path / "run.bat"
    f.write_text('@echo off\nset MSG="unterminated\n')
    result = batch_check(f)
    assert not result.success


def test_batch_check_ignores_double_colon_comments(tmp_path: Path):
    f = tmp_path / "run.bat"
    f.write_text(":: an unbalanced ( here\n@echo off\nset X=1\n")
    result = batch_check(f)
    assert result.success


def test_batch_check_does_not_treat_a_backslash_as_an_escape_character(tmp_path: Path):
    # \ is a literal Windows path separator in batch, not an escape char
    # -- a naive JS-style checker would misread the \" here as an
    # escaped quote and never see the string close.
    f = tmp_path / "run.bat"
    f.write_text('@echo off\nset "DIR=C:\\Program Files\\"\necho %DIR%\n')
    result = batch_check(f)
    assert result.success


def test_batch_check_fails_on_prose_with_no_commands(tmp_path: Path):
    f = tmp_path / "run.bat"
    f.write_text("Sure, here is the batch script you asked for.")
    result = batch_check(f)
    assert not result.success
    assert "no batch commands" in result.output


def test_powershell_check_passes_on_a_well_formed_script(tmp_path: Path):
    f = tmp_path / "run.ps1"
    f.write_text('$name = "world"\nfunction Greet {\n    Write-Host "hello $name"\n}\nGreet\n')
    result = powershell_check(f)
    assert result.success
    assert result.stage == "compile"


def test_powershell_check_fails_on_an_unclosed_brace(tmp_path: Path):
    f = tmp_path / "run.ps1"
    f.write_text('function Greet {\n    Write-Host "hi"\n')
    result = powershell_check(f)
    assert not result.success


def test_powershell_check_fails_on_an_unterminated_string(tmp_path: Path):
    f = tmp_path / "run.ps1"
    f.write_text("$x = 'unterminated\n")
    result = powershell_check(f)
    assert not result.success


def test_powershell_check_ignores_line_and_block_comments(tmp_path: Path):
    f = tmp_path / "run.ps1"
    f.write_text("# unbalanced { here\n<# also ( unbalanced #>\n$x = 1\n")
    result = powershell_check(f)
    assert result.success


def test_powershell_check_honours_backtick_escaping_inside_strings(tmp_path: Path):
    # PowerShell escapes with a backtick, not a backslash -- `" inside a
    # double-quoted string must not be read as closing it.
    f = tmp_path / "run.ps1"
    f.write_text('$msg = "she said `"hi`" to me"\nWrite-Host $msg\n')
    result = powershell_check(f)
    assert result.success


def test_powershell_check_does_not_treat_a_backslash_as_an_escape_character(tmp_path: Path):
    f = tmp_path / "run.ps1"
    f.write_text('$dir = "C:\\Users\\test"\nWrite-Host $dir\n')
    result = powershell_check(f)
    assert result.success


def test_powershell_check_fails_on_prose_with_no_statements(tmp_path: Path):
    f = tmp_path / "run.ps1"
    f.write_text("Sure, here is the PowerShell script you asked for.")
    result = powershell_check(f)
    assert not result.success
    assert "no PowerShell statements" in result.output


def test_verify_generated_file_routes_bat_cmd_and_ps1_by_extension(tmp_path: Path):
    bat = tmp_path / "run.bat"
    bat.write_text("@echo off\nset X=1\n")
    cmd = tmp_path / "run.cmd"
    cmd.write_text("@echo off\nset X=1\n")
    ps1 = tmp_path / "run.ps1"
    ps1.write_text("$x = 1\n")

    assert verify_generated_file(bat).success
    assert verify_generated_file(cmd).success
    assert verify_generated_file(ps1).success
