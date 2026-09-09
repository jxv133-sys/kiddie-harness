"""Deterministic verification of generated Python files.

This is the "gold standard" feedback signal for the fix loop: a real
compiler/interpreter result, never another LLM's opinion. Nothing here
calls the LLM.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path


@dataclasses.dataclass
class VerifyResult:
    success: bool
    stage: str  # "compile" or "run"
    output: str  # combined stdout+stderr, truncated for prompt reuse


def _truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def compile_check(path: Path) -> VerifyResult:
    """Syntax-check the file without executing it."""
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    combined = result.stdout + result.stderr
    return VerifyResult(success=result.returncode == 0, stage="compile", output=_truncate(combined))


def run_script(path: Path, timeout_seconds: int = 15) -> VerifyResult:
    """Execute the file and report whether it exited cleanly."""
    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        combined = (exc.stdout or "") + (exc.stderr or "")
        return VerifyResult(
            success=False,
            stage="run",
            output=_truncate(combined + f"\n[timed out after {timeout_seconds}s]"),
        )
    combined = result.stdout + result.stderr
    return VerifyResult(success=result.returncode == 0, stage="run", output=_truncate(combined))


def verify_python_file(path: Path, *, execute: bool = True, timeout_seconds: int = 15) -> VerifyResult:
    """Compile-check first (cheap, catches syntax errors fast), then run."""
    compiled = compile_check(path)
    if not compiled.success or not execute:
        return compiled
    return run_script(path, timeout_seconds=timeout_seconds)


def lint_check(path: Path) -> VerifyResult:
    """Static lint only -- never executes the file, safe even when sibling
    files it imports don't exist yet.

    Runs with --fix: ruff only applies fixes it considers safe (no
    semantic changes), so this can't change program behavior, and it
    means trivially-fixable nits (import sorting/formatting) never cost
    an LLM fix attempt -- only violations ruff can't fix itself count as
    a failure.
    """
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--fix", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    combined = result.stdout + result.stderr
    return VerifyResult(success=result.returncode == 0, stage="lint", output=_truncate(combined))


def import_check(path: Path, timeout_seconds: int = 15) -> VerifyResult:
    """Actually resolve the file's imports, without running its logic.

    Files are generated in the planner's declared dependency order, so by
    the time any file's own verify step runs, everything it can
    legitimately depend on already exists on disk -- it's safe to import
    it for real, not just syntax-check it. Uses runpy.run_path with a
    run_name other than "__main__" so any `if __name__ == "__main__":`
    block never executes (no side effects, only import-time resolution is
    checked). The path is passed as a real subprocess argument, never
    interpolated into the executed Python source text, so nothing
    planner-controlled ever becomes part of the code that actually runs.

    Resolved to absolute first: cwd is set to the file's own directory
    (empirically required for runpy.run_path's sibling-import resolution
    to work), and running against a caller-relative path (as `path` is
    in practice, since config/default.yaml's workspace root is relative)
    would otherwise get re-resolved against that new cwd and double up.
    """
    path = path.resolve()
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import runpy, sys; runpy.run_path(sys.argv[1], run_name='not_main')",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            cwd=path.parent,
        )
    except subprocess.TimeoutExpired as exc:
        combined = (exc.stdout or "") + (exc.stderr or "")
        return VerifyResult(
            success=False,
            stage="import",
            output=_truncate(combined + f"\n[timed out after {timeout_seconds}s]"),
        )
    combined = result.stdout + result.stderr
    return VerifyResult(success=result.returncode == 0, stage="import", output=_truncate(combined))


def verify_python_file_static(path: Path) -> VerifyResult:
    """Compile-check, lint-check, then import-check. Used while a
    multi-file project is still being assembled."""
    compiled = compile_check(path)
    if not compiled.success:
        return compiled
    linted = lint_check(path)
    if not linted.success:
        return linted
    return import_check(path)


def run_pytest(target: Path, timeout_seconds: int = 60) -> VerifyResult:
    """Run pytest against a directory (full suite) or a single test file."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(target), "-q"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        combined = (exc.stdout or "") + (exc.stderr or "")
        return VerifyResult(
            success=False,
            stage="pytest",
            output=_truncate(combined + f"\n[timed out after {timeout_seconds}s]"),
        )
    combined = result.stdout + result.stderr
    return VerifyResult(success=result.returncode == 0, stage="pytest", output=_truncate(combined))


def verify_test_file(path: Path, timeout_seconds: int = 30) -> VerifyResult:
    """Compile-check a single test file, then run just that file with pytest.

    Scoped to one file: the module it imports already exists on disk by the
    time this runs (its own generate/verify/fix loop already succeeded).
    """
    compiled = compile_check(path)
    if not compiled.success:
        return compiled
    return run_pytest(path, timeout_seconds=timeout_seconds)
