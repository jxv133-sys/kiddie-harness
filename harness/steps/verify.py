"""Deterministic verification of generated Python files.

This is the "gold standard" feedback signal for the fix loop: a real
compiler/interpreter result, never another LLM's opinion. Nothing here
calls the LLM.
"""

from __future__ import annotations

import ast
import dataclasses
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _clear_pycache(target: Path) -> None:
    """Remove any __pycache__ under target before a pytest run.

    Guards against a stale assertion-rewrite .pyc left by an earlier run
    (or by something other than pytest) whose (mtime, size) happens to
    match a freshly rewritten test file.
    """
    root = target if target.is_dir() else target.parent
    for cache_dir in root.rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)


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


def _is_main_guard(node: ast.stmt) -> bool:
    """True if node is an `if __name__ == "__main__":` block."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


_DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_CONTROL_FLOW_NODES = (
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.If,
    ast.Match,
)


def main_guard_check(path: Path) -> VerifyResult:
    """Fail if a module that defines reusable code also runs code at import.

    A multi-file project's files are imported for real by `import_check`,
    by their companion test, and at the integration step. A file that
    defines functions/classes (so it *will* be imported) but also parses
    `sys.argv` or calls its entry point at module level -- with no
    `if __name__ == "__main__":` guard -- executes, and often `sys.exit`s,
    the moment it's imported, which those steps then report as an opaque
    failure. Catch it here with an instruction the fixer can act on.

    A file with no defs at all is treated as a plain script and left
    alone; module-level imports, constants and docstrings are always
    fine.
    """
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as exc:  # compile_check already reports this properly
        return VerifyResult(success=False, stage="guard", output=str(exc))

    if not any(isinstance(node, _DEF_NODES) for node in tree.body):
        return VerifyResult(success=True, stage="guard", output="")

    flagged: list[str] = []
    for node in tree.body:
        if _is_main_guard(node):
            continue
        if isinstance(node, _CONTROL_FLOW_NODES):
            flagged.append(type(node).__name__)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            flagged.append("call")

    if flagged:
        kinds = ", ".join(sorted(set(flagged)))
        return VerifyResult(
            success=False,
            stage="guard",
            output=(
                f"This module defines functions or classes but also executes "
                f"code at import time (top-level {kinds} outside any function). "
                f"Move every executable top-level statement into an "
                f'`if __name__ == "__main__":` block, keeping the functions, '
                f"imports and constants at module level, so the file can be "
                f"imported without running or exiting."
            ),
        )
    return VerifyResult(success=True, stage="guard", output="")


def verify_python_file_static(path: Path) -> VerifyResult:
    """Compile-check, lint-check, main-guard-check, then import-check. Used
    while a multi-file project is still being assembled."""
    compiled = compile_check(path)
    if not compiled.success:
        return compiled
    linted = lint_check(path)
    if not linted.success:
        return linted
    guarded = main_guard_check(path)
    if not guarded.success:
        return guarded
    return import_check(path)


def run_pytest(target: Path, timeout_seconds: int = 60) -> VerifyResult:
    """Run pytest against a directory (full suite) or a single test file.

    The fix loop rewrites a test file in place and re-runs pytest against
    it. Two successive versions of that file can have the same size and an
    almost-identical mtime, and pytest's assertion-rewrite cache
    (`__pycache__/*.pyc`) is keyed on exactly (mtime, size) -- so pytest
    can silently execute the *previous* version's bytecode and report a
    phantom pass or failure. Clearing __pycache__ and setting
    PYTHONDONTWRITEBYTECODE makes pytest rewrite assertions in memory
    every run and never consult that cache; -p no:cacheprovider keeps it
    from littering the generated project with a .pytest_cache dir.
    """
    _clear_pycache(target)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(target), "-q", "-p", "no:cacheprovider"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
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
