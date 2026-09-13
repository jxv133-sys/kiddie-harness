"""Deterministic verification of generated files.

This is the "gold standard" feedback signal for the fix loop: a real
compiler/interpreter result, never another LLM's opinion. Nothing here
calls the LLM. Most of this file is Python-specific (py_compile, ruff,
pytest, real import resolution); `html_check`/`css_check`/`js_check` are
the equivalent for the other languages the planner can produce, using
hand-rolled structural checks since no such tooling exists in the
stdlib -- still real and deterministic, just less capable.
"""

from __future__ import annotations

import ast
import dataclasses
import os
import shutil
import subprocess
import sys
from html.parser import HTMLParser
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

    A multi-file project's files are imported for real by `import_check`
    and at the integration step. A file that
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


_VOID_HTML_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}


class _HTMLBalanceChecker(HTMLParser):
    """Tracks open/close tags to catch the structural mistake a small
    model actually makes with HTML: an unclosed, mismatched, or stray
    closing tag. `HTMLParser` itself is deliberately lenient (built to
    tolerate real-world markup, like a browser) and will not raise on
    almost anything -- it is not a validator on its own, so this walks
    its callbacks and does the balance-checking itself."""

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[tuple[str, int]] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() not in _VOID_HTML_TAGS:
            self.stack.append((tag.lower(), self.getpos()[0]))

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        line = self.getpos()[0]
        if not self.stack:
            self.errors.append(f"line {line}: closing tag </{tag}> with nothing open")
            return
        if self.stack[-1][0] == tag:
            self.stack.pop()
            return
        if tag not in (t for t, _ in self.stack):
            self.errors.append(f"line {line}: closing tag </{tag}> does not match any open tag")
            return
        while self.stack and self.stack[-1][0] != tag:
            unclosed_tag, unclosed_line = self.stack.pop()
            self.errors.append(f"line {unclosed_line}: <{unclosed_tag}> was never closed")
        if self.stack:
            self.stack.pop()  # the matching tag itself


def html_check(path: Path) -> VerifyResult:
    """Structural check only: are tags balanced and properly nested? Not
    full HTML validation (no such thing exists in the stdlib without a
    dependency) -- but real, deterministic, and catches the most common,
    consequential mistake: an unclosed or mismatched tag."""
    checker = _HTMLBalanceChecker()
    try:
        checker.feed(path.read_text())
    except Exception as exc:  # noqa: BLE001 -- HTMLParser can raise on truly pathological input
        return VerifyResult(success=False, stage="compile", output=f"could not parse: {exc}")
    errors = list(checker.errors)
    for tag, line in checker.stack:
        errors.append(f"line {line}: <{tag}> was never closed")
    return VerifyResult(success=not errors, stage="compile", output="\n".join(errors))


def _check_balance(
    text: str, *, line_comment: str | None, block_comment: tuple[str, str] | None
) -> list[str]:
    """Minimal, best-effort structural check shared by `css_check` and
    `js_check`: are brackets/braces/parens balanced, and are string
    literals terminated? Skips comments and escaped quote characters.
    Not a real parser -- a JS template literal's `${...}` interpolation
    isn't tracked inside the string, for one -- but it catches the most
    common, consequential mistake a small model makes: an unterminated
    string or an unclosed block.
    """
    pairs = {")": "(", "]": "[", "}": "{"}
    openers = set(pairs.values())
    stack: list[tuple[str, int]] = []
    errors: list[str] = []
    quote: str | None = None
    i, n, line = 0, len(text), 1
    while i < n:
        ch = text[i]
        if ch == "\n":
            line += 1
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if block_comment and text.startswith(block_comment[0], i):
            end = text.find(block_comment[1], i + len(block_comment[0]))
            if end == -1:
                errors.append(f"line {line}: unterminated comment")
                break
            i = end + len(block_comment[1])
            continue
        if line_comment and text.startswith(line_comment, i):
            nl = text.find("\n", i)
            i = nl if nl != -1 else n
            continue
        if ch in ("'", '"', "`"):
            quote = ch
            i += 1
            continue
        if ch in openers:
            stack.append((ch, line))
        elif ch in pairs:
            if not stack:
                errors.append(f"line {line}: unmatched '{ch}'")
            elif stack[-1][0] != pairs[ch]:
                errors.append(
                    f"line {line}: '{ch}' does not match the most recent "
                    f"'{stack[-1][0]}' (line {stack[-1][1]})"
                )
                stack.pop()
            else:
                stack.pop()
        i += 1
    if quote:
        errors.append(f"unterminated string starting with {quote}")
    errors.extend(f"line {ln}: '{ch}' was never closed" for ch, ln in stack)
    return errors


def css_check(path: Path) -> VerifyResult:
    """Structural check only: balanced braces, terminated strings. No CSS
    parser exists in the stdlib; this is deterministic and real, just
    far less capable than py_compile -- it can't catch an invalid
    property or selector, only a broken block."""
    errors = _check_balance(path.read_text(), line_comment=None, block_comment=("/*", "*/"))
    return VerifyResult(success=not errors, stage="compile", output="\n".join(errors))


def js_check(path: Path) -> VerifyResult:
    """Structural check only -- see `_check_balance`. No JS parser exists
    in the stdlib; this catches an unclosed block or an unterminated
    string, nothing about whether the code is actually valid JS."""
    errors = _check_balance(path.read_text(), line_comment="//", block_comment=("/*", "*/"))
    return VerifyResult(success=not errors, stage="compile", output="\n".join(errors))


_WEB_FILE_CHECKS = {".html": html_check, ".htm": html_check, ".css": css_check, ".js": js_check}


def verify_generated_file(path: Path) -> VerifyResult:
    """Routes to the right check for this file's extension: Python's full
    compile/lint/guard/import pipeline, or a lighter structural check for
    HTML/CSS/JS -- no Python-style tooling applies to those; the checks
    above are hand-rolled and deterministic, just less capable than
    py_compile/ruff. Anything else falls back to the Python pipeline
    (the planner is only ever allowed to hand back these four
    extensions, so this is only reached for a `.py` file in practice)."""
    checker = _WEB_FILE_CHECKS.get(path.suffix.lower())
    if checker is not None:
        return checker(path)
    return verify_python_file_static(path)


def run_pytest(
    target: Path, timeout_seconds: int = 60, *, ignore: list[Path] | None = None
) -> VerifyResult:
    """Run pytest against a directory (full suite) or a single test file.

    `ignore` drops specific test files from the run -- used at the
    integration step to leave out an advisory test that never passed, so
    one unverifiable test doesn't sink an otherwise-working project.

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
    command = [sys.executable, "-m", "pytest", str(target), "-q", "-p", "no:cacheprovider"]
    for path in ignore or []:
        command += ["--ignore", str(path)]
    try:
        result = subprocess.run(
            command,
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
    """Compile-check a `test_*.py` file, then run just that file with pytest.

    Used for a test file the planner asked for: unlike an implementation
    file (compile/lint/guard/import), a test only counts as passing if it
    actually runs green, so a wrong assertion is caught here and the file
    can be treated as advisory rather than sinking the integration run.
    """
    compiled = compile_check(path)
    if not compiled.success:
        return compiled
    return run_pytest(path, timeout_seconds=timeout_seconds)
