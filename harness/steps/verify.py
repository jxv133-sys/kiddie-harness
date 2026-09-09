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
