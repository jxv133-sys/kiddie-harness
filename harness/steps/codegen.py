"""Atomic code-generation and code-fix LLM calls.

Each function here does exactly one thing and returns exactly one output
contract (a raw file body). Neither function loops, retries, or decides
what happens next -- that's the orchestrator's job.

Language-aware: the planner can hand back .py, .html, .css, .js, .bat/
.cmd, or .ps1 files (see steps/plan.py), and each needs its own rules --
a Python file's main-guard/no-module-state conventions mean nothing for
a stylesheet.
`_LANGUAGE_RULES` is the only place that varies; the prompt structure and
the call itself are identical regardless of language.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

from ..llm_client import OllamaClient
from ..postprocess import strip_code_fences

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_CODEGEN_TEMPLATE = (_PROMPTS_DIR / "codegen.md").read_text()
_FIX_TEMPLATE = (_PROMPTS_DIR / "fix.md").read_text()

_LANGUAGE_BY_SUFFIX = {
    ".py": "Python",
    ".html": "HTML",
    ".css": "CSS",
    ".js": "JavaScript",
    ".bat": "Batch",
    ".cmd": "Batch",
    ".ps1": "PowerShell",
}

_LANGUAGE_RULES = {
    "Python": (
        "- The file must be runnable on its own with `python <file>`.\n"
        "- Put the program's work in named functions. If the file is an entry\n"
        '  point, give it a `main()` function that does the work and end the file\n'
        '  with `if __name__ == "__main__":` then `main()` -- nothing else runs at\n'
        "  module level, so the file can be imported (and its functions tested)\n"
        "  without running or exiting.\n"
        "- Do not keep mutable state (a list, dict, counter, open file) at module\n"
        "  level. Hold it in a class the caller instantiates, or pass it in and\n"
        "  out of functions, so importing the module twice or calling it many\n"
        "  times starts clean each time.\n"
        "- When the task lists other project modules, get everything you need from\n"
        "  them with `from <module> import <name>`. Never re-implement what a\n"
        "  project module already provides, and never import that behaviour from\n"
        "  the standard library instead (use the project's own `mean`, not\n"
        "  `statistics.mean`)."
    ),
    "HTML": (
        "- Produce a complete, well-formed HTML document: every tag that needs a\n"
        "  closing tag has one, correctly nested.\n"
        "- Reference sibling CSS/JS files by exactly the filename given in the\n"
        '  task (e.g. `<link rel="stylesheet" href="style.css">`,\n'
        '  `<script src="app.js"></script>`) -- never invent a filename you\n'
        "  weren't given."
    ),
    "CSS": (
        "- Output valid CSS rules only: selectors and declaration blocks with\n"
        "  matched braces, each declaration ending in a semicolon.\n"
        "- Use only the class/id names the task's spec actually calls for --\n"
        "  matching whatever the HTML file that references this stylesheet uses."
    ),
    "JavaScript": (
        "- Output plain browser JavaScript: no Node-only APIs, no `import`/\n"
        "  `require` of packages that don't exist in a browser, unless the task\n"
        "  says otherwise.\n"
        "- Reference only DOM elements, ids, and classes the task's spec says\n"
        "  exist in the HTML."
    ),
    "Batch": (
        "- Output a Windows Command Prompt batch script (`.bat`/`.cmd`), not a\n"
        "  Unix shell script -- `set VAR=value` not `VAR=value`, `%VAR%` not\n"
        "  `$VAR`, `if exist` not `if [ -e ]`, `rem` or `::` for comments.\n"
        "- Start with `@echo off` so commands aren't echoed to the console.\n"
        "- Use only commands built into `cmd.exe` (echo, set, if, for, goto,\n"
        "  call, exit) unless the task specifically names an external tool."
    ),
    "PowerShell": (
        "- Output a PowerShell script (`.ps1`) -- use PowerShell cmdlets\n"
        "  (`Write-Host`, `Get-ChildItem`, ...) and `$variable` syntax, not Unix\n"
        "  shell or batch syntax.\n"
        "- Use only cmdlets built into Windows PowerShell/PowerShell 7's\n"
        "  standard modules unless the task specifically names an external\n"
        "  module -- do not assume a package is installed.\n"
        "- Put the script's work in named functions where practical, matching\n"
        "  the PascalCase Verb-Noun convention (`Get-Todos`, not `getTodos`)."
    ),
}


def _language_for(path: str) -> str:
    return _LANGUAGE_BY_SUFFIX.get(Path(path).suffix.lower(), "Python")


@dataclasses.dataclass
class GeneratedCode:
    code: str
    truncated: bool


def generate_file(
    client: OllamaClient,
    goal: str,
    *,
    path: str,
    temperature: float,
    max_tokens: int,
    on_chunk: Callable[[str], None] | None = None,
) -> GeneratedCode:
    """Goal -> full file content. One-shot, no prior code, no error context.

    `path` (the target filename) picks which language's rules go in the
    prompt -- see `_LANGUAGE_RULES` -- and is never itself sent as
    something to write to; the model never sees a filename to copy.

    `on_chunk`, if given, is passed straight through to the client -- the
    raw text streams in as-is (reasoning block and fences included, if
    the model emits them); `strip_code_fences` only ever runs once, here,
    on the finished response.
    """
    language = _language_for(path)
    prompt = _CODEGEN_TEMPLATE.format(
        goal=goal, language=language, language_rules=_LANGUAGE_RULES[language]
    )
    response = client.generate(
        prompt, temperature=temperature, max_tokens=max_tokens, on_chunk=on_chunk
    )
    return GeneratedCode(code=strip_code_fences(response.text), truncated=response.truncated)


def fix_file(
    client: OllamaClient,
    *,
    code: str,
    error: str,
    stage: str,
    path: str,
    temperature: float,
    max_tokens: int,
    on_chunk: Callable[[str], None] | None = None,
) -> GeneratedCode:
    """Current file + exact error -> corrected full file. Nothing else in
    context. `path` only picks the language named in the prompt."""
    prompt = _FIX_TEMPLATE.format(code=code, error=error, stage=stage, language=_language_for(path))
    response = client.generate(
        prompt, temperature=temperature, max_tokens=max_tokens, on_chunk=on_chunk
    )
    return GeneratedCode(code=strip_code_fences(response.text), truncated=response.truncated)
