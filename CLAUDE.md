# kiddie-harness

A fully local, autonomous project-generator harness for small LLMs
(8B-20B params) served via [Ollama](https://ollama.com). The user states
a goal in plain language; the harness develops a working project against
it with no cloud API involved.

## Why it's built this way

A prior attempt at this exact idea failed because the model was given
complex, compound instructions ("write the code AND follow this exact
format AND explain your reasoning"). Small models are reliable at **one
narrow instruction with one narrow output contract** and unreliable the
moment a prompt asks for several things at once. This whole codebase
exists to route around that failure mode:

- A **deterministic Python orchestrator** owns all sequencing, retries,
  and stop conditions. The LLM is never asked "what should happen next?"
  — it only ever answers one of a handful of fixed, single-purpose
  prompts (plan the files, write a spec, write one file's code, write one
  file's test, fix one file given one exact error).
- Every LLM output is verified with **real tooling** (`py_compile`,
  `ruff`, `pytest`, actual import resolution) — never another LLM's
  opinion of whether the code is right.
- Context per call stays narrow: code generation only ever sees that
  file's own spec (+ its own prior content and error on a retry), never
  the whole growing project transcript.

If you're extending this project, preserve these properties. A change
that makes one LLM call do two things at once, or that lets the model
decide control flow instead of the orchestrator, is going against the
core design, not just style.

## Repo map

- `harness/orchestrator.py` — the state machine. `SingleFileLoop` (one
  file, no planning) and `MultiFileLoop` (plan → per-file spec/codegen/
  test → integration check) both build on a shared `_generate_and_fix`
  bounded-retry loop, which also short-circuits (`fix_noop`) when a fix
  call returns the exact bytes that just failed. `MultiFileLoop`'s
  test-writing step passes the finished module's on-disk source into both
  the generate and the fix prompt (`fix_context`), so a test matches what
  the code does, not an independent reading of the spec.
- `harness/steps/` — one atomic LLM call per concern: `plan.py` (JSON-
  schema-constrained file list), `spec.py` (per-file bullet spec),
  `codegen.py` (generate/fix a file; `fix_file` takes an optional
  `context` prepended verbatim, used only for test files), `testgen.py`
  (generate a test file, always a separate call from implementation).
- `harness/steps/verify.py` — deterministic checks only, **no LLM calls
  anywhere in this file**. `compile_check`, `lint_check` (runs `ruff
  check --fix`, so trivial nits get fixed for free instead of costing a
  fix attempt), `main_guard_check` (AST: a module that defines
  functions/classes must not also run control flow or bare calls at
  module level — that would `sys.exit` under `import_check`),
  `import_check` (actually resolves a file's imports via `runpy.run_path`,
  without executing `if __name__ == "__main__":` blocks), `run_pytest`
  (clears `__pycache__` + `PYTHONDONTWRITEBYTECODE` so an in-place test
  rewrite can't hit a stale assertion-rewrite `.pyc`). Composed into
  `verify_python_file` (single-file loop: compile then run),
  `verify_python_file_static` (multi-file implementation files: compile →
  lint → main-guard → import-check, in that order, stopping at the first
  failure), `verify_test_file` (compile → pytest on just that one test
  file).
- `harness/llm_client.py` — thin Ollama wrapper. Surfaces truncation
  (`done_reason == "length"`) so the fix loop can grow `max_tokens` and
  tell the model its last output was cut off, instead of treating it like
  an ordinary syntax error.
- `harness/session.py` — per-run JSONL transcript (`log.jsonl`), plus an
  optional `on_event` callback fired right after each write (used for
  live progress output; never lets a broken callback break a run).
- `harness/summary.py` / `harness/progress.py` — two views of the same
  event stream: `summary.py` parses a finished `log.jsonl` into a status
  table (used by `harness inspect` and at the end of every run);
  `progress.py` formats events into live one-line-per-step output while a
  run is in progress.
- `harness/cli.py` — `harness run [--multi-file] [--quiet] --goal "..."`
  and `harness inspect --run-id <id>`.
- `config/default.yaml` — model, host, temperature, token limits/ceiling,
  retry budgets, timeout.
- `tests/fakes.py` — shared `FakeClient`/`FakeResponse`/`make_config` test
  doubles used by every test file. No test needs a live Ollama server.

## Status

Phases 0-4 (single-file loop → multi-file planning → dedicated
test-writing step → hardening/observability) are complete. Earlier fixes
from real runs against `qwen2.5-coder:7b`:

- Auto-fix trivial lint issues (`ruff check --fix`) instead of burning
  LLM fix attempts on formatting noise.
- Catch bad cross-file imports (e.g. a typo'd module name) during the
  offending file's own generation, via `import_check` — not downstream,
  where a different file's fix loop has no way to fix it.
- A bug in `import_check` itself: it double-resolved relative paths
  because the subprocess's `cwd` was changed without first resolving the
  path to absolute. Fixed, with a regression test.

Second pass of real-run fixes (`qwen2.5-coder:7b` remote, plus reasoning
models `Ornith-1.5-9B` / `DeepSeek-V4` locally):

- `pyproject.toml` now pins `testpaths`/`norecursedirs` — a bare `pytest`
  was sweeping every generated `workspace/**/test_*.py` into this repo's
  own suite and erroring on the basename collisions.
- `postprocess.strip_code_fences` strips `<think>…</think>` reasoning
  blocks and pulls the last fenced block out of surrounding prose. Before
  this, a reasoning model's chain-of-thought was written straight to the
  file and never compiled.
- `_generate_and_fix` stops early when a fix attempt returns byte-for-byte
  what already failed verification (`fix_noop` event) instead of spending
  the whole retry budget on identical calls — weak models echo verbatim.
- `run_pytest` clears `__pycache__` and sets `PYTHONDONTWRITEBYTECODE`:
  the fix loop rewrites a test file in place, and two versions with the
  same size + near-identical mtime made pytest serve the stale
  assertion-rewrite `.pyc` — phantom pass/fail.
- The test-writer and test-fixer now get the module's **actual generated
  source**, not just the spec. A spec bullet like "handle errors
  gracefully" was being read one way by the impl call and another by the
  test call, producing a test whose contract the impl never met — and
  the test's own fix loop (which only sees the test + pytest output)
  could not reconcile it.
- New `verify.main_guard_check` (a stage between lint and import): a file
  that defines functions/classes but also runs `sys.argv` parsing or an
  entry call at module level with no `if __name__ == "__main__":` guard
  gets `sys.exit`'d the moment `import_check` imports it. Now flagged with
  an actionable message; `codegen.md` also asks for the guard up front.

End-to-end reality check (calculator: arithmetic module + argv main
script, `qwen2.5-coder:7b`): the harness reliably produces a correct
`arithmetic.py`, a correct guarded `main.py`, and a passing
`test_arithmetic.py`. The remaining failure mode is `test_main.py` — the
7B model still writes tests that call `main()` without importing it, or
assert a normal return where the code `sys.exit`s. The errors it gets are
clear and the loop behaves correctly (bounded retry, clean give-up); this
is model capability, not a harness defect. Worth retrying with a stronger
model, and worth deciding whether a broken *generated test* should fail
the whole run when the implementation itself is sound.

## The pattern worth repeating

At every step in this project, `pytest` (all `FakeClient`-based, no live
model) passed cleanly — and still, every real bug listed in Status only
ever surfaced from an actual run against a real local model over the real
network. Unit tests are necessary but not sufficient here. **Before
considering a change to the verify/fix loop done, run it for real**:
`harness run --goal "..."` and `harness run --multi-file --goal "..."`
against an actual Ollama host, not just `pytest`. Different model families
break it differently — a reasoning model (`<think>` output) exercises
paths a plain instruct model never touches, so test with both.

## Developing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pytest                          # fast, no live Ollama needed
ruff check harness tests
harness run --host http://<ollama-host>:<port> --model <model> \
  --goal "a script that prints the first 20 Fibonacci numbers"
harness run --multi-file --host ... --model ... --goal "..."
```

`harness inspect --run-id <id>` re-renders the summary table for any past
run from its `workspace/<run-id>/log.jsonl`.
