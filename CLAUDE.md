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
- Context per call stays narrow: code generation sees that file's own
  spec, the source of the sibling modules already built this run (capped;
  so its imports resolve instead of guessing), and — on a retry — its own
  prior content and the exact error. Never the whole growing transcript,
  never another file's spec.

If you're extending this project, preserve these properties. A change
that makes one LLM call do two things at once, or that lets the model
decide control flow instead of the orchestrator, is going against the
core design, not just style.

## Repo map

- `harness/orchestrator.py` — the state machine. `SingleFileLoop` (one
  file, no planning) and `MultiFileLoop` (plan → per-file spec/codegen →
  integration check) both build on a shared `_generate_and_fix`
  bounded-retry loop. `MultiFileLoop._generate_files` is a dispatcher:
  one worker thread per client in `pool_clients` (defaults to the one
  positional client = the old serial walk), each claims a file once every
  entry in its `depends_on` is built. `plan` and integration run on the
  positional client. A worker that hits `OllamaError` requeues its file
  and retires; the run aborts only when no worker can progress; a file
  whose dependency hard-failed is `skipped`. Each retry samples a little
  hotter than the last
  (`_retry_temperature`) — the prompt never changes, only the decoding
  randomness, which is what gives a stuck small model a real chance
  across `max_fix_attempts` tries instead of echoing itself; a verbatim
  repeat is logged (`fix_noop`) but no longer aborts. The small model is
  **not asked to write tests** (see Status). A `test_*.py` the planner
  itself listed is built like any other file but verified with pytest
  (`verify_test_file`); if it never passes it is **advisory** — recorded,
  excluded from the integration pytest run (`run_pytest(ignore=…)`), not
  counted against `overall_success`.
- `harness/steps/` — one atomic LLM call per concern: `plan.py`
  (schema-constrained file list, bounded retry, a final schema-free
  attempt parsed by `_parse_free_form`; also flattens to bare filenames,
  drops non-`.py`, dedups), `spec.py` (per-file bullet spec, reasoning
  stripped), `codegen.py` (`generate_file` / `fix_file` — a file body in,
  a file body out, nothing else).
- `harness/steps/verify.py` — deterministic checks only, **no LLM calls
  anywhere in this file**. `compile_check`, `lint_check` (runs `ruff
  check --fix`, so trivial nits get fixed for free instead of costing a
  fix attempt), `main_guard_check` (AST: a module that defines
  functions/classes must not also run control flow or bare calls at
  module level — that would `sys.exit` under `import_check`),
  `import_check` (actually resolves a file's imports via `runpy.run_path`,
  without executing `if __name__ == "__main__":` blocks), `run_pytest`
  (clears `__pycache__` + `PYTHONDONTWRITEBYTECODE` so an in-place test
  rewrite can't hit a stale assertion-rewrite `.pyc`; `ignore=` drops
  advisory tests from an integration run). Composed into
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
  live progress output; never lets a broken callback break a run). A
  `threading.Lock` makes each write + callback atomic for the parallel
  dispatcher.
- `harness/summary.py` / `harness/progress.py` — two views of the same
  event stream: `summary.py` parses a `log.jsonl` into a status table
  (used by `harness inspect` and at the end of every run); `progress.py`
  formats events into live one-line-per-step output while a run is in
  progress. Both loops log a terminal `run_result` event; a log without
  one is reported `INCOMPLETE` (the process was killed) rather than
  guessed at, and `harness inspect` exits non-zero for it.
- `harness/cli.py` — `harness run [--multi-file] [--quiet] --goal "..."`
  (`--model` / `--host` / `--max-retries` / `--timeout` / repeatable
  `--endpoint HOST,MODEL` override the config), `harness inspect --run-id
  <id>`, `harness gui`. An aborted run
  exits 2.
- `harness/gui.py` — a stdlib `http.server` GUI (`harness gui`). Pure
  core is testable: `available_models(host)`, `RunManager` (one run at a
  time, in a daemon thread; takes a `client_factory` seam for tests),
  `stream_events(log_path)` (tails `log.jsonl` → SSE frames using
  `progress.format_event`). The page is one embedded HTML string. No new
  deps; it reuses `progress` + `summary` and changes nothing elsewhere.
  `RunManager.cancel()` sets a cooperative `cancel_event` (checked in
  `orchestrator._generate_and_fix`, the multi-file worker loop, and the
  integration-fix loop -- it raises `RunCancelled` between calls, never
  mid-call) and immediately flips the manager back to `idle` so the GUI
  can start a new run without waiting for the old thread to notice; a
  `run_id` check in that thread's `finally` stops it from clobbering
  whatever run superseded it. `RequestTracker` (server-side) backs a
  small live "N requests active" readout on the page, keyed by a `_r=`
  id every client request tags its own URL with. A **settings screen**
  (gear icon) edits `temperature`/`max_tokens`/`max_tokens_ceiling`/
  `max_fix_attempts`/`max_total_iterations`/`timeout_seconds` for runs
  started after the change; saved to `config/gui_settings.json`
  (gitignored -- default.yaml keeps its comments) and merged on top of
  it at server start. A **Files panel** lists a run's generated `.py`
  files with their latest verify status, plus a top `Plan` row (when the
  run went through the planner) and a `spec` link on files that got one
  (single-file runs skip both). Clicking any of these opens a small
  modal window over the page (`#file-modal`) rather than an inline pane
  -- `/api/files/<run_id>` (list + `has_plan`/`has_spec` flags),
  `/api/file/<run_id>/<name>` (source), `/api/plan/<run_id>` (the
  planner's file list rendered as text), `/api/spec/<run_id>/<name>`
  (that file's spec). A window left open stays live as the run
  continues -- refreshed on the same poll as progress. `_read_run_file`
  confines reads to that run's own directory.
- `config/default.yaml` — model, host, temperature (the *base*; retries
  step up from it), token limits/ceiling, retry budgets, timeout, and an
  optional `endpoints:` list (`Config.Endpoint` / `resolved_endpoints()`)
  for the parallel dispatcher.
- `tests/fakes.py` — shared `FakeClient`/`FakeResponse`/`make_config` test
  doubles used by every test file. No test needs a live Ollama server.

## Status

Single-file loop → multi-file planning → hardening/observability →
minimal web GUI → parallel dual-endpoint dispatch are complete. The
dual-endpoint work is unit-tested but **not yet exercised against two
live models** — see `docs/design/2026-09-10-dual-endpoint-parallelism.md`.

**A real GUI run against `deepseek-r1:7b` surfaced the actual reason the
dual-endpoint feature looked broken in practice: `RunManager` had no way
to cancel a run, so a slow model stuck in its fix loop (or a model
returning a diff/patch format `codegen` doesn't understand, looping on
`fix_noop`) held the GUI's one-run-at-a-time lock indefinitely — no new
run, dual-endpoint or not, could start until it finished or the process
was killed.** Fixed with the cooperative `cancel_event` described under
`harness/gui.py` above (a Stop button in the GUI). Separately, the
planner's real depends_on output for that same run was a fully linear
chain (`core → auth → web`) even with two endpoints configured — the
dispatcher has nothing to parallelize when every file depends on the
one before it. `plan.md` now tells the model `depends_on` means "has an
`import` for," not "came after," since a weak model defaults to
narrative build order otherwise; unverified against a live model yet.

**The small model no longer writes tests.** An earlier phase had
`MultiFileLoop` generate a `test_*.py` for every implementation file
(from the module's real source, its own call). In practice a 7B model
wrote tests that asserted contracts the code didn't have, shared
module-global state between test functions, or imported names that
weren't exported — and those failures were structurally unfixable, so
they landed "advisory" every run: pure noise and an extra LLM call per
file. Removed (`testgen.py`, `testgen.md`, `verify_test_file`'s
companion-test role, `_generate_test_for`, the `fix_context` plumbing).
A generated project gets tests only when the goal/planner asks for a
`test_*.py` explicitly.

Earlier fixes from real runs against `qwen2.5-coder:7b`:

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
- `_generate_and_fix` raises the sampling temperature on each retry
  (`_retry_temperature`) so a stuck model gets real extra attempts rather
  than echoing itself; `max_fix_attempts` 3→5, `max_total_iterations`
  25→40. A verbatim repeat is logged (`fix_noop`) but no longer aborts.
  The prompt is byte-identical across retries — only the decoding
  randomness changes.
- `run_pytest` clears `__pycache__` and sets `PYTHONDONTWRITEBYTECODE`:
  the fix loop rewrites a test file in place, and two versions with the
  same size + near-identical mtime made pytest serve the stale
  assertion-rewrite `.pyc` — phantom pass/fail.
- (The test-writing step this list refers to was later removed entirely —
  see the top of Status. The `run_pytest` `__pycache__`/mtime fix above
  still matters: a planner-listed `test_*.py` is rewritten in place by
  its fix loop just the same.)
- New `verify.main_guard_check` (a stage between lint and import): a file
  that defines functions/classes but also runs `sys.argv` parsing or an
  entry call at module level with no `if __name__ == "__main__":` guard
  gets `sys.exit`'d the moment `import_check` imports it. Now flagged with
  an actionable message; `codegen.md` also asks for the guard up front.

Advisory tests (decision: a broken test the planner asked for must not
sink an otherwise-working project):

- A planner-listed `test_*.py` that never passes pytest is marked
  `advisory` on its `FileRunResult`, logged as `advisory_test`, and
  rendered `[advisory]` rather than `[FAILED]`.
- `MultiFileLoop.run` gates `overall_success` on the non-advisory files
  plus the integration check only. The integration `run_pytest` is given
  `ignore=<advisory paths>` so a failing test is left out; `has_tests`
  counts only *passing* tests, so an all-advisory project falls back to
  running its entry script.
- Exit code stays 0 on an otherwise-successful run; the CLI prints a
  `N advisory test(s) never passed` note to stderr and the summary line
  carries the count. `harness inspect` shows the same from the log.

More hardening from continued real runs:

- `plan_files` forces a flat file layout (`posixpath.basename` every
  entry) — a planned `pkg/core.py` broke its import-check (wrong cwd) and
  any sibling importing it by bare name. `plan.md` asks for it too.
- Both loops log a terminal `run_result` event; a log without one renders
  `INCOMPLETE` (the process was killed) instead of the old "no giving_up
  == success" guess, and `harness inspect` exits non-zero for it.
- A blank / whitespace-only generation is treated as a failure and
  retried — an empty file compiles and imports fine and was passing as a
  "success".
- `codegen.md` asks for no module-level mutable state (a store keeping
  its list in a module global made every generated test share it). This
  one is guidance a 7B model often ignores; a stronger model honours it.

Reasoning-model + resilience pass (found running `deepseek-r1:7b`, a
slow model that thinks in `<think>` blocks and fights the JSON grammar):

- `plan_files` now retries (3 attempts). Attempts 0..N-2 stay
  schema-constrained and just escalate temperature + a blunt "an empty
  list is not acceptable" line; the **last** attempt drops the schema so
  the model can think first, and `_parse_free_form` reads whatever comes
  back — JSON, or a numbered/markdown `*.py` list. Before this the
  planner had no retry at all and a single `{"files": []}` killed the
  whole run.
- Both loops catch `OllamaError` around the entire build: the model going
  unreachable mid-run (a 300s read timeout on one slow call) now logs
  `run_aborted`, keeps every completed file, and reports
  `Result: ABORTED …` + exit 2 — not a bare traceback and total loss.
- `--timeout <seconds>` CLI flag (and `Config.with_overrides`) so a slow
  reasoning model can be given more headroom without editing the yaml.
- `_run_integration`: when there are no passing tests and the entry
  script is run blind, a non-zero exit (a CLI that wanted argv) falls
  back to `import_check` — "we can't invoke it, but it and its cross-file
  imports load". A real cross-file break still fails `import_check`.

End-to-end reality check (`qwen2.5-coder:7b`), all SUCCESS — these
predate the test-writing removal, so the advisory notes are historical;
the implementation files and integration checks all passed:

- **1 file** (fib / count): 1 call, 0 fixes.
- **2 files** (calculator: arithmetic + argv main): correct `arithmetic.py`,
  guarded `main.py`, integration green.
- **3 files** (to-do: Task + store + argv CLI): all three implementation
  files clean, integration green, ~20 LLM calls (fewer now without the
  per-file test call).

`deepseek-r1:7b`: the planner's free-form fallback recovers the file
list; individual codegen calls then blow the 300s timeout and the run
aborts gracefully. Runnable with `--timeout 900`, just slowly.

**Re-verify a fresh multi-file run against a real model** after this
change — the removal touched the core `MultiFileLoop` and every
per-file test in the suite was rewritten. Unit tests pass; that has
never been sufficient here.

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
