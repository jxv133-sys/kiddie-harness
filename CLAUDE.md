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
  bounded-retry loop.
- `harness/steps/` — one atomic LLM call per concern: `plan.py` (JSON-
  schema-constrained file list), `spec.py` (per-file bullet spec),
  `codegen.py` (generate/fix a file), `testgen.py` (generate a test file,
  always a separate call from implementation).
- `harness/steps/verify.py` — deterministic checks only, **no LLM calls
  anywhere in this file**. `compile_check`, `lint_check` (runs `ruff
  check --fix`, so trivial nits get fixed for free instead of costing a
  fix attempt), `import_check` (actually resolves a file's imports via
  `runpy.run_path`, without executing `if __name__ == "__main__":`
  blocks), `run_pytest`. Composed into `verify_python_file` (single-file
  loop: compile then run), `verify_python_file_static` (multi-file
  implementation files: compile → lint → import-check, in that order,
  stopping at the first failure), `verify_test_file` (compile → pytest on
  just that one test file).
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
test-writing step → hardening/observability) are complete, plus three
fixes found by running against a real local Ollama instance
(`qwen2.5-coder:7b` at time of writing):

- Auto-fix trivial lint issues (`ruff check --fix`) instead of burning
  LLM fix attempts on formatting noise.
- Catch bad cross-file imports (e.g. a typo'd module name) during the
  offending file's own generation, via `import_check` — not downstream,
  where a different file's fix loop has no way to fix it.
- A bug in `import_check` itself: it double-resolved relative paths
  (`config/default.yaml`'s workspace root is relative) because the
  subprocess's `cwd` was changed without first resolving the path to
  absolute. Fixed, with a regression test reproducing the exact scenario.

A single-file run has been confirmed working end-to-end for real
(generate a Fibonacci script, 1 LLM call, 0 fix attempts, correct output).
A multi-file run is mid-debug as of this file being written — each retry
has surfaced one more real bug (all fixed so far), so don't assume the
multi-file path is fully proven yet; keep testing it for real.

## The pattern worth repeating

At every step in this project, `pytest` (all `FakeClient`-based, no live
model) passed cleanly — and still, three real bugs only ever surfaced
from an actual run against a real local model over the real network. Unit
tests are necessary but not sufficient here. **Before considering a
change to the verify/fix loop done, run it for real**: `harness run
--goal "..."` and `harness run --multi-file --goal "..."` against an
actual Ollama host, not just `pytest`.

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
