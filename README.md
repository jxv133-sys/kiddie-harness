# kiddie-harness

A fully local, autonomous project-generator harness for small local LLMs
(8B–20B params, e.g. `deepseek-coder:8b` via [Ollama](https://ollama.com)).

You give it a goal in plain language; a deterministic Python orchestrator
drives the model through small, single-purpose prompts and verifies every
output with real tooling (compiler/test runner), never the model's own
judgment. See `docs` below (or the design doc this repo was built from) for
the full architecture and rationale.

Small models are reliable at one narrow instruction with one narrow output
contract, and unreliable the moment a prompt asks for several things at
once (write code *and* follow this format *and* explain yourself). Every
LLM call in this harness does exactly one thing.

## Status

**Phase 0 + Phase 1:** scaffolding, config, an Ollama client wrapper, and
the single-file core loop:

```
goal -> generate one Python file -> verify (compile, then run)
  -> on failure, feed the exact error back for a scoped fix (bounded retries)
```

**Phase 2:** multi-file planning, layered on the same per-file loop:

```
goal -> plan (JSON-schema constrained list of files)
  -> for each file: write a short spec -> generate -> verify (compile, then lint, then main-guard, then import-check)
       -> on failure, feed the exact error back for a scoped fix (bounded retries)
```

```
  -> once every file passes its own verify: integration check -- run the
     entry file (falling back to import-checking it if a blind run without
     args exits non-zero), or pytest if the goal itself asked for a
     test_*.py file
       -> on failure, find which generated file the error names and fix just that file
          (bounded rounds, and a global iteration budget across the whole run)
```

The small model is **not** asked to write tests. An earlier version had
it generate a `test_*.py` for every implementation file; weak models
produced tests that asserted contracts the code didn't have or shared
module-global state between test functions, and those failures were
unfixable — pure noise. A generated project gets tests only when the
goal (and so the planner) explicitly calls for a `test_*.py` file; that
file is built like any other, and if the model can't get it green it is
*advisory* (reported, left out of the integration run) rather than
fatal.

Every step is still a single, narrow LLM call: the planner never writes
code, the spec writer never writes code. Code generation for one file
sees the *source of the sibling modules already built this run* (so its
imports resolve to the right module instead of a guess), but never a
spec or a plan it wasn't given — one file, one output.

**Phase 4 (this commit):** hardening and observability, on top of the
same loop shape -- no changes to what the model is asked to do:

- **Truncation-aware retries.** Ollama reports `done_reason: "length"`
  when a response was cut off by hitting `max_tokens` mid-file, a
  distinct failure mode from an ordinary syntax error. The next attempt
  for that file gets more room (`max_tokens` doubles, capped at
  `generation.max_tokens_ceiling`) and the fix prompt is told the
  previous output was cut off, instead of just being handed a confusing
  syntax error.
- **Graceful CLI errors.** An unreachable Ollama host or a malformed
  planner response now prints one clean line and exits with code `2`,
  instead of a raw Python traceback.
- **Run summary / inspector.** `harness/summary.py` turns a run's
  `log.jsonl` into a compact table (per-file pass/fail, fix-attempt
  count, truncation flag, integration result, total LLM calls). The same
  table prints at the end of every `harness run`, and `harness inspect
  --run-id <id>` re-renders it for any past run.

**Live progress output:** a multi-file run can sit silent for minutes on
modest hardware, so `harness run` prints a line per step as it happens
(`[plan]`, `[spec]`, `[codegen]`, `[verify:<stage>]`, `[fix]`,
`[integration:<stage>]`, ...) instead of only the final table. `--quiet`
suppresses these and prints only the final summary, for scripting/
log-parsing use. No changes to the orchestrator or prompts -- this taps
the same event stream `summary.py` already reads, live.

**Auto-fixable lint issues:** running against real local models surfaced
a fix loop that burned all its retries on a lint error ruff's own output
said was `[*] fixable with the --fix option` -- pure import-formatting
noise, not something that needed the model at all. The lint check now
runs `ruff check --fix`, which only ever applies fixes ruff considers
safe (no semantic changes), so trivial nits get resolved for free and
only genuinely unfixable violations cost an LLM fix attempt.

**Catch bad cross-file imports where they're caused (this commit):** a
real run had one file import a sibling module under the wrong name (a
typo). Its own verify (compile + lint) couldn't catch that -- neither
actually resolves an import -- so the bug surfaced downstream, in its
companion test file's fix loop, which has no bug of its own and no way to
fix a different file. An implementation file's own verify now also
import-checks it (`runpy.run_path` with a non-`"__main__"` run name, so
`if __name__ == "__main__":` blocks never execute -- only import-time
resolution is checked, no side effects). Since files are generated in the
planner's declared dependency order, anything a file legitimately depends
on already exists on disk by the time this runs, so it's safe -- and it
means this class of bug gets caught, correctly attributed, and fixed
during the file's own generation instead of wasting retries somewhere
else.

**Phase 5: what real runs against small models actually needed.** Every
item here was a real failure observed running the harness end to end
against a local Ollama model, not something `pytest` caught -- the
orchestrator and prompts are otherwise unchanged in spirit:

- **Reasoning-model output.** A model that emits a `<think>...</think>`
  chain-of-thought (or an orphan `</think>` when Ollama consumes the open
  tag) had that prose written straight to the file. `postprocess` now
  strips the reasoning block and, if a fenced code block is buried in
  surrounding prose, extracts it -- deterministically, never inventing
  code.
- **Escalating retries.** At the configured low temperature a stuck model
  returns byte-identical output every retry, so extra attempts explored
  nothing. Each retry now samples a little hotter (the prompt is
  unchanged -- only the decoding randomness). Default `--max-retries` is
  `5`.
- **Import-time side effects.** An entry script that parses `sys.argv` at
  module level `sys.exit`s the moment `import_check` imports it. A new
  AST `main_guard_check` stage flags a module that defines
  functions/classes but also runs code at module level, with an
  instruction the fixer can act on; codegen is asked for a `main()` +
  `if __name__ == "__main__":` up front.
- **No small-model test-writing.** Removed the step that had the model
  write a `test_*.py` for every implementation file — a reliable source
  of unfixable failures. Tests exist only when the goal calls for them.
- **Advisory tests.** A planner-requested test that still won't pass after the
  full retry budget is reported `[advisory]`, left out of the integration
  run, and does not fail the project -- an unverifiable test means
  "unverified", not "broken code". Only implementation files and the
  integration check gate a run.
- **pytest cache & collection.** `run_pytest` clears `__pycache__` and
  sets `PYTHONDONTWRITEBYTECODE` (an in-place test rewrite of the same
  size was hitting a stale assertion-rewrite `.pyc`); `pyproject.toml`
  pins `testpaths` so this repo's own `pytest` ignores generated
  `workspace/` projects.
- **Planner hygiene.** Non-`.py` entries (a README, a `requirements.txt`)
  and duplicate paths are dropped from the plan.

Multi-language support is a later phase (see the architecture doc) and
not implemented yet.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

You'll also need [Ollama](https://ollama.com) running locally with a model
pulled:

```bash
ollama pull deepseek-coder:8b
ollama serve   # if not already running
```

## Usage

```bash
# single file (Phase 1)
harness run --goal "a script that prints the first 20 Fibonacci numbers"

# multi-file project (Phase 2)
harness run --multi-file --goal "a CLI todo list app with add/remove/list commands"
```

Options:

- `--model` — override the model from `config/default.yaml` (default `deepseek-coder:8b`)
- `--host` — override the Ollama host (default `http://localhost:11434`)
- `--max-retries` — override the bounded fix-loop attempt count per file (default `5`; each retry samples a little hotter)
- `--timeout` — override the per-call Ollama timeout in seconds (default `800`; raise it further for slow reasoning models)
- `--filename` — output filename, single-file mode only (default `main.py`)
- `--multi-file` — plan and generate a multi-file project instead of one script
- `--quiet` — suppress live per-step progress lines; print only the final summary

Each run creates `workspace/<run-id>/` containing the generated file(s) and
a `log.jsonl` transcript of every prompt, response, and verifier result --
useful for seeing exactly where a small model went wrong. A summary table
prints at the end of every run; to see it again later (or for a run that
crashed before finishing), use:

```bash
harness inspect --run-id <run-id>
```

## Parallel endpoints

A multi-file run can be split across two (or more) Ollama backends. Each
planned file declares `depends_on` (the earlier files it imports from);
one worker per endpoint pulls a file whose dependencies are all built and
runs its spec → codegen → verify → fix against that endpoint, so
independent files are generated concurrently.

```bash
harness run --multi-file --goal "..." \
  --endpoint http://192.168.50.142:7869,qwen2.5-coder:7b \
  --endpoint http://localhost:11434,qwen2.5-coder:7b
```

or set an `endpoints:` list in `config/default.yaml`. With one endpoint
this is the same serial walk as before. If an endpoint goes unreachable
mid-run, its file is requeued to the other; the run only aborts when no
endpoint can finish. `plan` and the integration check always run on the
first endpoint. The realistic speed-up is ~1.5–2× — a file that depends
on every earlier one gets no parallelism.

## GUI

```bash
harness gui          # opens a browser at http://127.0.0.1:8765
```

A single minimalist page: pick a model (the list is pulled live from the
Ollama host; **↻ re-fetch models** after you pull one), point at a host,
type a goal, hit **Generate**. **+ second endpoint** adds a parallel
backend. The progress log streams in as it happens (`[plan]`,
`[codegen]`, `[verify:*]`, `[fix]`, ...), a small strip tracks files /
fix attempts / LLM calls / elapsed time, and the final verdict and
per-file table drop in when the run finishes (with two endpoints, the
log lines and each file's row show which one built it, e.g. `[codegen]
b.py @ http://localhost:11434`). A **Files** panel lists each generated
file with its pass/fail status as it's built, plus the overall **Plan**
and each file's **spec** for a multi-file run; clicking any of them
opens its content in a small window over the page, kept live while it's
open. **Stop** cancels a run in progress — the GUI
is free to start a new one right away even if the model is still mid-call
underneath. The gear icon opens a **settings** panel for the retry/token/
temperature/timeout knobs that are otherwise only in `config/default.yaml`;
changes apply to runs started after that point and persist across a GUI
restart. Reloading the page while a run is active picks its stream back
up instead of showing a blank form. One run at a time; stdlib
`http.server`, no new dependencies, binds to localhost only. `--port`
and `--no-browser` are
available.

```bash
pytest
```

Tests use a fake LLM client (queued canned responses) so the orchestrator
and verifier logic can be exercised without a running Ollama server.

## Configuration

Defaults live in `config/default.yaml`: model name, generation temperature
(kept low — small models drift more at higher temperature and every step
here needs one predictable output, not creative variety), token limits
(including `max_tokens_ceiling`, the cap on adaptive growth after a
truncated response), and retry budgets. The temperature is the *base*:
each fix retry samples a step hotter than the last (see
`orchestrator._retry_temperature`), so extra attempts are real second
chances rather than identical calls, while the first attempt at every
step stays at the low base.
