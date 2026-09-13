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
  opinion of whether the code is right. The one deliberate, narrow
  exception is the critic check (`harness/steps/critic.py`): "does this
  file actually do what its spec asked for" has no deterministic check,
  only a human or a model can judge it. Because that's an opinion and can
  be wrong, it never gets veto power the way real tooling does -- see
  `spec_flagged` in Status.
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
  counted against `overall_success`. `_with_critic` optionally wraps a
  file's `verify_fn` (see `harness/steps/critic.py` below): once it
  passes, one more call judges the file against its own spec; a
  disagreement becomes a `stage="critic"` verify failure, so it gets the
  same bounded fix attempts as any other stage, and if still unresolved
  at the end, the file is `spec_flagged` rather than failed (see Status).
  `_wait_if_paused` is the pause equivalent of `RunCancelled` -- a
  `pause_event` checked at the exact same three checkpoints as
  `cancel_event` (between fix attempts in `_generate_and_fix`, before a
  worker claims its next file in `_generate_files`, before each
  integration-fix round), blocking there until cleared, still honouring
  `cancel_event` while blocked so Stop works mid-pause. `record()` (the
  dispatcher's single choke point for "a file's build loop has actually
  concluded") logs a `file_result` event there -- the one place a live
  viewer (the GUI's dependency graph) can tell "still retrying, possibly
  paused" apart from "genuinely done"; a bare `verify` event alone can't,
  since a failed attempt with retries left looks identical in the log to
  one that just gave up.
- `harness/steps/` — one atomic LLM call per concern: `plan.py`
  (schema-constrained file list, bounded retry, a final schema-free
  attempt parsed by `_parse_free_form`; also flattens to bare filenames,
  drops anything outside `_ALLOWED_EXTENSIONS` (`.py`/`.html`/`.css`/
  `.js`), dedups), `spec.py` (per-file bullet spec, reasoning stripped),
  `codegen.py` (`generate_file` / `fix_file` — a file body in, a file
  body out, nothing else; language-aware, see below), `critic.py`
  (`critique_file` — a file's spec + contents in, a `{follows_spec,
  issues}` verdict out; fails open -- an unparseable or errored call is
  treated as "follows the spec," never as grounds to fail a file on its
  own).
- `harness/steps/codegen.py` — `_LANGUAGE_BY_SUFFIX` maps a target
  file's extension to a language name; `_LANGUAGE_RULES` holds that
  language's rule block (Python's main-guard/no-module-state
  conventions mean nothing for a stylesheet, so each language gets its
  own). One shared prompt structure (`codegen.md`/`fix.md`, both
  `{language}`-parameterized) formats in whichever rules apply — no
  per-language prompt duplication. `generate_file`/`fix_file` both take
  `path` (keyword-only) purely to pick the language; it's never sent as
  something to write to.
- `harness/steps/verify.py` — deterministic checks only, **no LLM calls
  anywhere in this file**. `verify_generated_file(path)` dispatches by
  suffix: `.py` goes to the existing `compile_check` → `lint_check` →
  `main_guard_check` → `import_check` pipeline (`verify_python_file_static`);
  `.html`/`.htm`/`.css`/`.js` go to hand-rolled structural checks instead
  -- `html_check` (a `html.parser.HTMLParser` subclass tracking a tag
  stack, since the stdlib parser itself never raises on malformed markup
  and can't be used for validation as-is) and `css_check`/`js_check`
  (a shared `_check_balance` state machine: bracket/brace/paren matching
  plus unterminated-string/comment detection, comment-aware). These are
  real tooling, just less capable than a real parser -- no new
  dependency was added; matches "never another LLM's opinion" even for a
  weaker check. `compile_check`, `lint_check` (runs `ruff check --fix`,
  so trivial nits get fixed for free instead of costing a fix attempt),
  `main_guard_check` (AST: a module that defines functions/classes must
  not also run control flow or bare calls at module level — that would
  `sys.exit` under `import_check`), `import_check` (actually resolves a
  file's imports via `runpy.run_path`, without executing `if __name__ ==
  "__main__":` blocks), `run_pytest` (clears `__pycache__` +
  `PYTHONDONTWRITEBYTECODE` so an in-place test rewrite can't hit a
  stale assertion-rewrite `.pyc`; `ignore=` drops advisory tests from an
  integration run). Composed into `verify_python_file` (single-file
  loop: compile then run), `verify_python_file_static` (multi-file `.py`
  implementation files: compile → lint → main-guard → import-check, in
  that order, stopping at the first failure), `verify_test_file`
  (compile → pytest on just that one test file).
- `harness/llm_client.py` — thin Ollama wrapper. Surfaces truncation
  (`done_reason == "length"`) so the fix loop can grow `max_tokens` and
  tell the model its last output was cut off, instead of treating it like
  an ordinary syntax error. `generate()`'s default (`on_chunk=None`) is
  one blocking call, unchanged; passing `on_chunk` switches to Ollama's
  streaming mode (confirmed live: schema-constrained decoding streams
  fine too) and calls it with the cumulative text after every line of
  the response -- `_generate_once` and `_generate_streaming` share the
  same `LLMResponse` contract, so nothing downstream needs to know which
  path ran. A broken `on_chunk` is swallowed (a live-view display bug
  must never abort a real generation); a dropped connection mid-stream
  or a stream that ends without a final `done` line both raise a normal
  `OllamaError`, same as any other call failure.
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
  whatever run superseded it. `RequestTracker` (server-side) tracks the
  browser's own HTTP requests to the GUI, keyed by a `_r=` id every
  client request tags its own URL with (`/api/requests`) -- used for
  stale-response guarding on the model re-fetch rows, not shown in the
  page itself. What *is* shown in the header is a live LLM-call bar,
  backed by a separate mechanism: `Session.track_call(kind, path,
  endpoint)` (a context manager wrapping every `client.generate()` call
  site in `orchestrator.py` -- plan/spec/codegen/fix/critic/
  integration_fix) marks a call in-flight for `Session.active_calls()`
  to report while it runs. `RunManager.active_calls(run_id)` exposes the
  current run's session for this (`/api/calls/<run_id>`), but only while
  `state == "running"` for that exact run_id -- `cancel()` is
  cooperative, so the old thread's call can genuinely still be running
  in the background after the GUI has moved on and shown a verdict, and
  it must not keep reporting that stray leftover as "active" once it has.
  `track_call` yields `update(text)`, threaded through as every step
  function's `on_chunk` (`plan.plan_files`, `spec.write_spec`,
  `codegen.generate_file`/`fix_file`, `critic.critique_file` all take it
  and pass it straight to `client.generate`) -- each in-flight entry
  carries a live `partial` field, so `active_calls()`/`/api/calls/` (and
  a click on a calls-bar row, reusing the Files panel's `#file-modal`)
  show the actual text streaming in from Ollama, not just that a call is
  running. Verified live against a real model: the modal updated with
  real generated code appearing line by line, and closed on its own the
  moment the call finished.
  A **settings screen**
  (gear icon) edits `temperature`/`max_tokens`/`max_tokens_ceiling`/
  `max_fix_attempts`/`max_total_iterations`/`timeout_seconds` for runs
  started after the change; saved to `config/gui_settings.json`
  (gitignored -- default.yaml keeps its comments) and merged on top of
  it at server start. Saving **while the active run is paused** also
  applies live: `RunManager._active_config` is the exact `Config`
  instance the running loop holds (a separate object from `self._config`,
  which is only "defaults for the next run"), and `update_config` calls
  `Config.apply_overrides` (mutates fields on `self` in place, unlike
  `with_overrides`'s fresh copy) on it when paused -- since every read
  site in `orchestrator.py` reads `config.<field>` live rather than a
  snapshot taken once, the change is in effect the moment the run
  resumes, no extra plumbing needed. **Pause/Resume** buttons
  (`RunManager.pause()`/`resume()`, `/api/run/pause`/`/api/run/resume`)
  set/clear a `pause_event` alongside the existing `cancel_event`, passed
  into the loop the same way; `/api/config`'s `state.paused` and the
  `run_paused`/`run_resumed` log events (the SSE stream already carries
  these live) drive the button swap and a "paused, changes apply on
  Resume" note. Verified live: pausing mid-fix-loop produces a
  `[paused] waiting to resume...` log line between two fix attempts, a
  `max_fix_attempts` bump applied while paused let a file survive past
  what the original budget would have allowed, and Resume picked the
  same file back up and finished the run. A **Files panel** renders every
  file the plan lists (not just ones already on disk) as a **layered SVG
  dependency graph** doubling as a to-do list -- `renderGraph` groups
  files into rows by `1 + max(dependency's row)` (0 for a leaf), draws
  `<path>` edges between them, and colors each node by status: `pending`
  (planned, unclaimed), `building` (has a spec/codegen event but no
  `file_result` yet -- *not* the same as "its last verify failed", see
  below), `ok`/`failed`/`advisory`/`flagged`/`skipped`. `_files_response`
  computes this from the `plan` event's own file list (falling back to a
  disk glob only for single-file runs, which have no plan) plus a
  `phase` field (`planning`/`building`/`integration`/`done`) a small
  stepper at the top renders. **Found live, fixed same session:** a
  verify failure isn't final by itself -- the fix loop may have retries
  left, or (with pause) be sitting on a just-failed attempt indefinitely
  -- so trusting the *last* verify event's success/fail showed a paused,
  still-retrying file as FAILED. `orchestrator.record()` now logs a
  `file_result` event exactly when a file's build loop actually
  concludes; `_files_response` only trusts a terminal status once that's
  been seen for the file, "building" otherwise. Clicking a node opens the
  same small modal window over the page (`#file-modal`) the calls bar and
  everything else use -- `/api/files/<run_id>` (the graph data +
  `has_plan`/`phase`), `/api/file/<run_id>/<name>` (source),
  `/api/plan/<run_id>` (the planner's file list rendered as text),
  `/api/spec/<run_id>/<name>` (that file's spec). A window left open
  stays live as the run continues -- refreshed on the same poll as
  progress. `_read_run_file` confines reads to that run's own directory.
  The **endpoint pool** in the form is an unbounded list, not a fixed
  primary+one: "+ add endpoint" (`addEndpointRow`) appends a row with its
  own suffixed ids and its own re-fetch/remove, any number of times --
  the backend already took an arbitrary `endpoints: list[dict]`, so this
  was a frontend-only change. Verified live with three endpoints (the
  primary plus two extras) configured at once.
- `config/default.yaml` — model, host, temperature (the *base*; retries
  step up from it), token limits/ceiling, retry budgets, timeout, and an
  optional `endpoints:` list (`Config.Endpoint` / `resolved_endpoints()`)
  for the parallel dispatcher.
- `tests/fakes.py` — shared `FakeClient`/`FakeResponse`/`make_config` test
  doubles used by every test file. No test needs a live Ollama server.
  `make_config`'s `critic_enabled` defaults to `False` (opposite of
  `config/default.yaml`'s `True`) so every pre-existing FakeClient test's
  queued response count is unaffected by the critic's extra call; tests
  for the critic itself opt in explicitly.

## Status

**Pause/resume, a live dependency graph, and an unbounded endpoint pool,
added on request.** Pause is cooperative, mirroring `cancel_event`
exactly: a `pause_event` checked at the same three checkpoints
(`orchestrator._wait_if_paused`), blocking between calls, never
mid-request. Its point is letting settings changes reach an *already
running* multi-file run -- `RunManager._active_config` is the loop's own
live `Config` object; `Config.apply_overrides` mutates it in place
(rather than `with_overrides`'s fresh copy) so a change made while
paused is visible to the loop's very next read, no extra plumbing. The
GUI's Files panel became a layered SVG dependency graph (doubles as a
to-do list: a planned-but-unclaimed file shows as `pending`, not
absent) plus a `planning -> building -> integration -> done` phase
stepper, both driven by an enriched `/api/files/<run_id>`. The "+ second
endpoint" row became "+ add endpoint", unbounded -- the dispatcher
already took an arbitrary `endpoints: list[dict]`, so supporting a third
(or more) box on the network was a frontend-only change (asked for
specifically: "I have a laptop I want to get in the generation pool").
**A real bug found live, in this same pass, testing pause itself:**
pausing a run mid-fix-loop showed the paused file as `FAILED` in the new
graph, because status was inferred from the *last* `verify` event's
success/fail alone -- indistinguishable in the log from a file that
had actually exhausted its retries and given up. Fixed by having
`orchestrator.record()` (the dispatcher's one choke point for "this
file's build loop concluded") log a dedicated `file_result` event;
`_files_response` now shows `building` for any file that hasn't gotten
one yet, however its last verify attempt went. Verified live end to end
against `llama3.2:latest`: paused a run mid-fix-loop (log line
`[paused] waiting to resume...` sitting between two `[fix]` attempts,
the graph correctly showing `building` rather than `FAILED` throughout),
confirmed the settings panel's "applies to the next run" message swaps
to "applied to the paused run" while paused, resumed, and watched the
same run finish to `SUCCESS` with the graph settling on all-green nodes
matching the final summary table. Also confirmed three endpoints (primary
+ two "+ add endpoint" rows) can be configured on one run at once.

**Multi-language web support, added on request** (the planner only ever
listed `.py` files, even for "make a web page" goals): the planner can
now emit `.html`/`.css`/`.js` files too (`plan.py`'s
`_ALLOWED_EXTENSIONS`), and `plan.md` tells it to include a small stdlib
`http.server` Python entry point for a web goal -- this harness only
ever runs and verifies things locally with Python, so a page needs
something runnable to check at all. `codegen.py` picks per-language
rules by the target file's extension (`_LANGUAGE_BY_SUFFIX`); one shared
`{language}`-parameterized prompt template covers all four languages
instead of duplicating `codegen.md`/`fix.md` per language.
`verify.py`'s new `verify_generated_file` dispatches `.html`/`.css`/
`.js` to hand-rolled, dependency-free structural checks (`html_check`,
`css_check`, `js_check` -- see `harness/steps/verify.py` above) since no
stdlib parser/linter exists for them; `.py` still gets the full
compile/lint/guard/import pipeline. No new dependency was added,
preserving "real tooling, never another LLM's opinion" even for a
weaker check. `_pick_entry_path` (`orchestrator.py`) only considers
`.py` files for the integration check, and returns `None` -- skipping
integration entirely -- for a goal that's pure HTML/CSS/JS with no
Python file at all. The GUI's Files panel (`_GENERATED_FILE_GLOBS`) and
`/api/files` list all four extensions, not just `.py`. **Verified live
end-to-end** against `llama3.2:latest`: goal "a simple web page with a
styled heading and a button that shows an alert when clicked" → planner
emitted `index.html` + `style.css` + `script.js` + a Python file; every
file passed codegen, critic, and its verify stage (including the new
`html_check`/`css_check`/`js_check`), and `[integration:run] -> ok` --
`SUCCESS`, 4/4 files, 0 fixes, 9 LLM calls. The live-streaming calls bar
and clickable call modal (see `harness/gui.py` below) both confirmed
working for non-Python generation too. One live finding, not a bug: a
weak model's own plan named a Python file `alert.py` with the purpose
"handle alert functionality" instead of using it as the server the
prompt asked for, and `script.js`'s `<script src>` in the generated HTML
didn't match the planned filename -- expected small-model unreliability
documented throughout this file, not something the harness pipeline
itself got wrong (every file still individually verified correctly).

**Critic check, added on request** ("make sure the file follows what the
spec was"): once a file passes every real check (compile/lint/guard/
import), one more call asks the model to judge its own output against
its own spec (`harness/steps/critic.py`, `orchestrator._with_critic`).
This is the project's one deliberate exception to "real tooling, never
another LLM's opinion" (see "Why it's built this way" above) -- resolved
in favor of that principle by making it advisory: a disagreement gets
the same bounded fix attempts as any other verify stage, but if it's
still unresolved when those run out, the file is reported `spec_flagged`
(`[flagged]` in the table, a distinct dot color in the GUI) rather than
`FAILED` -- it doesn't block the run, doesn't block a dependent file's
sibling-context, and doesn't fail the exit code. An agreeing critic call
returns the wrapped `verify_fn`'s own result unchanged (so it doesn't
show up as its own "verify" event), which would leave zero evidence the
call ever happened -- worth knowing since it looks identical to "critic
never ran" in the log otherwise; a dedicated `critic_check` event (always
logged, agree or not) and its own `[critic]` progress line exist
specifically to make that visible. Verified live against a real model
(`llama3.2:latest`): `[critic] main.py -> ok`, correctly counted in the
LLM-call total. On by default (`config/default.yaml`'s `critic.enabled`);
`--no-critic` / `RunManager.update_config` / the GUI's settings checkbox
all turn it off. Applies to single-file mode too (the goal itself stands
in for a spec there); not applied to `test_*.py` files -- a test's own
pass/fail against pytest already is its verification.

Single-file loop → multi-file planning → hardening/observability →
minimal web GUI → parallel dual-endpoint dispatch are complete. The
dual-endpoint work **is now verified against two live, real Ollama
backends** — `harness run --multi-file --endpoint http://<remote>,llama3.2:latest
--endpoint http://localhost:11434,<local-model>` against a goal with
genuinely independent files planned two leaf modules onto the two
endpoints concurrently, then built the dependent file after, all
`[ok]`. See `docs/design/2026-09-10-dual-endpoint-parallelism.md`.

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
narrative build order otherwise -- confirmed live: a later 3-file goal
with two genuinely independent modules got exactly that shape from the
planner and both endpoints fired at once.

**The other half of "not seeing it call the second endpoint": nothing
recorded which endpoint built which file.** Dispatch could be working
correctly and still look broken with no way to tell. `codegen`/`fix`/
`spec` log events now carry `endpoint=client.host`; the live progress
line shows it (`[codegen] b.py @ http://localhost:11434`, `spec` too --
`fix` omits it since a retry never changes which worker owns the file),
and the GUI's Files panel shows a small host badge per file, but only
once a run's files actually used more than one endpoint (a single-
endpoint run would just see the same badge on every row, so it's
suppressed).

**A round of real bugs found generating an actual project (a mock login
page) against `huihui_ai/deepseek-r1-abliterated:8b` on both endpoints:**

- **`strip_code_fences` left a guaranteed syntax error in place.** A
  response truncated mid-file, still inside the fence it opened (` ```python\n `
  with no closing ` ``` `), was returned verbatim -- the leading marker
  is not valid Python, so the fix loop's very first attempt was always a
  SyntaxError on a markdown artifact, not the model's actual mistake. Now
  drops a confirmed-unclosed opening fence and keeps everything after it
  (real, if incomplete, code) rather than leaving the marker in.
- **`plan_files` retried a dead connection like a bad response.** An
  `OllamaError` (host unreachable) was caught by the same retry loop as
  "the model returned nonsense," so a genuinely offline host cost 2-3x
  its own timeout before the run gave up. Now propagates immediately --
  only a content failure (empty/malformed plan) escalates temperature
  and retries.
- **A CLI `--endpoint` with no model silently ignored `--model`.** Config
  was loaded twice: once with the CLI overrides applied, once again
  (raw, un-overridden) just to resolve endpoint defaults, so `--model
  foo --endpoint host` gave that endpoint the yaml's default model, not
  `foo`. Endpoints now resolve against the already-overridden config.
- **An explicit `0` in `Config.with_overrides` was silently discarded.**
  `max_fix_attempts or self.max_fix_attempts` treats `0` the same as
  "not given" -- a real, meaningful override ("no retries, just report
  the first failure") was dropped in favor of the old value. Switched to
  explicit `is None` checks throughout.
- **No visibility when a worker's endpoint died mid-file.** `OllamaError`
  in the dispatcher requeues the file for another endpoint but never
  logged it -- a file's whole build (spec included) would just silently
  restart from scratch on a different host, with nothing in the log
  explaining why. Logged as `endpoint_retired` now, surfaced live as
  `[endpoint] <host> failed on <path> (<error>) -- requeued for another
  endpoint`.
- Added `--max-tokens` / `--max-tokens-ceiling` CLI flags (previously
  yaml-only) -- the fastest lever for a verbose reasoning model that
  keeps getting cut off mid-file, and exactly what surfaced the fence bug
  above in the first place.
- **An idle worker retired for good the instant `pending` was momentarily
  empty, even if another worker was still mid-build on the only
  remaining file.** With a single-file plan (or any moment where every
  pending file happens to already be claimed), the second endpoint would
  see nothing to claim on its very first check and exit permanently --
  so when the worker actually holding that file hit `OllamaError` and
  requeued it, nobody was left to pick it up and the whole run aborted,
  even with a perfectly good second endpoint sitting idle. This is the
  single biggest reason a run can fail to "use the second endpoint" at
  all: it's not that dispatch didn't try, it's that the backup worker
  had already given up before it was needed. Fixed with a `busy` counter
  -- a worker only retires when there's nothing pending *and* nobody
  else is mid-build (and so might fail and reissue more work). Regression
  test proven both ways: fails 5/5 against the pre-fix code, passes
  15/15 against the fix (`test_an_idle_worker_does_not_retire_just_because_the_only_file_is_already_claimed`).

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
