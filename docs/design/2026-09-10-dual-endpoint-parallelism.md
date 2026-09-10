# Dual-endpoint parallelism ("double the power")

Status: **implemented** (2026-09-10) — see the "What shipped" note at the end.
Date: 2026-09-10

## Goal

Run a multi-file generation across **two (or more) Ollama endpoints** —
e.g. the local machine plus the box on the LAN — so independent files are
generated concurrently and a run finishes in roughly half the wall-clock
time. One endpoint must keep working exactly as today.

## Why it isn't free

A multi-file run today is strictly serial:

```
plan -> (spec1 -> codegen1 -> verify1 -> fix...) -> (spec2 -> ...) -> integration
```

Two things make "just run every file in parallel" wrong:

1. **Files have a dependency order.** The planner emits files "in the
   order they should be created" (dependencies before dependents), and
   `MultiFileLoop._sibling_context` now feeds each file the *source of
   every file already built this run* so its imports resolve. So
   `main.py`'s codegen genuinely needs `core.py` finished first. Naive
   parallelism breaks sibling-context and produces `main.py` files whose
   imports don't resolve.
2. **The endpoints may run different models** of different quality;
   assigning a hard file to a weak model wastes the run.

## Design

A **bounded worker pool over a dependency DAG**, with the deterministic
orchestrator still owning all sequencing — workers *pull* ready tasks,
they never decide what happens next.

### 1. Plan declares dependencies (prerequisite, also a standalone win)

`plan.md` and `PLAN_SCHEMA` gain an optional `depends_on: [filename, ...]`
per file. The planner already half-does this by ordering the list;
asking for it explicitly is one extra, simple field.

`plan_files` normalises it: names not in the plan are dropped, a missing
or malformed `depends_on` falls back to **"depends on every earlier file
in the list"** (= today's conservative behaviour, still correct just less
parallel). A cycle is detected and broken by reverting that file to plan
order.

`_sibling_context` then narrows: a file's codegen sees the source of its
**declared dependencies only**, not all priors. This is a small quality
improvement on its own — less prompt bloat, sharper import hints.

### 2. Endpoint pool

`config/default.yaml` gains:

```yaml
endpoints:
  - host: "http://192.168.50.142:11434"
    model: "qwen2.5-coder:7b"
  - host: "http://localhost:11434"
    model: "qwen2.5-coder:7b"
```

`ollama.host` / `ollama.model` stay as the single-endpoint shorthand and
the fallback when `endpoints` is absent. CLI: `--endpoint HOST,MODEL`
(repeatable) overrides the list. One endpoint configured => pool size 1
=> today's serial path, unchanged.

Each endpoint gets one `OllamaClient` and one worker thread.

### 3. Dispatcher

```
ready   = files with no unmet dependency        (a thread-safe set)
done    = files whose per-file loop succeeded
failed  = files whose per-file loop gave up      (non-advisory)

each worker, until ready is empty and no file is in-flight:
    task = ready.pop()            # blocks briefly if empty but work remains
    run the existing per-file loop (spec -> codegen -> verify -> fix)
        against this worker's client
    on success: done.add(task); promote any file whose deps are now all in done
    on give-up: failed.add(task); its dependents can never run -> mark them skipped
    on OllamaError: return this file to `ready`, retire this worker
```

- `plan` runs first on `endpoints[0]`; `integration` runs last on
  `endpoints[0]` after every file is `done`/`advisory`.
- The **global iteration budget** becomes a shared counter guarded by a
  lock, checked before each LLM call; exceeding it stops new tasks (files
  in flight finish).
- **Advisory** (`test_*.py` the planner asked for) is unchanged — it just
  happens on whichever worker picked it up.

### 4. Failure isolation

- One endpoint goes unreachable (`OllamaError`) -> its worker retires,
  its in-flight file goes back to `ready`, the other worker drains the
  rest. The run still succeeds.
- **All** endpoints unreachable / retired with work left -> `run_aborted`
  (the existing machinery), keeping every completed file.

### 5. What does NOT change

`_generate_and_fix`, every `verify.*` stage, retry-temperature
escalation, the empty-generation guard, advisory handling, `run_result` /
graceful abort, the summary and the GUI's SSE stream (`log.jsonl` append
is safe from multiple threads; `Session.log` already takes no lock but
each `write` of one JSON line is atomic on POSIX for this size — confirm,
or add a lock). The dispatcher is a new layer *around* the per-file loop.

### 6. Determinism

The **set of files** produced and the **final verdict** stay
deterministic. What becomes non-deterministic: the interleaving of log
events, and which endpoint built which file. Acceptable and documented —
the verifiers are still deterministic and the output project is the same
regardless of interleaving.

## Testing

- `FakeClient` per endpoint. A latency-simulating `FakeClient` (sleeps in
  `generate`) + timestamps prove two independent files' codegen calls
  **overlap in time**.
- DAG ordering: a dependent's codegen never starts before its
  dependency's `verify -> success` is logged.
- One-endpoint config reproduces today's serial event order exactly
  (regression guard).
- Endpoint B raises `OllamaError` mid-run -> B's file completes on A, run
  succeeds.
- All endpoints raise -> `run_aborted`, completed files kept.

## Phasing

1. `depends_on` in plan + `_sibling_context` narrowing. Small, valuable
   alone, and unblocks the rest.
2. Endpoint pool + dispatcher (serial-equivalent at pool size 1 first,
   then real threads).
3. CLI `--endpoint` / config `endpoints:` / a second host+model row in
   the GUI with its own re-fetch.

## Open questions / risks

- **7B planner emitting a usable `depends_on`.** Mitigation: the
  fall-back to "all prior files" is always safe.
- **Style drift** between two different models. Accepted — verify +
  integration still gate; a single-model two-endpoint setup avoids it
  entirely and is the recommended config.
- **`Session.log` concurrency.** Likely fine (one small append per
  event) but add a `threading.Lock` around the write to be certain.
- **Speed-up ceiling.** A linear dependency chain (every file depends on
  the previous) gets no parallelism — the win scales with how "wide" the
  planner's DAG is. Typical 3-5 file projects have 2-3 independent leaves,
  so ~1.5-2x in practice, not a clean 2x.

## What shipped

All three phases, TDD, no live run yet.

- **`FileTask.depends_on`** — `plan.md` + `PLAN_SCHEMA` ask for it; the
  planner's list is restricted to *earlier* files (acyclic by
  construction); a missing list falls back to "every earlier file" (the
  prior behaviour). `_sibling_context` now feeds a file only its declared
  dependencies' source.
- **`Session._lock`** guards each log write + `on_event` callback.
- **`Config.endpoints` / `Endpoint` / `resolved_endpoints()`** — an
  optional `endpoints:` list in the yaml; `--endpoint HOST,MODEL`
  (repeatable) on the CLI; `resolved_endpoints()` returns the list or the
  single `ollama.host`/`model`.
- **`MultiFileLoop._generate_files`** — the dispatcher: a
  `threading.Condition`, one worker thread per client in `pool_clients`
  (defaults to `[client]` = the old serial walk), `claim()` hands out a
  file once its `depends_on` are all done, skips a file whose dependency
  hard-failed, stops seeding at the iteration budget. A worker that hits
  `OllamaError` requeues its file and retires; the run only aborts when
  no worker can make progress. `plan` and integration stay on the primary
  client.
- **GUI** — `RunManager.start(endpoints=[...])` builds one client per
  endpoint; a "+ second endpoint" row (host + model + its own re-fetch)
  posts `endpoints` to `/api/run`.

Tests: two workers provably overlap (a `Barrier`), a dependent waits for
its dependency and gets its source, a failed dependency skips its
dependents and fails the run, a dead endpoint doesn't sink a run another
can finish, single-endpoint is byte-for-byte the old serial path.
