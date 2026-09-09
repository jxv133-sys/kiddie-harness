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
  -> for each file: write a short spec -> generate -> verify (compile, then lint)
       -> on failure, feed the exact error back for a scoped fix (bounded retries)
```

**Phase 3 (this commit):** a dedicated test-writing step, added after each
implementation file:

```
  -> for each implementation file that passed: write a test file for it
       (its own call, its own prompt -- never combined with implementation)
       -> verify (compile, then run just that test), fix on failure the same way
  -> once every file (and its test) passes: integration check
       (run the whole test suite with pytest if any test succeeded, else run the entry file)
       -> on failure, find which generated file the error names and fix just that file
          (bounded rounds, and a global iteration budget across the whole run)
```

Every step above is still a single, narrow LLM call: the planner never
writes code, the spec writer never writes code, code generation for one
file never sees any other file's contents, and the test writer is always
a separate call from the implementation it's testing.

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
- `--max-retries` — override the bounded fix-loop attempt count per file (default `3`)
- `--filename` — output filename, single-file mode only (default `main.py`)
- `--multi-file` — plan and generate a multi-file project instead of one script

Each run creates `workspace/<run-id>/` containing the generated file(s) and
a `log.jsonl` transcript of every prompt, response, and verifier result --
useful for seeing exactly where a small model went wrong.

## Tests

```bash
pytest
```

Tests use a fake LLM client (queued canned responses) so the orchestrator
and verifier logic can be exercised without a running Ollama server.

## Configuration

Defaults live in `config/default.yaml`: model name, generation temperature
(kept low — small models drift more at higher temperature and every step
here needs one predictable output, not creative variety), token limits,
and retry budgets.
