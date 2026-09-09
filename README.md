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

**Phase 0 + Phase 1 (this commit):** scaffolding, config, an Ollama client
wrapper, and the single-file core loop:

```
goal -> generate one Python file -> verify (compile, then run)
  -> on failure, feed the exact error back for a scoped fix (bounded retries)
```

Multi-file planning, a dedicated test-writing step, and multi-language
support are later phases (see the architecture doc) and not implemented
yet.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

You'll also need [Ollama](https://ollama.com) running locally with a model
pulled:

```bash
ollama pull deepseek-coder:8b
ollama serve   # if not already running
```

## Usage

```bash
harness run --goal "a script that prints the first 20 Fibonacci numbers"
```

Options:

- `--model` — override the model from `config/default.yaml` (default `deepseek-coder:8b`)
- `--host` — override the Ollama host (default `http://localhost:11434`)
- `--max-retries` — override the bounded fix-loop attempt count (default `3`)
- `--filename` — output filename (default `main.py`)

Each run creates `workspace/<run-id>/` containing the generated file and a
`log.jsonl` transcript of every prompt, response, and verifier result --
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
