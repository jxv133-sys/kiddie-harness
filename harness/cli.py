"""`harness run --goal "..."` entrypoint."""

from __future__ import annotations

import argparse
import sys

from . import summary
from .config import Config
from .llm_client import OllamaClient, OllamaError
from .orchestrator import MultiFileLoop, SingleFileLoop
from .session import Session
from .steps.plan import PlanError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Generate a project from a goal")
    run.add_argument("--goal", required=True, help="Plain-language description of what to build")
    run.add_argument("--model", default=None, help="Override the Ollama model name")
    run.add_argument("--host", default=None, help="Override the Ollama server host")
    run.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Override the max fix attempts per file",
    )
    run.add_argument(
        "--filename",
        default="main.py",
        help="Output filename for the generated file (single-file mode only)",
    )
    run.add_argument(
        "--multi-file",
        action="store_true",
        help="Plan and generate a multi-file project instead of a single script",
    )

    inspect = subparsers.add_parser("inspect", help="Summarize a previous run from its log.jsonl")
    inspect.add_argument("--run-id", required=True, help="Run id (the workspace/<run-id> directory name)")

    return parser


def _run_single_file(client: OllamaClient, config: Config, session: Session, args) -> int:
    try:
        loop = SingleFileLoop(client, config, session)
        result = loop.run(args.goal, filename=args.filename)
    except (OllamaError, PlanError) as exc:
        print(f"ERROR: {exc}")
        return 2

    print(summary.render_table(summary.load_run_summary(session.log_path)))
    if not result.success:
        print("Last error:")
        print(result.last_output)
    print(f"Full transcript: {session.log_path}")
    return 0 if result.success else 1


def _run_multi_file(client: OllamaClient, config: Config, session: Session, args) -> int:
    try:
        loop = MultiFileLoop(client, config, session)
        result = loop.run(args.goal)
    except (OllamaError, PlanError) as exc:
        print(f"ERROR: {exc}")
        return 2

    print(summary.render_table(summary.load_run_summary(session.log_path)))
    if (
        not result.success
        and not result.stopped_early
        and result.integration is not None
        and not result.integration.success
    ):
        print("Last integration error:")
        print(result.integration.output)
    print(f"Full transcript: {session.log_path}")
    return 0 if result.success else 1


def _inspect(args) -> int:
    config = Config.load()
    log_path = config.workspace_root / args.run_id / "log.jsonl"
    if not log_path.exists():
        print(f"No such run: {args.run_id} (expected {log_path})")
        return 2

    run_summary = summary.load_run_summary(log_path)
    print(summary.render_table(run_summary))
    return 0 if run_summary.succeeded else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect":
        return _inspect(args)

    if args.command == "run":
        config = Config.load().with_overrides(
            model=args.model, host=args.host, max_fix_attempts=args.max_retries
        )
        client = OllamaClient(config.ollama_host, config.model, config.timeout_seconds)
        session = Session.create(config.workspace_root)

        print(f"Run {session.run_id}: goal = {args.goal!r}")
        print(f"Model: {config.model} @ {config.ollama_host}")

        if args.multi_file:
            return _run_multi_file(client, config, session, args)
        return _run_single_file(client, config, session, args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
