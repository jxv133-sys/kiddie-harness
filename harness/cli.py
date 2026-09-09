"""`harness run --goal "..."` entrypoint."""

from __future__ import annotations

import argparse
import sys

from .config import Config
from .llm_client import OllamaClient
from .orchestrator import SingleFileLoop
from .session import Session


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
        help="Override the max fix attempts for the single-file loop",
    )
    run.add_argument(
        "--filename",
        default="main.py",
        help="Output filename for the generated file (Phase 1: single-file only)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        config = Config.load().with_overrides(
            model=args.model, host=args.host, max_fix_attempts=args.max_retries
        )
        client = OllamaClient(config.ollama_host, config.model, config.timeout_seconds)
        session = Session.create(config.workspace_root)

        print(f"Run {session.run_id}: goal = {args.goal!r}")
        print(f"Model: {config.model} @ {config.ollama_host}")

        loop = SingleFileLoop(client, config, session)
        result = loop.run(args.goal, filename=args.filename)

        if result.success:
            print(f"SUCCESS after {result.attempts} fix attempt(s): {result.file_path}")
            return 0

        print(f"FAILED after {result.attempts} fix attempt(s): {result.file_path}")
        print("Last error:")
        print(result.last_output)
        print(f"Full transcript: {session.log_path}")
        return 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
