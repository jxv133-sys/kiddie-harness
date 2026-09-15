"""`harness run --goal "..."` entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import progress, summary
from .config import ROLES, Config, Endpoint
from .llm_client import OllamaClient, OllamaError
from .orchestrator import MultiFileLoop, SingleFileLoop, partition_clients_by_role
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
        "--timeout",
        type=int,
        default=None,
        help="Override the per-call Ollama timeout in seconds (raise it for slow reasoning models)",
    )
    run.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override the starting generation length per call (raise it for a model that keeps "
        "getting cut off mid-file, e.g. a verbose reasoning model)",
    )
    run.add_argument(
        "--max-tokens-ceiling",
        type=int,
        default=None,
        help="Override the cap on adaptive growth after a truncated response",
    )
    run.add_argument(
        "--no-critic",
        action="store_true",
        help="Skip the critic check (an extra LLM call per file judging it against its own "
        "spec; never hard-fails a file that compiles/lints/imports clean, but costs a call)",
    )
    run.add_argument(
        "--super-review",
        action="store_true",
        help="After every file is built and integration has run, review the whole project "
        "together for cross-file problems a per-file critic can never see (a second "
        "endpoint confirms each finding before it's reported); advisory only, never fails "
        "the run. Off by default.",
    )
    run.add_argument(
        "--branch-after-fixes",
        type=int,
        default=None,
        metavar="N",
        help="Once a file's fix loop reaches N attempts, an idle endpoint tagged smart or "
        "balanced (never quick) may start its own independent attempt at the same file in "
        "parallel; whichever finishes first wins. 0 (the default) disables this.",
    )
    run.add_argument(
        "--endpoint",
        action="append",
        metavar="HOST,MODEL[,ROLE]",
        help="Extra Ollama backend for parallel multi-file generation (repeatable). "
        "First --endpoint replaces the default primary; use it twice for two. "
        "Optional third field tags its role: smart (plan/critic/integration-fix), "
        "quick (the per-file spec/codegen/fix grind), or balanced (does either, "
        "the default) -- see the Model roles section of the README.",
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
    run.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress live per-step progress lines; print only the final summary",
    )

    inspect = subparsers.add_parser("inspect", help="Summarize a previous run from its log.jsonl")
    inspect.add_argument("--run-id", required=True, help="Run id (the workspace/<run-id> directory name)")

    gui_cmd = subparsers.add_parser("gui", help="Open a minimal local web GUI")
    gui_cmd.add_argument("--port", type=int, default=8765, help="Port to bind (default 8765)")
    gui_cmd.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind (default 127.0.0.1, localhost-only). Use 0.0.0.0 to accept "
        "connections from other devices on the local network -- there is no authentication, "
        "so anyone who can reach the port can start/stop runs and read generated code.",
    )
    gui_cmd.add_argument("--no-browser", action="store_true", help="Do not open a browser")

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
    if result.aborted:
        return 2
    return 0 if result.success else 1


def _run_multi_file(
    client: OllamaClient,
    config: Config,
    session: Session,
    args,
    pool_clients: list[OllamaClient] | None = None,
    critic_client: OllamaClient | None = None,
    branch_pool: list[OllamaClient] | None = None,
) -> int:
    try:
        loop = MultiFileLoop(
            client,
            config,
            session,
            pool_clients=pool_clients,
            critic_client=critic_client,
            branch_pool=branch_pool,
        )
        result = loop.run(args.goal)
    except (OllamaError, PlanError) as exc:
        print(f"ERROR: {exc}")
        return 2

    print(summary.render_table(summary.load_run_summary(session.log_path)))

    if not result.success and not result.stopped_early:
        for f in result.files:
            if not f.success and not f.advisory:
                print(f"Last error in {Path(f.path).name}:")
                print(f.last_output)
        if result.integration is not None and not result.integration.success:
            print("Last integration error:")
            print(result.integration.output)
    elif result.success:
        advisory = [f for f in result.files if f.advisory]
        if advisory:
            names = ", ".join(Path(f.path).name for f in advisory)
            print(
                f"Note: {len(advisory)} advisory test(s) never passed and were left out "
                f"of the integration check: {names}",
                file=sys.stderr,
            )

    print(f"Full transcript: {session.log_path}")
    if result.aborted:
        return 2
    return 0 if result.success else 1


def _parse_endpoints(specs: list[str], *, config: Config) -> tuple[Endpoint, ...]:
    endpoints = []
    for spec in specs:
        host, _, rest = spec.partition(",")
        model, _, role = rest.partition(",")
        role = role.strip() or "balanced"
        if role not in ROLES:
            raise ValueError(
                f"invalid --endpoint role {role!r} in {spec!r}: expected one of {', '.join(ROLES)}"
            )
        endpoints.append(
            Endpoint(
                host.strip(), model.strip() or config.model, config.timeout_seconds, role=role
            )
        )
    return tuple(endpoints)


def _inspect(args) -> int:
    config = Config.load()
    log_path = config.workspace_root / args.run_id / "log.jsonl"
    if not log_path.exists():
        print(f"No such run: {args.run_id} (expected {log_path})")
        return 2

    run_summary = summary.load_run_summary(log_path)
    print(summary.render_table(run_summary))
    return 0 if run_summary.finished and run_summary.succeeded else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect":
        return _inspect(args)

    if args.command == "gui":
        from . import gui

        gui.serve(
            Config.load(), host=args.host, port=args.port, open_browser=not args.no_browser
        )
        return 0

    if args.command == "run":
        # Apply --model/--host/--timeout first, then parse --endpoint
        # against *that* -- an --endpoint with no model given should
        # default to what the user just asked for on this run, not
        # silently fall back to the raw yaml default underneath it.
        config = Config.load().with_overrides(
            model=args.model,
            host=args.host,
            max_fix_attempts=args.max_retries,
            timeout_seconds=args.timeout,
            max_tokens=args.max_tokens,
            max_tokens_ceiling=args.max_tokens_ceiling,
            critic_enabled=False if args.no_critic else None,
            super_review_enabled=True if args.super_review else None,
            branch_after_fixes=args.branch_after_fixes,
        )
        if args.endpoint:
            try:
                config = config.with_overrides(
                    endpoints=_parse_endpoints(args.endpoint, config=config)
                )
            except ValueError as exc:
                print(f"ERROR: {exc}")
                return 2
        endpoints = config.resolved_endpoints()
        all_clients = [OllamaClient(e.host, e.model, e.timeout_seconds) for e in endpoints]
        client, pool_clients, critic_client, branch_pool = partition_clients_by_role(
            endpoints, all_clients
        )
        reporter = None if args.quiet else progress.console_reporter()
        session = Session.create(config.workspace_root, on_event=reporter)

        print(f"Run {session.run_id}: goal = {args.goal!r}", flush=True)
        if len(endpoints) == 1:
            print(f"Model: {endpoints[0].model} @ {endpoints[0].host}", flush=True)
        else:
            joined = ", ".join(f"{e.model}@{e.host} ({e.role})" for e in endpoints)
            print(f"Endpoints: {joined}", flush=True)

        if args.multi_file:
            return _run_multi_file(
                client,
                config,
                session,
                args,
                pool_clients=pool_clients,
                critic_client=critic_client,
                branch_pool=branch_pool,
            )
        return _run_single_file(all_clients[0], config, session, args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
