"""The deterministic state machine driving the harness.

The LLM never decides what happens next. This module owns all sequencing,
retries, and stop conditions; the LLM is only ever called through the
narrow, single-purpose functions in harness.steps.
"""

from __future__ import annotations

import dataclasses
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from .config import Config, Endpoint
from .llm_client import OllamaClient, OllamaError
from .session import Session
from .steps import codegen, critic, plan, spec, super_review, verify
from .steps.plan import FileTask
from .steps.verify import VerifyResult


class RunCancelled(Exception):
    """Raised when a `cancel_event` is set between two LLM calls.

    Cooperative only: a call already in flight to Ollama still has to
    return (or time out) before this is noticed -- there is no way to
    safely kill a thread mid-request. It fires at the next checkpoint
    (after a verify, before the next fix or integration round), which is
    also every place a long fix loop actually spends its time stuck.
    """


def _wait_if_paused(
    pause_event: threading.Event | None, cancel_event: threading.Event | None
) -> None:
    """Blocks here while paused -- the exact same "between calls, never
    mid-request" checkpoint `cancel_event` uses (see `RunCancelled`).

    A file's config is read fresh at every use, never snapshotted, so a
    settings change applied while paused (`Config.apply_overrides`) is
    guaranteed to be in effect by the time this returns and the next LLM
    call goes out. Still honours `cancel_event` while blocked, so Stop
    works even mid-pause."""
    if pause_event is None:
        return
    while pause_event.is_set():
        if cancel_event is not None and cancel_event.is_set():
            raise RunCancelled("cancelled by user")
        pause_event.wait(timeout=0.5)


_T = TypeVar("_T")
_ENDPOINT_RETRY_ATTEMPTS = 3
_ENDPOINT_RETRY_BACKOFF_SECONDS = 5.0


def _with_endpoint_retry(
    fn: Callable[[], _T],
    *,
    attempts: int = _ENDPOINT_RETRY_ATTEMPTS,
    cancel_event: threading.Event | None = None,
) -> _T:
    """Runs `fn` (one call into `harness.steps`, backed by one specific
    Ollama endpoint), retrying up to `attempts` times (default
    `_ENDPOINT_RETRY_ATTEMPTS`) with a short pause between attempts if it
    raises `OllamaError`, before letting the last one propagate. Pass a
    smaller `attempts` when the caller already spent one itself (see the
    worker dispatch loop) so the *total* tries against one endpoint stays
    `_ENDPOINT_RETRY_ATTEMPTS`, not that many again on top.

    This is a deliberate exception to `llm_client`'s own "never loops or
    retries on its own" -- that principle is about the *client* not
    deciding what happens next; this is the orchestrator doing exactly
    that, the same way it already decides to escalate temperature on a
    content failure. A dead connection is very often transient (Ollama
    mid-restart, a brief Wi-Fi drop) and clears up within a few seconds;
    only an endpoint that fails every attempt gets treated as genuinely
    unreachable -- which, in `MultiFileLoop`'s per-file dispatch, is what
    hands the file to a *different* endpoint if one exists (see
    `_generate_files`'s `endpoint_retired` handling) or aborts the run if
    it doesn't. This is the "just retry" underneath that: give the same
    endpoint a real chance before falling back to either of those.

    The backoff itself still honours `cancel_event`, polled in short
    slices rather than one long sleep, so Stop doesn't have to wait out
    the full pause. Deliberately does *not* check `cancel_event` before
    the first attempt -- that's not a retry gap, it's the original call,
    and cancellation there is `fn` itself's own responsibility (checked
    between its own internal calls, never before the first), exactly as
    if this wrapper didn't exist."""
    last_exc: OllamaError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except OllamaError as exc:
            last_exc = exc
            if attempt < attempts:
                deadline = time.monotonic() + _ENDPOINT_RETRY_BACKOFF_SECONDS
                while time.monotonic() < deadline:
                    if cancel_event is not None and cancel_event.is_set():
                        raise RunCancelled("cancelled by user") from exc
                    time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
    assert last_exc is not None
    raise last_exc


@dataclasses.dataclass
class RunResult:
    success: bool
    file_path: str
    attempts: int
    last_output: str
    # Set when the model became unreachable mid-run: the run stopped
    # without ever reaching a verdict on the code.
    aborted: bool = False
    abort_reason: str = ""
    # True when the file compiles/runs clean but the critic step never
    # signed off on it against the goal -- see FileRunResult.spec_flagged.
    spec_flagged: bool = False


@dataclasses.dataclass
class FileRunResult:
    path: str
    purpose: str
    success: bool
    attempts: int
    last_output: str
    # True for a test file that never passed its own verify loop. Such a
    # test is reported but does not fail the run or block the integration
    # check -- a test we couldn't get green means "unverified", not
    # "the code is broken".
    advisory: bool = False
    # True when the file compiles, lints, and imports clean, but the
    # critic step (another LLM's opinion, not real tooling -- see
    # harness/steps/critic.py) never signed off on it against its own
    # spec, even after the normal bounded fix attempts. Reported, but --
    # like `advisory` -- never fails the run or blocks a dependent file:
    # an opinion that might be wrong must not be able to sink code that
    # every deterministic check already passed.
    spec_flagged: bool = False


@dataclasses.dataclass
class MultiFileRunResult:
    success: bool
    run_dir: str
    files: list[FileRunResult]
    integration: VerifyResult | None
    total_iterations: int
    stopped_early: bool
    # Set when the model became unreachable partway through; whatever
    # files completed before the outage are kept in `files`.
    aborted: bool = False
    abort_reason: str = ""
    # Whole-project findings from steps/super_review.py, one dict per
    # issue: {"file", "description", "confirmed"}. Empty when
    # super_review_enabled is off, integration never ran, or the review
    # pass itself found nothing -- advisory only, never affects `success`.
    cross_file_issues: list[dict] = dataclasses.field(default_factory=list)


def _safe_relative_path(raw: str) -> Path:
    """Keep a model-provided file path confined to the run directory.

    The planner's output ultimately controls where files get written, so a
    stray "../" or absolute path must never be allowed to escape the
    session's workspace directory.
    """
    candidate = Path(raw.strip())
    parts = [p for p in candidate.parts if p not in ("..", ".", "/")]
    if not parts:
        parts = ["unnamed.py"]
    return Path(*parts)


_TRUNCATION_NOTE = (
    "[NOTE: the previous response was cut off before finishing -- "
    "write a shorter, complete file if possible.]\n"
)

# Each retry of a fix samples a little hotter than the last. The prompt
# never changes -- same single instruction, same output contract -- but a
# stuck model at the configured (low) temperature returns byte-identical
# output every time, so more retries of an identical call explore nothing.
# Nudging the temperature is the one lever that gives extra attempts a
# real chance without asking the model to do anything more complex.
_RETRY_TEMPERATURE_STEP = 0.15
_RETRY_TEMPERATURE_MAX = 0.9

# A spec response that doesn't look like a real spec (disclaimer prose
# instead of bullet points -- the same "vacuous pass" failure mode
# verify.py's html/css/js checks guard against, but one step earlier,
# before it ever reaches codegen) gets a few retries, each a little
# hotter than the last, same reasoning as fix attempts. Small on
# purpose: spec generation is cheap and a genuinely stuck model won't
# recover with more tries than that -- codegen and verify remain the
# real backstop either way.
_SPEC_RETRY_ATTEMPTS = 3


def _retry_temperature(base: float, fix_attempt: int) -> float:
    """Sampling temperature for fix attempt N (1 = first fix)."""
    return round(min(base + _RETRY_TEMPERATURE_STEP * fix_attempt, _RETRY_TEMPERATURE_MAX), 3)


def partition_clients_by_role(
    endpoints: list[Endpoint], clients: list[OllamaClient]
) -> tuple[OllamaClient, list[OllamaClient], OllamaClient | None]:
    """Splits a resolved endpoint pool into (the client for plan/critic/
    integration-fix -- the "judgement" calls), (the pool for per-file
    spec/codegen/fix dispatch -- the high-volume grind), and (an explicit
    override for critic, or None).

    Every endpoint works the per-file grind regardless of role -- a
    "smart" endpoint being more capable is a reason to *also* give it
    files to build, not to leave it idle except for plan/critic. Roles
    only decide which client handles the judgement calls: a "smart"
    endpoint is preferred there and, once tagged, becomes the critic
    override -- every file's critic check, regardless of which worker
    actually built it, goes to the one model asked to be careful, not
    whichever one happened to grab the file. Without a "smart" tag
    anywhere, a "balanced" endpoint can still fall back into that role
    (matching today's behaviour before roles existed: the first endpoint
    is the plan/integration client, and critic runs on whichever worker
    built the file, not a fixed override). A "quick" endpoint never
    becomes the plan/critic client, not even as a fallback -- that's the
    one thing tagging something "quick" actually opts it out of.

    No endpoint tagged "smart" and none tagged "quick" (every one left at
    the "balanced" default, today's only option before roles existed)
    reproduces today's exact behaviour byte for byte.
    """
    smart = [c for e, c in zip(endpoints, clients) if e.role == "smart"]
    # Anything that isn't "smart" or "quick" -- "balanced", or a typo'd/
    # unrecognised role -- can still stand in for "smart" as a fallback
    # plan/critic client, same as "balanced" always could.
    balanced = [c for e, c in zip(endpoints, clients) if e.role not in ("smart", "quick")]

    primary = smart[0] if smart else (balanced[0] if balanced else clients[0])
    critic_override = smart[0] if smart else None
    return primary, list(clients), critic_override


def _generate_and_fix(
    client: OllamaClient,
    config: Config,
    session: Session,
    file_path: Path,
    instruction: str,
    verify_fn: Callable[[Path], VerifyResult],
    *,
    cancel_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
) -> tuple[VerifyResult, int]:
    """Shared bounded-retry loop: generate once, verify, fix on failure.

    Used by the single-file loop and by every file of the multi-file
    loop, so the "atomic prompt + deterministic verify + bounded retry"
    pattern only has one implementation. Returns the final VerifyResult
    and how many fix attempts were used.

    Tracks a per-file max_tokens that doubles (capped at
    config.max_tokens_ceiling) whenever a generation was truncated -- a
    file that ran out of tokens needs more room on the next attempt, not
    just a generic "here's the error" retry.
    """
    max_tokens = config.max_tokens
    with session.track_call("codegen", str(file_path), client.host) as update:
        gen = codegen.generate_file(
            client,
            instruction,
            path=str(file_path),
            temperature=config.temperature,
            max_tokens=max_tokens,
            on_chunk=update,
        )
    code = gen.code
    session.log(
        "codegen", path=str(file_path), code=code, truncated=gen.truncated, endpoint=client.host
    )
    if gen.truncated:
        max_tokens = min(max_tokens * 2, config.max_tokens_ceiling)

    attempts = 0
    while True:
        file_path.write_text(code)
        if code.strip():
            result = verify_fn(file_path)
        else:
            # An empty file compiles and imports fine -- it would sail
            # through verification as a "success". Treat it as a failure
            # the fix loop can act on instead.
            result = VerifyResult(
                success=False,
                stage="generate",
                output="The model returned an empty file. Write the complete file.",
            )
        session.log(
            "verify",
            path=str(file_path),
            attempt=attempts,
            stage=result.stage,
            success=result.success,
            output=result.output,
        )

        if result.success or attempts >= config.max_fix_attempts:
            return result, attempts

        _wait_if_paused(pause_event, cancel_event)
        if cancel_event is not None and cancel_event.is_set():
            raise RunCancelled("cancelled by user")

        if result.stage == "lint":
            # ruff's --fix may have just rewritten the file in place; make
            # sure the fix prompt sees the current file, not stale
            # pre-autofix content (matters when a second, genuinely
            # unfixable issue remains alongside an auto-fixed one).
            code = file_path.read_text()

        error = result.output
        if gen.truncated:
            error = _TRUNCATION_NOTE + error

        failed_code = code
        with session.track_call("fix", str(file_path), client.host) as update:
            gen = codegen.fix_file(
                client,
                code=code,
                error=error,
                stage=result.stage,
                path=str(file_path),
                temperature=_retry_temperature(config.temperature, attempts + 1),
                max_tokens=max_tokens,
                on_chunk=update,
            )
        code = gen.code
        attempts += 1
        session.log(
            "fix",
            path=str(file_path),
            attempt=attempts,
            code=code,
            truncated=gen.truncated,
            endpoint=client.host,
        )
        if gen.truncated:
            max_tokens = min(max_tokens * 2, config.max_tokens_ceiling)

        if code == failed_code:
            # The fix returned the exact bytes that just failed. Not fatal
            # any more -- the next attempt samples hotter and may break the
            # rut -- but worth logging: a run full of these means the model
            # is stuck, not converging.
            session.log("fix_noop", path=str(file_path), attempt=attempts, stage=result.stage)


def _with_critic(
    verify_fn: Callable[[Path], VerifyResult],
    client: OllamaClient,
    config: Config,
    session: Session,
    spec_text: str,
    calls: list[int],
) -> Callable[[Path], VerifyResult]:
    """Wraps a real-tooling `verify_fn` so that, once it passes, one more
    call asks the model whether its own output holds up against `spec_text`
    -- composes into `_generate_and_fix`'s existing `verify_fn` contract
    with no changes to that loop at all. `calls` is a mutable single-item
    counter the caller reads back afterwards, so the critic's LLM calls
    count against the run's iteration budget like any other call.

    Logs its own `critic_check` event on *every* call, agree or not --
    when it agrees, `verify_fn`'s own result is returned unchanged (so a
    passing critic never shows up as its own "verify" event), and without
    this there would be no evidence a critic call was ever made at all."""

    def wrapped(path: Path) -> VerifyResult:
        result = verify_fn(path)
        if not result.success:
            return result
        calls[0] += 1
        with session.track_call("critic", str(path), client.host) as update:
            verdict = critic.critique_file(
                client,
                spec_text,
                path.read_text(),
                str(path.name),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                on_chunk=update,
            )
        session.log(
            "critic_check",
            path=str(path),
            follows_spec=verdict.follows_spec,
            issues=verdict.issues,
            endpoint=client.host,
        )
        if verdict.follows_spec:
            return result
        return VerifyResult(success=False, stage="critic", output=verdict.issues)

    return wrapped


class SingleFileLoop:
    """Phase 1: goal -> one Python file -> verify -> bounded fix loop.

    This is the smallest slice of the full architecture: it exists to prove
    that "atomic prompt + deterministic verify + bounded retry" converges
    with a small local model before multi-file planning is layered on top.
    """

    def __init__(
        self,
        client: OllamaClient,
        config: Config,
        session: Session,
        *,
        cancel_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
    ):
        self.client = client
        self.config = config
        self.session = session
        self._cancel = cancel_event
        self._pause = pause_event

    def run(self, goal: str, filename: str = "main.py") -> RunResult:
        self.session.log("goal", goal=goal)
        file_path = self.session.run_dir / filename
        verify_fn = verify.verify_python_file
        critic_calls = [0]
        if self.config.critic_enabled:
            # No separate spec step in single-file mode -- the goal itself
            # is the only specification there is to check against.
            verify_fn = _with_critic(
                verify_fn, self.client, self.config, self.session, goal, critic_calls
            )

        try:
            result, attempts = _with_endpoint_retry(
                lambda: _generate_and_fix(
                    self.client,
                    self.config,
                    self.session,
                    file_path,
                    goal,
                    verify_fn,
                    cancel_event=self._cancel,
                    pause_event=self._pause,
                ),
                cancel_event=self._cancel,
            )
        except (OllamaError, RunCancelled) as exc:
            reason = "cancelled by user" if isinstance(exc, RunCancelled) else str(exc)
            self.session.log("run_aborted", reason=reason)
            self.session.log("run_result", success=False)
            return RunResult(
                success=False,
                file_path=str(file_path),
                attempts=0,
                last_output=reason,
                aborted=True,
                abort_reason=reason,
            )

        # A critic disagreement is the one verify failure that must not
        # sink an otherwise-working file (see FileRunResult.spec_flagged)
        # -- everything real (compile/run) already passed by this point.
        spec_flagged = not result.success and result.stage == "critic"
        success = result.success or spec_flagged
        if spec_flagged:
            self.session.log("spec_flagged", path=str(file_path), issues=result.output)
        elif not success:
            self.session.log("giving_up", attempts=attempts)
        self.session.log("run_result", success=success)

        return RunResult(
            success=success,
            file_path=str(file_path),
            attempts=attempts,
            last_output=result.output,
            spec_flagged=spec_flagged,
        )


class MultiFileLoop:
    """Phase 2: goal -> planned file list -> per-file spec + codegen + fix
    -> integration verify.

    Every stage is a fixed, narrow prompt: the planner never writes code,
    the spec writer never writes code, the code generator only ever sees
    one file's spec (plus the source of its declared dependencies). The
    small model is not asked to write tests -- that was a reliable source
    of unfixable "advisory" failures for weak models and added an LLM call
    per file for no gate. Tests exist in a generated project only if the
    goal (and so the planner) calls for a `test_*.py` file explicitly;
    such a file is built like any other and, if the model can't get it
    green, it is advisory rather than fatal.

    Per-file work is dispatched over one worker per configured endpoint:
    a file is claimed once every file in its `depends_on` has been built,
    so with a single endpoint this is the same serial walk as before, and
    with two it builds independent files concurrently. `plan` and the
    integration check always run on the primary client. Roles
    (see `partition_clients_by_role`) decide what `client` and
    `pool_clients` actually are by the time they get here -- this class
    itself doesn't know about roles at all, only about a primary client
    and a worker pool, plus one optional override.
    """

    def __init__(
        self,
        client: OllamaClient,
        config: Config,
        session: Session,
        *,
        pool_clients: list[OllamaClient] | None = None,
        cancel_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
        critic_client: OllamaClient | None = None,
    ):
        self.client = client
        self.config = config
        self.session = session
        self._pool = list(pool_clients) if pool_clients else [client]
        self._cancel = cancel_event
        self._pause = pause_event
        # None (the default) means "no override" -- a file's critic check
        # runs on whichever worker built that file, same as ever. Set by
        # a "smart"-tagged endpoint (see partition_clients_by_role) to
        # funnel every file's critic check through that one model
        # instead, regardless of which "quick" worker wrote the code.
        self._critic_client = critic_client

    def run(self, goal: str) -> MultiFileRunResult:
        self.session.log("goal", goal=goal)

        try:
            with self.session.track_call("plan", "", self.client.host) as update:
                tasks = _with_endpoint_retry(
                    lambda: plan.plan_files(
                        self.client,
                        goal,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        on_chunk=update,
                    ),
                    cancel_event=self._cancel,
                )
        except (OllamaError, RunCancelled) as exc:
            reason = "cancelled by user" if isinstance(exc, RunCancelled) else str(exc)
            self.session.log("run_aborted", reason=reason)
            self.session.log("run_result", success=False)
            return MultiFileRunResult(
                success=False,
                run_dir=str(self.session.run_dir),
                files=[],
                integration=None,
                total_iterations=1,
                stopped_early=False,
                aborted=True,
                abort_reason=reason,
            )
        self.session.log(
            "plan", files=[dataclasses.asdict(t) for t in tasks], endpoint=self.client.host
        )

        file_results, stopped_early, abort_reason, iterations = self._generate_files(goal, tasks)

        integration: VerifyResult | None = None
        cross_file_issues: list[dict] = []
        advisory_paths = [Path(f.path) for f in file_results if f.advisory]
        # A run's success rides on its non-advisory, non-spec_flagged
        # files: the implementation, and any test that actually passed.
        # A spec_flagged file already passed every real check (compile,
        # lint, import) -- only the critic's opinion disagreed, and an
        # opinion that might be wrong must not be able to fail the run.
        required = [f for f in file_results if not f.advisory and not f.spec_flagged]
        all_required_ok = bool(required) and all(f.success for f in required)
        if all_required_ok and not stopped_early and abort_reason is None:
            # A passing test_*.py file (one the planner asked for) means
            # pytest is the right integration check; otherwise fall back
            # to running an entry file.
            has_tests = any(
                Path(f.path).suffix == ".py"
                and Path(f.path).name.startswith("test_")
                and f.success
                for f in file_results
            )
            generated_paths = [Path(f.path) for f in required]
            try:
                integration, iterations = self._run_integration_with_fixes(
                    tasks, generated_paths, has_tests, iterations, ignore_paths=advisory_paths
                )
            except OllamaError as exc:
                abort_reason = str(exc)
            except RunCancelled:
                abort_reason = "cancelled by user"

            if self.config.super_review_enabled and abort_reason is None:
                try:
                    cross_file_issues, iterations = self._run_super_review(
                        goal, required, iterations
                    )
                except (OllamaError, RunCancelled):
                    # Advisory only -- a failure in the review pass itself
                    # must never sink an otherwise-successful run, same
                    # contract as the per-file critic.
                    pass

        overall_success = (
            abort_reason is None
            and all_required_ok
            and not stopped_early
            and (integration is None or integration.success)
        )

        if not overall_success and abort_reason is None and self._cancel is not None and self._cancel.is_set():
            # The run didn't already succeed before the cancellation was
            # noticed -- attribute the (non-)result to the user's Stop,
            # not a generic "gave up".
            abort_reason = "cancelled by user"

        if abort_reason is not None:
            self.session.log("run_aborted", reason=abort_reason)
        elif not overall_success:
            self.session.log("giving_up", iterations=iterations, stopped_early=stopped_early)
        self.session.log("run_result", success=overall_success)

        return MultiFileRunResult(
            success=overall_success,
            run_dir=str(self.session.run_dir),
            files=file_results,
            integration=integration,
            total_iterations=iterations,
            stopped_early=stopped_early,
            aborted=abort_reason is not None,
            abort_reason=abort_reason or "",
            cross_file_issues=cross_file_issues,
        )

    def _generate_files(
        self, goal: str, tasks: list[FileTask]
    ) -> tuple[list[FileRunResult], bool, str | None, int]:
        """Dispatch every file over the endpoint pool, honouring
        `depends_on`. Returns the per-file results (in completion order),
        whether the iteration budget was hit, an abort reason if the pool
        ran out of working endpoints, and the total LLM-call count."""
        cv = threading.Condition()
        pending = list(tasks)
        results: dict[str, FileRunResult] = {}
        order: list[str] = []
        done: set[str] = set()  # bare names a dependent can be satisfied by
        hard_failed: set[str] = set()  # bare names whose file gave up
        iterations = [1]  # the planning call
        stopped_early = [False]
        last_error: list[str | None] = [None]
        abort_reason: list[str | None] = [None]
        active = [len(self._pool)]
        # Workers currently mid-build (outside the lock, in _build_one_file).
        # An idle worker must not give up just because `pending` happens to
        # be momentarily empty -- the file a busy worker is holding can
        # still fail and land right back in `pending`, and by then an idle
        # worker that already returned is gone for good and never claims it.
        busy = [0]

        def record(task: FileTask, result: FileRunResult) -> None:
            results[result.path] = result
            order.append(result.path)
            name = Path(task.path).name
            if result.success or result.advisory or result.spec_flagged:
                done.add(name)
            else:
                hard_failed.add(name)
            # The one place that knows a file's build loop has actually
            # concluded (success, exhausted its retries, or skipped) --
            # as opposed to a `verify` event that merely failed one of
            # possibly several attempts still to come. Lets a live
            # viewer (the GUI's dependency graph) tell "still retrying"
            # apart from "genuinely done", which a bare verify-event scan
            # can't -- especially with a paused run holding a failed
            # attempt open indefinitely before its next retry.
            self.session.log("file_result", path=result.path, success=result.success)

        def claim() -> FileTask | None:
            i = 0
            while i < len(pending):
                task = pending[i]
                if any(d in hard_failed for d in task.depends_on):
                    pending.pop(i)
                    skipped_path = str(self.session.run_dir / _safe_relative_path(task.path))
                    self.session.log(
                        "skipped", path=skipped_path, reason="a dependency did not build"
                    )
                    record(
                        task,
                        FileRunResult(
                            path=skipped_path,
                            purpose=task.purpose,
                            success=False,
                            attempts=0,
                            last_output="skipped: a dependency did not build",
                            advisory=False,
                        ),
                    )
                    continue
                if all(d in done for d in task.depends_on):
                    if iterations[0] >= self.config.max_total_iterations:
                        self.session.log(
                            "budget_exhausted", before=task.path, iterations=iterations[0]
                        )
                        stopped_early[0] = True
                        return None
                    return pending.pop(i)
                i += 1
            return None

        def worker(client: OllamaClient) -> None:
            try:
                while True:
                    with cv:
                        while True:
                            if abort_reason[0] is not None:
                                return
                            if self._cancel is not None and self._cancel.is_set():
                                abort_reason[0] = "cancelled by user"
                                cv.notify_all()
                                return
                            if self._pause is not None and self._pause.is_set():
                                # Don't claim a new file while paused -- but
                                # don't touch `pending`/`active`/`busy` either,
                                # so the "real stall" check below never fires
                                # just because everyone's sitting here waiting.
                                # A file another worker already has stays
                                # mid-build; it hits its own pause checkpoint
                                # in `_generate_and_fix` between fix attempts.
                                cv.wait(timeout=0.5)
                                continue
                            task = claim()
                            if task is not None:
                                busy[0] += 1
                                break
                            if stopped_early[0] or (not pending and busy[0] == 0):
                                # Nothing left to claim, and nobody else is
                                # mid-build to possibly fail and reissue more
                                # -- there is genuinely no more work for me.
                                return
                            if pending and active[0] <= 1:
                                # Something is still pending but blocked on a
                                # dependency, and I'm the last worker left --
                                # a real stall, not just "no work right now".
                                abort_reason[0] = (
                                    last_error[0]
                                    or "no endpoint could build the remaining files"
                                )
                                cv.notify_all()
                                return
                            cv.wait(timeout=0.5)
                        snapshot = list(results.values())
                    try:
                        result, used = self._build_one_file(client, goal, task, snapshot)
                    except OllamaError as first_exc:
                        with cv:
                            am_last_worker = active[0] <= 1
                        final_exc: OllamaError = first_exc
                        if am_last_worker:
                            # No other endpoint to hand this file to right
                            # now -- retry this one, with backoff, before
                            # giving up on it. Skipped when another worker
                            # is still active: that worker can pick this
                            # file up immediately once it's requeued below,
                            # which is strictly faster than waiting out a
                            # retry here that might not even be needed.
                            try:
                                result, used = _with_endpoint_retry(
                                    lambda task=task, snapshot=snapshot: self._build_one_file(
                                        client, goal, task, snapshot
                                    ),
                                    # One attempt already spent above --
                                    # this makes up the rest of
                                    # _ENDPOINT_RETRY_ATTEMPTS, not that
                                    # many more on top.
                                    attempts=_ENDPOINT_RETRY_ATTEMPTS - 1,
                                    cancel_event=self._cancel,
                                )
                            except OllamaError as retried_exc:
                                final_exc = retried_exc
                            except RunCancelled:
                                with cv:
                                    abort_reason[0] = abort_reason[0] or "cancelled by user"
                                    busy[0] -= 1
                                    cv.notify_all()
                                return
                            else:
                                with cv:
                                    iterations[0] += used
                                    busy[0] -= 1
                                    record(task, result)
                                    cv.notify_all()
                                continue
                        with cv:
                            last_error[0] = str(final_exc)
                            pending.insert(0, task)  # another endpoint may manage it
                            busy[0] -= 1
                            cv.notify_all()
                        # Otherwise this endpoint just silently vanishes from
                        # the log and the file's whole build (spec included)
                        # quietly restarts from scratch on another worker --
                        # confusing to watch live with no explanation.
                        self.session.log(
                            "endpoint_retired",
                            path=task.path,
                            endpoint=client.host,
                            reason=str(final_exc),
                        )
                        return
                    except RunCancelled:
                        with cv:
                            abort_reason[0] = abort_reason[0] or "cancelled by user"
                            busy[0] -= 1
                            cv.notify_all()
                        return
                    with cv:
                        iterations[0] += used
                        busy[0] -= 1
                        record(task, result)
                        cv.notify_all()
            finally:
                with cv:
                    active[0] -= 1
                    cv.notify_all()

        threads = [threading.Thread(target=worker, args=(c,), daemon=True) for c in self._pool]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if abort_reason[0] is None and pending and not stopped_early[0]:
            abort_reason[0] = last_error[0] or "the model became unreachable mid-run"

        return [results[p] for p in order], stopped_early[0], abort_reason[0], iterations[0]

    def _build_one_file(
        self, client: OllamaClient, goal: str, task: FileTask, built_so_far: list[FileRunResult]
    ) -> tuple[FileRunResult, int]:
        spec_text = ""
        spec_calls = 0
        for spec_attempt in range(1, _SPEC_RETRY_ATTEMPTS + 1):
            spec_calls += 1
            with self.session.track_call("spec", task.path, client.host) as update:
                spec_text = spec.write_spec(
                    client,
                    goal,
                    task,
                    temperature=_retry_temperature(self.config.temperature, spec_attempt - 1),
                    max_tokens=self.config.max_tokens,
                    on_chunk=update,
                )
            if spec.looks_like_a_spec(spec_text) or spec_attempt == _SPEC_RETRY_ATTEMPTS:
                break
            self.session.log(
                "spec_rejected",
                path=task.path,
                spec=spec_text,
                attempt=spec_attempt,
                endpoint=client.host,
            )
        self.session.log("spec", path=task.path, spec=spec_text, endpoint=client.host)

        file_path = self.session.run_dir / _safe_relative_path(task.path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        instruction = (
            f"Create the file `{task.path}`.\n"
            f"Purpose: {task.purpose}\n\n"
            f"Specification:\n{spec_text}"
            f"{self._sibling_context(task, built_so_far)}"
        )
        # test_*.py is a Python-only convention -- pytest can't run a
        # "test_button.js", so that verify path only ever applies to an
        # actual Python file named that way.
        is_test_file = Path(task.path).suffix == ".py" and Path(task.path).name.startswith("test_")
        verify_fn = verify.verify_test_file if is_test_file else verify.verify_generated_file
        critic_calls = [0]
        if self.config.critic_enabled and not is_test_file:
            # Not applied to test files: a test's own pass/fail against
            # pytest already is its verification: layering a second,
            # fallible opinion on top adds uncertainty without a clear
            # question for it to answer.
            verify_fn = _with_critic(
                verify_fn,
                self._critic_client or client,
                self.config,
                self.session,
                spec_text,
                critic_calls,
            )
        result, attempts = _generate_and_fix(
            client,
            self.config,
            self.session,
            file_path,
            instruction,
            verify_fn,
            cancel_event=self._cancel,
            pause_event=self._pause,
        )
        advisory = is_test_file and not result.success
        if advisory:
            self.session.log("advisory_test", path=str(file_path), last_error=result.output)

        spec_flagged = not result.success and result.stage == "critic"
        if spec_flagged:
            self.session.log("spec_flagged", path=str(file_path), issues=result.output)

        return (
            FileRunResult(
                path=str(file_path),
                purpose=task.purpose,
                success=result.success,
                attempts=attempts,
                last_output=result.output,
                advisory=advisory,
                spec_flagged=spec_flagged,
            ),
            spec_calls + 1 + attempts + critic_calls[0],  # spec attempts + codegen + fixes + critic
        )

    _SIBLING_CONTEXT_CHAR_CAP = 6000

    def _sibling_context(self, task: FileTask, file_results: list[FileRunResult]) -> str:
        """The content of the files this file declares it depends on, so
        it can build on them by name instead of guessing (or
        re-implementing what a dependency already provides). With no
        explicit `depends_on`, `task.depends_on` is every earlier file --
        the same as before this narrowing existed.

        Grouped by whether the dependency is itself a Python module --
        only those get the "import by module name" instruction and a
        ```python fence. Used to apply that framing to *every*
        dependency regardless of its real language: a Python file
        depending on `login_page.html` would see the HTML fenced as
        ```python and be told to `from login_page import ...` it, which
        is not just misleading but impossible (Python cannot import an
        .html file) -- a real, observed failure mode, not a hypothetical
        one. Anything non-Python is still included as real reference
        content (a server file legitimately benefits from seeing the
        exact HTML it needs to serve), just fenced under its own
        language and explicitly marked as not importable.
        """
        wanted = set(task.depends_on)
        py_blocks: list[str] = []
        other_blocks: list[str] = []
        used = 0
        for f in file_results:
            name = Path(f.path).name
            # A spec_flagged file already compiles/lints/imports clean --
            # only the critic disagreed -- so it's still safe, real source
            # for a dependent to build on.
            ok = f.success or f.spec_flagged
            if name not in wanted or not ok or f.advisory or name.startswith("test_"):
                continue
            source = Path(f.path).read_text()
            if used + len(source) > self._SIBLING_CONTEXT_CHAR_CAP:
                break
            used += len(source)
            suffix = Path(name).suffix.lower()
            if suffix == ".py":
                py_blocks.append(f"### {name}\n```python\n{source}\n```")
            else:
                other_blocks.append(f"### {name}\n```{suffix.lstrip('.') or 'text'}\n{source}\n```")

        sections = []
        if py_blocks:
            sections.append(
                "Modules already created in this project -- import what you "
                "need from them by module name (the filename without `.py`); "
                "do not re-implement what they already provide:\n"
                + "\n\n".join(py_blocks)
            )
        if other_blocks:
            sections.append(
                "Other project files already created that this file depends "
                "on, for reference only -- these are NOT Python modules and "
                "must never be `import`ed; read, serve, or reference them by "
                "their filename the way the task describes:\n"
                + "\n\n".join(other_blocks)
            )
        if not sections:
            return ""
        return "\n\n" + "\n\n".join(sections)

    def _run_integration_with_fixes(
        self,
        tasks: list[FileTask],
        generated_paths: list[Path],
        has_tests: bool,
        iterations: int,
        *,
        ignore_paths: list[Path] | None = None,
    ) -> tuple[VerifyResult | None, int]:
        result = self._run_integration(tasks, has_tests, ignore_paths)
        if result is None:
            return None, iterations

        rounds = 0
        max_rounds = self.config.max_fix_attempts
        while (
            not result.success
            and rounds < max_rounds
            and iterations < self.config.max_total_iterations
            and not (self._cancel is not None and self._cancel.is_set())
        ):
            _wait_if_paused(self._pause, self._cancel)
            target = self._find_implicated_file(result.output, generated_paths)
            if target is None:
                break

            with self.session.track_call(
                "integration_fix", str(target), self.client.host
            ) as update:
                gen = _with_endpoint_retry(
                    lambda target=target, result=result, rounds=rounds: codegen.fix_file(
                        self.client,
                        code=target.read_text(),
                        error=result.output,
                        stage=result.stage,
                        path=str(target),
                        temperature=_retry_temperature(self.config.temperature, rounds + 1),
                        max_tokens=self.config.max_tokens,
                        on_chunk=update,
                    ),
                    cancel_event=self._cancel,
                )
            target.write_text(gen.code)
            iterations += 1
            rounds += 1
            self.session.log("integration_fix", path=str(target), round=rounds, code=gen.code)
            result = self._run_integration(tasks, has_tests, ignore_paths)

        return result, iterations

    def _run_integration(
        self, tasks: list[FileTask], has_tests: bool, ignore_paths: list[Path] | None = None
    ) -> VerifyResult | None:
        if has_tests:
            result = verify.run_pytest(self.session.run_dir, ignore=ignore_paths)
        else:
            entry = self._pick_entry_path(tasks)
            if entry is None:
                return None
            entry_path = self.session.run_dir / entry
            result = verify.run_script(entry_path)
            if not result.success:
                # With no test to say how this program is meant to be run
                # (args? stdin? a file?), a non-zero exit from a blind run
                # is not proof it's broken. Fall back to confirming the
                # entry point and its cross-file imports at least load --
                # that is the part "integration" can actually verify.
                result = verify.import_check(entry_path)

        self.session.log(
            "integration_verify", stage=result.stage, success=result.success, output=result.output
        )
        return result

    def _run_super_review(
        self, goal: str, required: list[FileRunResult], iterations: int
    ) -> tuple[list[dict], int]:
        """Once every required file is built, hand all of them to the
        reviewer client (the same primary client used for plan/critic/
        integration-fix -- the existing "judgement calls" client) and ask
        it to find whole-project problems a per-file critic can never see,
        since it's only ever shown one file. Each finding is checked
        against a second, different pool client before being reported as
        confirmed; unconfirmed findings are still returned, just marked
        as such -- filtering silently would risk dropping a real issue
        just because two small models didn't happen to agree.

        Skips the confirm step (every finding comes back unconfirmed) when
        the pool has no second distinct client to ask -- a single-endpoint
        run has no "another agent" to check against."""
        if iterations >= self.config.max_total_iterations:
            return [], iterations
        confirm_client = next((c for c in self._pool if c is not self.client), None)
        files = [(Path(f.path).name, Path(f.path).read_text()) for f in required]

        with self.session.track_call("super_review", "", self.client.host) as update:
            found = super_review.find_issues(
                self.client,
                goal,
                files,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                on_chunk=update,
            )
        iterations += 1
        self.session.log(
            "super_review",
            issues=[dataclasses.asdict(i) for i in found],
            endpoint=self.client.host,
        )

        results: list[dict] = []
        for issue in found:
            confirmed = False
            if confirm_client is not None and iterations < self.config.max_total_iterations:
                with self.session.track_call(
                    "super_review_confirm", issue.file, confirm_client.host
                ) as update:
                    confirmed = super_review.confirm_issue(
                        confirm_client,
                        files,
                        issue,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        on_chunk=update,
                    )
                iterations += 1
                self.session.log(
                    "super_review_confirm",
                    file=issue.file,
                    description=issue.description,
                    confirmed=confirmed,
                    endpoint=confirm_client.host,
                )
            results.append(
                {"file": issue.file, "description": issue.description, "confirmed": confirmed}
            )
        return results, iterations

    def _pick_entry_path(self, tasks: list[FileTask]) -> Path | None:
        """The file `run_script`/`import_check` treats as the program's
        entry point -- always a `.py` file, since that's the only thing
        this harness can actually execute. A goal that's pure HTML/CSS/JS
        (a static page with no Python server) has no entry point at all;
        returning None here means `_run_integration` just skips the
        integration check rather than trying to `python file.html`."""
        py_tasks = [t for t in tasks if Path(t.path).suffix == ".py"]
        for task in py_tasks:
            if Path(task.path).name == "main.py":
                return _safe_relative_path(task.path)
        return _safe_relative_path(py_tasks[0].path) if py_tasks else None

    def _find_implicated_file(self, error_text: str, candidate_paths: list[Path]) -> Path | None:
        """Best-effort: a traceback almost always names the file it failed
        in, so look for one of the generated files' names in the error
        text. If none matches, there's nothing safe to scope a fix to.

        The name must sit on a boundary -- not preceded by a word char or
        a dot -- so "main.py" doesn't match inside "domain.py"."""
        for path in candidate_paths:
            if re.search(rf"(?<![\w.]){re.escape(path.name)}", error_text):
                return path
        return None
