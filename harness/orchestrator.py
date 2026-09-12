"""The deterministic state machine driving the harness.

The LLM never decides what happens next. This module owns all sequencing,
retries, and stop conditions; the LLM is only ever called through the
narrow, single-purpose functions in harness.steps.
"""

from __future__ import annotations

import dataclasses
import re
import threading
from collections.abc import Callable
from pathlib import Path

from .config import Config
from .llm_client import OllamaClient, OllamaError
from .session import Session
from .steps import codegen, critic, plan, spec, verify
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


def _retry_temperature(base: float, fix_attempt: int) -> float:
    """Sampling temperature for fix attempt N (1 = first fix)."""
    return round(min(base + _RETRY_TEMPERATURE_STEP * fix_attempt, _RETRY_TEMPERATURE_MAX), 3)


def _generate_and_fix(
    client: OllamaClient,
    config: Config,
    session: Session,
    file_path: Path,
    instruction: str,
    verify_fn: Callable[[Path], VerifyResult],
    *,
    cancel_event: threading.Event | None = None,
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
    gen = codegen.generate_file(
        client, instruction, temperature=config.temperature, max_tokens=max_tokens
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
        gen = codegen.fix_file(
            client,
            code=code,
            error=error,
            stage=result.stage,
            temperature=_retry_temperature(config.temperature, attempts + 1),
            max_tokens=max_tokens,
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
        verdict = critic.critique_file(
            client,
            spec_text,
            path.read_text(),
            str(path.name),
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        session.log(
            "critic_check", path=str(path), follows_spec=verdict.follows_spec, issues=verdict.issues
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
    ):
        self.client = client
        self.config = config
        self.session = session
        self._cancel = cancel_event

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
            result, attempts = _generate_and_fix(
                self.client,
                self.config,
                self.session,
                file_path,
                goal,
                verify_fn,
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
    integration check always run on the primary client.
    """

    def __init__(
        self,
        client: OllamaClient,
        config: Config,
        session: Session,
        *,
        pool_clients: list[OllamaClient] | None = None,
        cancel_event: threading.Event | None = None,
    ):
        self.client = client
        self.config = config
        self.session = session
        self._pool = list(pool_clients) if pool_clients else [client]
        self._cancel = cancel_event

    def run(self, goal: str) -> MultiFileRunResult:
        self.session.log("goal", goal=goal)

        tasks = plan.plan_files(
            self.client, goal, temperature=self.config.temperature, max_tokens=self.config.max_tokens
        )
        self.session.log("plan", files=[dataclasses.asdict(t) for t in tasks])

        file_results, stopped_early, abort_reason, iterations = self._generate_files(goal, tasks)

        integration: VerifyResult | None = None
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
                Path(f.path).name.startswith("test_") and f.success for f in file_results
            )
            generated_paths = [Path(f.path) for f in required]
            try:
                integration, iterations = self._run_integration_with_fixes(
                    tasks, generated_paths, has_tests, iterations, ignore_paths=advisory_paths
                )
            except OllamaError as exc:
                abort_reason = str(exc)

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
                    except OllamaError as exc:
                        with cv:
                            last_error[0] = str(exc)
                            pending.insert(0, task)  # another endpoint may manage it
                            busy[0] -= 1
                            cv.notify_all()
                        # Otherwise this endpoint just silently vanishes from
                        # the log and the file's whole build (spec included)
                        # quietly restarts from scratch on another worker --
                        # confusing to watch live with no explanation.
                        self.session.log(
                            "endpoint_retired", path=task.path, endpoint=client.host, reason=str(exc)
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
        spec_text = spec.write_spec(
            client,
            goal,
            task,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
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
        is_test_file = Path(task.path).name.startswith("test_")
        verify_fn = verify.verify_test_file if is_test_file else verify.verify_python_file_static
        critic_calls = [0]
        if self.config.critic_enabled and not is_test_file:
            # Not applied to test files: a test's own pass/fail against
            # pytest already is its verification: layering a second,
            # fallible opinion on top adds uncertainty without a clear
            # question for it to answer.
            verify_fn = _with_critic(
                verify_fn, client, self.config, self.session, spec_text, critic_calls
            )
        result, attempts = _generate_and_fix(
            client,
            self.config,
            self.session,
            file_path,
            instruction,
            verify_fn,
            cancel_event=self._cancel,
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
            2 + attempts + critic_calls[0],  # spec + codegen + fixes + critic checks
        )

    _SIBLING_CONTEXT_CHAR_CAP = 6000

    def _sibling_context(self, task: FileTask, file_results: list[FileRunResult]) -> str:
        """The source of the modules this file declares it depends on, so
        its codegen imports from them by name instead of guessing (or
        re-implementing what a dependency already provides). With no
        explicit `depends_on`, `task.depends_on` is every earlier file --
        the same as before this narrowing existed."""
        wanted = set(task.depends_on)
        blocks: list[str] = []
        used = 0
        for f in file_results:
            name = Path(f.path).name
            # A spec_flagged file already compiles/lints/imports clean --
            # only the critic disagreed -- so it's still safe, real source
            # for a dependent to import from.
            ok = f.success or f.spec_flagged
            if name not in wanted or not ok or f.advisory or name.startswith("test_"):
                continue
            source = Path(f.path).read_text()
            if used + len(source) > self._SIBLING_CONTEXT_CHAR_CAP:
                break
            used += len(source)
            blocks.append(f"### {name}\n```python\n{source}\n```")
        if not blocks:
            return ""
        return (
            "\n\nModules already created in this project -- import what you "
            "need from them by module name (the filename without `.py`); do "
            "not re-implement what they already provide:\n" + "\n\n".join(blocks)
        )

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
            target = self._find_implicated_file(result.output, generated_paths)
            if target is None:
                break

            gen = codegen.fix_file(
                self.client,
                code=target.read_text(),
                error=result.output,
                stage=result.stage,
                temperature=_retry_temperature(self.config.temperature, rounds + 1),
                max_tokens=self.config.max_tokens,
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

    def _pick_entry_path(self, tasks: list[FileTask]) -> Path | None:
        for task in tasks:
            if Path(task.path).name == "main.py":
                return _safe_relative_path(task.path)
        return _safe_relative_path(tasks[0].path) if tasks else None

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
