"""The deterministic state machine driving the harness.

The LLM never decides what happens next. This module owns all sequencing,
retries, and stop conditions; the LLM is only ever called through the
narrow, single-purpose functions in harness.steps.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from pathlib import Path

from .config import Config
from .llm_client import OllamaClient, OllamaError
from .session import Session
from .steps import codegen, plan, spec, testgen, verify
from .steps.plan import FileTask
from .steps.verify import VerifyResult


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
    generate_fn: Callable[..., codegen.GeneratedCode] = codegen.generate_file,
    fix_context: str = "",
) -> tuple[VerifyResult, int]:
    """Shared bounded-retry loop: generate once, verify, fix on failure.

    Used by the single-file loop and by each implementation/test file of
    the multi-file loop, so the "atomic prompt + deterministic verify +
    bounded retry" pattern only has one implementation. generate_fn swaps
    in testgen.generate_test_file for test files; fixing always reuses
    codegen.fix_file since a fix only ever needs the current code and the
    exact error, regardless of what kind of file it is. Returns the final
    VerifyResult and how many fix attempts were used.

    Tracks a per-file max_tokens that doubles (capped at
    config.max_tokens_ceiling) whenever a generation was truncated -- a
    file that ran out of tokens needs more room on the next attempt, not
    just a generic "here's the error" retry.
    """
    max_tokens = config.max_tokens
    gen = generate_fn(client, instruction, temperature=config.temperature, max_tokens=max_tokens)
    code = gen.code
    session.log("codegen", path=str(file_path), code=code, truncated=gen.truncated)
    if gen.truncated:
        max_tokens = min(max_tokens * 2, config.max_tokens_ceiling)

    attempts = 0
    while True:
        file_path.write_text(code)
        if code.strip():
            result = verify_fn(file_path)
        else:
            # An empty file compiles and imports fine -- it would sail
            # through verification as a "success" and leave a blank module
            # for its companion test to import. Treat it as a failure the
            # fix loop can act on instead.
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
            context=fix_context,
        )
        code = gen.code
        attempts += 1
        session.log("fix", path=str(file_path), attempt=attempts, code=code, truncated=gen.truncated)
        if gen.truncated:
            max_tokens = min(max_tokens * 2, config.max_tokens_ceiling)

        if code == failed_code:
            # The fix returned the exact bytes that just failed. Not fatal
            # any more -- the next attempt samples hotter and may break the
            # rut -- but worth logging: a run full of these means the model
            # is stuck, not converging.
            session.log("fix_noop", path=str(file_path), attempt=attempts, stage=result.stage)


class SingleFileLoop:
    """Phase 1: goal -> one Python file -> verify -> bounded fix loop.

    This is the smallest slice of the full architecture: it exists to prove
    that "atomic prompt + deterministic verify + bounded retry" converges
    with a small local model before multi-file planning is layered on top.
    """

    def __init__(self, client: OllamaClient, config: Config, session: Session):
        self.client = client
        self.config = config
        self.session = session

    def run(self, goal: str, filename: str = "main.py") -> RunResult:
        self.session.log("goal", goal=goal)
        file_path = self.session.run_dir / filename

        try:
            result, attempts = _generate_and_fix(
                self.client, self.config, self.session, file_path, goal, verify.verify_python_file
            )
        except OllamaError as exc:
            self.session.log("run_aborted", reason=str(exc))
            self.session.log("run_result", success=False)
            return RunResult(
                success=False,
                file_path=str(file_path),
                attempts=0,
                last_output=str(exc),
                aborted=True,
                abort_reason=str(exc),
            )

        if not result.success:
            self.session.log("giving_up", attempts=attempts)
        self.session.log("run_result", success=result.success)

        return RunResult(
            success=result.success,
            file_path=str(file_path),
            attempts=attempts,
            last_output=result.output,
        )


class MultiFileLoop:
    """Phase 2/3: goal -> planned file list -> per-file spec + codegen + fix
    (each followed by a dedicated test-writing step) -> integration verify.

    Every stage is a fixed, narrow prompt: the planner never writes code,
    the spec writer never writes code, the code generator only ever sees
    one file's spec at a time, and the test writer is a separate call from
    implementation -- never combined, since a compound "write the code and
    its own test" instruction is exactly the kind of prompt small models
    handle unreliably. The test writer does see the finished module's
    source (read from disk, not the same call), so its tests match what
    the code actually does rather than an independent reading of the spec.
    """

    def __init__(self, client: OllamaClient, config: Config, session: Session):
        self.client = client
        self.config = config
        self.session = session

    def run(self, goal: str) -> MultiFileRunResult:
        self.session.log("goal", goal=goal)

        tasks = plan.plan_files(
            self.client, goal, temperature=self.config.temperature, max_tokens=self.config.max_tokens
        )
        self.session.log("plan", files=[dataclasses.asdict(t) for t in tasks])

        iterations = 1  # the planning call itself
        file_results: list[FileRunResult] = []
        stopped_early = False
        integration: VerifyResult | None = None
        abort_reason: str | None = None

        try:
            for task in tasks:
                if iterations >= self.config.max_total_iterations:
                    self.session.log("budget_exhausted", before=task.path, iterations=iterations)
                    stopped_early = True
                    break

                spec_text = spec.write_spec(
                    self.client,
                    goal,
                    task,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                iterations += 1
                self.session.log("spec", path=task.path, spec=spec_text)

                file_path = self.session.run_dir / _safe_relative_path(task.path)
                file_path.parent.mkdir(parents=True, exist_ok=True)

                instruction = (
                    f"Create the file `{task.path}`.\n"
                    f"Purpose: {task.purpose}\n\n"
                    f"Specification:\n{spec_text}"
                )
                result, attempts = _generate_and_fix(
                    self.client,
                    self.config,
                    self.session,
                    file_path,
                    instruction,
                    verify.verify_python_file_static,
                )
                iterations += 1 + attempts

                is_test_file = Path(task.path).name.startswith("test_")
                planner_test_advisory = is_test_file and not result.success
                if planner_test_advisory:
                    self.session.log("advisory_test", path=str(file_path), last_error=result.output)

                file_results.append(
                    FileRunResult(
                        path=str(file_path),
                        purpose=task.purpose,
                        success=result.success,
                        attempts=attempts,
                        last_output=result.output,
                        advisory=planner_test_advisory,
                    )
                )

                if (
                    result.success
                    and not is_test_file
                    and iterations < self.config.max_total_iterations
                ):
                    test_result, iterations = self._generate_test_for(task, spec_text, iterations)
                    file_results.append(test_result)

            advisory_paths = [Path(f.path) for f in file_results if f.advisory]
            # A run's success rides on its non-advisory files: the
            # implementation, and any test that actually passed.
            required = [f for f in file_results if not f.advisory]
            all_required_ok = bool(required) and all(f.success for f in required)
            if all_required_ok and not stopped_early:
                # A passing test_*.py file -- planner-provided or generated
                # by _generate_test_for -- means pytest is the right
                # integration check; otherwise fall back to an entry file.
                has_tests = any(
                    Path(f.path).name.startswith("test_") and f.success for f in file_results
                )
                generated_paths = [Path(f.path) for f in required]
                integration, iterations = self._run_integration_with_fixes(
                    tasks, generated_paths, has_tests, iterations, ignore_paths=advisory_paths
                )
        except OllamaError as exc:
            # The model went unreachable partway through. Keep every file
            # that finished; report the run as aborted, not failed.
            abort_reason = str(exc)

        required = [f for f in file_results if not f.advisory]
        all_required_ok = bool(required) and all(f.success for f in required)
        overall_success = (
            abort_reason is None
            and all_required_ok
            and not stopped_early
            and (integration is None or integration.success)
        )

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

    def _generate_test_for(
        self, task: FileTask, spec_text: str, iterations: int
    ) -> tuple[FileRunResult, int]:
        module_name = Path(task.path).stem
        module_path = self.session.run_dir / _safe_relative_path(task.path)
        module_source = module_path.read_text()
        test_file_path = self.session.run_dir / _safe_relative_path(f"test_{module_name}.py")

        # The test writer sees the module's *actual* generated source, not
        # just the spec: a spec bullet like "handle errors gracefully" is
        # ambiguous, and a test written from the spec alone routinely
        # asserts a contract the implementation chose not to honour --
        # a mismatch the test's own fix loop (which never sees the module)
        # then can't resolve.
        module_block = f"```python\n{module_source}\n```"
        instruction = (
            f"Write tests for the module `{module_name}` (file `{task.path}`).\n"
            f"Purpose: {task.purpose}\n\n"
            f"Specification of what it does:\n{spec_text}\n\n"
            f"Actual current source of `{task.path}` -- test the behavior this "
            f"code really has, not what the spec implies it might:\n{module_block}\n\n"
            f"Import it with `from {module_name} import ...` -- "
            f"the module file is in the same directory as the test."
        )
        fix_context = (
            f"The module under test, `{task.path}`, has exactly this source. "
            f"The test must match the behavior this code actually has; if the "
            f"test expects something the module does not do, change the test:\n"
            f"{module_block}"
        )
        result, attempts = _generate_and_fix(
            self.client,
            self.config,
            self.session,
            test_file_path,
            instruction,
            verify.verify_test_file,
            generate_fn=testgen.generate_test_file,
            fix_context=fix_context,
        )
        iterations += 1 + attempts

        if not result.success:
            self.session.log(
                "advisory_test", path=str(test_file_path), last_error=result.output
            )

        return (
            FileRunResult(
                path=str(test_file_path),
                purpose=f"tests for {task.path}",
                success=result.success,
                attempts=attempts,
                last_output=result.output,
                advisory=not result.success,
            ),
            iterations,
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
