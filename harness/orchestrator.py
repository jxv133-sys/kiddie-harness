"""The deterministic state machine driving the harness.

The LLM never decides what happens next. This module owns all sequencing,
retries, and stop conditions; the LLM is only ever called through the
narrow, single-purpose functions in harness.steps.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

from .config import Config
from .llm_client import OllamaClient
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


@dataclasses.dataclass
class FileRunResult:
    path: str
    purpose: str
    success: bool
    attempts: int
    last_output: str


@dataclasses.dataclass
class MultiFileRunResult:
    success: bool
    run_dir: str
    files: list[FileRunResult]
    integration: VerifyResult | None
    total_iterations: int
    stopped_early: bool


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
        result = verify_fn(file_path)
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
            temperature=config.temperature,
            max_tokens=max_tokens,
            context=fix_context,
        )
        code = gen.code
        attempts += 1
        session.log("fix", path=str(file_path), attempt=attempts, code=code, truncated=gen.truncated)
        if gen.truncated:
            max_tokens = min(max_tokens * 2, config.max_tokens_ceiling)

        if code == failed_code:
            # The fix call returned the exact bytes we just verified as
            # failing. Re-verifying them would burn the rest of the fix
            # budget on a guaranteed-identical failure -- weak models at a
            # low temperature routinely echo their last output verbatim.
            # Stop now with the failure we already have.
            session.log("fix_noop", path=str(file_path), attempt=attempts, stage=result.stage)
            return result, attempts


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

        result, attempts = _generate_and_fix(
            self.client, self.config, self.session, file_path, goal, verify.verify_python_file
        )

        if not result.success:
            self.session.log("giving_up", attempts=attempts)

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

            file_results.append(
                FileRunResult(
                    path=str(file_path),
                    purpose=task.purpose,
                    success=result.success,
                    attempts=attempts,
                    last_output=result.output,
                )
            )

            is_test_file = Path(task.path).name.startswith("test_")
            if result.success and not is_test_file and iterations < self.config.max_total_iterations:
                test_result, iterations = self._generate_test_for(task, spec_text, iterations)
                file_results.append(test_result)

        integration = None
        all_files_ok = bool(file_results) and all(f.success for f in file_results)
        if all_files_ok and not stopped_early:
            # Any successful test_*.py file -- planner-provided or
            # generated by _generate_test_for -- means pytest is the right
            # integration check; otherwise fall back to running an entry file.
            has_tests = any(
                Path(f.path).name.startswith("test_") and f.success for f in file_results
            )
            generated_paths = [Path(f.path) for f in file_results]
            integration, iterations = self._run_integration_with_fixes(
                tasks, generated_paths, has_tests, iterations
            )

        overall_success = (
            all_files_ok and not stopped_early and (integration is None or integration.success)
        )

        if not overall_success:
            self.session.log("giving_up", iterations=iterations, stopped_early=stopped_early)

        return MultiFileRunResult(
            success=overall_success,
            run_dir=str(self.session.run_dir),
            files=file_results,
            integration=integration,
            total_iterations=iterations,
            stopped_early=stopped_early,
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

        return (
            FileRunResult(
                path=str(test_file_path),
                purpose=f"tests for {task.path}",
                success=result.success,
                attempts=attempts,
                last_output=result.output,
            ),
            iterations,
        )

    def _run_integration_with_fixes(
        self,
        tasks: list[FileTask],
        generated_paths: list[Path],
        has_tests: bool,
        iterations: int,
    ) -> tuple[VerifyResult | None, int]:
        result = self._run_integration(tasks, has_tests)
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
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            target.write_text(gen.code)
            iterations += 1
            rounds += 1
            self.session.log("integration_fix", path=str(target), round=rounds, code=gen.code)
            result = self._run_integration(tasks, has_tests)

        return result, iterations

    def _run_integration(self, tasks: list[FileTask], has_tests: bool) -> VerifyResult | None:
        if has_tests:
            result = verify.run_pytest(self.session.run_dir)
        else:
            entry = self._pick_entry_path(tasks)
            if entry is None:
                return None
            result = verify.run_script(self.session.run_dir / entry)

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
        text. If none matches, there's nothing safe to scope a fix to."""
        for path in candidate_paths:
            if path.name in error_text:
                return path
        return None
