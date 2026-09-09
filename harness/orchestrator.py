"""The deterministic state machine driving the harness.

The LLM never decides what happens next. This module owns all sequencing,
retries, and stop conditions; the LLM is only ever called through the
narrow, single-purpose functions in harness.steps.
"""

from __future__ import annotations

import dataclasses

from .config import Config
from .llm_client import OllamaClient
from .session import Session
from .steps import codegen, verify


@dataclasses.dataclass
class RunResult:
    success: bool
    file_path: str
    attempts: int
    last_output: str


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

        code = codegen.generate_file(
            self.client,
            goal,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        self.session.log("codegen", code=code)

        attempts = 0
        result = None
        while attempts <= self.config.max_fix_attempts:
            file_path.write_text(code)
            result = verify.verify_python_file(file_path)
            self.session.log(
                "verify",
                attempt=attempts,
                stage=result.stage,
                success=result.success,
                output=result.output,
            )

            if result.success:
                return RunResult(
                    success=True,
                    file_path=str(file_path),
                    attempts=attempts,
                    last_output=result.output,
                )

            if attempts == self.config.max_fix_attempts:
                break

            code = codegen.fix_file(
                self.client,
                code=code,
                error=result.output,
                stage=result.stage,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            self.session.log("fix", attempt=attempts + 1, code=code)
            attempts += 1

        self.session.log("giving_up", attempts=attempts)
        return RunResult(
            success=False,
            file_path=str(file_path),
            attempts=attempts,
            last_output=result.output if result else "",
        )
