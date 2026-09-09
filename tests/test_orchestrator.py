from pathlib import Path

from harness.config import Config
from harness.orchestrator import SingleFileLoop
from harness.session import Session


class FakeResponse:
    def __init__(self, text: str):
        self.text = text
        self.raw = {}


class FakeClient:
    """Stands in for OllamaClient: returns queued responses in order.

    No real network/model calls -- keeps these tests fast and deterministic
    while exercising the exact same orchestrator/codegen code paths a real
    Ollama-backed run would use.
    """

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[str] = []

    def generate(self, prompt: str, *, system=None, json_schema=None, temperature=0.2, max_tokens=2048):
        self.calls.append(prompt)
        if not self._responses:
            raise AssertionError("FakeClient ran out of queued responses")
        return FakeResponse(self._responses.pop(0))


def make_config(tmp_path: Path, max_fix_attempts: int = 3) -> Config:
    return Config(
        ollama_host="http://unused",
        model="fake-model",
        timeout_seconds=1,
        temperature=0.2,
        max_tokens=512,
        max_fix_attempts=max_fix_attempts,
        max_total_iterations=25,
        workspace_root=tmp_path / "workspace",
    )


def test_succeeds_on_first_generation(tmp_path: Path):
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    client = FakeClient(["print('hello world')\n"])

    result = SingleFileLoop(client, config, session).run("print hello world")

    assert result.success
    assert result.attempts == 0
    assert Path(result.file_path).read_text() == "print('hello world')"


def test_recovers_after_one_fix(tmp_path: Path):
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    client = FakeClient(
        [
            "print('unterminated\n",  # syntax error
            "print('hello world')\n",  # fixed
        ]
    )

    result = SingleFileLoop(client, config, session).run("print hello world")

    assert result.success
    assert result.attempts == 1
    assert len(client.calls) == 2


def test_gives_up_after_max_fix_attempts(tmp_path: Path):
    config = make_config(tmp_path, max_fix_attempts=2)
    session = Session.create(config.workspace_root)
    broken = "def broken(:\n"
    client = FakeClient([broken, broken, broken])  # 1 initial + 2 fixes, all still broken

    result = SingleFileLoop(client, config, session).run("do something impossible")

    assert not result.success
    assert result.attempts == 2
    assert len(client.calls) == 3


def test_strips_markdown_fence_from_model_output(tmp_path: Path):
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    client = FakeClient(["```python\nprint('hello world')\n```"])

    result = SingleFileLoop(client, config, session).run("print hello world")

    assert result.success
    assert Path(result.file_path).read_text() == "print('hello world')"
