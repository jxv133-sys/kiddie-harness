from pathlib import Path

from harness.orchestrator import SingleFileLoop
from harness.session import Session

from .fakes import FakeClient, make_config


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
