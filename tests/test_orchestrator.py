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
    # Each attempt is broken but distinct, so the no-op short-circuit does
    # not fire and the loop runs the full fix budget.
    client = FakeClient(["def broken(:\n", "def broke(:\n", "def brok(:\n"])

    result = SingleFileLoop(client, config, session).run("do something impossible")

    assert not result.success
    assert result.attempts == 2
    assert len(client.calls) == 3


def test_retries_run_at_a_rising_temperature_when_a_fix_repeats_itself(tmp_path: Path):
    config = make_config(tmp_path, max_fix_attempts=3)
    session = Session.create(config.workspace_root)
    broken = "def broken(:\n"
    client = FakeClient([broken] * 5)

    result = SingleFileLoop(client, config, session).run("do something impossible")

    assert not result.success
    # An identical repeat no longer aborts the loop: it uses the whole
    # fix budget, raising the sampling temperature each attempt so the
    # model has a real chance to produce something different.
    assert result.attempts == 3
    assert len(client.calls) == 4
    assert client.temperature_calls[0] == config.temperature
    assert client.temperature_calls == sorted(client.temperature_calls)
    assert client.temperature_calls[-1] > client.temperature_calls[0]


def test_an_empty_generation_is_retried_not_accepted(tmp_path: Path):
    config = make_config(tmp_path, max_fix_attempts=2)
    session = Session.create(config.workspace_root)
    client = FakeClient(["   \n  ", "print('hello world')\n"])

    result = SingleFileLoop(client, config, session).run("print hello world")

    assert result.success
    assert result.attempts == 1
    assert Path(result.file_path).read_text() == "print('hello world')"


def test_a_run_that_only_ever_returns_blank_fails(tmp_path: Path):
    config = make_config(tmp_path, max_fix_attempts=1)
    session = Session.create(config.workspace_root)
    client = FakeClient(["", "  "])

    result = SingleFileLoop(client, config, session).run("do a thing")

    assert not result.success
    assert "empty" in result.last_output.lower()


def test_strips_markdown_fence_from_model_output(tmp_path: Path):
    config = make_config(tmp_path)
    session = Session.create(config.workspace_root)
    client = FakeClient(["```python\nprint('hello world')\n```"])

    result = SingleFileLoop(client, config, session).run("print hello world")

    assert result.success
    assert Path(result.file_path).read_text() == "print('hello world')"


def test_grows_max_tokens_and_notes_truncation_after_a_cut_off_response(tmp_path: Path):
    config = make_config(tmp_path, max_tokens=512, max_tokens_ceiling=4096)
    session = Session.create(config.workspace_root)
    client = FakeClient(
        [
            ("print('unterminated", "length"),  # cut off mid-string, syntax error
            "print('hello world')\n",  # fixed, complete
        ]
    )

    result = SingleFileLoop(client, config, session).run("print hello world")

    assert result.success
    assert result.attempts == 1
    assert client.max_tokens_calls == [512, 1024]
    assert "cut off" in client.calls[1]


def test_max_tokens_growth_is_capped_by_ceiling(tmp_path: Path):
    config = make_config(tmp_path, max_tokens=3000, max_tokens_ceiling=4000, max_fix_attempts=2)
    session = Session.create(config.workspace_root)
    client = FakeClient(
        [
            ("bad(", "length"),
            ("also bad(", "length"),
            "print('ok')\n",
        ]
    )

    result = SingleFileLoop(client, config, session).run("goal")

    assert result.success
    assert result.attempts == 2
    assert client.max_tokens_calls == [3000, 4000, 4000]
