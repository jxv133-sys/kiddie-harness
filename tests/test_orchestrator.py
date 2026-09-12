import json
import threading
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


def test_a_cancelled_run_stops_before_the_next_fix_call(tmp_path: Path):
    config = make_config(tmp_path, max_fix_attempts=5)
    session = Session.create(config.workspace_root)
    # Never actually reached: cancellation fires right after the first
    # verify fails, before a second call would be made.
    client = FakeClient(["def broken(:\n", "print('should not be requested')\n"])
    cancel_event = threading.Event()
    cancel_event.set()

    result = SingleFileLoop(client, config, session, cancel_event=cancel_event).run("goal")

    assert not result.success
    assert result.aborted
    assert result.abort_reason == "cancelled by user"
    assert len(client.calls) == 1  # the fix call never went out


def test_critic_disabled_by_default_never_calls_the_model_a_third_time(tmp_path: Path):
    config = make_config(tmp_path)  # critic_enabled=False
    session = Session.create(config.workspace_root)
    client = FakeClient(["print('hi')\n"])

    result = SingleFileLoop(client, config, session).run("print hi")

    assert result.success
    assert not result.spec_flagged
    assert len(client.calls) == 1  # codegen only -- no critic call queued or made


def test_critic_lets_a_passing_file_through_untouched(tmp_path: Path):
    config = make_config(tmp_path, critic_enabled=True)
    session = Session.create(config.workspace_root)
    client = FakeClient(["print('hi')\n", '{"follows_spec": true, "issues": ""}'])

    result = SingleFileLoop(client, config, session).run("print hi")

    assert result.success
    assert not result.spec_flagged
    assert len(client.calls) == 2  # codegen + one critic check
    # Proof the critic call actually happened -- an agreeing critic
    # returns the original verify result unchanged, so without its own
    # log event there would be no other evidence of it in the log.
    events = [json.loads(line) for line in session.log_path.read_text().splitlines()]
    checks = [e for e in events if e["event"] == "critic_check"]
    assert len(checks) == 1
    assert checks[0]["follows_spec"] is True


def test_a_file_the_critic_disagrees_with_is_flagged_not_failed(tmp_path: Path):
    config = make_config(tmp_path, critic_enabled=True, max_fix_attempts=2)
    session = Session.create(config.workspace_root)
    # Compiles and runs fine every time -- the critic is the only thing
    # that never signs off, across every attempt.
    disagree = '{"follows_spec": false, "issues": "never greets the user by name"}'
    client = FakeClient(
        [
            "print('hi')\n", disagree,  # initial generation + critic
            "print('hi')\n", disagree,  # fix attempt 1 + critic
            "print('hi')\n", disagree,  # fix attempt 2 + critic
        ]
    )

    result = SingleFileLoop(client, config, session).run("greet the user by name")

    # The code genuinely runs -- a critic opinion, which can be wrong,
    # must not be able to fail a file that passes every real check.
    assert result.success
    assert result.spec_flagged
    assert "never greets" in result.last_output
    events = [json.loads(line) for line in session.log_path.read_text().splitlines()]
    assert any(e["event"] == "spec_flagged" for e in events)
    assert events[-1]["event"] == "run_result" and events[-1]["success"] is True
