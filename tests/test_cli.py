from pathlib import Path

from harness import cli
from harness.config import Config
from harness.llm_client import OllamaError
from harness.session import Session

from .fakes import FakeClient, make_config


class RaisingClient:
    """A client whose .generate() always raises, simulating an
    unreachable Ollama host without touching real network code."""

    def __init__(self, exc: Exception):
        self._exc = exc

    def generate(self, *args, **kwargs):
        raise self._exc


def _patch_config(monkeypatch, tmp_path: Path, **overrides):
    config = make_config(tmp_path, **overrides)
    monkeypatch.setattr(Config, "load", classmethod(lambda cls, path=None: config))
    return config


def test_main_reports_clean_error_when_ollama_unreachable(tmp_path, monkeypatch, capsys):
    _patch_config(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "OllamaClient", lambda *a, **k: RaisingClient(OllamaError("could not connect")))

    exit_code = cli.main(["run", "--goal", "anything"])

    assert exit_code == 2
    out = capsys.readouterr().out
    assert "ABORTED" in out
    assert "could not connect" in out


def test_main_reports_clean_error_on_bad_planner_json(tmp_path, monkeypatch, capsys):
    _patch_config(monkeypatch, tmp_path)
    # planner never produces usable JSON, across all its retries
    monkeypatch.setattr(cli, "OllamaClient", lambda *a, **k: FakeClient(["not json"] * 5))

    exit_code = cli.main(["run", "--multi-file", "--goal", "anything"])

    assert exit_code == 2
    out = capsys.readouterr().out
    assert "ERROR" in out


def test_main_success_path_prints_summary_table_and_returns_zero(tmp_path, monkeypatch, capsys):
    _patch_config(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "OllamaClient", lambda *a, **k: FakeClient(["print('hi')\n"]))

    exit_code = cli.main(["run", "--goal", "print hi"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Result: SUCCESS" in out


def test_inspect_reports_summary_for_existing_run(tmp_path, monkeypatch, capsys):
    config = _patch_config(monkeypatch, tmp_path)
    session = Session.create(config.workspace_root, run_id="myrun")
    session.log("codegen", path="main.py", code="print(1)", truncated=False)
    session.log("verify", path="main.py", attempt=0, stage="run", success=True, output="")
    session.log("run_result", success=True)

    exit_code = cli.main(["inspect", "--run-id", "myrun"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Run myrun" in out
    assert "[ok] main.py" in out


def test_inspect_reports_incomplete_for_a_killed_run(tmp_path, monkeypatch, capsys):
    config = _patch_config(monkeypatch, tmp_path)
    session = Session.create(config.workspace_root, run_id="killed")
    session.log("codegen", path="main.py", code="x = 1", truncated=False)
    session.log("verify", path="main.py", attempt=0, stage="compile", success=True, output="")
    # no run_result / giving_up -- process was killed

    exit_code = cli.main(["inspect", "--run-id", "killed"])

    assert exit_code == 1
    assert "INCOMPLETE" in capsys.readouterr().out


def test_inspect_reports_error_for_missing_run(tmp_path, monkeypatch, capsys):
    _patch_config(monkeypatch, tmp_path)

    exit_code = cli.main(["inspect", "--run-id", "nope"])

    assert exit_code == 2
    assert "No such run" in capsys.readouterr().out


def test_main_run_prints_live_progress_by_default(tmp_path, monkeypatch, capsys):
    _patch_config(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "OllamaClient", lambda *a, **k: FakeClient(["print('hi')\n"]))

    exit_code = cli.main(["run", "--goal", "print hi"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "[codegen]" in out
    assert "[verify:" in out
    assert out.index("[codegen]") < out.index("Result:")


def test_main_run_quiet_suppresses_live_progress(tmp_path, monkeypatch, capsys):
    _patch_config(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "OllamaClient", lambda *a, **k: FakeClient(["print('hi')\n"]))

    exit_code = cli.main(["run", "--quiet", "--goal", "print hi"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "[codegen]" not in out
    assert "[verify:" not in out
    assert "Result: SUCCESS" in out


def test_main_multi_file_prints_the_failing_file_error(tmp_path, monkeypatch, capsys):
    _patch_config(monkeypatch, tmp_path, max_fix_attempts=1)
    monkeypatch.setattr(
        cli,
        "OllamaClient",
        lambda *a, **k: FakeClient(
            [
                '{"files": [{"path": "bad.py", "purpose": "x"}]}',
                "- do a thing",
                "def broken(:\n",
                "def broken(:\n",
            ]
        ),
    )

    exit_code = cli.main(["run", "--multi-file", "--goal", "x"])

    assert exit_code == 1
    out = capsys.readouterr().out
    assert "bad.py" in out
    assert "SyntaxError" in out


def test_main_multi_file_notes_advisory_tests_on_an_otherwise_successful_run(
    tmp_path, monkeypatch, capsys
):
    _patch_config(monkeypatch, tmp_path, max_fix_attempts=1)
    bad_test = 'from greeter import greet\n\ndef test_greet():\n    assert greet() == "bye"\n'
    monkeypatch.setattr(
        cli,
        "OllamaClient",
        lambda *a, **k: FakeClient(
            [
                (
                    '{"files": [{"path": "greeter.py", "purpose": "greet"},'
                    ' {"path": "test_greeter.py", "purpose": "tests"}]}'
                ),
                "- expose greet",
                'def greet():\n    return "hi"\n',
                "- test greet",
                bad_test,
                bad_test,
            ]
        ),
    )

    exit_code = cli.main(["run", "--multi-file", "--goal", "x"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Result: SUCCESS" in captured.out
    assert "advisory" in (captured.out + captured.err).lower()


def test_max_tokens_flags_override_the_configured_generation_length(tmp_path, monkeypatch, capsys):
    config = _patch_config(monkeypatch, tmp_path)
    client = FakeClient(["print('hi')\n"])
    monkeypatch.setattr(cli, "OllamaClient", lambda *a, **k: client)

    exit_code = cli.main(
        ["run", "--goal", "print hi", "--max-tokens", "4096", "--max-tokens-ceiling", "16384"]
    )

    assert exit_code == 0
    assert client.max_tokens_calls == [4096]
    assert config.max_tokens != 4096  # the yaml default is untouched


def test_no_critic_flag_skips_the_critic_call(tmp_path, monkeypatch, capsys):
    _patch_config(monkeypatch, tmp_path, critic_enabled=True)
    client = FakeClient(["print('hi')\n"])
    monkeypatch.setattr(cli, "OllamaClient", lambda *a, **k: client)

    exit_code = cli.main(["run", "--goal", "print hi", "--no-critic"])

    assert exit_code == 0
    assert len(client.calls) == 1  # codegen only -- no critic call


def test_critic_runs_by_default_when_the_config_enables_it(tmp_path, monkeypatch, capsys):
    _patch_config(monkeypatch, tmp_path, critic_enabled=True)
    client = FakeClient(["print('hi')\n", '{"follows_spec": true, "issues": ""}'])
    monkeypatch.setattr(cli, "OllamaClient", lambda *a, **k: client)

    exit_code = cli.main(["run", "--goal", "print hi"])

    assert exit_code == 0
    assert len(client.calls) == 2  # codegen + critic


def test_endpoint_with_no_model_defaults_to_the_run_overridden_model(tmp_path, monkeypatch, capsys):
    # --endpoint host (no ",model") should default to what --model just
    # asked for on this run, not silently fall back to the raw yaml
    # default ("fake-model", from make_config) underneath it.
    _patch_config(monkeypatch, tmp_path)
    made: list[tuple[str, str, int]] = []

    def factory(host, model, timeout):
        made.append((host, model, timeout))
        return FakeClient(["print('hi')\n"])

    monkeypatch.setattr(cli, "OllamaClient", factory)

    exit_code = cli.main(
        ["run", "--goal", "print hi", "--model", "custom-model", "--endpoint", "http://a:11434"]
    )

    assert exit_code == 0
    assert made == [("http://a:11434", "custom-model", made[0][2])]


def test_main_multi_file_prints_plan_and_spec_progress(tmp_path, monkeypatch, capsys):
    _patch_config(monkeypatch, tmp_path)
    plan_json = '{"files": [{"path": "main.py", "purpose": "entry point"}]}'
    monkeypatch.setattr(
        cli,
        "OllamaClient",
        lambda *a, **k: FakeClient(
            [
                plan_json,
                "- print hello",  # spec
                "print('hello')\n",  # codegen
                "def test_placeholder():\n    assert True\n",  # test for main.py
            ]
        ),
    )

    exit_code = cli.main(["run", "--multi-file", "--goal", "print hello"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "[plan] 1 file(s) planned" in out
    assert "[spec] main.py" in out
