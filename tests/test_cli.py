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
    assert "ERROR" in out
    assert "could not connect" in out


def test_main_reports_clean_error_on_bad_planner_json(tmp_path, monkeypatch, capsys):
    _patch_config(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "OllamaClient", lambda *a, **k: FakeClient(["not json"]))

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

    exit_code = cli.main(["inspect", "--run-id", "myrun"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Run myrun" in out
    assert "[ok] main.py" in out


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
