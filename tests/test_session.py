from pathlib import Path

from harness.session import Session


def test_log_invokes_on_event_callback_with_event_and_fields(tmp_path: Path):
    calls = []
    session = Session.create(tmp_path, on_event=lambda e, f: calls.append((e, f)))

    session.log("verify", path="x.py", attempt=0, stage="run", success=True, output="")

    assert calls == [
        ("verify", {"path": "x.py", "attempt": 0, "stage": "run", "success": True, "output": ""})
    ]


def test_log_without_on_event_does_not_raise(tmp_path: Path):
    session = Session.create(tmp_path)

    session.log("goal", goal="do a thing")

    assert len(session.log_path.read_text().splitlines()) == 1


def test_log_writes_to_file_before_invoking_callback(tmp_path: Path):
    seen_lines = []

    def on_event(event, fields):
        seen_lines.append(len(session.log_path.read_text().splitlines()))

    session = Session.create(tmp_path, on_event=on_event)
    session.log("goal", goal="do a thing")

    assert seen_lines == [1]


def test_log_swallows_exceptions_from_on_event(tmp_path: Path):
    def on_event(event, fields):
        raise RuntimeError("boom")

    session = Session.create(tmp_path, on_event=on_event)

    session.log("goal", goal="do a thing")

    assert len(session.log_path.read_text().splitlines()) == 1
