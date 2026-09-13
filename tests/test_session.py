import threading
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


def test_active_calls_is_empty_when_nothing_is_in_flight(tmp_path: Path):
    session = Session.create(tmp_path)

    assert session.active_calls() == []


def test_track_call_reports_and_clears_an_in_flight_call(tmp_path: Path):
    session = Session.create(tmp_path)

    with session.track_call("spec", "core.py", "http://h:11434"):
        active = session.active_calls()
        assert len(active) == 1
        assert active[0]["kind"] == "spec"
        assert active[0]["path"] == "core.py"
        assert active[0]["endpoint"] == "http://h:11434"
        assert active[0]["elapsed"] >= 0
        assert active[0]["partial"] == ""  # nothing streamed in yet

    assert session.active_calls() == []


def test_track_call_reports_streamed_partial_text_as_it_updates(tmp_path: Path):
    session = Session.create(tmp_path)

    with session.track_call("codegen", "a.py", "http://h") as update:
        assert session.active_calls()[0]["partial"] == ""
        update("def f")
        assert session.active_calls()[0]["partial"] == "def f"
        update("def f(x):\n    return x\n")
        assert session.active_calls()[0]["partial"] == "def f(x):\n    return x\n"

    assert session.active_calls() == []


def test_active_calls_each_have_a_distinct_stable_id(tmp_path: Path):
    session = Session.create(tmp_path)

    with (
        session.track_call("codegen", "a.py", "http://x") as update_a,
        session.track_call("codegen", "b.py", "http://y") as update_b,
    ):
        update_a("partial a")
        update_b("partial b")
        active = {c["path"]: c for c in session.active_calls()}
        assert active["a.py"]["partial"] == "partial a"
        assert active["b.py"]["partial"] == "partial b"
        assert active["a.py"]["id"] != active["b.py"]["id"]


def test_track_call_clears_on_an_exception_inside_the_block(tmp_path: Path):
    session = Session.create(tmp_path)

    try:
        with session.track_call("codegen", "a.py", "http://h"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert session.active_calls() == []


def test_two_concurrent_calls_both_show_up_independently(tmp_path: Path):
    session = Session.create(tmp_path)
    entered1, entered2 = threading.Event(), threading.Event()
    release = threading.Event()

    def worker(kind, path, endpoint, entered):
        with session.track_call(kind, path, endpoint):
            entered.set()
            release.wait(timeout=5)

    t1 = threading.Thread(target=worker, args=("codegen", "a.py", "http://x", entered1))
    t2 = threading.Thread(target=worker, args=("codegen", "b.py", "http://y", entered2))
    t1.start()
    t2.start()
    assert entered1.wait(timeout=5)
    assert entered2.wait(timeout=5)

    active = session.active_calls()
    assert {c["path"] for c in active} == {"a.py", "b.py"}
    assert {c["endpoint"] for c in active} == {"http://x", "http://y"}

    release.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert session.active_calls() == []
