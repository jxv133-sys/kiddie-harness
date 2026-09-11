import json
from pathlib import Path

import pytest

from harness import gui, summary

from .fakes import FakeClient, make_config


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_available_models_returns_sorted_names(monkeypatch):
    payload = {"models": [{"name": "qwen2.5-coder:7b"}, {"name": "llama3.2"}, {"name": "abc"}]}
    monkeypatch.setattr(gui.requests, "get", lambda *a, **k: _FakeResp(payload))

    assert gui.available_models("http://h") == ["abc", "llama3.2", "qwen2.5-coder:7b"]


def test_available_models_is_empty_when_the_host_is_unreachable(monkeypatch):
    def boom(*a, **k):
        raise OSError("no route to host")

    monkeypatch.setattr(gui.requests, "get", boom)

    assert gui.available_models("http://nope") == []


def test_run_manager_runs_a_multi_file_job_to_completion(tmp_path: Path):
    config = make_config(tmp_path)
    client = FakeClient(
        [
            json.dumps({"files": [{"path": "main.py", "purpose": "entry"}]}),
            "- print hi",
            "print('hi')\n",
        ]
    )
    manager = gui.RunManager(config, client_factory=lambda host, model, timeout: client)

    run_id = manager.start(goal="say hi", model="m", host="http://h", multi_file=True)
    manager.wait(timeout=10)

    status = manager.status()
    assert status["state"] == "done"
    assert status["run_id"] == run_id
    log_path = config.workspace_root / run_id / "log.jsonl"
    assert summary.load_run_summary(log_path).succeeded


def test_run_manager_builds_one_client_per_endpoint(tmp_path: Path):
    config = make_config(tmp_path)
    made: list[tuple[str, str]] = []

    def factory(host, model, timeout):
        made.append((host, model))
        return FakeClient(
            [
                json.dumps(
                    {
                        "files": [
                            {"path": "a.py", "purpose": "leaf", "depends_on": []},
                            {"path": "b.py", "purpose": "leaf", "depends_on": []},
                        ]
                    }
                ),
                *(["- s", "def f():\n    return 1\n"] * 2),
            ]
        )

    manager = gui.RunManager(config, client_factory=factory)
    manager.start(
        goal="x",
        model="m",
        host="h",
        multi_file=True,
        endpoints=[{"host": "http://a", "model": "m1"}, {"host": "http://b", "model": "m2"}],
    )
    manager.wait(timeout=10)

    assert ("http://a", "m1") in made
    assert ("http://b", "m2") in made
    assert manager.status()["state"] == "done"


def test_run_manager_rejects_a_second_run_while_one_is_active(tmp_path: Path):
    config = make_config(tmp_path)
    # a client that blocks forever on the first call keeps the run "active"
    import threading

    gate = threading.Event()

    class _Blocking:
        def generate(self, *a, **k):
            gate.wait()
            raise AssertionError("unblocked")

    manager = gui.RunManager(config, client_factory=lambda *a, **k: _Blocking())
    manager.start(goal="x", model="m", host="h", multi_file=True)

    with pytest.raises(RuntimeError):
        manager.start(goal="y", model="m", host="h", multi_file=True)

    gate.set()
    manager.wait(timeout=10)


def test_run_manager_records_a_crash_as_a_failed_run(tmp_path: Path):
    config = make_config(tmp_path)

    class _Broken:
        def generate(self, *a, **k):
            raise RuntimeError("kaboom")

    manager = gui.RunManager(config, client_factory=lambda *a, **k: _Broken())
    run_id = manager.start(goal="x", model="m", host="h", multi_file=True)
    manager.wait(timeout=10)

    assert manager.status()["state"] == "done"
    log_path = config.workspace_root / run_id / "log.jsonl"
    result = summary.load_run_summary(log_path)
    assert not result.succeeded
    assert result.finished


def test_stream_events_formats_log_lines_and_ends_on_run_result(tmp_path: Path):
    log_path = tmp_path / "log.jsonl"
    records = [
        {"event": "goal", "goal": "x"},
        {"event": "plan", "files": [{"path": "a.py", "purpose": "p"}]},
        {"event": "codegen", "path": "a.py", "code": "x", "truncated": False},
        {"event": "run_result", "success": True},
    ]
    log_path.write_text("".join(json.dumps(r) + "\n" for r in records))

    chunks = list(gui.stream_events(log_path, poll_interval=0, idle_timeout=0.1))
    text = "".join(chunks)

    assert "[plan] 1 file(s) planned" in text
    assert "[codegen] a.py" in text
    assert chunks[-1].strip() == "event: done\ndata: {}"
    assert "id: 3\n" in text  # each data frame carries the line count


def test_stream_events_resumes_from_a_start_offset(tmp_path: Path):
    log_path = tmp_path / "log.jsonl"
    records = [
        {"event": "plan", "files": []},
        {"event": "codegen", "path": "a.py", "code": "x", "truncated": False},
        {"event": "run_result", "success": True},
    ]
    log_path.write_text("".join(json.dumps(r) + "\n" for r in records))

    text = "".join(gui.stream_events(log_path, start=2, poll_interval=0, idle_timeout=0.1))

    assert "[plan]" not in text  # line 0 skipped
    assert "run_result" in text  # line 2 still delivered


def test_stream_events_keeps_polling_while_the_run_is_active(tmp_path: Path):
    log_path = tmp_path / "log.jsonl"
    log_path.write_text(json.dumps({"event": "plan", "files": []}) + "\n")
    active = iter([True, True, False])  # run stops after two idle polls

    chunks = list(
        gui.stream_events(
            log_path, is_active=lambda: next(active, False), poll_interval=0, idle_timeout=999
        )
    )

    # never bailed on idle_timeout; ended only once is_active went False
    assert chunks[-1].strip() == "event: done\ndata: {}"


def test_request_tracker_reports_and_clears_an_active_request():
    tracker = gui.RequestTracker()

    tracker.start("r1", "GET", "/api/models?host=h&_r=r1")
    active = tracker.active()
    assert len(active) == 1
    assert active[0]["id"] == "r1"
    assert active[0]["method"] == "GET"
    assert active[0]["elapsed"] >= 0

    tracker.finish("r1")
    assert tracker.active() == []


def test_request_tracker_ignores_a_blank_id():
    tracker = gui.RequestTracker()

    tracker.start("", "GET", "/api/config")
    tracker.finish("")

    assert tracker.active() == []


def test_requests_endpoint_reports_a_request_while_it_is_in_flight(tmp_path: Path, monkeypatch):
    import threading
    import urllib.request

    config = make_config(tmp_path)
    server = gui.build_server(config, host="127.0.0.1", port=0)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    gate = threading.Event()
    release = threading.Event()

    def slow_get(*a, **k):
        gate.set()
        release.wait(timeout=5)
        raise OSError("simulated: never reaches a real host")

    monkeypatch.setattr(gui.requests, "get", slow_get)

    try:
        caller = threading.Thread(
            target=lambda: urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/models?host=h&_r=track-me", timeout=5
            ).read()
        )
        caller.start()
        assert gate.wait(timeout=5)

        active = json.loads(
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/requests", timeout=5).read()
        )["active"]
        assert any(r["id"] == "track-me" for r in active)

        release.set()
        caller.join(timeout=5)

        active_after = json.loads(
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/requests", timeout=5).read()
        )["active"]
        assert not any(r["id"] == "track-me" for r in active_after)
    finally:
        server.shutdown()
        t.join(timeout=5)


def test_the_index_page_and_config_endpoint_serve(tmp_path: Path):
    import urllib.request

    config = make_config(tmp_path)
    server = gui.build_server(config, host="127.0.0.1", port=0)
    port = server.server_address[1]
    import threading

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read().decode()
        assert "<title" in page.lower()
        assert "kiddie-harness" in page.lower()

        cfg = json.loads(
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/config", timeout=5).read()
        )
        assert cfg["host"] == config.ollama_host
        assert cfg["model"] == config.model
    finally:
        server.shutdown()
        t.join(timeout=5)
