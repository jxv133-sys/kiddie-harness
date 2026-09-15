"""A minimal local web GUI for the harness.

`harness gui` starts a stdlib `http.server` on localhost and opens a
browser to one page: pick a model, type a goal, watch the live progress
and the final stats. It is a thin shell over the same pieces the CLI
uses -- `RunManager` drives `SingleFileLoop` / `MultiFileLoop`,
`progress.format_event` formats the live lines, `summary.load_run_summary`
builds the final table.

One run at a time. No auth (it binds to 127.0.0.1). No new dependencies.
"""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from collections.abc import Callable, Iterator
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

from . import progress, summary
from .config import DEFAULT_CONFIG_PATH, Config, Endpoint
from .llm_client import OllamaClient
from .orchestrator import MultiFileLoop, SingleFileLoop, partition_clients_by_role
from .session import Session

# Config fields the settings screen can change. Persisted alongside the
# repo's own config/default.yaml (never rewriting that file -- it keeps
# its comments) so they survive a `harness gui` restart.
_SETTINGS_KEYS = (
    "temperature",
    "max_tokens",
    "max_tokens_ceiling",
    "max_fix_attempts",
    "max_total_iterations",
    "timeout_seconds",
    "branch_after_fixes",
)
# Boolean settings, handled separately from the numeric ones above (no
# float()/int() parsing -- the value is already a real JSON boolean).
_BOOL_SETTINGS_KEYS = ("critic_enabled", "super_review_enabled")
_ALL_SETTINGS_KEYS = _SETTINGS_KEYS + _BOOL_SETTINGS_KEYS
_SETTINGS_PATH = DEFAULT_CONFIG_PATH.parent / "gui_settings.json"


def _load_settings_overrides() -> dict:
    try:
        return json.loads(_SETTINGS_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_settings_overrides(values: dict) -> None:
    try:
        _SETTINGS_PATH.write_text(json.dumps(values, indent=2) + "\n")
    except OSError:
        pass  # best-effort persistence -- the setting still applies this session


def available_models(host: str) -> list[str]:
    """Model names Ollama reports at `host`, sorted. Empty if unreachable."""
    try:
        resp = requests.get(f"{host.rstrip('/')}/api/tags", timeout=5)
        resp.raise_for_status()
        return sorted(m["name"] for m in resp.json().get("models", []))
    except Exception:  # noqa: BLE001
        return []


class RunManager:
    """Owns at most one active harness run, in a daemon thread."""

    def __init__(self, config: Config, client_factory=None):
        self._config = config
        self._client_factory = client_factory or (
            lambda host, model, timeout: OllamaClient(host, model, timeout)
        )
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._run_id: str | None = None
        self._state = "idle"  # idle | running | done
        self._cancel_event: threading.Event | None = None
        self._pause_event: threading.Event | None = None
        # The exact Config object the active run's loop holds (a
        # different instance than `self._config`, which is only "defaults
        # for the next run" -- see `start()`). Set only while a run is in
        # progress; `update_config()` mutates it in place, live, while
        # paused (see `Config.apply_overrides`).
        self._active_config: Config | None = None
        self._session: Session | None = None

    @property
    def config(self) -> Config:
        with self._lock:
            return self._config

    def active_calls(self, run_id: str) -> list[dict]:
        """In-flight LLM calls for `run_id` -- empty for any run that
        isn't the current *running* one. Requiring `state == "running"`
        (not just a matching run_id) matters after `cancel()`: it flips
        the state to "idle" right away, but the old thread's in-flight
        call is cooperative-only and can genuinely still be running in
        the background for a while. Once the GUI has moved on and shown
        a verdict for this run, it must not keep reporting that run's
        stray leftover call as if it were still part of an active run."""
        with self._lock:
            if run_id != self._run_id or self._session is None or self._state != "running":
                return []
            session = self._session
        return session.active_calls()

    def update_config(self, **overrides) -> Config:
        """Apply settings-screen overrides for every run started from now
        on. Also applied live, in place, to the *active* run's own Config
        -- but only while that run is paused; a change made while it's
        still going would silently affect whichever call happens to fire
        next, with no clear moment the user asked for that. Paused, there
        is no call in flight to be surprised by one."""
        with self._lock:
            self._config = self._config.with_overrides(**overrides)
            if (
                self._active_config is not None
                and self._pause_event is not None
                and self._pause_event.is_set()
            ):
                self._active_config.apply_overrides(**overrides)
            return self._config

    def pause(self) -> bool:
        """Ask the active run to stop claiming new work at its next
        checkpoint (see `orchestrator._wait_if_paused`) and block there --
        a call already in flight still has to finish, same as `cancel()`."""
        with self._lock:
            if self._state != "running" or self._pause_event is None:
                return False
            self._pause_event.set()
        if self._session is not None:
            self._session.log("run_paused")
        return True

    def resume(self) -> bool:
        with self._lock:
            if self._pause_event is None or not self._pause_event.is_set():
                return False
            self._pause_event.clear()
        if self._session is not None:
            self._session.log("run_resumed")
        return True

    def status(self) -> dict:
        with self._lock:
            paused = self._pause_event.is_set() if self._pause_event else False
            return {"state": self._state, "run_id": self._run_id, "paused": paused}

    def start(
        self,
        *,
        goal: str,
        model: str,
        host: str,
        multi_file: bool,
        endpoints: list[dict] | None = None,
    ) -> str:
        with self._lock:
            if self._state == "running":
                raise RuntimeError("a run is already in progress")
            config = self._config.with_overrides(model=model, host=host)
            eps = endpoints or [{"host": host, "model": model}]
            session = Session.create(config.workspace_root)
            self._run_id = session.run_id
            self._session = session
            self._state = "running"
            self._cancel_event = threading.Event()
            self._pause_event = threading.Event()
            self._active_config = config
            self._thread = threading.Thread(
                target=self._run,
                args=(config, session, goal, multi_file, eps, self._cancel_event, self._pause_event),
                daemon=True,
            )
            self._thread.start()
            return session.run_id

    def cancel(self) -> bool:
        """Ask the active run to stop, and free the GUI to start a new one
        right away. The old run's thread keeps going until it notices the
        cancellation at its next checkpoint (at most one in-flight LLM
        call's worth of delay) and writes its own `run_aborted` -- but it
        no longer holds the whole GUI hostage while that happens."""
        with self._lock:
            if self._state != "running" or self._cancel_event is None:
                return False
            self._cancel_event.set()
            if self._pause_event is not None:
                # A paused worker is asleep waiting on this event, not on
                # cancel_event directly -- clear it so `_wait_if_paused`'s
                # loop wakes on its next 0.5s poll and sees cancel is set,
                # rather than staying parked until something resumes it.
                self._pause_event.clear()
            self._state = "idle"
            return True

    def log_path(self, run_id: str) -> Path:
        return self._config.workspace_root / run_id / "log.jsonl"

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(
        self,
        config: Config,
        session: Session,
        goal: str,
        multi_file: bool,
        endpoints: list[dict],
        cancel_event: threading.Event,
        pause_event: threading.Event,
    ) -> None:
        run_id = session.run_id
        try:
            eps = [
                Endpoint(
                    e["host"],
                    e.get("model") or config.model,
                    config.timeout_seconds,
                    role=e.get("role") or "balanced",
                )
                for e in endpoints
            ]
            pool = [self._client_factory(e.host, e.model, e.timeout_seconds) for e in eps]
            if multi_file:
                primary, workers, critic_client, branch_pool = partition_clients_by_role(eps, pool)
                MultiFileLoop(
                    primary,
                    config,
                    session,
                    pool_clients=workers,
                    cancel_event=cancel_event,
                    pause_event=pause_event,
                    critic_client=critic_client,
                    branch_pool=branch_pool,
                ).run(goal)
            else:
                SingleFileLoop(
                    pool[0], config, session, cancel_event=cancel_event, pause_event=pause_event
                ).run(goal)
        except Exception as exc:  # noqa: BLE001 -- a GUI run must never crash silently
            try:
                session.log("run_aborted", reason=f"{type(exc).__name__}: {exc}")
                session.log("run_result", success=False)
            except Exception:  # noqa: BLE001, S110
                pass
        finally:
            with self._lock:
                # Only the still-current run gets to flip state back to
                # "done" -- a cancelled run's thread finishing late must
                # not clobber whatever run superseded it.
                if self._run_id == run_id:
                    self._state = "done"
                    self._active_config = None
                    if self._pause_event is not None:
                        self._pause_event.clear()


def stream_events(
    log_path: Path,
    *,
    start: int = 0,
    is_active: Callable[[], bool] | None = None,
    poll_interval: float = 0.4,
    idle_timeout: float = 1800.0,
) -> Iterator[str]:
    """Tail a run's log.jsonl and yield Server-Sent-Event chunks.

    Each `data:` frame carries an `id:` (the count of log lines consumed)
    and `{"line": <formatted or null>, "event": <raw>, "fields": {...}}`.
    On reconnect the browser sends the last id as `Last-Event-ID`; pass it
    as `start` and the stream resumes without replaying. The stream ends
    with an `event: done` frame once a `run_result` is logged. `is_active`,
    when given, says whether the run is still going: while it returns True
    the stream keeps polling no matter how long a single LLM call takes;
    `idle_timeout` is only a safety net for an orphaned stream.
    """
    seen = max(start, 0)
    last_activity = time.monotonic()
    grace = 5  # polls to keep reading after the run stops, for a late run_result
    while True:
        try:
            lines = Path(log_path).read_text().splitlines()
        except FileNotFoundError:
            lines = []

        emitted = False
        while seen < len(lines):
            raw = lines[seen].strip()
            if not raw:
                seen += 1
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                break  # a half-written final line; try again next poll
            seen += 1
            emitted = True
            event = record.get("event")
            fields = {k: v for k, v in record.items() if k not in ("event", "ts")}
            formatted = progress.format_event(event, fields) if event else None
            payload = json.dumps({"line": formatted, "event": event, "fields": fields})
            yield f"id: {seen}\ndata: {payload}\n\n"
            if event == "run_result":
                yield "event: done\ndata: {}\n\n"
                return

        if emitted:
            last_activity = time.monotonic()
            grace = 5
            continue

        if is_active is not None:
            if is_active():
                time.sleep(poll_interval)  # run still going, no matter how slow
                continue
            grace -= 1  # run stopped; a few more reads for a late run_result
            if grace <= 0:
                yield "event: done\ndata: {}\n\n"
                return
        elif time.monotonic() - last_activity > idle_timeout:
            yield "event: done\ndata: {}\n\n"
            return
        time.sleep(poll_interval)


class RequestTracker:
    """Tracks in-flight HTTP requests by the `_r` id the page puts on
    every URL, so `/api/requests` (and the page's own small readout) can
    show what's currently being served -- in particular that a run's SSE
    stream is still open, not just whether the run itself is done."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[str, dict] = {}

    def start(self, req_id: str, method: str, path: str) -> None:
        if not req_id:
            return
        with self._lock:
            self._active[req_id] = {"method": method, "path": path, "started": time.monotonic()}

    def finish(self, req_id: str) -> None:
        if not req_id:
            return
        with self._lock:
            self._active.pop(req_id, None)

    def active(self, *, exclude: str = "") -> list[dict]:
        """Snapshot of in-flight requests. `exclude` leaves out the caller's
        own id -- otherwise a GET /api/requests always includes itself,
        since it's still "in flight" until its own handler returns."""
        with self._lock:
            now = time.monotonic()
            return [
                {
                    "id": rid,
                    "method": v["method"],
                    "path": v["path"],
                    "elapsed": round(now - v["started"], 1),
                }
                for rid, v in self._active.items()
                if rid != exclude
            ]


class _Handler(BaseHTTPRequestHandler):
    server_version = "kiddie-harness-gui"

    def log_message(self, *args) -> None:  # keep the console quiet
        pass

    # --- helpers -----------------------------------------------------------
    def _send_json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, content_type: str = "text/html") -> None:
        body = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @property
    def _config(self) -> Config:
        # RunManager is the single source of truth once the settings
        # screen can change it at runtime; `server.config` is only the
        # value it was constructed with.
        return self._runs.config

    @property
    def _runs(self) -> RunManager:
        return self.server.run_manager  # type: ignore[attr-defined]

    @property
    def _tracker(self) -> RequestTracker:
        return self.server.requests  # type: ignore[attr-defined]

    def _request_id(self) -> str:
        """The `_r` the page tags every URL with, for `/api/requests` and
        the stale-response guards on the client -- empty for a request
        that didn't set one (e.g. a plain curl)."""
        return (parse_qs(urlparse(self.path).query).get("_r") or [""])[0]

    # --- routes ----------------------------------------------------------
    def do_GET(self) -> None:
        req_id = self._request_id()
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/events/"):
            # Long-lived: stays "active" for the whole SSE stream, not
            # just this dispatch, so _serve_events finishes it itself.
            self._tracker.start(req_id, "GET", self.path)
            self._serve_events(path.rsplit("/", 1)[-1], req_id)
            return

        self._tracker.start(req_id, "GET", self.path)
        try:
            if path == "/":
                self._send_text(_INDEX_HTML)
            elif path == "/api/config":
                self._send_json(
                    {
                        "host": self._config.ollama_host,
                        "model": self._config.model,
                        "endpoints": [
                            {"host": e.host, "model": e.model, "role": e.role}
                            for e in self._config.endpoints
                        ],
                        "state": self._runs.status(),
                    }
                )
            elif path == "/api/models":
                host = parse_qs(parsed.query).get("host", [self._config.ollama_host])[0]
                self._send_json({"models": available_models(host)})
            elif path == "/api/requests":
                self._send_json({"active": self._tracker.active(exclude=req_id)})
            elif path.startswith("/api/calls/"):
                run_id = path[len("/api/calls/") :]
                self._send_json({"active": self._runs.active_calls(run_id)})
            elif path == "/api/settings":
                c = self._config
                self._send_json({k: getattr(c, k) for k in _ALL_SETTINGS_KEYS})
            elif path.startswith("/api/files/"):
                run_id = path[len("/api/files/") :]
                self._send_json(self._files_response(run_id))
            elif path.startswith("/api/file/"):
                content = self._read_run_file(path[len("/api/file/") :])
                if content is None:
                    self._send_json({"error": "no such file"}, status=404)
                else:
                    self._send_json({"content": content})
            elif path.startswith("/api/plan/"):
                run_id = path[len("/api/plan/") :]
                text = self._read_plan_text(run_id)
                if text is None:
                    self._send_json({"error": "no plan for this run"}, status=404)
                else:
                    self._send_json({"content": text})
            elif path.startswith("/api/spec/"):
                run_id, _, filename = path[len("/api/spec/") :].partition("/")
                text = self._read_spec_text(run_id, filename) if filename else None
                if text is None:
                    self._send_json({"error": "no spec for this file"}, status=404)
                else:
                    self._send_json({"content": text})
            elif path.startswith("/api/summary/"):
                run_id = path.rsplit("/", 1)[-1]
                log_path = self._runs.log_path(run_id)
                if not log_path.exists():
                    self._send_json({"error": "no such run"}, status=404)
                else:
                    self._send_json(asdict(summary.load_run_summary(log_path)))
            else:
                self._send_json({"error": "not found"}, status=404)
        finally:
            self._tracker.finish(req_id)

    def do_POST(self) -> None:
        req_id = self._request_id()
        self._tracker.start(req_id, "POST", self.path)
        try:
            path = urlparse(self.path).path
            if path == "/api/run":
                self._post_run()
            elif path == "/api/run/cancel":
                self._send_json({"cancelled": self._runs.cancel()})
            elif path == "/api/run/pause":
                self._send_json({"paused": self._runs.pause()})
            elif path == "/api/run/resume":
                self._send_json({"resumed": self._runs.resume()})
            elif path == "/api/settings":
                self._post_settings()
            else:
                self._send_json({"error": "not found"}, status=404)
        finally:
            self._tracker.finish(req_id)

    def _read_json_body(self) -> dict | None:
        """Parses the request body as JSON, or sends a 400 and returns
        None -- callers just bail out when they get None back."""
        length = int(self.headers.get("Content-Length", 0))
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON"}, status=400)
            return None

    def _post_run(self) -> None:
        body = self._read_json_body()
        if body is None:
            return
        goal = (body.get("goal") or "").strip()
        if not goal:
            self._send_json({"error": "goal is required"}, status=400)
            return
        endpoints = body.get("endpoints") or None
        if isinstance(endpoints, list):
            endpoints = [e for e in endpoints if isinstance(e, dict) and e.get("host")]
        try:
            run_id = self._runs.start(
                goal=goal,
                model=body.get("model") or self._config.model,
                host=body.get("host") or self._config.ollama_host,
                multi_file=bool(body.get("multi_file", True)),
                endpoints=endpoints or None,
            )
        except RuntimeError as exc:
            self._send_json({"error": str(exc)}, status=409)
            return
        self._send_json({"run_id": run_id})

    def _post_settings(self) -> None:
        body = self._read_json_body()
        if body is None:
            return
        updates: dict[str, float | int | bool] = {}
        bad_keys: list[str] = []
        for key in _SETTINGS_KEYS:
            if key not in body:
                continue
            try:
                updates[key] = float(body[key]) if key == "temperature" else int(body[key])
            except (TypeError, ValueError):
                bad_keys.append(key)
        for key in _BOOL_SETTINGS_KEYS:
            if key in body:
                updates[key] = bool(body[key])
        if bad_keys:
            self._send_json({"error": f"invalid value(s) for: {', '.join(bad_keys)}"}, status=400)
            return
        raw_endpoints = body.get("endpoints")
        if raw_endpoints is not None:
            if not isinstance(raw_endpoints, list) or not all(
                isinstance(e, dict) and e.get("host") for e in raw_endpoints
            ):
                self._send_json({"error": "invalid endpoints"}, status=400)
                return
            current = self._runs.config
            updates["endpoints"] = tuple(
                Endpoint(
                    e["host"],
                    e.get("model") or current.model,
                    current.timeout_seconds,
                    role=e.get("role") or "balanced",
                )
                for e in raw_endpoints
            )
        new_config = self._runs.update_config(**updates)
        values = {k: getattr(new_config, k) for k in _ALL_SETTINGS_KEYS}
        values["endpoints"] = [
            {"host": e.host, "model": e.model, "role": e.role} for e in new_config.endpoints
        ]
        _save_settings_overrides(values)
        self._send_json(values)

    _MAX_FILE_VIEW_BYTES = 200_000

    def _iter_log_events(self, run_id: str) -> Iterator[dict]:
        log_path = self._runs.log_path(run_id)
        if not log_path.exists():
            return
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

    _GENERATED_FILE_GLOBS = ("*.py", "*.html", "*.css", "*.js")

    def _files_response(self, run_id: str) -> dict:
        """Every file this run is building, `.py`/`.html`/`.css`/`.js`
        (everything the planner may produce, see steps/plan.py) -- not
        just the ones that exist on disk yet. When the run went through
        the planner (`has_plan`), the plan's own file list is the source
        of truth, in plan order: a file the planner listed but no worker
        has claimed yet shows up as "pending" rather than being absent,
        which is the whole point for a to-do list -- and each entry
        carries `purpose`/`depends_on` straight from the plan, so the
        page can render the dependency graph without a second fetch. A
        single-file run has no plan at all; falls back to whatever's on
        disk, as before. `phase` is a coarse "where is this run right
        now" for an at-a-glance header: planning -> building ->
        integration -> done."""
        run_dir = self._runs.log_path(run_id).parent
        if not run_dir.is_dir():
            return {"files": [], "has_plan": False, "phase": "planning"}
        # In-flight, not-yet-logged spec/critic/fix calls -- a file only
        # shows up in the completed log's own event once a call returns,
        # but the graph should show what's actually happening to it right
        # now, not just after. One active_calls() snapshot so all three
        # sets agree on the same instant. active_calls() already scopes to
        # this exact run while it's the one actually running, so all three
        # are empty for good on a finished/other run with no extra
        # bookkeeping. "integration_fix" counts as fixing too -- it's the
        # same kind of in-place repair, just triggered by an integration
        # error instead of the file's own verify loop.
        active_calls = self._runs.active_calls(run_id)
        speccing_names = {
            Path(c["path"]).name for c in active_calls if c.get("kind") == "spec" and c.get("path")
        }
        criticizing_names = {
            Path(c["path"]).name for c in active_calls if c.get("kind") == "critic" and c.get("path")
        }
        fixing_names = {
            Path(c["path"]).name
            for c in active_calls
            if c.get("kind") in ("fix", "integration_fix") and c.get("path")
        }
        # A branch attempt (orchestrator._generate_files) codegens/fixes
        # under a scratch filename (".branch-<real name>") so it can
        # never collide on disk with the original attempt still racing
        # it -- strip that prefix back off to attribute the activity to
        # the real file. Its brief initial spec phase isn't caught here
        # (spec calls are tracked under the bare task path, same as the
        # original's, since specs don't touch a file path at all) -- a
        # small, acceptable gap since branching exists for files stuck
        # deep in the fix loop, where that's where almost all of a
        # branch's time is actually spent.
        branching_names = {
            Path(c["path"]).name.removeprefix(".branch-")
            for c in active_calls
            if c.get("path") and Path(c["path"]).name.startswith(".branch-")
        }
        status_by_name: dict[str, str] = {}
        fixes_by_name: dict[str, int] = {}
        spec_names: set[str] = set()
        endpoint_by_name: dict[str, str] = {}
        # has a spec/codegen event, or a spec call in flight right now --
        # "building", not "pending". Without the speccing_names seed, a
        # file whose spec call just started would show as untouched
        # ("pending", dashed border) underneath its own pulsing
        # "writing spec" dot -- a contradiction.
        started_names: set[str] = set(speccing_names)
        finished_names: set[str] = set()  # its build loop has actually concluded, see file_result
        plan_entries: list[dict] = []
        has_plan = False
        log_path = run_dir / "log.jsonl"
        run_summary = None
        if log_path.exists():
            run_summary = summary.load_run_summary(log_path)
            for f in run_summary.files:
                name = Path(f.path).name
                if f.spec_flagged:
                    status_by_name[name] = "flagged"
                elif f.advisory:
                    status_by_name[name] = "advisory"
                elif f.last_error.startswith("skipped:"):
                    status_by_name[name] = "skipped"
                else:
                    status_by_name[name] = "ok" if f.success else "failed"
                # Same source the final run-summary table's "N fixes" column
                # already uses -- live here too, since load_run_summary
                # tallies a file's `fix` events as soon as they're logged,
                # not only once the file's build has concluded.
                fixes_by_name[name] = f.attempts
            for record in self._iter_log_events(run_id):
                event = record.get("event")
                if event == "plan":
                    has_plan = True
                    plan_entries = [
                        {
                            "name": Path(pf.get("path", "")).name,
                            "purpose": pf.get("purpose", ""),
                            "depends_on": pf.get("depends_on") or [],
                        }
                        for pf in record.get("files", [])
                    ]
                elif event == "spec":
                    name = Path(record.get("path", "")).name
                    spec_names.add(name)
                    started_names.add(name)
                elif event == "codegen":
                    name = Path(record.get("path", "")).name
                    started_names.add(name)
                    if record.get("endpoint"):
                        # The same worker (client/endpoint) owns a file for
                        # its whole build, so this is set once and stays --
                        # last-write-wins is only relevant if it ever isn't.
                        endpoint_by_name[name] = record["endpoint"]
                elif event == "file_result":
                    finished_names.add(Path(record.get("path", "")).name)

        phase = "planning"
        if has_plan:
            phase = "building"
            if run_summary is not None and run_summary.integration is not None:
                phase = "integration"
            if run_summary is not None and run_summary.super_review_started:
                phase = "reviewing"
        if run_summary is not None and run_summary.finished:
            phase = "done"

        if plan_entries:
            files = []
            for pf in plan_entries:
                name = pf["name"]
                # A verify event's own success/fail isn't final by itself
                # -- the fix loop may still have retries left (and, while
                # paused, may be sitting on a just-failed attempt for a
                # while before trying again). Only trust status_by_name
                # once file_result says this file's build has actually
                # concluded; until then it's still "building".
                status = status_by_name.get(name) if name in finished_names else None
                if status is None:
                    status = "building" if name in started_names else "pending"
                disk_path = run_dir / name
                files.append(
                    {
                        "name": name,
                        "purpose": pf["purpose"],
                        "depends_on": pf["depends_on"],
                        "status": status,
                        "size": disk_path.stat().st_size if disk_path.exists() else 0,
                        "has_spec": name in spec_names,
                        "speccing": name in speccing_names,
                        "criticizing": name in criticizing_names,
                        "fixing": name in fixing_names,
                        "branching": name in branching_names,
                        "fixes": fixes_by_name.get(name, 0),
                        "endpoint": endpoint_by_name.get(name, ""),
                    }
                )
            return {"files": files, "has_plan": True, "phase": phase}

        found = (p for pattern in self._GENERATED_FILE_GLOBS for p in run_dir.glob(pattern))
        files = []
        for p in sorted(found):
            files.append(
                {
                    "name": p.name,
                    "purpose": "",
                    "depends_on": [],
                    "status": status_by_name.get(p.name, "pending"),
                    "size": p.stat().st_size,
                    "has_spec": p.name in spec_names,
                    "speccing": p.name in speccing_names,
                    "criticizing": p.name in criticizing_names,
                    "fixing": p.name in fixing_names,
                    "branching": p.name in branching_names,
                    "fixes": fixes_by_name.get(p.name, 0),
                    "endpoint": endpoint_by_name.get(p.name, ""),
                }
            )
        return {"files": files, "has_plan": has_plan, "phase": phase}

    def _read_plan_text(self, run_id: str) -> str | None:
        """The most recent `plan` event, rendered as plain text -- None
        when this run never went through the planner (single-file)."""
        plan_event = None
        for record in self._iter_log_events(run_id):
            if record.get("event") == "plan":
                plan_event = record
        if plan_event is None:
            return None
        lines = []
        for f in plan_event.get("files", []):
            deps = f.get("depends_on") or []
            dep_note = f"  (depends on: {', '.join(deps)})" if deps else ""
            lines.append(f"{f.get('path', '?')} -- {f.get('purpose', '')}{dep_note}")
        return "\n".join(lines) if lines else "(empty plan)"

    def _read_spec_text(self, run_id: str, filename: str) -> str | None:
        """The most recent `spec` event for `filename` -- None when this
        file never got one (single-file runs skip spec-writing)."""
        spec_text = None
        for record in self._iter_log_events(run_id):
            if record.get("event") == "spec" and Path(record.get("path", "")).name == filename:
                spec_text = record.get("spec", "")
        return spec_text

    def _read_run_file(self, rest: str) -> str | None:
        """`rest` is `<run_id>/<filename>`. Confines the read to that
        run's own directory -- `filename` ultimately comes from the URL,
        so a `../` must never be able to walk it anywhere else."""
        run_id, _, filename = rest.partition("/")
        if not filename:
            return None
        run_dir = self._runs.log_path(run_id).parent.resolve()
        target = (run_dir / filename).resolve()
        if run_dir not in target.parents:
            return None
        if not target.is_file() or target.stat().st_size > self._MAX_FILE_VIEW_BYTES:
            return None
        return target.read_text(errors="replace")

    def _serve_events(self, run_id: str, req_id: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def still_running() -> bool:
            st = self._runs.status()
            return st["run_id"] == run_id and st["state"] == "running"

        last_id = self.headers.get("Last-Event-ID", "")
        start = int(last_id) if last_id.isdigit() else 0
        try:
            for chunk in stream_events(
                self._runs.log_path(run_id), start=start, is_active=still_running
            ):
                self.wfile.write(chunk.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ValueError):
            pass
        finally:
            self._tracker.finish(req_id)


def build_server(config: Config, *, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    overrides = _load_settings_overrides()
    if overrides:
        raw_endpoints = overrides.pop("endpoints", None)
        if raw_endpoints:
            overrides["endpoints"] = tuple(
                Endpoint(
                    e["host"],
                    e.get("model") or config.model,
                    config.timeout_seconds,
                    role=e.get("role") or "balanced",
                )
                for e in raw_endpoints
                if e.get("host")
            )
        config = config.with_overrides(**overrides)
    server = ThreadingHTTPServer((host, port), _Handler)
    server.config = config  # type: ignore[attr-defined]
    server.run_manager = RunManager(config)  # type: ignore[attr-defined]
    server.requests = RequestTracker()  # type: ignore[attr-defined]
    return server


def serve(config: Config, *, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    server = build_server(config, host=host, port=port)
    url = f"http://{host}:{server.server_address[1]}"
    print(f"kiddie-harness GUI on {url}  (Ctrl-C to stop)", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>kiddie-harness</title>
<style>
  :root { --fg:#1c1c1e; --bg:#fbfbfa; --muted:#8a8a8e; --line:#e4e4e2;
          --accent:#3a6adf; --ok:#1f9d55; --bad:#d1453b; --warn:#c47f17;
          --spec:#8a5cf6; --critic:#0891b2; --branch:#db2777; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#eaeaea; --bg:#181818; --muted:#8a8a8e; --line:#333;
            --accent:#6f9bff; --ok:#57c97f; --bad:#ff6b60; --warn:#e0a24a;
            --spec:#b39bfa; --critic:#22d3ee; --branch:#f472b6; }
  }
  * { box-sizing:border-box; }
  [hidden] { display:none !important; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  main { max-width:760px; margin:0 auto; padding:44px 20px 80px; }
  h1 { font-size:19px; font-weight:600; letter-spacing:-.01em; margin:0 0 24px;
       display:flex; align-items:baseline; gap:0; }
  h1 span { color:var(--muted); font-weight:400; }
  h1 .calls { font-size:11px; }
  .gear { margin-left:auto; background:none; border:0; padding:0 0 0 6px; width:auto;
          color:var(--muted); font-size:14px; cursor:pointer; }
  .gear:hover { color:var(--accent); }
  .calls-list, .settings-panel { margin:2px 0 20px; padding:7px 10px; border:1px solid var(--line);
               border-radius:8px; background:color-mix(in srgb, var(--fg) 4%, transparent); }
  .calls-list { font:11px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--muted); }
  .calls-list div { display:flex; justify-content:space-between; gap:10px; cursor:pointer;
                     border-radius:4px; padding:0 4px; margin:0 -4px; }
  .calls-list div:hover { background:color-mix(in srgb, var(--fg) 7%, transparent); }
  .calls-list div.active { background:color-mix(in srgb, var(--accent) 16%, transparent); }
  .calls-list .kind { color:var(--fg); font-weight:600; }
  .calls-list .t { color:var(--fg); opacity:.7; flex:0 0 auto; }
  .settings-panel { padding:12px 14px 4px; }
  .settings-panel .row > div { margin-bottom:10px; }
  .settings-panel label { margin:0 0 4px; font-size:10px; }
  .settings-panel input { padding:6px 8px; font-size:13px; }
  .settings-actions { display:flex; align-items:center; gap:10px; margin:-2px 0 10px; }
  .settings-actions button { margin:0; width:auto; padding:6px 14px; }
  .settings-msg { font-size:12px; color:var(--muted); }
  label { display:block; font-size:12px; text-transform:uppercase; letter-spacing:.06em;
          color:var(--muted); margin:16px 0 6px; }
  select, input[type=text], textarea {
    width:100%; padding:9px 11px; border:1px solid var(--line); border-radius:8px;
    background:var(--bg); color:var(--fg); font:inherit; }
  textarea { resize:vertical; min-height:76px; }
  .row { display:flex; gap:12px; }
  .row > div { flex:1; }
  .row > .role-col { flex:0 0 96px; }
  .role-col select { padding:9px 4px; text-align:center; }
  .check { display:flex; align-items:center; gap:8px; margin-top:14px; font-size:13px; color:var(--muted); }
  .check input { width:auto; }
  button { margin-top:20px; width:100%; padding:11px; border:0; border-radius:8px;
           background:var(--accent); color:#fff; font:inherit; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.45; cursor:default; }
  button.link { display:block; margin-top:8px; width:auto; padding:2px 0; background:none;
                color:var(--muted); font-weight:400; font-size:12px; text-align:left; }
  button.link:hover:not(:disabled) { color:var(--accent); }
  #stop { background:var(--bad); }
  .stats { display:flex; flex-wrap:wrap; gap:6px 18px; margin:26px 0 10px;
           font-size:13px; color:var(--muted); min-height:20px; }
  .stats b { color:var(--fg); font-weight:600; }
  .verdict { display:inline-block; padding:2px 9px; border-radius:6px; font-weight:600;
             font-size:12px; letter-spacing:.03em; }
  .verdict.ok { background:color-mix(in srgb, var(--ok) 18%, transparent); color:var(--ok); }
  .verdict.bad { background:color-mix(in srgb, var(--bad) 18%, transparent); color:var(--bad); }
  .verdict.warn { background:color-mix(in srgb, var(--warn) 20%, transparent); color:var(--warn); }
  pre#log { margin:8px 0 0; padding:14px; background:color-mix(in srgb, var(--fg) 5%, transparent);
            border:1px solid var(--line); border-radius:8px; font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
            white-space:pre-wrap; word-break:break-word; max-height:340px; overflow:auto; }
  pre#log:empty { display:none; }
  .files-panel { margin-top:14px; }
  .files-panel > label { margin:0 0 6px; }
  .file-list { border:1px solid var(--line); border-radius:8px; overflow:hidden;
               background:color-mix(in srgb, var(--fg) 3%, transparent); }
  .file-list div { padding:8px 12px; font-size:13px; cursor:pointer; display:flex;
                    align-items:center; gap:8px; border-top:1px solid var(--line); }
  .file-list div:first-child { border-top:0; }
  .file-list div:hover { background:color-mix(in srgb, var(--fg) 7%, transparent); }
  .file-list div .fname { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .file-list div .fspec { font-size:10.5px; color:var(--muted); cursor:pointer; flex:0 0 auto;
                           text-decoration:underline; text-underline-offset:2px; }
  .file-list div .fspec:hover { color:var(--accent); }
  .file-list div .fsize { color:var(--muted); font-size:11px; flex:0 0 auto; }
  .file-list div .fep { font-size:10px; color:var(--muted); flex:0 0 auto; padding:1px 6px;
                         border-radius:4px; background:color-mix(in srgb, var(--fg) 8%, transparent); }
  .file-dot { width:7px; height:7px; border-radius:50%; flex:0 0 auto; background:var(--muted); }
  .file-dot.ok { background:var(--ok); }
  .file-dot.failed { background:var(--bad); }
  .file-dot.advisory { background:var(--warn); }
  .file-dot.flagged { background:var(--accent); }
  .file-modal { position:fixed; inset:0; background:rgba(0,0,0,.45); display:flex;
                align-items:center; justify-content:center; padding:30px; z-index:10; }
  .file-modal-box { width:100%; max-width:700px; max-height:100%; display:flex;
                     flex-direction:column; background:var(--bg); border:1px solid var(--line);
                     border-radius:10px; box-shadow:0 20px 60px rgba(0,0,0,.35); overflow:hidden; }
  .file-modal-head { display:flex; align-items:center; justify-content:space-between;
                      padding:10px 14px; border-bottom:1px solid var(--line);
                      font:600 13px ui-monospace,SFMono-Regular,Menlo,monospace; }
  .file-modal-head button { margin:0; width:auto; padding:2px 8px; background:none;
                             color:var(--muted); font-size:18px; line-height:1; }
  .file-modal-head button:hover { color:var(--bad); }
  .file-modal-body { margin:0; padding:14px; overflow:auto; font-size:12px; line-height:1.5;
                      font-family:ui-monospace,SFMono-Regular,Menlo,monospace; white-space:pre-wrap;
                      word-break:break-word; color:var(--fg); }
  table { width:100%; border-collapse:collapse; margin-top:14px; font-size:13px; }
  td { padding:5px 8px; border-top:1px solid var(--line); }
  td.s-ok { color:var(--ok); } td.s-bad { color:var(--bad); } td.s-adv { color:var(--warn); }
  td.s-flag { color:var(--accent); }
  td.s-review-confirmed { color:var(--critic); } td.s-review-unconfirmed { color:var(--muted); }
  .s-branched { color:var(--branch); font-weight:600; font-size:11.5px; }
  .review-findings { margin-top:14px; }
  .review-findings b { font-size:11px; text-transform:uppercase; letter-spacing:.03em;
                         color:var(--muted); }
  .err { color:var(--bad); font-size:13px; margin-top:10px; }
  pre.reason { margin:12px 0 0; padding:12px 14px; border:1px solid var(--bad);
               border-radius:8px; background:color-mix(in srgb, var(--bad) 8%, transparent);
               color:var(--fg); font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
               white-space:pre-wrap; word-break:break-word; max-height:280px; overflow:auto; }

  /* -- phase stepper: "where is this run right now", at a glance -- */
  .phase-stepper { display:flex; margin:0 0 22px; border:1px solid var(--line);
                    border-radius:8px; overflow:hidden; }
  .phase-step { flex:1; text-align:center; padding:8px 6px; font-size:10.5px; font-weight:700;
                text-transform:uppercase; letter-spacing:.05em; color:var(--muted);
                background:color-mix(in srgb, var(--fg) 3%, transparent);
                border-right:1px solid var(--line); transition:background .25s,color .25s; }
  .phase-step:last-child { border-right:0; }
  .phase-step.past { color:var(--ok); background:color-mix(in srgb, var(--ok) 12%, transparent); }
  .phase-step.current { color:#fff; background:var(--accent); }
  .phase-step.aborted { color:#fff; background:var(--bad); }

  /* -- run controls: Stop / Pause / Resume side by side -- */
  .run-actions { display:flex; gap:8px; }
  .run-actions button { flex:1; width:auto; }
  #pause { background:var(--warn); }
  #resume { background:var(--ok); }
  .pause-note { display:flex; align-items:center; gap:6px; font-size:12px; color:var(--warn);
                 margin-top:10px; }
  .pause-note .dot { width:6px; height:6px; border-radius:50%; background:var(--warn);
                       animation:pulse 1.4s ease-in-out infinite; }
  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:.3; } }

  /* -- endpoint pool: primary + any number of extras -- */
  .endpoint-row { position:relative; padding-right:22px; }
  .endpoint-row .endpoint-remove { position:absolute; top:0; right:0; background:none; width:auto;
                margin:16px 0 0; padding:4px 2px; color:var(--muted); font-size:16px; line-height:1; }
  .endpoint-row .endpoint-remove:hover { color:var(--bad); }

  /* -- dependency graph: doubles as the to-do list -- */
  .graph-wrap { overflow-x:auto; padding:2px 0 4px; }
  .graph-empty { color:var(--muted); font-size:13px; padding:4px 0; }
  .graph-hint { color:var(--muted); font-size:10.5px; margin:0 0 6px; }
  .dep-svg { display:block; margin:0 auto; }
  .dep-node-rect { fill:var(--bg); stroke:var(--line); stroke-width:1.5; cursor:pointer;
                    transition:opacity .15s ease, stroke-width .15s ease; }
  .dep-node-rect:hover { stroke:var(--accent); }
  .dep-node-rect.pending { stroke-dasharray:4 3; }
  .dep-node-rect.building { stroke:var(--accent); stroke-width:2; }
  .dep-node-rect.ok { stroke:var(--ok); }
  .dep-node-rect.failed { stroke:var(--bad); }
  .dep-node-rect.advisory { stroke:var(--warn); }
  .dep-node-rect.flagged { stroke:var(--accent); stroke-dasharray:2 2; }
  .dep-node-rect.skipped { stroke:var(--muted); stroke-dasharray:2 2; }
  .dep-node-group { cursor:pointer; }
  .dep-node-name { font:600 11px ui-monospace,SFMono-Regular,Menlo,monospace; fill:var(--fg);
                    pointer-events:none; transition:opacity .15s ease; }
  .dep-node-status { font:600 9px ui-monospace,SFMono-Regular,Menlo,monospace; pointer-events:none;
                       text-transform:uppercase; letter-spacing:.03em; transition:opacity .15s ease; }
  .dep-node-status.ok { fill:var(--ok); }
  .dep-node-status.failed { fill:var(--bad); }
  .dep-node-status.advisory { fill:var(--warn); }
  .dep-node-status.flagged { fill:var(--accent); }
  .dep-node-status.building { fill:var(--accent); }
  .dep-node-status.pending, .dep-node-status.skipped { fill:var(--muted); }
  /* Small "N fix attempts so far" badge, bottom-left -- only rendered
     when non-zero, same amber as the live "fixing" dot since it's the
     same underlying signal (the model needed another try), just a
     running count instead of a moment-in-time indicator. */
  .dep-node-fixes { font:600 9px ui-monospace,SFMono-Regular,Menlo,monospace;
                      fill:var(--warn); pointer-events:none; }
  /* Hovering a node highlights its own edges/neighbours (set via JS) and
     dims everything else -- with more than a handful of files, which
     line belongs to which box stops being obvious from color alone. */
  .dep-node-group.dim .dep-node-rect,
  .dep-node-group.dim .dep-node-name,
  .dep-node-group.dim .dep-node-status,
  .dep-node-group.dim .dep-spec-dot,
  .dep-node-group.dim .dep-speccing-dot,
  .dep-node-group.dim .dep-criticizing-dot,
  .dep-node-group.dim .dep-fixing-dot,
  .dep-node-group.dim .dep-branching-dot,
  .dep-node-group.dim .dep-node-fixes { opacity:.3; }
  .dep-node-group.hl .dep-node-rect { stroke-width:2.5; }
  .dep-edge { fill:none; stroke:var(--muted); stroke-width:1.6; opacity:.65;
               transition:opacity .15s ease, stroke .15s ease, stroke-width .15s ease; }
  .dep-edge.hl { stroke:var(--accent); stroke-width:2.4; opacity:1; }
  .dep-edge.dim { opacity:.08; }
  .dep-arrowhead { fill:var(--muted); }
  /* A small dot marking "this file has a spec" -- independent of status
     (the rect's own color/border), and its own click target to jump
     straight to the spec instead of the code. */
  .dep-spec-dot { fill:var(--spec); cursor:pointer; transition:opacity .15s ease; }
  .dep-spec-dot:hover { stroke:var(--spec); stroke-width:2; }
  .dep-speccing-dot { fill:var(--spec); pointer-events:none; animation:pulse 1.2s ease-in-out infinite; }
  .dep-criticizing-dot { fill:var(--critic); pointer-events:none; animation:pulse 1.2s ease-in-out infinite; }
  .dep-fixing-dot { fill:var(--warn); pointer-events:none; animation:pulse 1.2s ease-in-out infinite; }
  .dep-branching-dot { fill:var(--branch); pointer-events:none; animation:pulse 1.2s ease-in-out infinite; }
  .graph-legend { display:flex; flex-wrap:wrap; gap:4px 14px; margin-top:8px; font-size:10.5px;
                   color:var(--muted); }
  .graph-legend span { display:inline-flex; align-items:center; gap:4px; }
  .graph-legend i { width:8px; height:8px; border-radius:2px; display:inline-block;
                     border:1.5px solid var(--muted); }
  .graph-legend i.ok { border-color:var(--ok); } .graph-legend i.failed { border-color:var(--bad); }
  .graph-legend i.advisory { border-color:var(--warn); }
  .graph-legend i.flagged { border-color:var(--accent); }
  .graph-legend i.building { border-color:var(--accent); border-width:2px; }
  .graph-legend i.pending { border-style:dashed; }
  .graph-legend i.spec { border-radius:50%; border-color:var(--spec); background:var(--spec); }
  .graph-legend i.speccing { border-radius:50%; border-color:var(--spec); background:var(--spec);
                              animation:pulse 1.2s ease-in-out infinite; }
  .graph-legend i.criticizing { border-radius:50%; border-color:var(--critic); background:var(--critic);
                                  animation:pulse 1.2s ease-in-out infinite; }
  .graph-legend i.fixing { border-radius:50%; border-color:var(--warn); background:var(--warn);
                             animation:pulse 1.2s ease-in-out infinite; }
  .graph-legend i.branching { border-radius:50%; border-color:var(--branch); background:var(--branch);
                                animation:pulse 1.2s ease-in-out infinite; }
</style>
</head>
<body>
<main>
  <h1>kiddie-harness <span>&mdash; generate a project</span> <span id="calls" class="calls"></span>
    <button type="button" id="settings-btn" class="gear" title="settings">&#9881;</button>
  </h1>
  <div id="calls-list" class="calls-list" hidden></div>
  <div id="phase-stepper" class="phase-stepper" hidden>
    <div class="phase-step" data-phase="planning">Plan</div>
    <div class="phase-step" data-phase="building">Build</div>
    <div class="phase-step" data-phase="integration">Integrate</div>
    <div class="phase-step" data-phase="reviewing">Review</div>
    <div class="phase-step" data-phase="done">Done</div>
  </div>
  <div id="settings-panel" class="settings-panel" hidden>
    <div class="row">
      <div><label for="s-temperature">Temperature</label><input type="text" id="s-temperature"></div>
      <div><label for="s-max_tokens">Max tokens</label><input type="text" id="s-max_tokens"></div>
    </div>
    <div class="row">
      <div><label for="s-max_tokens_ceiling">Max tokens ceiling</label><input type="text" id="s-max_tokens_ceiling"></div>
      <div><label for="s-max_fix_attempts">Fix attempts</label><input type="text" id="s-max_fix_attempts"></div>
    </div>
    <div class="row">
      <div><label for="s-max_total_iterations">Max LLM calls</label><input type="text" id="s-max_total_iterations"></div>
      <div><label for="s-timeout_seconds">Call timeout (s)</label><input type="text" id="s-timeout_seconds"></div>
    </div>
    <div class="row">
      <div><label for="s-branch_after_fixes" title="Once a file's fix loop reaches this many attempts, an idle Smart or Balanced endpoint may start its own independent attempt at the same file in parallel -- whichever finishes first wins. 0 disables this.">Branch after N fixes (0 = off)</label><input type="text" id="s-branch_after_fixes"></div>
    </div>
    <div class="check" style="margin-top:0">
      <input type="checkbox" id="s-critic_enabled">
      <label for="s-critic_enabled" style="margin:0;text-transform:none;letter-spacing:0;font-size:13px">critic check (one extra call per file, judges it against its own spec)</label>
    </div>
    <div class="check">
      <input type="checkbox" id="s-super_review_enabled">
      <label for="s-super_review_enabled" style="margin:0;text-transform:none;letter-spacing:0;font-size:13px" title="Once every file is built and integration has run, review the whole project together for cross-file problems a per-file critic can never see. A second endpoint confirms each finding before it's reported. Advisory only.">whole-project review (finds cross-file issues once everything's built; a second endpoint confirms each finding)</label>
    </div>
    <div class="settings-actions">
      <button type="button" id="settings-save">Save</button>
      <span class="settings-msg" id="settings-msg"></span>
    </div>
  </div>

  <label for="model">Model</label>
  <div class="row">
    <div><select id="model"></select></div>
    <div><input type="text" id="host" placeholder="http://localhost:11434"></div>
    <div class="role-col">
      <select id="role" title="Balanced does both plan/critic and per-file work. Smart handles plan/critic/integration-fix only. Quick does the per-file spec/codegen/fix grind only.">
        <option value="balanced" selected>Balanced</option>
        <option value="smart">Smart</option>
        <option value="quick">Quick</option>
      </select>
    </div>
  </div>
  <button type="button" id="refresh" class="link">&#8635; re-fetch models</button>

  <div id="endpoints-extra"></div>
  <button type="button" id="add-ep" class="link">+ add endpoint <span style="text-transform:none;letter-spacing:0">(parallel, multi-file only)</span></button>
  <button type="button" id="save-ep" class="link">&#9733; save as default</button>
  <span class="settings-msg" id="ep-msg"></span>

  <label for="goal">Goal</label>
  <textarea id="goal" placeholder="a command-line to-do list with add / list / done subcommands"></textarea>

  <div class="check">
    <input type="checkbox" id="multi" checked>
    <label for="multi" style="margin:0;text-transform:none;letter-spacing:0;font-size:13px">multi-file project</label>
  </div>

  <button id="go">Generate</button>
  <div class="run-actions" id="run-actions" hidden>
    <button type="button" id="pause">Pause</button>
    <button type="button" id="resume" hidden>Resume</button>
    <button type="button" id="stop">Stop</button>
  </div>
  <div class="pause-note" id="pause-note" hidden>
    <span class="dot"></span> Paused -- change settings above, they'll apply on Resume.
  </div>
  <div class="err" id="err" hidden></div>

  <div class="stats" id="stats"></div>
  <pre id="log"></pre>
  <div class="files-panel" id="files-panel" hidden>
    <label>Files <span id="plan-link-wrap" style="text-transform:none;letter-spacing:0;font-size:11px"></span></label>
    <div class="graph-hint" id="graph-hint" hidden>hover a file to trace what it depends on and what needs it &mdash; an arrow points from a dependency to the file that needs it</div>
    <div class="graph-wrap" id="graph-wrap"></div>
    <div class="graph-legend" id="graph-legend" hidden>
      <span><i class="pending"></i>pending</span>
      <span><i class="building"></i>building</span>
      <span><i class="ok"></i>ok</span>
      <span><i class="failed"></i>failed</span>
      <span><i class="advisory"></i>advisory</span>
      <span><i class="flagged"></i>flagged</span>
      <span><i class="spec"></i>has a spec</span>
      <span><i class="speccing"></i>writing spec&hellip;</span>
      <span><i class="criticizing"></i>critic reviewing&hellip;</span>
      <span><i class="fixing"></i>fixing&hellip;</span>
      <span><i class="branching"></i>branching to another endpoint&hellip;</span>
    </div>
  </div>
  <div id="summary"></div>
</main>

<div class="file-modal" id="file-modal" hidden>
  <div class="file-modal-box">
    <div class="file-modal-head">
      <span id="file-modal-name"></span>
      <button type="button" id="file-modal-close">&times;</button>
    </div>
    <pre class="file-modal-body" id="file-modal-body"></pre>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
const logEl = $("#log"), statsEl = $("#stats"), goBtn = $("#go"), errEl = $("#err");
const stopBtn = $("#stop"), pauseBtn = $("#pause"), resumeBtn = $("#resume");
let started = 0, calls = 0, fixes = 0, filesDone = 0, filesTotal = 0, tick = null, es = null;
let defaultModel = "";
let currentRunId = null;

// Every request to the server carries a unique id in the URL (_r=...):
// the server exposes what's currently in flight at /api/requests, and
// the page uses the id itself to drop a response that's been superseded
// by a newer request for the same field (e.g. two quick re-fetch clicks).
let reqSeq = 0;
function tagUrl(url) {
  const id = "r" + (++reqSeq) + "-" + Date.now().toString(36);
  return { url: url + (url.includes("?") ? "&" : "?") + "_r=" + id, id };
}
const rowReq = { "": 0 };

// Any number of extra endpoints beyond the primary -- each row gets its
// own suffixed ids ("-1", "-2", ...) so refreshRow(p) (built for the
// original fixed host/host2 pair) works unchanged for a dynamic row too.
let extraEpSeq = 0;
const extraEpIds = [];

function addEndpointRow(seed) {
  extraEpSeq++;
  const suffix = "-" + extraEpSeq;
  extraEpIds.push(suffix);
  const row = document.createElement("div");
  row.className = "endpoint-row";
  row.innerHTML =
    `<label>Endpoint <span style="text-transform:none;letter-spacing:0">(parallel, multi-file only)</span></label>` +
    `<div class="row"><div><select id="model${suffix}"></select></div>` +
    `<div><input type="text" id="host${suffix}" placeholder="http://localhost:11434"></div>` +
    `<div class="role-col"><select id="role${suffix}">` +
    `<option value="balanced" selected>Balanced</option>` +
    `<option value="smart">Smart</option>` +
    `<option value="quick">Quick</option>` +
    `</select></div></div>` +
    `<button type="button" id="refresh${suffix}" class="link">&#8635; re-fetch models</button>` +
    `<button type="button" class="endpoint-remove" title="remove this endpoint">&times;</button>`;
  $("#endpoints-extra").appendChild(row);
  const hostInput = $("#host" + suffix);
  // Default to localhost, not a copy of the primary host -- an endpoint
  // pointed at the same host as another isn't a separate endpoint at
  // all, just two workers queuing on one server. A seed (restoring a
  // saved default) overrides that.
  hostInput.value = (seed && seed.host) || "http://localhost:11434";
  if (seed && seed.role) $("#role" + suffix).value = seed.role;
  hostInput.addEventListener("change", () => refreshRow(suffix));
  $("#refresh" + suffix).addEventListener("click", () => refreshRow(suffix));
  row.querySelector(".endpoint-remove").addEventListener("click", () => {
    row.remove();
    const i = extraEpIds.indexOf(suffix);
    if (i !== -1) extraEpIds.splice(i, 1);
  });
  refreshRow(suffix, seed && seed.model);
}

async function loadConfig() {
  const { url } = tagUrl("/api/config");
  const c = await (await fetch(url)).json();
  // A saved default endpoint list (see #save-ep) takes over the whole
  // form -- it's the primary row plus zero or more extras, same shape
  // the "Generate" button itself sends. Falls back to the bare
  // host/model when nothing's been saved yet.
  const eps = c.endpoints && c.endpoints.length ? c.endpoints : null;
  $("#host").value = eps ? eps[0].host : c.host;
  defaultModel = eps ? eps[0].model : c.model;
  if (eps && eps[0].role) $("#role").value = eps[0].role;
  await refreshRow("", eps ? eps[0].model : undefined);
  if (eps) eps.slice(1).forEach(e => addEndpointRow(e));
  $("#host").addEventListener("change", () => refreshRow(""));
  $("#refresh").addEventListener("click", () => refreshRow(""));
  $("#add-ep").addEventListener("click", () => addEndpointRow());
  $("#save-ep").addEventListener("click", saveEndpointsAsDefault);
  pollCalls();
  setInterval(pollCalls, 3000);
  loadSettings();

  // A run started before this page load (or before a reload) is still
  // going -- pick its stream back up instead of showing an idle form
  // that silently rejects "Generate" with "already in progress".
  if (c.state && c.state.state === "running" && c.state.run_id) {
    beginTracking();
    setPausedUI(!!c.state.paused);
    watchRun(c.state.run_id);
  }
}

const SETTINGS_KEYS = [
  "temperature", "max_tokens", "max_tokens_ceiling",
  "max_fix_attempts", "max_total_iterations", "timeout_seconds",
  "branch_after_fixes",
];
const BOOL_SETTINGS_KEYS = ["critic_enabled", "super_review_enabled"];

async function loadSettings() {
  try {
    const { url } = tagUrl("/api/settings");
    const s = await (await fetch(url)).json();
    SETTINGS_KEYS.forEach(k => { if (k in s) $("#s-" + k).value = s[k]; });
    BOOL_SETTINGS_KEYS.forEach(k => { if (k in s) $("#s-" + k).checked = s[k]; });
  } catch (e) { /* settings panel just stays blank */ }
}

function currentEndpoints() {
  return [
    { host: $("#host").value.trim(), model: $("#model").value, role: $("#role").value },
    ...extraEpIds.map(suffix => ({
      host: $("#host" + suffix).value.trim(),
      model: $("#model" + suffix).value,
      role: $("#role" + suffix).value,
    })),
  ].filter(e => e.host);
}

async function saveEndpointsAsDefault() {
  const msg = $("#ep-msg");
  msg.textContent = "saving\\u2026";
  try {
    const { url } = tagUrl("/api/settings");
    const res = await fetch(url, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ endpoints: currentEndpoints() }),
    });
    const data = await res.json();
    msg.textContent = res.ok ? "saved as default" : (data.error || "save failed");
  } catch (e) { msg.textContent = "could not reach the server"; }
  setTimeout(() => { if (msg.textContent.startsWith("saved")) msg.textContent = ""; }, 3000);
}

$("#settings-btn").addEventListener("click", () => {
  $("#settings-panel").hidden = !$("#settings-panel").hidden;
});

$("#settings-save").addEventListener("click", async () => {
  const body = {};
  SETTINGS_KEYS.forEach(k => {
    const v = $("#s-" + k).value.trim();
    if (v !== "") body[k] = v;
  });
  BOOL_SETTINGS_KEYS.forEach(k => { body[k] = $("#s-" + k).checked; });
  const msg = $("#settings-msg");
  msg.textContent = "saving\\u2026";
  try {
    const { url } = tagUrl("/api/settings");
    const res = await fetch(url, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) { msg.textContent = data.error || "save failed"; return; }
    SETTINGS_KEYS.forEach(k => { if (k in data) $("#s-" + k).value = data[k]; });
    BOOL_SETTINGS_KEYS.forEach(k => { if (k in data) $("#s-" + k).checked = data[k]; });
    msg.textContent = "saved \\u2014 "
      + (isPaused() ? "applied to the paused run" : "applies to the next run");
    setTimeout(() => { if (msg.textContent.startsWith("saved")) msg.textContent = ""; }, 3000);
  } catch (e) { msg.textContent = "could not reach the server"; }
});

function isPaused() {
  return !resumeBtn.hidden;
}

function setPausedUI(paused) {
  pauseBtn.hidden = paused; pauseBtn.disabled = false;
  resumeBtn.hidden = !paused; resumeBtn.disabled = false;
  $("#pause-note").hidden = !paused;
}

pauseBtn.addEventListener("click", async () => {
  pauseBtn.disabled = true;
  try {
    const { url } = tagUrl("/api/run/pause");
    await fetch(url, { method: "POST" });
  } catch (e) { /* the SSE stream will still carry run_paused when it lands */ }
});

resumeBtn.addEventListener("click", async () => {
  resumeBtn.disabled = true;
  try {
    const { url } = tagUrl("/api/run/resume");
    await fetch(url, { method: "POST" });
  } catch (e) { /* the SSE stream will still carry run_resumed when it lands */ }
});

const PHASE_ORDER = ["planning", "building", "integration", "reviewing", "done"];
function updatePhaseStepper(phase, aborted) {
  const idx = PHASE_ORDER.indexOf(phase);
  $("#phase-stepper").querySelectorAll(".phase-step").forEach(el => {
    const i = PHASE_ORDER.indexOf(el.dataset.phase);
    el.classList.remove("past", "current", "aborted");
    if (i < idx) el.classList.add("past");
    else if (i === idx) el.classList.add(aborted ? "aborted" : (phase === "done" ? "past" : "current"));
  });
}
async function refreshRow(p, wantOverride) {
  const host = $("#host" + p).value.trim();
  const sel = $("#model" + p);
  const want = wantOverride || sel.value || defaultModel;
  const btn = $("#refresh" + p);
  sel.disabled = true; btn.disabled = true; btn.textContent = "\\u21bb fetching\\u2026";
  const { url, id } = tagUrl("/api/models?host=" + encodeURIComponent(host));
  rowReq[p] = id;
  let models = [];
  try {
    ({ models } = await (await fetch(url)).json());
  } catch (e) { /* leave empty; UI falls back to the wanted model */ }
  if (rowReq[p] !== id) return; // a newer re-fetch for this row beat us back
  sel.innerHTML = "";
  const list = models.length ? models : (want ? [want] : []);
  list.forEach(m => {
    const o = document.createElement("option");
    o.value = o.textContent = m;
    if (m === want) o.selected = true;
    sel.appendChild(o);
  });
  sel.disabled = false; btn.disabled = false; btn.textContent = "\\u21bb re-fetch models";
}

function fmtElapsed(s) {
  const m = Math.floor(s / 60), r = Math.floor(s % 60);
  return m + ":" + String(r).padStart(2, "0");
}
function renderStats(done, verdict) {
  const el = fmtElapsed((Date.now() - started) / 1000);
  let s = `<span>files <b>${filesDone}${filesTotal ? "/" + filesTotal : ""}</b></span>`
        + `<span>fixes <b>${fixes}</b></span>`
        + `<span>LLM calls <b>${calls}</b></span>`
        + `<span><b>${el}</b></span>`;
  if (done && verdict) s = `<span class="verdict ${verdict.cls}">${verdict.text}</span> ` + s;
  statsEl.innerHTML = s;
}

function shortHost(h) {
  return h.replace("https://", "").replace("http://", "");
}

async function pollCalls() {
  // Nothing to show outside a run -- and no run_id to even ask about.
  if (!currentRunId) {
    $("#calls").textContent = "";
    $("#calls-list").hidden = true;
    $("#calls-list").innerHTML = "";
    return;
  }
  try {
    const { url } = tagUrl("/api/calls/" + currentRunId);
    const { active } = await (await fetch(url)).json();
    const badge = $("#calls"), list = $("#calls-list");
    if (!active.length) {
      badge.textContent = "";
      list.hidden = true;
      list.innerHTML = "";
      return;
    }
    badge.textContent = `\\u00b7 ${active.length} call${active.length === 1 ? "" : "s"} active`;
    list.hidden = false;
    list.innerHTML = active
      .sort((a, b) => b.elapsed - a.elapsed)
      .map(c => {
        const kind = `<span class="kind">${esc(c.kind)}</span>`;
        const where = c.path ? `${kind} ${esc(c.path.split("/").pop())}` : kind;
        const active_cls = openWindow && openWindow.kind === "call" && openWindow.id === c.id
          ? "active" : "";
        return `<div class="${active_cls}" data-call-id="${c.id}"><span>${where}</span>`
          + `<span class="t">${esc(shortHost(c.endpoint))} \\u00b7 ${c.elapsed}s</span></div>`;
      })
      .join("");
    list.querySelectorAll("[data-call-id]").forEach(el => {
      el.addEventListener("click", () => {
        const call = active.find(c => String(c.id) === el.dataset.callId);
        if (call) showCallWindow(call);
      });
    });
    // A live view left open for a call still running gets the same
    // treatment as the Files panel's open-window refresh below: show
    // the new text as it streams in, not what it had when first opened.
    if (openWindow && openWindow.kind === "call") {
      const still = active.find(c => c.id === openWindow.id);
      if (still) showCallWindow(still);
      else closeFileWindow();  // the call finished or vanished
    }
  } catch (e) { /* not worth surfacing */ }
}

// Live view of one in-flight call -- reuses the same modal the Files
// panel uses for static code/spec/plan content, distinguished by
// openWindow.kind === "call". Unlike those, no fetch: the calls bar's
// own poll already carries the full partial text.
function showCallWindow(call) {
  openWindow = { kind: "call", id: call.id };
  $("#file-modal").hidden = false;
  const name = call.path ? call.path.split("/").pop() : "";
  const label = name ? `${call.kind} ${name}` : call.kind;
  $("#file-modal-name").textContent = `${label} \\u2014 generating\\u2026`;
  $("#file-modal-body").textContent = call.partial || "(waiting for the first chunk\\u2026)";
}

async function refreshFiles(runId) {
  if (!runId) return;
  let files = [], hasPlan = false, phase = "planning";
  try {
    const { url } = tagUrl("/api/files/" + runId);
    ({ files, has_plan: hasPlan, phase } = await (await fetch(url)).json());
  } catch (e) { return; }
  updatePhaseStepper(phase, false);
  const panel = $("#files-panel");
  if (!files.length && !hasPlan) { panel.hidden = true; return; }
  panel.hidden = false;

  const planWrap = $("#plan-link-wrap");
  if (hasPlan) {
    planWrap.innerHTML = `<span class="fspec" style="text-decoration:underline;cursor:pointer">view plan</span>`;
    planWrap.firstChild.addEventListener("click", () => openFileWindow(runId, "plan", ""));
  } else {
    planWrap.innerHTML = "";
  }

  $("#graph-legend").hidden = false;
  $("#graph-hint").hidden = false;
  renderGraph(files, runId);

  // A window left open while its file is still being rewritten stays
  // live -- refetch it on the same poll instead of freezing on the
  // content it had when it was first opened. A "call" window (a live
  // in-flight generation) is refreshed by pollCalls() instead -- it has
  // no file on disk to refetch yet.
  if (
    openWindow && openWindow.kind !== "call" &&
    (openWindow.kind !== "code" || files.some(f => f.name === openWindow.name))
  ) {
    openFileWindow(runId, openWindow.kind, openWindow.name);
  }
}

const _STATUS_LABEL = {
  pending: "queued", building: "\\u2026", ok: "ok", failed: "failed",
  advisory: "advisory", flagged: "flagged", skipped: "skipped",
};

// Renders every planned file as a node in a layered dependency graph --
// layer = 1 + the deepest dependency's layer (0 for a file with none),
// so the graph reads top-to-bottom in build order and doubles as a
// to-do list: a "pending" node is one nobody has claimed yet. Plain SVG,
// laid out here rather than via flexbox-then-measure, so edges can be
// drawn as simple paths between known coordinates in one pass.
function renderGraph(files, runId) {
  const wrap = $("#graph-wrap");
  if (!files.length) {
    wrap.innerHTML = `<div class="graph-empty">waiting for the plan\\u2026</div>`;
    return;
  }

  const layerOf = {};
  files.forEach(f => {
    const deps = f.depends_on || [];
    layerOf[f.name] = deps.length ? 1 + Math.max(...deps.map(d => layerOf[d] ?? 0)) : 0;
  });
  const maxLayer = Math.max(0, ...files.map(f => layerOf[f.name]));
  const rows = [];
  for (let i = 0; i <= maxLayer; i++) rows.push([]);
  files.forEach(f => rows[layerOf[f.name]].push(f));

  const NODE_W = 132, NODE_H = 40, GAP_X = 18, GAP_Y = 38, PAD = 16;
  const maxCols = Math.max(...rows.map(r => r.length));
  const width = PAD * 2 + maxCols * NODE_W + Math.max(0, maxCols - 1) * GAP_X;
  const height = PAD * 2 + rows.length * NODE_H + Math.max(0, rows.length - 1) * GAP_Y;

  const pos = {};
  rows.forEach((row, r) => {
    const rowWidth = row.length * NODE_W + (row.length - 1) * GAP_X;
    const startX = (width - rowWidth) / 2;
    row.forEach((f, c) => {
      const x = startX + c * (NODE_W + GAP_X);
      const y = PAD + r * (NODE_H + GAP_Y);
      pos[f.name] = { x, cx: x + NODE_W / 2, top: y, bottom: y + NODE_H };
    });
  });

  // Reverse map ("needed by") purely for the tooltip -- the arrows
  // themselves already show it, but a native title works without a
  // precise hover and reads fine on a trackpad or a small graph alike.
  const neededBy = {};
  files.forEach(f => {
    (f.depends_on || []).forEach(dep => {
      (neededBy[dep] || (neededBy[dep] = [])).push(f.name);
    });
  });

  let edges = "";
  files.forEach(f => {
    (f.depends_on || []).forEach(dep => {
      const from = pos[dep], to = pos[f.name];
      if (!from || !to) return;
      const midY = (from.bottom + to.top) / 2;
      edges += `<path class="dep-edge" marker-end="url(#dep-arrow)" `
        + `data-from="${esc(dep)}" data-to="${esc(f.name)}" `
        + `d="M${from.cx},${from.bottom} C${from.cx},${midY} ${to.cx},${midY} ${to.cx},${to.top}" />`;
    });
  });

  let nodes = "";
  files.forEach(f => {
    const p = pos[f.name];
    const deps = f.depends_on || [];
    const dependents = neededBy[f.name] || [];
    let title = f.purpose ? `${f.name} \\u2014 ${f.purpose}` : f.name;
    title += deps.length ? `\\ndepends on: ${deps.join(", ")}` : "\\ndepends on: (nothing)";
    title += dependents.length ? `\\nneeded by: ${dependents.join(", ")}` : "\\nneeded by: (nothing yet)";
    title += `\\n${f.fixes} fix${f.fixes === 1 ? "" : "es"}`;
    const label = f.name.length > 16 ? f.name.slice(0, 14) + "\\u2026" : f.name;
    // Bottom-left, opposite corner from the spec/critic/fix dots -- only
    // shown once a file has actually needed a fix, so the common case
    // (0 fixes) stays uncluttered.
    const fixesBadge = f.fixes > 0
      ? `<text class="dep-node-fixes" x="${p.x + 6}" y="${p.top + NODE_H - 6}" text-anchor="start">`
        + `\\u21bb${f.fixes}<title>${f.fixes} fix${f.fixes === 1 ? "" : "es"}</title></text>`
      : "";
    // One corner, one dot at a time. Live activity always wins over the
    // static "has a spec" dot: has_spec stays true for the rest of the
    // file's life once logged, so without a priority order it would
    // permanently mask whatever's actually happening to the file right
    // now. "branching" comes first -- a file racing two concurrent
    // attempts is the most exceptional thing that can be happening to
    // it, worth flagging over the (also-true) fact that one side of
    // that race happens to be speccing/critiquing/fixing at this instant.
    const live = f.branching
      ? { cls: "dep-branching-dot", title: "branching to another endpoint\\u2026" }
      : f.speccing
      ? { cls: "dep-speccing-dot", title: "writing spec\\u2026" }
      : f.criticizing
      ? { cls: "dep-criticizing-dot", title: "critic reviewing\\u2026" }
      : f.fixing
      ? { cls: "dep-fixing-dot", title: "fixing\\u2026" }
      : null;
    const cornerDot = live
      ? `<circle class="${live.cls}" cx="${p.x + NODE_W - 8}" cy="${p.top + 8}" r="4">`
        + `<title>${live.title}</title></circle>`
      : f.has_spec
      ? `<circle class="dep-spec-dot" data-spec-name="${esc(f.name)}" `
        + `cx="${p.x + NODE_W - 8}" cy="${p.top + 8}" r="4"><title>view spec</title></circle>`
      : "";
    nodes += `<g class="dep-node-group" data-name="${esc(f.name)}">`
      + `<rect class="dep-node-rect ${f.status}" x="${p.x}" y="${p.top}" `
      + `width="${NODE_W}" height="${NODE_H}" rx="7"><title>${esc(title)}</title></rect>`
      + `<text class="dep-node-name" x="${p.cx}" y="${p.top + 17}" text-anchor="middle">${esc(label)}</text>`
      + `<text class="dep-node-status ${f.status}" x="${p.cx}" y="${p.top + 30}" text-anchor="middle">`
      + `${_STATUS_LABEL[f.status] || esc(f.status)}</text>${cornerDot}${fixesBadge}</g>`;
  });

  const defs = `<defs><marker id="dep-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" `
    + `markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">`
    + `<path class="dep-arrowhead" d="M0,0 L10,5 L0,10 z"></path></marker></defs>`;
  wrap.innerHTML = `<svg class="dep-svg" viewBox="0 0 ${width} ${height}" `
    + `width="${width}" height="${height}">${defs}${edges}${nodes}</svg>`;

  const edgeEls = wrap.querySelectorAll(".dep-edge");
  const nodeEls = wrap.querySelectorAll(".dep-node-group");
  const clearHighlight = () => {
    edgeEls.forEach(e => e.classList.remove("hl", "dim"));
    nodeEls.forEach(n => n.classList.remove("hl", "dim"));
  };
  nodeEls.forEach(el => {
    const name = el.dataset.name;
    el.addEventListener("mouseenter", () => {
      const related = new Set([name]);
      edgeEls.forEach(e => {
        if (e.dataset.from === name || e.dataset.to === name) {
          e.classList.add("hl");
          related.add(e.dataset.from);
          related.add(e.dataset.to);
        } else {
          e.classList.add("dim");
        }
      });
      nodeEls.forEach(n => n.classList.add(related.has(n.dataset.name) ? "hl" : "dim"));
    });
    el.addEventListener("mouseleave", clearHighlight);
    el.addEventListener("click", () => openFileWindow(runId, "code", name));
  });
  wrap.querySelectorAll(".dep-spec-dot").forEach(dot => {
    dot.addEventListener("click", e => {
      e.stopPropagation(); // don't also trigger the node's own code-view click
      openFileWindow(runId, "spec", dot.dataset.specName);
    });
  });
}

let openWindow = null; // { kind: "code" | "spec" | "plan", name }

async function openFileWindow(runId, kind, name) {
  openWindow = { kind, name };
  const modal = $("#file-modal");
  modal.hidden = false;
  $("#file-modal-name").textContent =
    kind === "plan" ? "Plan" : kind === "spec" ? name + " \\u2014 spec" : name;
  let url;
  if (kind === "plan") ({ url } = tagUrl("/api/plan/" + runId));
  else if (kind === "spec") ({ url } = tagUrl("/api/spec/" + runId + "/" + encodeURIComponent(name)));
  else ({ url } = tagUrl("/api/file/" + runId + "/" + encodeURIComponent(name)));
  try {
    const data = await (await fetch(url)).json();
    $("#file-modal-body").textContent = data.error ? "(not available)" : data.content;
  } catch (e) { $("#file-modal-body").textContent = "(could not reach the server)"; }
}

function closeFileWindow() {
  openWindow = null;
  $("#file-modal").hidden = true;
}
$("#file-modal-close").addEventListener("click", closeFileWindow);
$("#file-modal").addEventListener("click", e => {
  if (e.target.id === "file-modal") closeFileWindow();
});
document.addEventListener("keydown", e => {
  if (e.key === "Escape" && !$("#file-modal").hidden) closeFileWindow();
});

function onEvent(d) {
  if (d.line) { logEl.textContent += d.line + "\\n"; logEl.scrollTop = logEl.scrollHeight; }
  const CALLS = ["plan", "spec", "codegen", "fix", "integration_fix"];
  if (CALLS.includes(d.event)) calls++;
  if (d.event === "fix") fixes++;
  if (d.event === "plan") filesTotal = (d.fields.files || []).length;
  if (d.event === "verify" && d.fields.success) filesDone++;
  if (d.event === "run_paused") setPausedUI(true);
  if (d.event === "run_resumed") setPausedUI(false);
  renderStats(false);
}

function esc(s) {
  return String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

function whyFailed(s) {
  if (s.aborted) {
    return "The endpoint became unreachable mid-run"
      + (s.abort_reason ? ":\\n" + s.abort_reason : ".");
  }
  if (!s.finished) return "The run was killed before it reached a verdict.";
  if (s.succeeded) return "";
  if (s.stopped_early) return "Hit the iteration budget before every file was built.";
  const bad = (s.files || []).find(f => !f.success && !f.advisory);
  if (bad && bad.last_error) {
    return bad.path.split("/").pop() + " never passed verification:\\n" + bad.last_error;
  }
  if (s.integration && !s.integration.success) {
    return "Integration check (" + s.integration.stage + ") failed:\\n"
      + (s.integration.output || "(no output)");
  }
  return "One or more files could not be built.";
}

async function showSummary(runId) {
  const { url: sumUrl } = tagUrl("/api/summary/" + runId);
  const s = await (await fetch(sumUrl)).json();
  let verdict = { cls: "ok", text: "SUCCESS" };
  if (s.aborted) verdict = { cls: "bad", text: "ABORTED" };
  else if (!s.finished) verdict = { cls: "warn", text: "INCOMPLETE" };
  else if (!s.succeeded) verdict = s.stopped_early
      ? { cls: "warn", text: "STOPPED" } : { cls: "bad", text: "FAILED" };
  renderStats(true, verdict);

  let rows = (s.files || []).map(f => {
    const cls = f.success ? "s-ok" : (f.spec_flagged ? "s-flag" : (f.advisory ? "s-adv" : "s-bad"));
    const tag = f.success ? "ok" : (f.spec_flagged ? "flagged" : (f.advisory ? "advisory" : "FAILED"));
    const name = f.path.split("/").pop();
    const branchNote = f.branched
      ? ` <span class="s-branched" title="won by a branch to another endpoint">(branched)</span>`
      : "";
    return `<tr><td class="${cls}">${tag}</td><td>${esc(name)}</td>`
      + `<td>${f.attempts} fix${f.attempts === 1 ? "" : "es"}${branchNote}</td></tr>`;
  }).join("");
  if (s.integration) {
    const ic = s.integration.success ? "s-ok" : "s-bad";
    rows += `<tr><td class="${ic}">${s.integration.success ? "ok" : "FAILED"}</td><td>integration (${esc(s.integration.stage)})</td><td></td></tr>`;
  }
  const why = whyFailed(s);
  const reason = why ? `<pre class="reason">${esc(why)}</pre>` : "";
  let issuesHtml = "";
  if (s.cross_file_issues && s.cross_file_issues.length) {
    const issueRows = s.cross_file_issues.map(i => {
      const cls = i.confirmed ? "s-review-confirmed" : "s-review-unconfirmed";
      const tag = i.confirmed ? "confirmed" : "unconfirmed";
      return `<tr><td class="${cls}">${tag}</td><td>${esc(i.file)}</td><td>${esc(i.description)}</td></tr>`;
    }).join("");
    issuesHtml = `<div class="review-findings"><b>Whole-project review</b>`
      + `<table>${issueRows}</table></div>`;
  }
  $("#summary").innerHTML = (rows ? `<table>${rows}</table>` : "") + issuesHtml + reason;
  return s;
}

// Shared by both a fresh "Generate" click and resuming a run that was
// already in progress when the page loaded -- everything that resets
// the display for "a run is now live", with no network call of its own.
function beginTracking() {
  errEl.hidden = true;
  goBtn.disabled = true;
  $("#run-actions").hidden = false;
  stopBtn.disabled = false;
  setPausedUI(false);
  logEl.textContent = ""; $("#summary").innerHTML = "";
  $("#files-panel").hidden = true; $("#graph-wrap").innerHTML = "";
  $("#graph-legend").hidden = true; $("#graph-hint").hidden = true;
  $("#phase-stepper").hidden = false;
  updatePhaseStepper("planning", false);
  closeFileWindow();
  calls = fixes = filesDone = filesTotal = 0; started = Date.now();
  renderStats(false);
  tick = setInterval(() => { renderStats(false); pollCalls(); refreshFiles(currentRunId); }, 1000);
}

// Opens the SSE stream for `runId` and wires it up to the log/stats/
// summary. With no Last-Event-ID this always replays the run's log from
// the top, so resuming after a page reload rebuilds the same counters a
// tab that had been open the whole time would show.
function watchRun(runId) {
  currentRunId = runId;
  const finish = async () => {
    es.close(); clearInterval(tick);
    $("#run-actions").hidden = true;
    const s = await showSummary(runId);
    await refreshFiles(runId);
    if (s && s.aborted) {
      $("#phase-stepper").querySelectorAll(".phase-step.current").forEach(el => {
        el.classList.replace("current", "aborted");
      });
    }
    await pollCalls();  // clears the calls bar right away, not up to 3s late
    goBtn.disabled = false;
  };
  const { url: evUrl } = tagUrl("/api/events/" + runId);
  es = new EventSource(evUrl);
  es.onmessage = e => onEvent(JSON.parse(e.data));
  es.addEventListener("done", finish);
  es.onerror = async () => {
    // EventSource reconnects on its own; only wrap up if the run is
    // actually over -- a dropped connection is not a finished run.
    try {
      const { url } = tagUrl("/api/config");
      const c = await (await fetch(url)).json();
      if (c.state.run_id === runId && c.state.state === "running") return;
    } catch (e) { return; }
    finish();
  };
}

$("#go").addEventListener("click", async () => {
  const goal = $("#goal").value.trim();
  if (!goal) return;
  beginTracking();

  const body = {
    goal, model: $("#model").value, host: $("#host").value.trim(),
    multi_file: $("#multi").checked,
    endpoints: currentEndpoints(),
  };
  const { url: runUrl } = tagUrl("/api/run");
  let res;
  try {
    res = await fetch(runUrl, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (e) { return fail("could not reach the server"); }
  const data = await res.json();
  if (!res.ok) return fail(data.error || "run failed to start");
  watchRun(data.run_id);
});

stopBtn.addEventListener("click", async () => {
  stopBtn.disabled = true;
  try {
    const { url } = tagUrl("/api/run/cancel");
    await fetch(url, { method: "POST" });
  } catch (e) { /* the poll loop will notice the run is over either way */ }
});

function fail(msg) {
  clearInterval(tick); goBtn.disabled = false; $("#run-actions").hidden = true;
  errEl.textContent = msg; errEl.hidden = false;
}

loadConfig();
</script>
</body>
</html>
"""
