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
from .config import DEFAULT_CONFIG_PATH, Config
from .llm_client import OllamaClient
from .orchestrator import MultiFileLoop, SingleFileLoop
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
)
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

    @property
    def config(self) -> Config:
        with self._lock:
            return self._config

    def update_config(self, **overrides) -> Config:
        """Apply settings-screen overrides for every run started from now
        on. Never touches a run already in progress -- its own Config was
        captured at `start()` time."""
        with self._lock:
            self._config = self._config.with_overrides(**overrides)
            return self._config

    def status(self) -> dict:
        return {"state": self._state, "run_id": self._run_id}

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
            self._state = "running"
            self._cancel_event = threading.Event()
            self._thread = threading.Thread(
                target=self._run,
                args=(config, session, goal, multi_file, eps, self._cancel_event),
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
    ) -> None:
        run_id = session.run_id
        try:
            pool = [
                self._client_factory(
                    e["host"], e.get("model") or config.model, config.timeout_seconds
                )
                for e in endpoints
            ]
            if multi_file:
                MultiFileLoop(
                    pool[0], config, session, pool_clients=pool, cancel_event=cancel_event
                ).run(goal)
            else:
                SingleFileLoop(pool[0], config, session, cancel_event=cancel_event).run(goal)
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
                        "state": self._runs.status(),
                    }
                )
            elif path == "/api/models":
                host = parse_qs(parsed.query).get("host", [self._config.ollama_host])[0]
                self._send_json({"models": available_models(host)})
            elif path == "/api/requests":
                self._send_json({"active": self._tracker.active(exclude=req_id)})
            elif path == "/api/settings":
                c = self._config
                self._send_json({k: getattr(c, k) for k in _SETTINGS_KEYS})
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
        updates: dict[str, float | int] = {}
        bad_keys: list[str] = []
        for key in _SETTINGS_KEYS:
            if key not in body:
                continue
            try:
                updates[key] = float(body[key]) if key == "temperature" else int(body[key])
            except (TypeError, ValueError):
                bad_keys.append(key)
        if bad_keys:
            self._send_json({"error": f"invalid value(s) for: {', '.join(bad_keys)}"}, status=400)
            return
        new_config = self._runs.update_config(**updates)
        values = {k: getattr(new_config, k) for k in _SETTINGS_KEYS}
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

    def _files_response(self, run_id: str) -> dict:
        """The `.py` files on disk for a run, each tagged with its latest
        known verify outcome from the log (or "pending" while it hasn't
        been verified yet, e.g. mid-generation) and whether a spec was
        written for it -- single-file runs never have one. `has_plan`
        says whether the run went through the planner at all (multi-file
        only), so the page knows whether to offer a Plan window."""
        run_dir = self._runs.log_path(run_id).parent
        if not run_dir.is_dir():
            return {"files": [], "has_plan": False}
        status_by_name: dict[str, str] = {}
        spec_names: set[str] = set()
        endpoint_by_name: dict[str, str] = {}
        has_plan = False
        log_path = run_dir / "log.jsonl"
        if log_path.exists():
            for f in summary.load_run_summary(log_path).files:
                name = Path(f.path).name
                if f.advisory:
                    status_by_name[name] = "advisory"
                else:
                    status_by_name[name] = "ok" if f.success else "failed"
            for record in self._iter_log_events(run_id):
                event = record.get("event")
                if event == "plan":
                    has_plan = True
                elif event == "spec":
                    spec_names.add(Path(record.get("path", "")).name)
                elif event == "codegen" and record.get("endpoint"):
                    # The same worker (client/endpoint) owns a file for
                    # its whole build, so this is set once and stays --
                    # last-write-wins is only relevant if it ever isn't.
                    endpoint_by_name[Path(record.get("path", "")).name] = record["endpoint"]
        files = []
        for p in sorted(run_dir.glob("*.py")):
            files.append(
                {
                    "name": p.name,
                    "status": status_by_name.get(p.name, "pending"),
                    "size": p.stat().st_size,
                    "has_spec": p.name in spec_names,
                    "endpoint": endpoint_by_name.get(p.name, ""),
                }
            )
        return {"files": files, "has_plan": has_plan}

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
          --accent:#3a6adf; --ok:#1f9d55; --bad:#d1453b; --warn:#c47f17; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#eaeaea; --bg:#181818; --muted:#8a8a8e; --line:#333;
            --accent:#6f9bff; --ok:#57c97f; --bad:#ff6b60; --warn:#e0a24a; }
  }
  * { box-sizing:border-box; }
  [hidden] { display:none !important; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  main { max-width:680px; margin:0 auto; padding:44px 20px 80px; }
  h1 { font-size:19px; font-weight:600; letter-spacing:-.01em; margin:0 0 24px;
       display:flex; align-items:baseline; gap:0; }
  h1 span { color:var(--muted); font-weight:400; }
  h1 .reqs { font-size:11px; }
  .gear { margin-left:auto; background:none; border:0; padding:0 0 0 6px; width:auto;
          color:var(--muted); font-size:14px; cursor:pointer; }
  .gear:hover { color:var(--accent); }
  .reqs-list, .settings-panel { margin:2px 0 20px; padding:7px 10px; border:1px solid var(--line);
               border-radius:8px; background:color-mix(in srgb, var(--fg) 4%, transparent); }
  .reqs-list { font:11px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--muted); }
  .reqs-list div { display:flex; justify-content:space-between; gap:10px; }
  .reqs-list .t { color:var(--fg); opacity:.7; flex:0 0 auto; }
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
  .err { color:var(--bad); font-size:13px; margin-top:10px; }
  pre.reason { margin:12px 0 0; padding:12px 14px; border:1px solid var(--bad);
               border-radius:8px; background:color-mix(in srgb, var(--bad) 8%, transparent);
               color:var(--fg); font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
               white-space:pre-wrap; word-break:break-word; max-height:280px; overflow:auto; }
</style>
</head>
<body>
<main>
  <h1>kiddie-harness <span>&mdash; generate a project</span> <span id="reqs" class="reqs"></span>
    <button type="button" id="settings-btn" class="gear" title="settings">&#9881;</button>
  </h1>
  <div id="reqs-list" class="reqs-list" hidden></div>
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
    <div class="settings-actions">
      <button type="button" id="settings-save">Save</button>
      <span class="settings-msg" id="settings-msg"></span>
    </div>
  </div>

  <label for="model">Model</label>
  <div class="row">
    <div><select id="model"></select></div>
    <div><input type="text" id="host" placeholder="http://localhost:11434"></div>
  </div>
  <button type="button" id="refresh" class="link">&#8635; re-fetch models</button>

  <div id="ep2" hidden>
    <label for="model2">Second endpoint <span style="text-transform:none;letter-spacing:0">(parallel, multi-file only)</span></label>
    <div class="row">
      <div><select id="model2"></select></div>
      <div><input type="text" id="host2" placeholder="http://localhost:11434"></div>
    </div>
    <button type="button" id="refresh2" class="link">&#8635; re-fetch models</button>
  </div>
  <button type="button" id="add-ep" class="link">+ second endpoint</button>

  <label for="goal">Goal</label>
  <textarea id="goal" placeholder="a command-line to-do list with add / list / done subcommands"></textarea>

  <div class="check">
    <input type="checkbox" id="multi" checked>
    <label for="multi" style="margin:0;text-transform:none;letter-spacing:0;font-size:13px">multi-file project</label>
  </div>

  <button id="go">Generate</button>
  <button type="button" id="stop" hidden>Stop</button>
  <div class="err" id="err" hidden></div>

  <div class="stats" id="stats"></div>
  <pre id="log"></pre>
  <div class="files-panel" id="files-panel" hidden>
    <label>Files</label>
    <div class="file-list" id="file-list"></div>
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
const stopBtn = $("#stop");
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
const rowReq = { "": 0, "2": 0 };

async function loadConfig() {
  const { url } = tagUrl("/api/config");
  const c = await (await fetch(url)).json();
  $("#host").value = c.host;
  defaultModel = c.model;
  await refreshRow("");
  $("#host").addEventListener("change", () => refreshRow(""));
  $("#refresh").addEventListener("click", () => refreshRow(""));
  $("#host2").addEventListener("change", () => refreshRow("2"));
  $("#refresh2").addEventListener("click", () => refreshRow("2"));
  $("#add-ep").addEventListener("click", () => {
    $("#ep2").hidden = false;
    $("#add-ep").hidden = true;
    if (!$("#host2").value) $("#host2").value = c.host;
    refreshRow("2");
  });
  pollActive();
  setInterval(pollActive, 3000);
  loadSettings();

  // A run started before this page load (or before a reload) is still
  // going -- pick its stream back up instead of showing an idle form
  // that silently rejects "Generate" with "already in progress".
  if (c.state && c.state.state === "running" && c.state.run_id) {
    beginTracking();
    watchRun(c.state.run_id);
  }
}

const SETTINGS_KEYS = [
  "temperature", "max_tokens", "max_tokens_ceiling",
  "max_fix_attempts", "max_total_iterations", "timeout_seconds",
];

async function loadSettings() {
  try {
    const { url } = tagUrl("/api/settings");
    const s = await (await fetch(url)).json();
    SETTINGS_KEYS.forEach(k => { if (k in s) $("#s-" + k).value = s[k]; });
  } catch (e) { /* settings panel just stays blank */ }
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
    msg.textContent = "saved \\u2014 applies to the next run";
    setTimeout(() => { if (msg.textContent.startsWith("saved")) msg.textContent = ""; }, 3000);
  } catch (e) { msg.textContent = "could not reach the server"; }
});
async function refreshRow(p) {
  const host = $("#host" + p).value.trim();
  const sel = $("#model" + p);
  const want = sel.value || defaultModel;
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

function shortPath(p) {
  try {
    const u = new URL(p, location.origin);
    u.searchParams.delete("_r");
    return u.pathname + u.search;
  } catch (e) { return p; }
}

function shortHost(h) {
  return h.replace("https://", "").replace("http://", "");
}

async function pollActive() {
  try {
    const { url } = tagUrl("/api/requests");
    const { active } = await (await fetch(url)).json();
    const badge = $("#reqs"), list = $("#reqs-list");
    if (!active.length) {
      badge.textContent = "";
      list.hidden = true;
      list.innerHTML = "";
      return;
    }
    badge.textContent = `\\u00b7 ${active.length} request${active.length === 1 ? "" : "s"} active`;
    list.hidden = false;
    list.innerHTML = active
      .sort((a, b) => b.elapsed - a.elapsed)
      .map(r => `<div><span>${esc(r.method)} ${esc(shortPath(r.path))}</span><span class="t">${r.elapsed}s</span></div>`)
      .join("");
  } catch (e) { /* not worth surfacing */ }
}

async function refreshFiles(runId) {
  if (!runId) return;
  let files = [], hasPlan = false;
  try {
    const { url } = tagUrl("/api/files/" + runId);
    ({ files, has_plan: hasPlan } = await (await fetch(url)).json());
  } catch (e) { return; }
  const panel = $("#files-panel"), list = $("#file-list");
  if (!files.length && !hasPlan) { panel.hidden = true; return; }
  panel.hidden = false;

  // Only worth a badge once there's actually more than one endpoint in
  // play for this run -- the common single-endpoint case would just see
  // the same host repeated on every row.
  const showEndpoints = new Set(files.map(f => f.endpoint).filter(Boolean)).size > 1;

  let html = hasPlan
    ? `<div data-kind="plan"><span class="file-dot"></span><span class="fname">Plan</span></div>`
    : "";
  html += files.map(f => {
    const spec = f.has_spec
      ? `<span class="fspec" data-kind="spec" data-name="${esc(f.name)}">spec</span>`
      : "";
    const ep = showEndpoints && f.endpoint
      ? `<span class="fep" title="${esc(f.endpoint)}">${esc(shortHost(f.endpoint))}</span>`
      : "";
    return `<div data-kind="code" data-name="${esc(f.name)}">`
      + `<span class="file-dot ${f.status}"></span>`
      + `<span class="fname">${esc(f.name)}</span>${spec}${ep}`
      + `<span class="fsize">${f.size}b</span></div>`;
  }).join("");
  list.innerHTML = html;

  list.querySelectorAll("[data-kind='plan'], [data-kind='code']").forEach(el => {
    el.addEventListener("click", () => openFileWindow(runId, el.dataset.kind, el.dataset.name));
  });
  list.querySelectorAll("[data-kind='spec']").forEach(el => {
    el.addEventListener("click", e => {
      e.stopPropagation();
      openFileWindow(runId, "spec", el.dataset.name);
    });
  });

  // A window left open while its file is still being rewritten stays
  // live -- refetch it on the same poll instead of freezing on the
  // content it had when it was first opened.
  if (openWindow && (openWindow.kind !== "code" || files.some(f => f.name === openWindow.name))) {
    openFileWindow(runId, openWindow.kind, openWindow.name);
  }
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
    const cls = f.success ? "s-ok" : (f.advisory ? "s-adv" : "s-bad");
    const tag = f.success ? "ok" : (f.advisory ? "advisory" : "FAILED");
    const name = f.path.split("/").pop();
    return `<tr><td class="${cls}">${tag}</td><td>${esc(name)}</td><td>${f.attempts} fix${f.attempts === 1 ? "" : "es"}</td></tr>`;
  }).join("");
  if (s.integration) {
    const ic = s.integration.success ? "s-ok" : "s-bad";
    rows += `<tr><td class="${ic}">${s.integration.success ? "ok" : "FAILED"}</td><td>integration (${esc(s.integration.stage)})</td><td></td></tr>`;
  }
  const why = whyFailed(s);
  const reason = why ? `<pre class="reason">${esc(why)}</pre>` : "";
  $("#summary").innerHTML = (rows ? `<table>${rows}</table>` : "") + reason;
}

// Shared by both a fresh "Generate" click and resuming a run that was
// already in progress when the page loaded -- everything that resets
// the display for "a run is now live", with no network call of its own.
function beginTracking() {
  errEl.hidden = true;
  goBtn.disabled = true;
  stopBtn.hidden = false; stopBtn.disabled = false;
  logEl.textContent = ""; $("#summary").innerHTML = "";
  $("#files-panel").hidden = true; $("#file-list").innerHTML = "";
  closeFileWindow();
  calls = fixes = filesDone = filesTotal = 0; started = Date.now();
  renderStats(false);
  tick = setInterval(() => { renderStats(false); pollActive(); refreshFiles(currentRunId); }, 1000);
}

// Opens the SSE stream for `runId` and wires it up to the log/stats/
// summary. With no Last-Event-ID this always replays the run's log from
// the top, so resuming after a page reload rebuilds the same counters a
// tab that had been open the whole time would show.
function watchRun(runId) {
  currentRunId = runId;
  const finish = async () => {
    es.close(); clearInterval(tick);
    stopBtn.hidden = true;
    await showSummary(runId);
    await refreshFiles(runId);
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
  };
  if (!$("#ep2").hidden && $("#host2").value.trim()) {
    body.endpoints = [
      { host: body.host, model: body.model },
      { host: $("#host2").value.trim(), model: $("#model2").value },
    ];
  }
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
  clearInterval(tick); goBtn.disabled = false; stopBtn.hidden = true;
  errEl.textContent = msg; errEl.hidden = false;
}

loadConfig();
</script>
</body>
</html>
"""
