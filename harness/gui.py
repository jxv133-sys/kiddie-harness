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
from collections.abc import Iterator
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

from . import progress, summary
from .config import Config
from .llm_client import OllamaClient
from .orchestrator import MultiFileLoop, SingleFileLoop
from .session import Session


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

    def status(self) -> dict:
        return {"state": self._state, "run_id": self._run_id}

    def start(self, *, goal: str, model: str, host: str, multi_file: bool) -> str:
        with self._lock:
            if self._state == "running":
                raise RuntimeError("a run is already in progress")
            config = self._config.with_overrides(model=model, host=host)
            session = Session.create(config.workspace_root)
            self._run_id = session.run_id
            self._state = "running"
            self._thread = threading.Thread(
                target=self._run,
                args=(config, session, goal, multi_file),
                daemon=True,
            )
            self._thread.start()
            return session.run_id

    def log_path(self, run_id: str) -> Path:
        return self._config.workspace_root / run_id / "log.jsonl"

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self, config: Config, session: Session, goal: str, multi_file: bool) -> None:
        try:
            client = self._client_factory(
                config.ollama_host, config.model, config.timeout_seconds
            )
            loop_cls = MultiFileLoop if multi_file else SingleFileLoop
            loop_cls(client, config, session).run(goal)
        except Exception as exc:  # noqa: BLE001 -- a GUI run must never crash silently
            try:
                session.log("run_aborted", reason=f"{type(exc).__name__}: {exc}")
                session.log("run_result", success=False)
            except Exception:  # noqa: BLE001, S110
                pass
        finally:
            self._state = "done"


def stream_events(
    log_path: Path, *, poll_interval: float = 0.4, idle_timeout: float = 600.0
) -> Iterator[str]:
    """Tail a run's log.jsonl and yield Server-Sent-Event chunks.

    Each `data:` frame is `{"line": <formatted or null>, "event": <raw>,
    "fields": {...}}`; the stream ends with an `event: done` frame once a
    `run_result` is logged (or after `idle_timeout` with no new lines).
    """
    seen = 0
    last_activity = time.monotonic()
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
            yield f"data: {payload}\n\n"
            if event == "run_result":
                yield "event: done\ndata: {}\n\n"
                return

        if emitted:
            last_activity = time.monotonic()
        elif time.monotonic() - last_activity > idle_timeout:
            yield "event: done\ndata: {}\n\n"
            return
        else:
            time.sleep(poll_interval)


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
        return self.server.config  # type: ignore[attr-defined]

    @property
    def _runs(self) -> RunManager:
        return self.server.run_manager  # type: ignore[attr-defined]

    # --- routes ----------------------------------------------------------
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

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
        elif path.startswith("/api/summary/"):
            run_id = path.rsplit("/", 1)[-1]
            log_path = self._runs.log_path(run_id)
            if not log_path.exists():
                self._send_json({"error": "no such run"}, status=404)
            else:
                self._send_json(asdict(summary.load_run_summary(log_path)))
        elif path.startswith("/api/events/"):
            self._serve_events(path.rsplit("/", 1)[-1])
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/run":
            self._send_json({"error": "not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON"}, status=400)
            return
        goal = (body.get("goal") or "").strip()
        if not goal:
            self._send_json({"error": "goal is required"}, status=400)
            return
        try:
            run_id = self._runs.start(
                goal=goal,
                model=body.get("model") or self._config.model,
                host=body.get("host") or self._config.ollama_host,
                multi_file=bool(body.get("multi_file", True)),
            )
        except RuntimeError as exc:
            self._send_json({"error": str(exc)}, status=409)
            return
        self._send_json({"run_id": run_id})

    def _serve_events(self, run_id: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for chunk in stream_events(self._runs.log_path(run_id)):
                self.wfile.write(chunk.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ValueError):
            pass


def build_server(config: Config, *, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), _Handler)
    server.config = config  # type: ignore[attr-defined]
    server.run_manager = RunManager(config)  # type: ignore[attr-defined]
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
  body { margin:0; background:var(--bg); color:var(--fg);
         font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  main { max-width:640px; margin:0 auto; padding:44px 20px 80px; }
  h1 { font-size:19px; font-weight:600; letter-spacing:-.01em; margin:0 0 24px; }
  h1 span { color:var(--muted); font-weight:400; }
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
  button.link { margin-top:8px; width:auto; padding:2px 0; background:none; color:var(--muted);
                font-weight:400; font-size:12px; }
  button.link:hover:not(:disabled) { color:var(--accent); }
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
  table { width:100%; border-collapse:collapse; margin-top:14px; font-size:13px; }
  td { padding:5px 8px; border-top:1px solid var(--line); }
  td.s-ok { color:var(--ok); } td.s-bad { color:var(--bad); } td.s-adv { color:var(--warn); }
  .err { color:var(--bad); font-size:13px; margin-top:10px; }
</style>
</head>
<body>
<main>
  <h1>kiddie-harness <span>&mdash; generate a project</span></h1>

  <label for="model">Model</label>
  <div class="row">
    <div><select id="model"></select></div>
    <div><input type="text" id="host" placeholder="http://localhost:11434"></div>
  </div>
  <button type="button" id="refresh" class="link">&#8635; re-fetch models</button>

  <label for="goal">Goal</label>
  <textarea id="goal" placeholder="a command-line to-do list with add / list / done subcommands"></textarea>

  <div class="check">
    <input type="checkbox" id="multi" checked>
    <label for="multi" style="margin:0;text-transform:none;letter-spacing:0;font-size:13px">multi-file project</label>
  </div>

  <button id="go">Generate</button>
  <div class="err" id="err" hidden></div>

  <div class="stats" id="stats"></div>
  <pre id="log"></pre>
  <div id="summary"></div>
</main>

<script>
const $ = s => document.querySelector(s);
const logEl = $("#log"), statsEl = $("#stats"), goBtn = $("#go"), errEl = $("#err");
let started = 0, calls = 0, fixes = 0, filesDone = 0, filesTotal = 0, tick = null, es = null;
let defaultModel = "";

async function loadConfig() {
  const c = await (await fetch("/api/config")).json();
  $("#host").value = c.host;
  defaultModel = c.model;
  await loadModels();
  $("#host").addEventListener("change", loadModels);
  $("#refresh").addEventListener("click", loadModels);
}
async function loadModels() {
  const host = $("#host").value.trim();
  const sel = $("#model");
  const want = sel.value || defaultModel;
  const btn = $("#refresh");
  sel.disabled = true; btn.disabled = true; btn.textContent = "\\u21bb fetching\\u2026";
  let models = [];
  try {
    ({ models } = await (await fetch("/api/models?host=" + encodeURIComponent(host))).json());
  } catch (e) { /* leave empty; UI falls back to the wanted model */ }
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

function onEvent(d) {
  if (d.line) { logEl.textContent += d.line + "\\n"; logEl.scrollTop = logEl.scrollHeight; }
  const CALLS = ["plan", "spec", "codegen", "fix", "integration_fix"];
  if (CALLS.includes(d.event)) calls++;
  if (d.event === "fix") fixes++;
  if (d.event === "plan") filesTotal = (d.fields.files || []).length;
  if (d.event === "verify" && d.fields.success) filesDone++;
  renderStats(false);
}

async function showSummary(runId) {
  const s = await (await fetch("/api/summary/" + runId)).json();
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
    return `<tr><td class="${cls}">${tag}</td><td>${name}</td><td>${f.attempts} fix${f.attempts === 1 ? "" : "es"}</td></tr>`;
  }).join("");
  if (s.integration) {
    const ic = s.integration.success ? "s-ok" : "s-bad";
    rows += `<tr><td class="${ic}">${s.integration.success ? "ok" : "FAILED"}</td><td>integration (${s.integration.stage})</td><td></td></tr>`;
  }
  $("#summary").innerHTML = rows ? `<table>${rows}</table>` : "";
}

$("#go").addEventListener("click", async () => {
  const goal = $("#goal").value.trim();
  if (!goal) return;
  errEl.hidden = true;
  goBtn.disabled = true;
  logEl.textContent = ""; $("#summary").innerHTML = "";
  calls = fixes = filesDone = filesTotal = 0; started = Date.now();
  renderStats(false);
  tick = setInterval(() => renderStats(false), 1000);

  let res;
  try {
    res = await fetch("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ goal, model: $("#model").value, host: $("#host").value.trim(), multi_file: $("#multi").checked }),
    });
  } catch (e) { return fail("could not reach the server"); }
  const data = await res.json();
  if (!res.ok) return fail(data.error || "run failed to start");

  es = new EventSource("/api/events/" + data.run_id);
  es.onmessage = e => onEvent(JSON.parse(e.data));
  es.addEventListener("done", async () => {
    es.close(); clearInterval(tick);
    await showSummary(data.run_id);
    goBtn.disabled = false;
  });
  es.onerror = () => { es.close(); clearInterval(tick); goBtn.disabled = false; };
});

function fail(msg) {
  clearInterval(tick); goBtn.disabled = false;
  errEl.textContent = msg; errEl.hidden = false;
}

loadConfig();
</script>
</body>
</html>
"""
