"""Per-run transcript logging.

Every prompt, response, and verifier result is appended to a JSONL file so
an 8B model's failure can be inspected after the fact -- without this,
debugging why a small model went wrong is guesswork.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import itertools
import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any


def new_run_id() -> str:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


@dataclasses.dataclass
class Session:
    run_id: str
    run_dir: Path
    log_path: Path
    on_event: Callable[[str, dict], None] | None = None
    # The multi-file loop can drive several endpoints in parallel threads;
    # one lock keeps each event's file write + on_event callback atomic.
    _lock: threading.Lock = dataclasses.field(
        default_factory=threading.Lock, compare=False, repr=False
    )
    # In-flight LLM calls (plan/spec/codegen/fix/critic/...), so a live
    # viewer can show what's actually happening right now -- which step,
    # which file, which endpoint, for how long -- not just that *a*
    # request to the GUI is open. Separate lock from `_lock`: this is
    # touched far more often (every call, not every log line) and never
    # needs to be atomic with a log write.
    _calls_lock: threading.Lock = dataclasses.field(
        default_factory=threading.Lock, compare=False, repr=False
    )
    _active_calls: dict[int, dict] = dataclasses.field(
        default_factory=dict, compare=False, repr=False
    )
    _call_ids: Iterator[int] = dataclasses.field(
        default_factory=itertools.count, compare=False, repr=False
    )

    @classmethod
    def create(
        cls,
        workspace_root: Path,
        run_id: str | None = None,
        on_event: Callable[[str, dict], None] | None = None,
    ) -> Session:
        run_id = run_id or new_run_id()
        run_dir = workspace_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "log.jsonl"
        return cls(run_id=run_id, run_dir=run_dir, log_path=log_path, on_event=on_event)

    def log(self, event: str, **fields: Any) -> None:
        record = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        line = json.dumps(record) + "\n"
        with self._lock:
            with self.log_path.open("a") as f:
                f.write(line)
            if self.on_event is not None:
                try:
                    self.on_event(event, fields)
                except Exception:  # noqa: S110, BLE001 -- a broken reporter must never take down a run
                    pass

    @contextlib.contextmanager
    def track_call(self, kind: str, path: str, endpoint: str) -> Iterator[None]:
        """Marks one LLM call (`kind`: "plan"/"spec"/"codegen"/"fix"/
        "critic"/"integration_fix") as in-flight for the duration of the
        `with` block. `active_calls()` reads this back -- it's how the GUI
        shows what's actually happening right now, not just that some
        request to the GUI itself is open."""
        call_id = next(self._call_ids)
        with self._calls_lock:
            self._active_calls[call_id] = {
                "kind": kind,
                "path": path,
                "endpoint": endpoint,
                "started": time.monotonic(),
            }
        try:
            yield
        finally:
            with self._calls_lock:
                self._active_calls.pop(call_id, None)

    def active_calls(self) -> list[dict]:
        """Snapshot of in-flight LLM calls, most recently started last."""
        with self._calls_lock:
            now = time.monotonic()
            return [
                {
                    "kind": v["kind"],
                    "path": v["path"],
                    "endpoint": v["endpoint"],
                    "elapsed": round(now - v["started"], 1),
                }
                for v in self._active_calls.values()
            ]
