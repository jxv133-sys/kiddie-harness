"""Per-run transcript logging.

Every prompt, response, and verifier result is appended to a JSONL file so
an 8B model's failure can be inspected after the fact -- without this,
debugging why a small model went wrong is guesswork.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import uuid
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

    @classmethod
    def create(cls, workspace_root: Path, run_id: str | None = None) -> Session:
        run_id = run_id or new_run_id()
        run_dir = workspace_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "log.jsonl"
        return cls(run_id=run_id, run_dir=run_dir, log_path=log_path)

    def log(self, event: str, **fields: Any) -> None:
        record = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
