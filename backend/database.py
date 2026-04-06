"""
database.py — JSON-backed store for debug logs.
Swap out for PostgreSQL by replacing load/save with SQLAlchemy calls.
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "logs.json"


def _load() -> list[dict]:
    """Load all logs from disk."""
    if not DB_PATH.exists():
        return []
    with open(DB_PATH) as f:
        return json.load(f)


def _save(logs: list[dict]) -> None:
    """Persist logs to disk atomically."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = DB_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(logs, f, indent=2, default=str)
    tmp.replace(DB_PATH)


def save_log(log: dict) -> str:
    """Insert a new debug log entry; returns its assigned ID."""
    logs = _load()
    log_id = str(uuid.uuid4())
    log["id"] = log_id
    log.setdefault("timestamp", datetime.utcnow().isoformat())
    logs.append(log)
    _save(logs)
    return log_id


def get_all_logs() -> list[dict]:
    """Return all logs, newest first."""
    return list(reversed(_load()))


def get_log(log_id: str) -> dict | None:
    """Retrieve a single log by ID."""
    for log in _load():
        if log.get("id") == log_id:
            return log
    return None


def clear_logs() -> int:
    """Delete all logs; returns count deleted."""
    logs = _load()
    count = len(logs)
    _save([])
    return count