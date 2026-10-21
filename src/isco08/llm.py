from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ChatResult:
    text: str
    usage: dict = field(default_factory=dict)
    cached: bool = False
    dry_run: bool = False
    model: str = ""
    latency_s: float = 0.0
    reasoning_content: str = ""
    finish_reason: str = ""
    timings: dict = field(default_factory=dict)


class UsageTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.api_calls = 0
        self.cache_hits = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.latency_s = 0.0

    def add(
        self,
        usage: dict | None,
        cached: bool = False,
        latency_s: float = 0.0,
    ) -> None:
        usage = usage or {}
        with self._lock:
            if cached:
                self.cache_hits += 1
                return
            self.api_calls += 1
            self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            details = usage.get("completion_tokens_details") or {}
            if isinstance(details, dict):
                self.reasoning_tokens += int(details.get("reasoning_tokens", 0) or 0)
            self.latency_s += float(latency_s)

    def summary(self, model: str | None = None, dry_run: bool = False) -> dict:
        with self._lock:
            return {
                "model": model,
                "dry_run": dry_run,
                "api_calls": self.api_calls,
                "cache_hits": self.cache_hits,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "total_latency_s": round(self.latency_s, 3),
            }


class ResponseCache:
    """Cache SQLite de respuestas para reanudar sin repetir llamadas."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _init_db(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    key TEXT PRIMARY KEY,
                    response TEXT NOT NULL,
                    usage TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def get(self, key: str) -> tuple[str, dict] | None:
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    "SELECT response, usage FROM responses WHERE key = ?", (key,)
                ).fetchone()
            finally:
                connection.close()
        if row is None:
            return None
        return str(row[0]), json.loads(row[1]) if row[1] else {}

    def set(self, key: str, response: str, usage: dict) -> None:
        with self._lock:
            connection = self._connect()
            try:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO responses (key, response, usage, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        key,
                        response,
                        json.dumps(usage, ensure_ascii=False),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                connection.commit()
            finally:
                connection.close()

    def count(self) -> int:
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute("SELECT COUNT(*) FROM responses").fetchone()
            finally:
                connection.close()
        return int(row[0]) if row else 0
