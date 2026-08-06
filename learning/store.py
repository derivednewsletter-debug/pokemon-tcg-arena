"""Game-record storage for the learning loop.

Two backends:

* **Memory** (default, local dev): an in-process list. Survives for the
  lifetime of the server process only.
* **Vercel KV** (production): when ``KV_REST_API_URL`` /
  ``KV_REST_API_TOKEN`` are set, records are persisted in Vercel KV
  (key ``tcg_arena_records``) so learning survives cold starts and is
  shared across serverless instances.

The store is deliberately defensive: any KV failure degrades to memory
so the game never breaks because telemetry hiccuped.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
import urllib.error

KV_KEY = "tcg_arena_records"


class Store:
    def __init__(self):
        self._lock = threading.RLock()
        self._records: list[dict] = []
        self._kv_url = os.environ.get("KV_REST_API_URL", "").strip()
        self._kv_token = os.environ.get("KV_REST_API_TOKEN", "").strip()
        if self._kv_url and self._kv_token:
            self._load_from_kv()

    # ------------------------------------------------------------------
    # persistence helpers
    # ------------------------------------------------------------------
    def _kv_request(self, method: str, body: str | None = None) -> str | None:
        url = self._kv_url.rstrip("/") + "/" + KV_KEY
        req = urllib.request.Request(url, method=method)
        req.add_header("Authorization", "Bearer " + self._kv_token)
        if body is not None:
            req.add_header("Content-Type", "application/json")
            req.data = body.encode("utf-8")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw = resp.read().decode("utf-8", "replace")
                return raw if raw else None
        except Exception:
            return None

    def _load_from_kv(self):
        raw = self._kv_request("GET")
        if raw is None:
            return
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                self._records = data
        except Exception:
            pass

    def _save_to_kv(self):
        try:
            self._kv_request("PUT", json.dumps(self._records))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def add(self, record: dict) -> None:
        with self._lock:
            record = dict(record)
            record.setdefault("ts", time.time())
            self._records.append(record)
            # keep memory and KV in sync (records are small; no hard cap)
            if self._kv_url:
                self._save_to_kv()

    def records(self) -> list[dict]:
        with self._lock:
            if self._kv_url:
                # pick up games recorded by other serverless instances
                self._load_from_kv()
            return list(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records = []
            if self._kv_url:
                self._kv_request("DELETE")
