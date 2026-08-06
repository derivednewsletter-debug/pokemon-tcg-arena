"""Shared learning-store singleton.

``game.py`` (writes records) and ``learn.py`` (reads aggregates) both
import this module, so in local development they share one in-memory
store. On Vercel each function runs in its own process, but the store's
Vercel-KV backend keeps them consistent across processes and cold
starts when KV credentials are configured.
"""
from __future__ import annotations

from learning.store import Store

store = Store()
