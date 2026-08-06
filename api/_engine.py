"""Per-game wrapper around the official competition engine.

The stock ``cg.game`` module keeps a single *global* battle pointer,
which only supports one game per process. The web arena needs many
concurrent games, so this module drives the same C exports directly and
keeps one battle pointer per :class:`Game` instance.

``libcg.so`` (x86-64 Linux) is what Vercel loads; locally the correct
binary is picked automatically by ``cg.sim`` based on the platform.
"""
from __future__ import annotations

import ctypes
import json
import threading

from cg.sim import lib  # noqa: F401  (engine init + restype setup)


class EngineError(Exception):
    """Raised when the engine rejects a move (should never happen for
    validated picks — a bug if it does)."""


class Game:
    """One live battle backed by its own engine battle pointer."""

    _select_lock = threading.Lock()

    def __init__(self, deck0: list[int], deck1: list[int]):
        if len(deck0) != 60 or len(deck1) != 60:
            raise EngineError("decks must have exactly 60 cards")
        cards = list(deck0) + list(deck1)
        arg = (ctypes.c_int * len(cards))(*cards)
        sd = lib.BattleStart(arg)
        if not sd.battlePtr:
            raise EngineError(
                f"battle_start failed (player {sd.errorPlayer}, type {sd.errorType})")
        self.ptr = sd.battlePtr
        self.obs = self._data()
        self.finished = False

    # ------------------------------------------------------------------
    def _data(self) -> dict:
        sd = lib.GetBattleData(self.ptr)
        obs = json.loads(sd.json.decode())
        obs["search_begin_input"] = ctypes.string_at(sd.data, sd.count).decode("ascii")
        return obs

    def select(self, picks: list[int]) -> dict:
        """Apply option indices and return the next observation."""
        picks = [int(p) for p in picks]
        arg = (ctypes.c_int * len(picks))(*picks)
        with self._select_lock:
            err = lib.Select(self.ptr, arg, len(picks))
            if err != 0:
                raise EngineError(f"select error {err}")
            self.obs = self._data()
        return self.obs

    def finish(self) -> None:
        if self.finished:
            return
        self.finished = True
        try:
            if self.ptr:
                lib.BattleFinish(self.ptr)
        except Exception:
            pass
        self.ptr = None
