"""One game session: engine + human policy + AI agent + learning record.

Flow
----
``/api/game/new`` creates a :class:`GameSession` and calls
:meth:`advance`, which plays every *automatic* selection (the AI's
turns, the human's nested selections like search/discard/retreat-cost)
through the real engine and stops at the first selection the human must
make (MAIN, setup picks, go-first). ``/api/game/act`` feeds the human's
picks in and advances again until the next human choice — or the end of
the game, at which point a learning record is stored.

The AI agent is built with the learned ``opp_profile`` so its lookahead
models the opponent using aggregated human behaviour.
"""
from __future__ import annotations

import time
import uuid

from cg.api import SelectContext, SelectType
from agent import Agent
from learning.profiles import build_profile
from learning.store import Store

from _engine import Game, EngineError
from _view import build_menu, describe_main_option, option_card_id

DIFFICULTIES = {
    "easy":   {"use_lookahead": False},
    "medium": {"use_lookahead": True, "worlds": 1, "rounds": 1,
               "lookahead_budget_ms": 60},
    "hard":   {"use_lookahead": True, "worlds": 2, "rounds": 2,
               "lookahead_budget_ms": 110},
}

MAX_STEPS = 4000  # hard guard against runaway engine loops


def classify_main(type_id: int | None) -> str | None:
    """Map a MAIN option type to a learning action label."""
    return {
        7: "play_pokemon", 8: "attach", 9: "evolve", 10: "ability",
        11: "discard", 12: "retreat", 13: "attack", 14: "end",
    }.get(type_id)


class GameSession:
    def __init__(self, human_deck: list[int], ai_deck: list[int],
                 human_deck_id: str, ai_deck_id: str,
                 difficulty: str = "medium", store: Store | None = None):
        self.game_id = uuid.uuid4().hex[:12]
        self.game = Game(human_deck, ai_deck)
        self.human_index = 0          # we always seed decks[0] = human
        self.human_agent = Agent(human_deck, use_lookahead=False)
        profile = build_profile(store.records()) if store else None
        ai_kwargs = dict(DIFFICULTIES.get(difficulty, DIFFICULTIES["medium"]))
        ai_kwargs["opp_profile"] = profile if profile and profile["n_games"] else None
        self.ai_agent = Agent(ai_deck, go_first=True, **ai_kwargs)
        self.human_agent.begin_game()
        self.ai_agent.begin_game()

        self.human_deck_id = human_deck_id
        self.ai_deck_id = ai_deck_id
        self.difficulty = difficulty
        self.store = store

        self.log: list[str] = []
        self.human_actions: list[str] = []
        self.done = False
        self.winner: int | None = None
        self._start = time.time()
        self._steps = 0
        self._deck_names = {}

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def advance(self) -> dict:
        """Run automatic selections until the human must act (or the
        game ends). Returns the view to send the client."""
        guard = 0
        while guard < MAX_STEPS:
            obs = self.game.obs
            sel, cur = obs.get("select"), obs.get("current")
            if sel is None or cur is None:
                return self._finish(obs, "select lost")
            if cur.get("result", -1) != -1:
                return self._finish(obs, "result")
            who = cur["yourIndex"]
            if who == self.human_index and self._human_choice(sel):
                return self._view(obs)
            self._auto_step(obs, who)
            guard += 1
        self.done = True
        return self._view(self.game.obs)

    def act(self, picks: list[int]) -> dict:
        """The human answered the current selection; continue the game."""
        obs = self.game.obs
        self._log_human_pick(obs, picks)
        try:
            self.game.select(picks)
        except EngineError as exc:
            # illegal pick (shouldn't happen — UI only offers legal ones)
            self.log.append(f"⚠ invalid pick rejected ({exc})")
            return self._view(self.game.obs)
        return self.advance()

    # ------------------------------------------------------------------
    def _human_choice(self, sel: dict) -> bool:
        t = sel["type"]
        if t == SelectType.MAIN:
            return True
        if t == SelectType.YES_NO and sel.get("context") == SelectContext.IS_FIRST:
            return True
        if t == SelectType.CARD and sel.get("context") in (
                SelectContext.SETUP_ACTIVE_POKEMON,
                SelectContext.SETUP_BENCH_POKEMON):
            return True
        return False

    def _auto_step(self, obs: dict, who: int) -> None:
        agent = self.human_agent if who == self.human_index else self.ai_agent
        picks = agent.choose(obs)
        self.game.select(picks)
        self._steps += 1
        if who != self.human_index:
            self._log_ai_pick(obs, picks)

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------
    def _log_human_pick(self, obs: dict, picks: list[int]) -> None:
        sel, cur = obs.get("select"), obs.get("current")
        if sel is None or cur is None:
            return
        who = cur["yourIndex"]
        state = cur
        t = sel["type"]

        if t == SelectType.MAIN:
            for p in picks:
                if not (0 <= p < len(sel["option"])):
                    continue
                o = sel["option"][p]
                self.log.append("You — " + describe_main_option(
                    state, who, o, state["players"][who].get("hand") or []))
                cls = classify_main(o.get("type"))
                if cls:
                    self.human_actions.append(cls)
            return

        if t == SelectType.YES_NO and sel.get("context") == SelectContext.IS_FIRST:
            self.log.append("You chose to go "
                            + ("first" if picks and picks[0] == 0 else "second"))
            return

        if t == SelectType.CARD and sel.get("context") in (
                SelectContext.SETUP_ACTIVE_POKEMON,
                SelectContext.SETUP_BENCH_POKEMON):
            hand = state["players"][who].get("hand") or []
            names = []
            for p in picks:
                o = sel["option"][p] if 0 <= p < len(sel["option"]) else None
                if o is None:
                    continue
                cid = option_card_id(sel, o, state, who)
                if cid:
                    from card_db import card as _c
                    c = _c(cid)
                    names.append(c.name if c else f"#{cid}")
            if names:
                kind = "Active Pokémon" if sel["context"] == \
                    SelectContext.SETUP_ACTIVE_POKEMON else "Bench"
                self.log.append(f"You put {', '.join(names)} into play ({kind})")
            return

    def _log_ai_pick(self, obs: dict, picks: list[int]) -> None:
        sel, cur = obs.get("select"), obs.get("current")
        if sel is None or cur is None or sel["type"] != SelectType.MAIN:
            return
        who = cur["yourIndex"]
        state = cur
        for p in picks:
            if not (0 <= p < len(sel["option"])):
                continue
            o = sel["option"][p]
            self.log.append("AI — " + describe_main_option(
                state, who, o, state["players"][who].get("hand") or []))

    # ------------------------------------------------------------------
    # finish + record
    # ------------------------------------------------------------------
    def _finish(self, obs: dict, why: str) -> dict:
        if self.done:
            return self._view(obs)
        self.done = True
        cur = obs.get("current") or {}
        self.winner = cur.get("result")
        if self.winner is None or self.winner not in (0, 1):
            self.winner = 1  # treat engine weirdness as AI win; keep playing state
        self.log.append(
            "You win! 🎉" if self.winner == self.human_index
            else "The AI wins this one.")
        self._record(obs)
        return self._view(obs)

    def _record(self, obs: dict) -> None:
        cur = obs.get("current") or {}
        human_ps = cur.get("players", [{}, {}])[self.human_index]
        ai_ps = cur.get("players", [{}, {}])[1 - self.human_index]
        record = {
            "winner": self.winner,                       # 0 human, 1 AI
            "human_deck": self.human_deck_id,
            "ai_deck": self.ai_deck_id,
            "difficulty": self.difficulty,
            "go_first_human": cur.get("firstPlayer") == self.human_index,
            "turns": cur.get("turn"),
            "human_actions": list(self.human_actions),
            "human_prizes_left": len(human_ps.get("prize") or []),
            "ai_prizes_left": len(ai_ps.get("prize") or []),
            "duration_s": round(time.time() - self._start, 1),
        }
        if self.store is not None:
            try:
                self.store.add(record)
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _view(self, obs: dict) -> dict:
        cur = obs.get("current") or {}
        human_ps = cur.get("players", [{}, {}])[self.human_index]
        ai_ps = cur.get("players", [{}, {}])[1 - self.human_index]
        from _view import player_view
        menu = None if self.done else build_menu(obs, self.human_index)
        return {
            "game_id": self.game_id,
            "turn": cur.get("turn"),
            "yourIndex": cur.get("yourIndex"),
            "firstPlayer": cur.get("firstPlayer"),
            "result": cur.get("result"),
            "human": player_view(human_ps, True),
            "ai": player_view(ai_ps, False),
            "menu": menu,
            "log": self.log[-80:],
            "over": self.done,
            "winner": self.winner,
            "decks": {"human": self.human_deck_id, "ai": self.ai_deck_id,
                      "difficulty": self.difficulty},
            "human_prizes_left": len(human_ps.get("prize") or []),
            "ai_prizes_left": len(ai_ps.get("prize") or []),
        }
