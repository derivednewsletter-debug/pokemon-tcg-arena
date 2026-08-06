"""Selection dispatcher — turns any engine ``Observation`` into option picks.

This is the layer that maps every ``SelectType`` / ``SelectContext`` the
engine can produce onto a decision:

* **MAIN** — the tactical core: :func:`strategy.choose_main_action`
  with engine-search lookahead.
* **CARD / ENERGY / ATTACK / COUNT / YES_NO / ...** — nested selections
  (setup, searches, damage placement, retreat costs, coin flips, ...)
  via rule-based pickers.

Every path is wrapped so an unexpected engine edge case degrades to a
valid random pick instead of crashing the match.
"""
from __future__ import annotations

import random

from card_db import (
    CardData, Observation, Option, SelectContext, SelectData, SelectType,
    State, best_potential_attack, card,
)
from strategy import (
    OPT_END, choose_main_action, option_card_id, rank_cards,
)
from tracker import Tracker

try:
    from lookahead import Lookahead
except Exception:  # pragma: no cover - lookahead is optional at import time
    Lookahead = None  # type: ignore


class Agent:
    def __init__(self, deck: list[int], go_first: bool = True,
                 lookahead_budget_ms: float = 110.0, use_lookahead: bool = True,
                 worlds: int = 2, rounds: int = 2,
                 opp_profile: dict | None = None):
        self.deck = list(deck)
        self.go_first = go_first
        self.opp_profile = opp_profile  # learned human tendencies (web arena)
        self.tracker = Tracker(deck)
        self.lookahead = (Lookahead(self.tracker, budget_ms=lookahead_budget_ms,
                                    go_first=go_first, worlds=worlds,
                                    rounds=rounds, opp_profile=opp_profile)
                          if use_lookahead and Lookahead is not None else None)
        self._rng = random.Random(1234)
        self.stats = {"decisions": 0, "main": 0, "lookahead_ms": 0.0}

    # ------------------------------------------------------------------
    def begin_game(self) -> None:
        self.tracker.begin_game()

    def choose(self, obs_dict: dict) -> list[int]:
        """Public entry: dict observation -> list of option indices."""
        from cg.api import to_observation_class
        obs = to_observation_class(obs_dict)
        self.stats["decisions"] += 1

        if obs.select is None:
            # initial deck selection
            self.begin_game()
            return self.deck

        self.tracker.observe(obs)
        return self._handle(obs)

    # ------------------------------------------------------------------
    def _handle(self, obs: Observation) -> list[int]:
        sel: SelectData = obs.select
        state: State = obs.current
        who = state.yourIndex if state is not None else 0
        n = len(sel.option)

        try:
            if sel.type == SelectType.MAIN:
                self.stats["main"] += 1
                import time
                t0 = time.time()
                idx = choose_main_action(obs, self.tracker, self.lookahead)
                self.stats["lookahead_ms"] += (time.time() - t0) * 1000
                return self._sanitize([idx], sel)
            picks = self._dispatch_nested(sel, state, who)
        except Exception:
            picks = self._fallback(sel)
        return self._sanitize(picks, sel)

    # ------------------------------------------------------------------
    def _dispatch_nested(self, sel: SelectData, state: State, who: int) -> list[int]:
        opts = sel.option
        n = len(opts)
        ctx = sel.context
        minc = sel.minCount
        maxc = sel.maxCount

        # -- CARD-based selections --------------------------------------
        if sel.type == SelectType.CARD:
            ids = [option_card_id(sel, o, state) for o in opts]
            ranked = sorted(range(n), key=lambda i: rank_cards([ids[i]])[0][1],
                            reverse=True)
            if ctx in (SelectContext.SETUP_ACTIVE_POKEMON,
                       SelectContext.SETUP_BENCH_POKEMON,
                       SelectContext.TO_BENCH, SelectContext.TO_ACTIVE,
                       SelectContext.TO_FIELD, SelectContext.TO_HAND,
                       SelectContext.LOOK, SelectContext.EVOLVES_TO):
                k = self._count(sel)
                return ranked[:k]
            if ctx in (SelectContext.DISCARD, SelectContext.TO_DECK,
                       SelectContext.TO_DECK_BOTTOM, SelectContext.TO_PRIZE):
                if minc == 0:
                    return []  # optional discard: keep everything
                return ranked[::-1][: self._count(sel)]
            if ctx == SelectContext.SWITCH:
                return [self._best_bench(opts, state)]
            if ctx in (SelectContext.ATTACH_TO, SelectContext.ATTACH_FROM):
                return [self._best_attach_target(opts, state)]
            if ctx in (SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER,
                       SelectContext.DAMAGE_COUNTER_ANY):
                return self._damage_pick(sel, state, who)
            if ctx in (SelectContext.HEAL, SelectContext.REMOVE_DAMAGE_COUNTER):
                return self._heal_pick(opts, state, who)
            if ctx == SelectContext.EVOLVES_FROM:
                return list(range(min(maxc, n)))
            if ctx == SelectContext.EFFECT_TARGET:
                return [self._opp_active(opts, state, who)]
            k = self._count(sel)
            return ranked[:k]

        # -- Energy / attached-card --------------------------------------
        if sel.type == SelectType.CARD_OR_ATTACHED_CARD:
            ids = [option_card_id(sel, o, state) for o in opts]
            ranked = sorted(range(n), key=lambda i: rank_cards([ids[i]])[0][1],
                            reverse=True)
            return ranked[: self._count(sel)]
        if sel.type in (SelectType.ENERGY, SelectType.ATTACHED_CARD):
            return list(range(self._count(sel)))

        # -- Attack -------------------------------------------------------
        if sel.type == SelectType.ATTACK:
            return [best_attack_option_pick(opts)]

        # -- Evolve -------------------------------------------------------
        if sel.type == SelectType.EVOLVE:
            return list(range(min(maxc, n)))

        # -- Count --------------------------------------------------------
        if sel.type == SelectType.COUNT:
            if ctx == SelectContext.DRAW_COUNT:
                # opponent-mulligan bonus: draw as many as allowed
                return [max(minc, maxc)]
            return [max(minc, min(maxc, n - 1))]

        # -- Yes/No -------------------------------------------------------
        if sel.type == SelectType.YES_NO:
            if ctx == SelectContext.IS_FIRST:
                return [0 if self.go_first else 1]
            return [0]  # yes

        # -- Skill / special condition ------------------------------------
        if sel.type in (SelectType.SKILL, SelectType.SPECIAL_CONDITION):
            return list(range(self._count(sel)))

        k = self._count(sel)
        return list(range(k))

    # ------------------------------------------------------------------
    # nested pickers
    # ------------------------------------------------------------------
    def _count(self, sel: SelectData) -> int:
        n = len(sel.option)
        return max(sel.minCount, min(sel.maxCount, n))

    def _pokemon(self, state: State, o: Option):
        if o.area not in (4, 5) or o.playerIndex is None or not (0 <= o.playerIndex <= 1):
            return None
        ps = state.players[o.playerIndex]
        if o.area == 4:
            return ps.active[0] if ps.active and ps.active[0] is not None else None
        bench = ps.bench or []
        if o.index is not None and 0 <= o.index < len(bench):
            return bench[o.index]
        return None

    def _best_bench(self, opts, state: State) -> int:
        best_i, best_v = 0, -1.0
        for i, o in enumerate(opts):
            pok = self._pokemon(state, o)
            if pok is None:
                continue
            cd = card(pok.id)
            best = best_potential_attack(cd) if cd else None
            d = (best.damage or 0) if best else 0
            v = d + pok.hp * 0.5
            if v > best_v:
                best_v, best_i = v, i
        return best_i

    def _best_attach_target(self, opts, state: State) -> int:
        best_i, best_v = 0, -1.0
        for i, o in enumerate(opts):
            pok = self._pokemon(state, o)
            v = 0.0
            if pok is not None:
                cd = card(pok.id)
                best = best_potential_attack(cd) if cd else None
                if best:
                    need = max(0, len(best.energies or []) - len(pok.energies or []))
                    v = (best.damage or 0) * 0.3 - need * 12.0
                    if need == 1:
                        v += 30.0
                v += pok.hp * 0.1
            if v > best_v:
                best_v, best_i = v, i
        return best_i

    def _damage_pick(self, sel: SelectData, state: State, who: int) -> list[int]:
        remain = (sel.remainDamageCounter or 0) * 10
        for i, o in enumerate(sel.option):
            if o.playerIndex != who:
                pok = self._pokemon(state, o)
                if pok is not None and pok.hp <= remain:
                    return [i]
        for i, o in enumerate(sel.option):
            if o.playerIndex != who and o.area in (4, 5):
                return [i]
        return [0]

    def _heal_pick(self, opts, state: State, who: int) -> list[int]:
        for i, o in enumerate(opts):
            if o.playerIndex == who and o.area == 4:
                return [i]
        for i, o in enumerate(opts):
            if o.playerIndex == who:
                return [i]
        return [0]

    def _opp_active(self, opts, state: State, who: int) -> int:
        for i, o in enumerate(opts):
            if o.playerIndex != who:
                return i
        return 0

    # ------------------------------------------------------------------
    # safety
    # ------------------------------------------------------------------
    def _sanitize(self, picks: list[int], sel: SelectData) -> list[int]:
        n = len(sel.option)
        out: list[int] = []
        for i in picks:
            if 0 <= i < n and i not in out:
                out.append(i)
        # satisfy minCount
        if len(out) < sel.minCount:
            for i in range(n):
                if i not in out:
                    out.append(i)
                if len(out) >= sel.minCount:
                    break
        # cap at maxCount
        if len(out) > sel.maxCount:
            out = out[: sel.maxCount]
        return out

    def _fallback(self, sel: SelectData) -> list[int]:
        n = len(sel.option)
        if n == 0:
            return []
        k = max(0, min(sel.maxCount, n))
        if sel.minCount > 0:
            k = max(k, sel.minCount)
        pool = list(range(n))
        self._rng.shuffle(pool)
        return pool[:k]


def best_attack_option_pick(opts: list[Option]) -> int:
    best, bd = 0, -1
    for i, o in enumerate(opts):
        if o.attackId is None:
            continue
        from card_db import attack
        a = attack(o.attackId)
        d = a.damage or 0 if a else 0
        if d > bd:
            best, bd = i, d
    return best
