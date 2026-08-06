"""Engine-search lookahead.

Wraps the official engine's ``search_begin`` / ``search_step`` API to
score candidate MAIN actions under a *predicted* hidden world (deck
order, prizes, opponent hand/active — supplied by the Tracker). Each
candidate is played out through the real rules engine; nested selections
are resolved by rule-based policies, the opponent's full turn is played
heuristically, and the resulting state is scored with
:func:`strategy.evaluate`.

Because ``search_step`` costs ~0.1 ms, a candidate + opponent turn
resolves in a few milliseconds — lookahead is affordable on every MAIN
decision.
"""
from __future__ import annotations

import threading

from cg.api import search_begin, search_end, search_step

from card_db import (
    CardData, Observation, Option, Pokemon, SelectContext, SelectData,
    SelectType, State,
    best_potential_attack, card,
)
from strategy import (
    LOSE, OPT_END, WIN, choose_main_action, evaluate, option_card_id,
    rank_cards,
)

STEP_BUDGET = 160          # max engine search_steps per candidate
DEFAULT_GO_FIRST = True    # empirically tunable

# The engine's search API keeps one process-global agent. Serialize
# lookaheads so concurrent games on one instance can't corrupt it.
_SEARCH_LOCK = threading.RLock()


class Lookahead:
    def __init__(self, tracker, budget_ms: float = 120.0,
                 go_first: bool = DEFAULT_GO_FIRST,
                 worlds: int = 2, rounds: int = 2,
                 opp_profile: dict | None = None):
        self.tracker = tracker
        self.budget_ms = budget_ms
        self.go_first = go_first
        self.worlds = max(1, worlds)
        self.rounds = max(1, rounds)
        self.opp_profile = opp_profile  # learned human tendencies
        self._attack_cache = {}

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------
    def score_candidate(self, obs: Observation, option_idx: int, who: int,
                        worlds: list | None = None) -> float | None:
        """Score playing ``option_idx``, averaged over hidden worlds.

        ``worlds`` should be the same prediction set for every candidate
        in a decision (sampled once by the caller). When ``None``, one
        world is sampled here.
        """
        if worlds is None:
            worlds = [self.tracker.predictions(obs)]
        scores = []
        for w in worlds:
            try:
                s = self._score_once(obs, option_idx, who, w)
            except Exception:
                s = None
            if s is not None:
                scores.append(s)
        if not scores:
            return None
        return sum(scores) / len(scores)

    def _score_once(self, obs: Observation, option_idx: int, who: int,
                    preds: dict) -> float:
        with _SEARCH_LOCK:
            return self._score_once_locked(obs, option_idx, who, preds)

    def _score_once_locked(self, obs: Observation, option_idx: int, who: int,
                           preds: dict) -> float:
        st = search_begin(
            obs,
            list(preds["your_deck"]),
            list(preds["your_prize"]),
            list(preds["opponent_deck"]),
            list(preds["opponent_prize"]),
            list(preds["opponent_hand"]),
            list(preds["opponent_active"]),
            False,
        )
        st = search_step(st.searchId, [option_idx])
        seen_opp = False
        passes = 0
        steps = 0
        try:
            while steps < STEP_BUDGET:
                o2 = st.observation
                sel: SelectData | None = o2.select
                cur: State | None = o2.current
                if cur is None or sel is None:
                    return evaluate(o2, who)
                if cur.result != -1:
                    return WIN if cur.result == who else LOSE
                if sel.type == SelectType.MAIN:
                    p = cur.yourIndex
                    if p == who:
                        if seen_opp:
                            passes += 1
                            if passes >= self.rounds:
                                return evaluate(o2, who)
                            seen_opp = False
                        idx = choose_main_action(o2, lookahead=None)
                    else:
                        seen_opp = True
                        # opponent model: biased by what humans actually do
                        idx = choose_main_action(o2, lookahead=None,
                                                profile=self.opp_profile)
                    st = search_step(st.searchId, [idx])
                else:
                    picks = self.pick_context(sel, cur, cur.yourIndex)
                    if not picks:
                        n = len(sel.option)
                        k = min(sel.maxCount, max(sel.minCount, 0), n)
                        picks = list(range(k))
                    st = search_step(st.searchId, picks)
                steps += 1
            return evaluate(st.observation, who)
        finally:
            try:
                search_end()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # nested-selection policy
    # ------------------------------------------------------------------
    def pick_context(self, sel: SelectData, state: State, who: int) -> list[int]:
        """Rule-based pick for a nested (non-MAIN) selection.

        ``who`` is the player whose selection this is.
        """
        opts: list[Option] = sel.option
        n = len(opts)
        if n == 0:
            return []
        ctx = sel.context
        maxc = sel.maxCount
        minc = sel.minCount

        if sel.type == SelectType.CARD:
            ids = [option_card_id(sel, o, state) for o in opts]
            ranked = sorted(range(n), key=lambda i: rank_cards([ids[i]])[0][1],
                            reverse=True)
            if ctx in (SelectContext.TO_HAND, SelectContext.TO_ACTIVE,
                       SelectContext.TO_FIELD, SelectContext.LOOK,
                       SelectContext.SETUP_ACTIVE_POKEMON,
                       SelectContext.EVOLVES_TO):
                return ranked[: self._pick_count(sel)]
            if ctx in (SelectContext.DISCARD, SelectContext.TO_DECK,
                       SelectContext.TO_DECK_BOTTOM, SelectContext.TO_PRIZE,
                       SelectContext.NOT_MOVE):
                return ranked[::-1][: self._pick_count(sel)]
            if ctx in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH):
                return ranked[: self._pick_count(sel)]
            if ctx == SelectContext.SWITCH:
                return [self._best_bench_index(sel, state, who, 0)]
            if ctx in (SelectContext.ATTACH_TO, SelectContext.ATTACH_FROM):
                return [self._best_attach_target(sel, state, who)]
            if ctx in (SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER,
                       SelectContext.DAMAGE_COUNTER_ANY):
                return self._damage_pick(sel, state, who)
            if ctx in (SelectContext.HEAL, SelectContext.REMOVE_DAMAGE_COUNTER):
                return self._heal_pick(sel, state, who)
            if ctx == SelectContext.EVOLVES_FROM:
                return list(range(min(maxc, n)))
            if ctx == SelectContext.EFFECT_TARGET:
                return [self._opp_active_index(sel, state, who)]
            return ranked[: self._pick_count(sel)]

        if sel.type == SelectType.ENERGY:
            return list(range(self._pick_count(sel)))

        if sel.type == SelectType.ATTACHED_CARD:
            return list(range(self._pick_count(sel)))

        if sel.type == SelectType.CARD_OR_ATTACHED_CARD:
            ids = [option_card_id(sel, o, state) for o in opts]
            ranked = sorted(range(n), key=lambda i: rank_cards([ids[i]])[0][1],
                            reverse=True)
            return ranked[: self._pick_count(sel)]

        if sel.type == SelectType.ATTACK:
            best, bd = 0, -1
            for i, o in enumerate(opts):
                a = self._attack(o.attackId)
                d = a.damage or 0 if a else 0
                if d > bd:
                    best, bd = i, d
            return [best]

        if sel.type == SelectType.EVOLVE:
            return list(range(min(maxc, n)))

        if sel.type == SelectType.COUNT:
            if ctx == SelectContext.DRAW_COUNT:
                return [max(minc, maxc)]  # mulligan bonus: draw max
            k = max(minc, min(maxc, n - 1))
            return [k]

        if sel.type == SelectType.YES_NO:
            if ctx == SelectContext.IS_FIRST:
                return [0 if self.go_first else 1]
            return [0]

        if sel.type in (SelectType.SKILL, SelectType.SPECIAL_CONDITION):
            return list(range(self._pick_count(sel)))

        return list(range(self._pick_count(sel)))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _pick_count(self, sel: SelectData) -> int:
        n = len(sel.option)
        return max(sel.minCount, min(sel.maxCount, n))

    def _attack(self, aid):
        if aid in self._attack_cache:
            return self._attack_cache[aid]
        from card_db import attack
        a = attack(aid)
        self._attack_cache[aid] = a
        return a

    def _pokemon(self, state: State, o: Option) -> Pokemon | None:
        """Resolve the Pokemon an option points at (if any)."""
        if o.area not in (4, 5) or o.playerIndex is None:
            return None
        if o.playerIndex < 0 or o.playerIndex > 1:
            return None
        ps = state.players[o.playerIndex]
        if o.area == 4:
            if ps.active and ps.active[0] is not None:
                return ps.active[0]
            return None
        bench = ps.bench or []
        if o.index is not None and 0 <= o.index < len(bench):
            return bench[o.index]
        return None

    def _best_bench_index(self, sel: SelectData, state: State, who: int, _tie: int) -> int:
        best_i, best_v = 0, -1.0
        for i, o in enumerate(sel.option):
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

    def _best_attach_target(self, sel: SelectData, state: State, who: int) -> int:
        best_i, best_v = 0, -1.0
        for i, o in enumerate(sel.option):
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
        """Place damage: finish KOs (prefer opponent active), else hit active."""
        remain = (sel.remainDamageCounter or 0) * 10  # counters -> HP
        for i, o in enumerate(sel.option):
            if o.playerIndex != who:
                pok = self._pokemon(state, o)
                if pok is not None and pok.hp <= remain:
                    return [i]
        for i, o in enumerate(sel.option):
            if o.playerIndex != who and o.area in (4, 5):
                return [i]
        return [0]

    def _heal_pick(self, sel: SelectData, state: State, who: int) -> list[int]:
        for i, o in enumerate(sel.option):
            if o.playerIndex == who and o.area == 4:
                return [i]
        for i, o in enumerate(sel.option):
            if o.playerIndex == who:
                return [i]
        return [0]

    def _opp_active_index(self, sel: SelectData, state: State, who: int) -> int:
        for i, o in enumerate(sel.option):
            if o.playerIndex != who:
                return i
        return 0
