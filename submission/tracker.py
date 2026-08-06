"""Hidden-information tracker.

The agent sees its own hand, both players' discard piles, both players'
Pokemon in play (card IDs, HP, attached energy), prize *counts* and
deck *counts*. It does not see: the opponent's hand, either player's
deck order, or face-down prize/active cards.

This module keeps a per-game model of what each player's deck still
contains (a multiset of card IDs), plus predictions (fixed samples)
that the lookahead engine search needs: deck order, prize contents,
opponent hand, and a possibly face-down opponent Active.

Key contract: :meth:`Tracker.predictions` returns a dict of lists used
verbatim by every ``search_begin`` call within one decision, so
candidate actions are compared under the same hidden world.
"""
from __future__ import annotations

import random
from collections import Counter

from card_db import card, CardType

Game = "game"


class Tracker:
    def __init__(self, deck: list[int], seed: int = 0):
        self.deck = list(deck)
        self.total = len(deck)
        self._seen: list[int] = []       # card IDs revealed anywhere (order = reveal order)
        self._rng = random.Random(seed)  # deterministic within a game if seeded
        self.reset()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Start a fresh game model."""
        self.logs: list = []
        self.my_known: Counter = Counter()       # cards in my hand / discard / in-play / prizes
        self.opp_known: Counter = Counter()      # cards seen in opponent discard / in-play / prizes
        self._predictions: dict | None = None

    def begin_game(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    # observation intake
    # ------------------------------------------------------------------
    def observe(self, obs) -> None:
        """Accumulate logs and reconcile visible piles."""
        if obs is None:
            return
        for lg in obs.logs or []:
            self.logs.append(lg)

        state = obs.current
        if state is None:
            return
        me = state.players[state.yourIndex]
        opp = state.players[1 - state.yourIndex]

        # My known cards: hand + discard + prizes (revealed ones) + in play.
        my_pile: Counter = Counter()
        if me.hand:
            my_pile.update(c.id for c in me.hand)
        my_pile.update(c.id for c in me.discard or [])
        for p in (me.prize or []):
            if p is not None:
                my_pile[p.id] += 1
        my_pile.update(_in_play_ids(me))
        self.my_known = my_pile

        # Opponent known: discard + revealed prizes + in play.
        opp_pile: Counter = Counter()
        opp_pile.update(c.id for c in opp.discard or [])
        for p in (opp.prize or []):
            if p is not None:
                opp_pile[p.id] += 1
        opp_pile.update(_in_play_ids(opp))
        self.opp_known = opp_pile

        for cid, n in my_pile.items():
            if cid > 0:
                self._seen.append(cid)
        for cid, n in opp_pile.items():
            if cid > 0:
                self._seen.append(cid)

    # ------------------------------------------------------------------
    # composition queries
    # ------------------------------------------------------------------
    def my_deck_composition(self, state) -> Counter:
        """Multiset of card IDs still in my deck (by count)."""
        known = Counter(self.my_known)
        # remove anything currently in hand/discard/prize/in-play from the
        # 60-card total (best-effort; overlaps are fine).
        return _deck_remaining(self.deck, known)

    def opp_deck_composition(self, state) -> Counter:
        """Multiset of card IDs likely still in opponent's deck.

        We assume the opponent's 60-card list is a copy of ours (mirror)
        unless their revealed cards prove otherwise. Cards revealed in
        their discard/in-play are subtracted; their unknown hand/prizes
        are modelled by sampling from the remainder.
        """
        base = Counter(self.deck)
        base.subtract(self.opp_known)
        base = +base  # drop negatives
        return base

    # ------------------------------------------------------------------
    # predictions for the engine search API
    # ------------------------------------------------------------------
    def sample_worlds(self, obs, n: int) -> list[dict]:
        """Sample ``n`` independent hidden worlds (fresh predictions).

        All candidate actions within one decision should be scored under
        the *same* set of worlds for a fair comparison.
        """
        return [self.predictions(obs) for _ in range(max(1, n))]

    def predictions(self, obs):
        """Build (and cache per-decision) the hidden-world prediction.

        Returns a dict with keys: your_deck, your_prize, opponent_deck,
        opponent_prize, opponent_hand, opponent_active. Deterministic
        within one decision; refreshed on the next agent() call.
        """
        state = obs.current
        me = state.players[state.yourIndex]
        opp = state.players[1 - state.yourIndex]

        my_comp = self.my_deck_composition(state)
        opp_comp = self.opp_deck_composition(state)

        my_prize_n = len(me.prize or [])
        opp_prize_n = len(opp.prize or [])
        opp_hand_n = opp.handCount or 0
        my_deck_n = me.deckCount or 0
        opp_deck_n = opp.deckCount or 0

        pool_my = list(my_comp.elements())
        pool_opp = list(opp_comp.elements())
        if len(pool_my) < my_deck_n + my_prize_n:
            # fall back: sample from the full deck list
            pool_my = list(self.deck)
        if len(pool_opp) < opp_deck_n + opp_prize_n + opp_hand_n:
            pool_opp = list(self.deck)

        rng = self._rng
        # Draw my deck order first, then my prizes from what remains.
        shuffled_my = pool_my[:]
        rng.shuffle(shuffled_my)
        your_deck = shuffled_my[:my_deck_n]
        rest_my = shuffled_my[my_deck_n:]
        your_prize = rest_my[:my_prize_n]
        while len(your_prize) < my_prize_n:
            your_prize.append(rng.choice(pool_my))

        shuffled_opp = pool_opp[:]
        rng.shuffle(shuffled_opp)
        opponent_deck = shuffled_opp[:opp_deck_n]
        rest_opp = shuffled_opp[opp_deck_n:]
        opponent_prize = rest_opp[:opp_prize_n]
        opponent_hand = rest_opp[opp_prize_n:opp_prize_n + opp_hand_n]
        while len(opponent_prize) < opp_prize_n:
            opponent_prize.append(rng.choice(pool_opp))
        while len(opponent_hand) < opp_hand_n:
            opponent_hand.append(rng.choice(pool_opp))

        # Opponent active: only needed when facedown.
        opponent_active: list[int] = []
        act = opp.active
        if act and act[0] is None:
            candidates = [cid for cid in pool_opp if _is_pokemon(cid)]
            if not candidates:
                candidates = [cid for cid in self.deck if _is_pokemon(cid)]
            opponent_active = [rng.choice(candidates)] if candidates else []

        self._predictions = dict(
            your_deck=your_deck,
            your_prize=your_prize,
            opponent_deck=opponent_deck,
            opponent_prize=opponent_prize,
            opponent_hand=opponent_hand,
            opponent_active=opponent_active,
        )
        return self._predictions


def _in_play_ids(ps) -> list[int]:
    ids = []
    for p in ps.active or []:
        if p is not None:
            ids.append(p.id)
    for p in ps.bench or []:
        ids.append(p.id)
    # attached energies / tools are separate Card objects but their ids
    # also came from the deck; count them too
    for p in list(ps.active or []) + list(ps.bench or []):
        if p is None:
            continue
        for c in p.energyCards or []:
            ids.append(c.id)
        for c in p.tools or []:
            ids.append(c.id)
    return ids


def _deck_remaining(total: list[int], known: Counter) -> Counter:
    base = Counter(total)
    base.subtract(known)
    return +base


def _is_pokemon(cid: int) -> bool:
    c = card(cid)
    return c is not None and c.cardType == CardType.POKEMON
