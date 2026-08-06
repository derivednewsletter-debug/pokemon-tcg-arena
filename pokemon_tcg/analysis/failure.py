"""Failure analysis — categorize why we lost a match.

Tournament results are most useful when we can attribute losses to
specific decision categories. This module walks the event log and
classifies each loss into one of:

* NO_KO         — Never KO'd the opponent's active in 6+ turns
* POOR_SETUP    — Never bench-deep enough / energy starved
* PRIZE_TRADE   — Took fewer prizes than expected after a KO
* DECK_OUT      — Lost by being unable to draw
* OVER_EXT      — Lost all board to a single big attack
* ENERGY_STARVED — Attached energy, never reached KO power
* NOT_CATEGORIZED — default bucket

The output is a dict {category: count} so the leaderboard can annotate
loans with the dominant failure mode per matchup.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

from ..game_state import GameState
from ..logging_utils.game_log import MatchRecord


CATEGORIES = (
    "NO_KO", "POOR_SETUP", "PRIZE_TRADE", "DECK_OUT",
    "OVER_EXT", "ENERGY_STARVED", "NOT_CATEGORIZED",
)


def categorize_failure(record: MatchRecord, perspective: int = 0) -> str:
    """Pick the most likely failure category for `record` from
    `perspective`'s point of view (0 or 1)."""
    if record.winner == perspective:
        return "WIN"
    events = record.events
    # Deck out?
    if any(e.get("reason") == "deck_out" for e in events):
        return "DECK_OUT"
    # No KO ever
    kos = [e for e in events if e["kind"] == "PRIZE_TAKEN"]
    if len(kos) < 2:
        # Less than 2 prizes taken across the whole match is POOR_SETUP
        return "NO_KO" if not kos else "POOR_SETUP"
    # Energy starved — multiple ATTACH_ENERGY but few ATTACK to KO
    attacks = [e for e in events if e["kind"] == "ATTACK" and e["player"] == perspective]
    attaches = [e for e in events if e.get("action", {}).get("kind") == "ATTACH_ENERGY"
                and e.get("player") == perspective]
    if len(attaches) > len(attacks) * 2 and len(attacks) > 0:
        return "ENERGY_STARVED"
    # Bench evolution coverage — never evolved a basic
    evolves = [e for e in events if e.get("action", {}).get("kind") == "EVOLVE"]
    if not evolves:
        return "POOR_SETUP"
    return "NOT_CATEGORIZED"


class FailureAnalyzer:
    """Aggregates failure categories across many matches."""

    def __init__(self):
        self._counts: Counter = Counter()
        self._per_matchup: dict = defaultdict(Counter)

    def add(self, record: MatchRecord, perspective: int, opponent_name: str) -> None:
        cat = categorize_failure(record, perspective)
        if cat == "WIN":
            return
        self._counts[cat] += 1
        self._per_matchup[opponent_name][cat] += 1

    def totals(self) -> dict[str, int]:
        return dict(self._counts)

    def per_opponent(self) -> dict[str, dict[str, int]]:
        return {k: dict(v) for k, v in self._per_matchup.items()}

    def dominant_failure(self) -> str:
        if not self._counts:
            return "NONE"
        return self._counts.most_common(1)[0][0]


def summarize_failures(records: Iterable, perspective: int = 0,
                        opponent_name: str = "?") -> FailureAnalyzer:
    fa = FailureAnalyzer()
    for r in records:
        fa.add(r, perspective, opponent_name)
    return fa
