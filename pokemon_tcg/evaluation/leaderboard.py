"""Leaderboard / multi-strategy comparison."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Leaderboard:
    """Aggregated head-to-head matrix and per-strategy Elo."""
    matchup_wins: dict = field(default_factory=dict)   # (a, b) -> (wins of a vs b, games)
    elo: dict = field(default_factory=dict)
    names: list = field(default_factory=list)

    def record(self, winner: int, a: str, b: str) -> None:
        key = (a, b)
        wins, games = self.matchup_wins.get(key, (0, 0))
        games += 1
        if winner == 0:
            wins += 1
        self.matchup_wins[key] = (wins, games)
        # Update Elo
        self._update_elo(a, b, win_idx=(0 if winner == 0 else 1))

    def _update_elo(self, a: str, b: str, win_idx: int, k: float = 24.0) -> None:
        from math import log10, sqrt, exp  # noqa: F401  (used inline)
        if a not in self.elo:
            self.elo[a] = 1500.0
        if b not in self.elo:
            self.elo[b] = 1500.0
        sa = self.elo[a]
        sb = self.elo[b]
        # Logistic expected scores (Riemann–Glickman approximation)
        ea = _exp_score(sa, sb)
        eb = 1 - ea
        sa = sa + k * ((1.0 if win_idx == 0 else 0.0) - ea)
        sb = sb + k * ((1.0 if win_idx == 1 else 0.0) - eb)
        self.elo[a] = sa
        self.elo[b] = sb

    def top(self, n: int = 5) -> list:
        return sorted(self.elo.items(), key=lambda kv: -kv[1])[:n]

    def to_dict(self) -> dict:
        return {
            "elo": dict(self.elo),
            "matchup": {f"{a}_vs_{b}": list(v) for (a, b), v in self.matchup_wins.items()},
        }


def _exp_score(a: float, b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((b - a) / 400.0))


def compare_strategies(agents: list, deck_pool, seeds: list,
                       games_per_pair: int = 4) -> Leaderboard:
    """Round-robin: every agent vs every other (and self) over many seeds."""
    from ..simulator import simulate_match
    from ..experiments.deck_pool import build_deck_pool  # cycle
    import random

    lb = Leaderboard()
    lb.names = [a.name for a in agents]
    decks = build_deck_pool(deck_pool, len(seeds) * 2)

    for i, a in enumerate(agents):
        for j, b in enumerate(agents):
            for k, seed in enumerate(seeds):
                deck_a = decks[(k * 2) % len(decks)]
                deck_b = decks[(k * 2 + 1) % len(decks)]
                for g in range(games_per_pair):
                    seed_k = seed * 1000 + g + i * 131 + j * 17
                    result = simulate_match(deck_a, deck_b, [a, b],
                                             seed=seed_k, log=False)
                    winner = result["winner"]
                    # Reverse if i == j (side swap for fairness)
                    if i == j and g % 2 == 1:
                        winner = 1 - winner if winner is not None else None
                    lb.record(winner if winner is not None else 0,
                              a.name, b.name)
    return lb

# Avoid circular import: keep this lazy inside the function above
