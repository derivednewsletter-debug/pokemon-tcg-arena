"""Tournament — round-robin between agents.

Each pair plays multiple matches over multiple seeds, with the deck
pool choosing sides deterministically. Elo is updated for each game.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from ..agents.base import Agent
from ..evaluation.leaderboard import Leaderboard, _exp_score
from ..logging_utils.game_log import MatchRecord, MatchLogger
from ..simulator import simulate_match
from .deck_pool import DeckPool, build_deck_pool


@dataclass
class TournamentResult:
    leaderboard: Leaderboard
    records: list[MatchRecord] = field(default_factory=list)
    per_matchup_metrics: dict = field(default_factory=dict)

    def markdown(self) -> str:
        rows = []
        rows.append("| Agent | Elo |")
        rows.append("|-------|----:|")
        for name, elo in sorted(self.leaderboard.elo.items(), key=lambda kv: -kv[1]):
            rows.append(f"| {name} | {elo:.1f} |")
        rows.append("\n## Matchup table\n")
        rows.append("| | " + " | ".join(self.leaderboard.names) + " |")
        rows.append("|---|" + "---|" * len(self.leaderboard.names))
        matchup = dict(self.leaderboard.matchup_wins)
        for a in self.leaderboard.names:
            row = [a]
            for b in self.leaderboard.names:
                wins, games = matchup.get((a, b), (0, 0))
                row.append(f"{wins}/{games}")
            rows.append("| " + " | ".join(row) + " |")
        return "\n".join(rows)


def run_tournament(agents: list[Agent], seeds: list[int],
                   pool_mode: str = "themed", games_per_pair: int = 4,
                   out_dir: str = "results") -> TournamentResult:
    decks = build_deck_pool(pool_mode, n=8)
    lb = Leaderboard(names=[a.name for a in agents])
    records: list[MatchRecord] = []
    log_path = os.path.join(out_dir, "tournament")
    os.makedirs(log_path, exist_ok=True)
    logger = MatchLogger(log_path)

    for i, a in enumerate(agents):
        for j, b in enumerate(agents):
            for s_i, seed in enumerate(seeds):
                deck_a = decks.decks[(s_i * 2 + i) % len(decks.decks)]
                deck_b = decks.decks[(s_i * 2 + j + 1) % len(decks.decks)]
                for g in range(games_per_pair):
                    seed_k = seed * 10000 + g + i * 31 + j * 17
                    result = simulate_match(deck_a, deck_b, [a, b], seed=seed_k,
                                             log=False, max_turns=80)
                    winner = result["winner"]
                    if i == j and g % 2 == 1:
                        winner = 1 - winner if winner is not None else None
                    lb.record(winner if winner is not None else 0, a.name, b.name)
                    rec = MatchRecord(seed=seed_k, winner=winner,
                                       turns=result["turns"],
                                       events=result["log"],
                                       metadata={"i": i, "j": j,
                                                 "p0": a.name,
                                                 "p1": b.name})
                    records.append(rec)
    return TournamentResult(leaderboard=lb, records=records)
