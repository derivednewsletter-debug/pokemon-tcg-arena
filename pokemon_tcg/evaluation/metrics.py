"""Match metrics — extract structured statistics from a finished match.

The evaluator runs a batch of matches and emits a `Metrics` blob that
captures the things the experiment framework reads:
  * win rate by player index
  * mean turn count on a win vs a loss
  * mean prize differential
  * average hand sizes per turn
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..game_state import GameState
from ..simulator import new_game


@dataclass
class Metrics:
    games: int = 0
    wins_by_player: dict = field(default_factory=lambda: {0: 0, 1: 0})
    avg_turns_winner: float = 0.0
    avg_turns_loser: float = 0.0
    avg_prize_diff_winner: float = 0.0
    avg_hand_size_per_turn: list = field(default_factory=list)
    decision_latency_p50: float = 0.0
    decision_latency_p95: float = 0.0
    agent: str = ""
    opponent: str = ""
    seeds: list = field(default_factory=list)

    def add(self, record) -> None:
        self.games += 1
        if record.winner is not None:
            self.wins_by_player[record.winner] = self.wins_by_player.get(record.winner, 0) + 1
        self._update_turns(record)
        self._update_prizes(record)
        self._update_latency(record)

    def _update_turns(self, record) -> None:
        # We record turn count per game as `record.turns`
        if not hasattr(self, "_w_turns"):
            self._w_turns = []
            self._l_turns = []
        if record.winner == 0:
            self._w_turns.append(record.turns)
        elif record.winner == 1:
            self._l_turns.append(record.turns)

    def _update_prizes(self, record) -> None:
        if not hasattr(self, "_prize_diff"):
            self._prize_diff = []
        # Use the events list to back-count prize differences
        # Walk events: each PRIZE_TAKEN event has a `remaining` field
        state = getattr(record, "state", None)
        if state is None:
            return
        my_prize = state.players[0].prize_count
        opp_prize = state.players[1].prize_count
        if record.winner == 0:
            diff = opp_prize - my_prize  # opponent has more remaining, I took more
        else:
            diff = my_prize - opp_prize
        self._prize_diff.append(diff)

    def _update_latency(self, record) -> None:
        lats = []
        for e in record.events:
            if e.get("kind") == "ACTION" and "elapsed_ms" in e:
                lats.append(e["elapsed_ms"])
        if not hasattr(self, "_all_latencies"):
            self._all_latencies = []
        self._all_latencies.extend(lats)

    def finalize(self) -> None:
        if hasattr(self, "_w_turns") and self._w_turns:
            self.avg_turns_winner = sum(self._w_turns) / len(self._w_turns)
        if hasattr(self, "_l_turns") and self._l_turns:
            self.avg_turns_loser = sum(self._l_turns) / len(self._l_turns)
        if hasattr(self, "_prize_diff") and self._prize_diff:
            self.avg_prize_diff_winner = sum(self._prize_diff) / len(self._prize_diff)
        if hasattr(self, "_all_latencies") and self._all_latencies:
            sorted_lat = sorted(self._all_latencies)
            self.decision_latency_p50 = sorted_lat[int(0.5 * len(sorted_lat))]
            self.decision_latency_p95 = sorted_lat[int(0.95 * len(sorted_lat))]

    def win_rate(self, player: int = 0) -> float:
        if self.games == 0:
            return 0.0
        return self.wins_by_player.get(player, 0) / self.games

    @property
    def wins_p0(self) -> int:
        return self.wins_by_player.get(0, 0)

    @property
    def wins_p1(self) -> int:
        return self.wins_by_player.get(1, 0)

    def to_dict(self) -> dict:
        return {
            "games": self.games,
            "wins_p0": self.wins_by_player.get(0, 0),
            "wins_p1": self.wins_by_player.get(1, 0),
            "win_rate_p0": self.win_rate(0),
            "win_rate_p1": self.win_rate(1),
            "avg_turns_winner": self.avg_turns_winner,
            "avg_turns_loser": self.avg_turns_loser,
            "avg_prize_diff_winner": self.avg_prize_diff_winner,
            "decision_latency_p50_ms": self.decision_latency_p50,
            "decision_latency_p95_ms": self.decision_latency_p95,
            "agent": self.agent, "opponent": self.opponent,
            "seeds": self.seeds,
        }


def summarize(records: Iterable) -> Metrics:
    """Compute Metrics from an iterable of MatchRecord-like objects."""
    m = Metrics()
    for r in records:
        m.add(r)
    m.finalize()
    return m
