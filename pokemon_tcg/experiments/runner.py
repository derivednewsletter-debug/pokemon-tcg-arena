"""Experiment runner — batch-match execution and metrics reporting."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

from ..agents.base import Agent
from ..evaluation.metrics import Metrics
from ..logging_utils.game_log import MatchRecord, MatchLogger
from ..simulator import simulate_match
from .deck_pool import build_deck_pool, DeckPool


@dataclass
class ExperimentResult:
    agent: str
    opponent: str
    records: list[MatchRecord] = field(default_factory=list)
    metrics: Metrics = field(default_factory=Metrics)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            fh.write(json.dumps({
                "agent": self.agent, "opponent": self.opponent,
                "records": [r.to_json() for r in self.records],
                "metrics": self.metrics.to_dict(),
            }, default=str))


class ExperimentRunner:
    """Run one agent vs many opponents across many seeds."""

    def __init__(self, out_dir: str = "results", max_turns: int = 80):
        self.out_dir = out_dir
        self.logger = MatchLogger(out_dir)
        self.max_turns = max_turns

    def run(self, agent_a: Agent, agent_b: Agent, decks: DeckPool,
            seeds: Iterable[int], games_per_pair: int = 4) -> ExperimentResult:
        a_records: list[MatchRecord] = []
        m = Metrics(agent=agent_a.name, opponent=agent_b.name, seeds=list(seeds))
        for s_i, seed in enumerate(seeds):
            for g in range(games_per_pair):
                deck_a = decks.decks[(s_i * 2 + g) % len(decks.decks)]
                deck_b = decks.decks[(s_i * 2 + g + 1) % len(decks.decks)]
                log_path = os.path.join(self.out_dir, "logs")
                os.makedirs(log_path, exist_ok=True)
                t0 = time.time()
                result = simulate_match(deck_a, deck_b,
                                        [agent_a, agent_b],
                                        seed=seed * 1000 + g, log=False,
                                        max_turns=self.max_turns)
                elapsed = time.time() - t0
                rec = MatchRecord(
                    seed=seed * 1000 + g, winner=result["winner"],
                    turns=result["turns"], events=result["log"], metadata={
                        "deck_a_seed": seed * 1000 + g,
                        "elapsed_ms": int(elapsed * 1000),
                    },
                )
                a_records.append(rec)
                m.add(rec)
        m.finalize()
        return ExperimentResult(agent=agent_a.name, opponent=agent_b.name,
                                records=a_records, metrics=m)


def run_batch(agent: Agent, opponents: list[Agent], seeds: Iterable[int],
              out_dir: str = "results", games_per_pair: int = 4) -> list[ExperimentResult]:
    """Convenience: run `agent` against all `opponents` over `seeds`."""
    decks = build_deck_pool("random", n=8)
    runner = ExperimentRunner(out_dir=out_dir)
    out = []
    for opp in opponents:
        out.append(runner.run(agent, opp, decks, seeds, games_per_pair))
    return out
