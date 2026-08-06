"""Structured game-log writers.

A single `simulator.match` produces a list of dict events; this module
serializes them to JSONL for offline analysis, and provides a thin
`MatchLogger` wrapper that bundles per-game stats.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MatchRecord:
    seed: int
    winner: Optional[int]
    turns: int
    events: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "seed": self.seed,
            "winner": self.winner,
            "turns": self.turns,
            "events": self.events,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_json(d: dict) -> "MatchRecord":
        return MatchRecord(
            seed=d["seed"], winner=d["winner"], turns=d["turns"],
            events=d.get("events", []), metadata=d.get("metadata", {}),
        )


class MatchLogger:
    """Wraps a match dict with helpers to label and persist it."""

    def __init__(self, out_dir: str = "results"):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def write(self, record: MatchRecord, name: Optional[str] = None) -> str:
        name = name or f"match_seed{record.seed}.jsonl"
        path = os.path.join(self.out_dir, name)
        with open(path, "w") as fh:
            fh.write(json.dumps(record.to_json(), default=str))
        return path

    def write_batch(self, records: list[MatchRecord], tag: str = "batch") -> str:
        path = os.path.join(self.out_dir, f"{tag}.jsonl")
        with open(path, "w") as fh:
            for r in records:
                fh.write(json.dumps(r.to_json(), default=str))
                fh.write("\n")
        return path


def write_match_json(path: str, record: MatchRecord) -> None:
    mkdirs(path)
    with open(path, "w") as fh:
        fh.write(json.dumps(record.to_json(), default=str))


def read_match_json(path: str) -> MatchRecord:
    with open(path) as fh:
        return MatchRecord.from_json(json.loads(fh.read()))


def mkdirs(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
