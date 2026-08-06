"""Deck pool — shared pre-built decks used across many matches.

A tournament must produce reproducible matchups; using one canonical
deck per type for each seed keeps every A vs B matchup deterministic
across re-runs and CI pipelines.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

from ..cards import Card, load_cards, POKEMON_TYPES
from ..deck import build_random_deck, build_themed_deck


@dataclass
class DeckPool:
    """Holds N pre-built decks that experimental agents draw from."""
    decks: list[list[Card]] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    seed: int = 0

    def __len__(self) -> int:
        return len(self.decks)


def build_deck_pool(mode: str = "random", n: int = 8, base_seed: int = 42) -> DeckPool:
    """Build a shared pool of `n` decks.

    `mode` can be:
      * "random" — `n` random decks (default)
      * "themed" — `n` decks, one per type (cycles if n > 8)
    """
    cards = load_cards()
    pool = DeckPool(decks=[], names=[], seed=base_seed)

    if mode == "themed":
        for i in range(n):
            t = POKEMON_TYPES[i % len(POKEMON_TYPES)]
            d = build_themed_deck(cards, t, seed=base_seed + i * 7)
            pool.decks.append(d)
            pool.names.append(f"{t}-themed-{i}")
    else:
        for i in range(n):
            d = build_random_deck(cards, seed=base_seed + i * 31)
            pool.decks.append(d)
            pool.names.append(f"random-{i}")
    return pool


def default_pool(n: int = 8) -> DeckPool:
    """Cached pool reused across CLI runs."""
    cache_path = f"_deck_cache_{n}.pkl"  # placeholder
    return build_deck_pool("random", n=n)
