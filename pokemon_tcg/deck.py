"""Deck construction utilities.

A "deck" is a list of Card instances (60 cards in real TCG). We expose
two strategies here:

  * `build_random_deck(cards, seed)`           sample random decks
  * `build_themed_deck(cards, ptype)`          build a focused deck around a type
  * `pad_deck(deck, rng, target)`              ensure exact 60-card deck size

Decks are deterministic given the seed, so test runs and experiments
reproduce exactly.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Optional

from .cards import Card, EnergyCard, PokemonCard, COLORLESS, load_cards


def shuffle(deck: list[Card], rng: random.Random) -> list[Card]:
    rng.shuffle(deck)
    return deck


def pad_deck(deck: list[Card], rng: random.Random, target: int = 60) -> list[Card]:
    """Pad a deck with basics/energy until it has exactly `target` cards."""
    while len(deck) < target:
        # Always pad with basic energy (always safe)
        deck.append(Card(card_id="",
                          energy=EnergyCard(name="Basic {C} Energy",
                                            provides=COLORLESS,
                                            is_special=False, text="")))
    rng.shuffle(deck)
    return deck[:target]


def build_random_deck(cards: dict[str, Card], seed: int, deck_size: int = 60,
                      energy_bias: float = 0.30, only_basics: bool = False) -> list[Card]:
    """Sample a random legal-ish deck.

    Energy share is set to ~30% (the standard); the rest is Pokemon +
    trainers. We pick Basic Pokemon as starters by default
    (``only_basics=False`` includes Stage 1/2 for completeness).
    """
    rng = random.Random(seed)
    basics = [c for c in cards.values() if c.pokemon and c.pokemon.stage == "Basic"
              and c.pokemon.moves and any((m.damage or 0) > 0 for m in c.pokemon.moves)]
    stages1 = [c for c in cards.values() if c.pokemon and c.pokemon.stage == "Stage 1"
               and c.pokemon.moves]
    stages2 = [c for c in cards.values() if c.pokemon and c.pokemon.stage == "Stage 2"
               and c.pokemon.moves]
    energies = [c for c in cards.values() if c.energy]
    items = [c for c in cards.values() if c.trainer and c.trainer.category == "Item"]
    supporters = [c for c in cards.values() if c.trainer and c.trainer.category == "Supporter"]

    if only_basics:
        stages1 = []
        stages2 = []

    n_energy = int(deck_size * energy_bias)
    n_basics = (deck_size - n_energy) * 2 // 5
    n_stages = (deck_size - n_energy) // 5
    n_items = max(1, (deck_size - n_energy) // 8)
    n_supps = max(1, (deck_size - n_energy) // 12)

    deck: list[Card] = []
    deck += _sample(basics, n_basics, rng)
    deck += _sample(stages1, n_stages // 2, rng)
    deck += _sample(stages2, n_stages // 2, rng)
    deck += _sample(_basic_energies(energies), int(n_energy * 0.7), rng)
    deck += _sample(_special_energies(energies), n_energy - int(n_energy * 0.7), rng)
    deck += _sample(items, n_items, rng)
    deck += _sample(supporters, n_supps, rng)
    return pad_deck(deck, rng, target=deck_size)


def _sample(pool: list, n: int, rng: random.Random) -> list[Card]:
    if not pool:
        return []
    n = min(n, len(pool) * 4)  # allow duplicates
    return [rng.choice(pool) for _ in range(n)]


def _basic_energies(energies: list[Card]) -> list[Card]:
    return [c for c in energies if c.energy and not c.energy.is_special]


def _special_energies(energies: list[Card]) -> list[Card]:
    return [c for c in energies if c.energy and c.energy.is_special]


def build_themed_deck(cards: dict[str, Card], ptype: str, seed: int,
                      deck_size: int = 60, focus_basics_only: bool = False) -> list[Card]:
    """Build a deck focused on a single Pokemon type."""
    rng = random.Random(seed)
    chosen_basics = [c for c in cards.values()
                     if c.pokemon and c.pokemon.stage == "Basic"
                     and c.pokemon.ptype == ptype
                     and c.pokemon.moves
                     and any((m.damage or 0) > 0 for m in c.pokemon.moves)]
    chosen_stages1 = [c for c in cards.values()
                      if c.pokemon and c.pokemon.stage == "Stage 1"
                      and c.pokemon.ptype == ptype and c.pokemon.moves]
    chosen_stages2 = [c for c in cards.values()
                      if c.pokemon and c.pokemon.stage == "Stage 2"
                      and c.pokemon.ptype == ptype and c.pokemon.moves]
    energies = _basic_energies([c for c in cards.values() if c.energy])
    type_energy = [c for c in energies if c.energy.provides == ptype]
    if not type_energy:
        # Treat colorless energy as our "any color" energy if specific not found
        type_energy = [c for c in energies if c.energy.provides == COLORLESS] or energies[:1]

    deck: list[Card] = []
    n_basics = deck_size * 2 // 5
    deck += _sample(chosen_basics, n_basics, rng)
    if not focus_basics_only:
        n_stages1 = deck_size // 8
        n_stages2 = deck_size // 10
        deck += _sample(chosen_stages1, n_stages1, rng)
        deck += _sample(chosen_stages2, n_stages2, rng)
    n_energy = deck_size // 4
    deck += _sample(type_energy, n_energy, rng)
    # Pad until target
    while len(deck) < deck_size:
        deck.append(rng.choice(type_energy))
    rng.shuffle(deck)
    return deck[:deck_size]
