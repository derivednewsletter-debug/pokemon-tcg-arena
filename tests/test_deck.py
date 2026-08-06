"""Deck construction tests."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pokemon_tcg.cards import load_cards
from pokemon_tcg.deck import build_random_deck, build_themed_deck, shuffle


def test_random_deck_size():
    cards = load_cards()
    d = build_random_deck(cards, seed=42, deck_size=60)
    assert len(d) == 60


def test_random_deck_deterministic():
    cards = load_cards()
    d1 = build_random_deck(cards, seed=42, deck_size=60)
    d2 = build_random_deck(cards, seed=42, deck_size=60)
    assert [c.pokemon.name if c.pokemon else c.energy.name if c.energy else c.trainer.name
            for c in d1] == \
           [c.pokemon.name if c.pokemon else c.energy.name if c.energy else c.trainer.name
            for c in d2]


def test_themed_deck_construction():
    cards = load_cards()
    d = build_themed_deck(cards, "R", seed=42, deck_size=60)
    assert len(d) == 60
    # All basics should be Fire type
    basics = [c for c in d if c.pokemon and c.pokemon.stage == "Basic"]
    if basics:
        for c in basics:
            assert c.pokemon.ptype == "R", \
                f"theme broken: {c.pokemon.name} type={c.pokemon.ptype}"


def test_only_basics_filter():
    cards = load_cards()
    d = build_random_deck(cards, seed=42, deck_size=60, only_basics=True)
    assert len(d) == 60
    for c in d:
        if c.pokemon:
            assert c.pokemon.stage == "Basic"


def test_shuffle_deterministic():
    import random
    d = list(range(20))
    a = shuffle(list(d), random.Random(42))
    b = shuffle(list(d), random.Random(42))
    assert a == b
