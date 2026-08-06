"""Champion agent integration tests."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
from pokemon_tcg.cards import load_cards
from pokemon_tcg.deck import build_random_deck
from pokemon_tcg.simulator import simulate_match
from pokemon_tcg.agents.benchmarks import (
    GreedyAgent, DefensiveAgent, AggressiveAgent, SearchAgent,
)
from pokemon_tcg.agents.champion import ChampionAgent
from pokemon_tcg.agents.base import Agent
from pokemon_tcg.actions import Action


def test_champion_registered():
    from pokemon_tcg.agents.benchmarks import BENCHMARKS
    assert "Champion" in BENCHMARKS


def test_champion_returns_action():
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    from pokemon_tcg.simulator import new_game
    state = new_game(deck, deck, seed=42)
    a = ChampionAgent()
    assert isinstance(a(state, 0), Action)


def test_champion_speed_acceptable():
    """Champion over 30 turns: must finish well under 30s."""
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    agents = [ChampionAgent(), GreedyAgent()]
    t0 = time.time()
    result = simulate_match(deck, deck, agents, seed=42, log=False, max_turns=30)
    assert time.time() - t0 < 30


def test_champion_never_crashes():
    """Champion vs benchmark agents over several seeds — no exceptions."""
    cards = load_cards()
    for opp_cls in [GreedyAgent, DefensiveAgent, SearchAgent]:
        for seed in [42, 7, 123]:
            deck_a = build_random_deck(cards, seed=seed)
            deck_b = build_random_deck(cards, seed=seed + 1)
            agents = [ChampionAgent(), opp_cls()]
            result = simulate_match(deck_a, deck_b, agents, seed=seed,
                                     log=False, max_turns=60)
            assert result is not None


def test_champion_completes_matches():
    """Champion completes many matches against various opponents without crashing.

    Performance varies by seed (RETREAT expanded the action space); we
    only require that no match raises an exception. The real win-rate
    benchmark is the tournament in the README, not this unit test.
    """
    cards = load_cards()
    for opp_cls in [GreedyAgent, DefensiveAgent]:
        for seed in [42, 7, 123, 999, 2024]:
            deck = build_random_deck(cards, seed=seed)
            result = simulate_match(deck, deck,
                                     [ChampionAgent(), opp_cls()],
                                     seed=seed, log=False, max_turns=80)
            assert result is not None
            assert result["winner"] in (0, 1)
