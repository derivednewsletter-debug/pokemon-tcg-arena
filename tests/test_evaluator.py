"""Evaluator + search tests."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pokemon_tcg.cards import load_cards
from pokemon_tcg.deck import build_random_deck
from pokemon_tcg.simulator import simulate_match, new_game
from pokemon_tcg.evaluator import score, EvaluatorConfig
from pokemon_tcg.search import greedy_1ply, alpha_beta_2ply
from pokemon_tcg.actions import Action
from pokemon_tcg.agents.benchmarks import GreedyAgent


def test_score_returns_finite():
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    s = score(state, 0)
    assert isinstance(s, float)
    assert abs(s) < 1e6


def test_score_terminal_is_extreme():
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    state.winner = 0
    # Score >= 1_000_000 means victory (terminal) for `who`
    assert score(state, 0) >= 1_000_000
    assert score(state, 1) <= -1_000_000


def test_score_handles_empty_moves():
    """Evaluator must not crash on Pokemon with empty move lists."""
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    # Force one or both actives to be a fossil with no moves
    from pokemon_tcg.cards import PokemonCard
    fossils = [c.pokemon for c in cards.values()
                if c.pokemon and not c.pokemon.moves]
    if fossils and state.players[0].active:
        state.players[0].active.base = fossils[0]
    s0 = score(state, 0)
    s1 = score(state, 1)
    assert isinstance(s0, float)
    assert isinstance(s1, float)


def test_score_prize_diff_matters():
    """Lowering ME's prize_count (I lose prizes) should decrease the
    score from MY perspective. Setting opp.prize_count down means OPP
    has lost more prizes (good for me) — score should INCREASE."""
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    base = score(state, 0)
    # I lose 2 prizes -> my prize_count goes 6->4 -> score should DROP
    state.players[0].prize_count = 4
    losing = score(state, 0)
    # We become worse when we have fewer prizes left
    assert losing < base, \
        f"expected lower score after losing prizes, base={base}, losing={losing}"


def test_score_opp_prize_change_is_positive():
    """If opp loses 2 prizes (opp.prize_count 6->4), score (from my view) should RISE."""
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    base = score(state, 0)
    state.players[1].prize_count = 4
    winning = score(state, 0)
    assert winning > base, \
        f"expected higher score after opp loses prizes, base={base}, winning={winning}"


def test_greedy_returns_action():
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    a, info = greedy_1ply(state, 0)
    assert isinstance(a, Action)
    assert info["candidates"] > 0
    assert info["best_delta"] is not None


def test_alpha_beta_runs():
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    a, info = alpha_beta_2ply(state, 0, beam=4)
    assert isinstance(a, Action)
    assert info["beam"] == 4
    assert info["depth"] == 2


def test_search_stable_under_seed():
    """Two search calls with the same state should return the same action."""
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    s1 = new_game(deck, deck, seed=42)
    s2 = new_game(deck, deck, seed=42)
    a1, _ = greedy_1ply(s1, 0)
    a2, _ = greedy_1ply(s2, 0)
    assert a1.to_json() == a2.to_json()


def test_score_with_custom_config():
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    cfg = EvaluatorConfig(prize_value=200.0)
    s = score(state, 0, cfg)
    assert isinstance(s, float)
