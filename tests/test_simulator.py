"""Simulator + game-state tests."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pokemon_tcg.cards import load_cards
from pokemon_tcg.deck import build_random_deck
from pokemon_tcg.simulator import simulate_match, new_game, step
from pokemon_tcg.actions import Action, legal_actions
from pokemon_tcg.game_state import GameState, MAX_BENCH
from pokemon_tcg.agents.benchmarks import GreedyAgent, DefensiveAgent, AggressiveAgent


def test_new_game_initial_state():
    """After `new_game`, 7 cards are drawn into the hand and the first
    Basic is auto-promoted to active. So the hand has 6 cards (one was
    moved to active), and the deck has been depleted accordingly."""
    from pokemon_tcg.cards import load_cards
    cards = load_cards()
    deck = build_random_deck(cards, seed=42, deck_size=60)
    state = new_game(deck, deck, seed=42)
    assert state.turn == 1
    assert state.active_player in (0, 1)
    assert not state.is_terminal()
    assert state.players[0].prize_count == 6
    assert state.players[1].prize_count == 6
    # If a Basic was in the starting hand, total is 7 - 1 = 6
    # Otherwise, 7 (no promotion)
    if state.players[0].active is not None:
        # 6 because one card was promoted
        assert len(state.players[0].hand) == 6
    else:
        assert len(state.players[0].hand) == 7
    if state.players[1].active is not None:
        assert len(state.players[1].hand) == 6
    else:
        assert len(state.players[1].hand) == 7


def test_new_game_promotes_basic():
    """If at least one Basic is in hand, it should be the initial active."""
    from pokemon_tcg.cards import load_cards
    cards = load_cards()
    # Force a hand with a specific basic
    basics = [c for c in cards.values() if c.pokemon and c.pokemon.stage == "Basic"
              and c.pokemon.moves and any((m.damage or 0) > 0 for m in c.pokemon.moves)]
    assert basics, "expected basics"
    # Put that basic 8 copies in the deck so a random hand always contains it
    target = basics[0]
    deck = [target] * 55 + [basics[1] if len(basics) > 1 else basics[0]] * 5
    state = new_game(deck, deck, seed=42)
    assert state.players[0].active is not None
    assert state.players[0].active.base.name == target.pokemon.name


def test_legal_actions_pass_present():
    """Every turn should include at least one PASS option."""
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    la = legal_actions(state, 0)
    assert any(a.kind == "PASS" for a in la)


def test_legal_actions_play_pokemon():
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    la = legal_actions(state, 0)
    play = [a for a in la if a.kind == "PLAY_POKEMON"]
    # Hand has 6-7 cards; if any are basics a PLAY_POKEMON should be in there
    has_basic_in_hand = any(c.pokemon and c.pokemon.stage == "Basic"
                            for c in state.players[0].hand)
    if has_basic_in_hand:
        assert play, "expected at least one PLAY_POKEMON"


def test_match_runs_to_terminal():
    cards = load_cards()
    deck_a = build_random_deck(cards, seed=42)
    deck_b = build_random_deck(cards, seed=99)
    agents = [GreedyAgent(), DefensiveAgent()]
    result = simulate_match(deck_a, deck_b, agents, seed=42, log=False, max_turns=80)
    assert result["winner"] in (0, 1)
    assert result["turns"] >= 1
    assert result["turns"] <= 80


def test_match_reproducible():
    cards = load_cards()
    deck_a = build_random_deck(cards, seed=42)
    deck_b = build_random_deck(cards, seed=99)
    agents = [GreedyAgent(), GreedyAgent()]
    r1 = simulate_match(deck_a, deck_b, agents, seed=42, log=False, max_turns=80)
    r2 = simulate_match(deck_a, deck_b, agents, seed=42, log=False, max_turns=80)
    # Same decks + same agents + same seed → same outcome
    assert r1["winner"] == r2["winner"]
    assert r1["turns"] == r2["turns"]


def test_ko_resolves_to_prize():
    """If we KO the opponent's only Pokemon, opponent must take a prize."""
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    # Manually KO opponent's active
    opp = state.players[1].active
    if opp is not None:
        opp.hp = 0
        from pokemon_tcg.simulator import _resolve_ko
        _resolve_ko(state, 1)
        # The player who KOs (we, player 0) takes a prize from opponent's pile,
        # so OUR prize_count decreases (we collect their card). Wait —
        # actually re-read the game: when opponent's pokemon is KO'd, the
        # ACTIVE player (us) takes the prize. So state.players[0].prize_count
        # should decrement.
        assert state.players[0].prize_count < 6 or state.winner is not None


def test_match_no_agent_error():
    """Three different agent classes should all complete a match cleanly."""
    cards = load_cards()
    deck_a = build_random_deck(cards, seed=42)
    deck_b = build_random_deck(cards, seed=99)
    for cls in [GreedyAgent, DefensiveAgent, AggressiveAgent]:
        agents = [cls(), cls()]
        result = simulate_match(deck_a, deck_b, agents, seed=7, log=False, max_turns=40)
        # Could finish or could cap out — either way, log shouldn't error
        assert "log" in result


def test_simulator_step_increments_turn():
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    start_turn = state.turn
    start_player = state.active_player
    next_state = step(state.deepcopy(), Action("PASS"))
    assert next_state.turn == start_turn + 1, \
        f"expected turn to increment, got {next_state.turn}"
    assert next_state.active_player != start_player, \
        f"expected player to flip, was {start_player}"


def test_legal_actions_no_active():
    """With no active Pokemon, only PASS is legal (and per-play actions)."""
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    state.players[0].active = None
    la = legal_actions(state, 0)
    assert any(a.kind == "PASS" for a in la)
    assert not any(a.kind == "ATTACK" for a in la)
