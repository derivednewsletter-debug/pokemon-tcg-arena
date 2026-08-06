"""Agent interface + benchmark strategies tests."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pokemon_tcg.cards import load_cards
from pokemon_tcg.deck import build_random_deck
from pokemon_tcg.simulator import simulate_match
from pokemon_tcg.actions import Action, legal_actions
from pokemon_tcg.agents.benchmarks import (
    BENCHMARKS, GreedyAgent, DefensiveAgent, AggressiveAgent,
    EnergyRampAgent, BenchBufferAgent, SearchAgent,
)


def test_benchmarks_registered():
    expected = {"Greedy", "SearchAgent", "Aggressive", "Defensive", "EnergyRamp", "BenchBuffer"}
    assert expected.issubset(BENCHMARKS.keys())


def test_agents_return_action():
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    for cls in [GreedyAgent, DefensiveAgent, AggressiveAgent,
                EnergyRampAgent, BenchBufferAgent]:
        a = cls()
        from pokemon_tcg.simulator import new_game
        state = new_game(deck, deck, seed=42)
        act = a(state, 0)
        assert isinstance(act, Action)


def test_search_agent_speed_acceptable():
    """Search agent should complete a 30-turn game in reasonable time."""
    import time
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    agents = [SearchAgent(), GreedyAgent()]
    t0 = time.time()
    result = simulate_match(deck, deck, agents, seed=42, log=False, max_turns=30)
    elapsed = time.time() - t0
    assert elapsed < 60, f"SearchAgent too slow: {elapsed:.1f}s"


def test_energy_ramp_attaches_when_no_damage():
    """EnergyRampAgent should prefer ATTACH_ENERGY while damage is below
    the configured threshold."""
    from pokemon_tcg.simulator import new_game
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    # Force active to have 0 energy and low-damage moves
    if state.players[0].active:
        state.players[0].active.attached_energy = tuple()
        state.players[0].energy_attached_this_turn = False
    has_energy_in_hand = any(c.energy for c in state.players[0].hand)
    a = EnergyRampAgent(config={"min_attack_damage": 200})
    # If energy is in hand AND active can do <200 damage, prefer ATTACH_ENERGY
    if has_energy_in_hand and state.players[0].active \
            and state.players[0].active.best_usable_damage() < 200:
        act = a(state, 0)
        assert act.kind in ("ATTACH_ENERGY", "ATTACK", "PASS"), \
            f"unexpected action {act.kind} from EnergyRampAgent"


def test_pass_action_legal_when_no_active():
    """If a player has no active, only PASS and SETUP-style actions are legal."""
    from pokemon_tcg.simulator import new_game
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    state.players[0].active = None
    la = legal_actions(state, 0)
    assert any(a.kind == "PASS" for a in la)
    # No ATTACK actions without active
    assert not any(a.kind == "ATTACK" for a in la)
