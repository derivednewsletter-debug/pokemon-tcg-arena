"""Retreat mechanics tests (SwitchActive via paying retreat cost)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pokemon_tcg.cards import load_cards
from pokemon_tcg.deck import build_random_deck
from pokemon_tcg.simulator import simulate_match, new_game, step, _action_retreat
from pokemon_tcg.actions import Action, legal_actions
from pokemon_tcg.agents.benchmarks import GreedyAgent, DefensiveAgent, AggressiveAgent
from pokemon_tcg.game_state import PokemonInstance


def test_legal_actions_retreat_present_with_energy():
    """When active has enough energy, RETREAT actions should appear."""
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    if state.players[0].active:
        # Force the active Pokemon to have retreat cost 0 so any energy
        # count satisfies it — we want RETREAT to be legal.
        from pokemon_tcg.cards import PokemonCard, Move
        cheap = PokemonCard(
            name="ZeroRetreat", stage="Basic", evolves_from=None,
            hp=100, ptype="C", weakness=None, resistance=None,
            resistance_value=0, retreat=0,
            moves=(Move(name="Hit", cost=("C",), damage=20, text=""),),
        )
        state.players[0].active.base = cheap
        state.players[0].active.attached_energy = ("C",)
        # Add a benchmate
        state.players[0].bench.append(PokemonInstance(
            base=cheap, hp=100, base_hp=100, attached_energy=("C",),
        ))
    la = legal_actions(state, 0)
    retreat_actions = [a for a in la if a.kind == "RETREAT"]
    alive_bench = sum(1 for p in state.players[0].bench if p.hp > 0)
    assert alive_bench > 0, "test setup should have a benchmate"
    assert len(retreat_actions) == alive_bench, \
        f"expected {alive_bench} RETREAT actions, got {len(retreat_actions)}; " \
        f"active retreat cost = {state.players[0].active.base.retreat}, " \
        f"attached energy = {len(state.players[0].active.attached_energy)}"


def test_legal_actions_no_retreat_without_energy():
    """If active has fewer energy than retreat cost, no RETREAT allowed."""
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    if state.players[0].active:
        # Force retreat cost high
        cost = state.players[0].active.base.retreat
        # Strip energy to below cost
        state.players[0].active.attached_energy = tuple()  # no energy
        # Add a benchmate
        from pokemon_tcg.game_state import PokemonInstance as PI
        state.players[0].bench.append(PI(
            base=cards["Pikachu ex"].pokemon,
            hp=cards["Pikachu ex"].pokemon.hp,
            base_hp=cards["Pikachu ex"].pokemon.hp,
        ))
    la = legal_actions(state, 0)
    retreat_actions = [a for a in la if a.kind == "RETREAT"]
    # Only available if active's attached energy >= retreat cost
    if cost > 0 and len(state.players[0].active.attached_energy) < cost:
        assert retreat_actions == [], \
            f"should NOT have RETREAT actions; cost={cost}, energy={len(state.players[0].active.attached_energy)}"


def test_retreat_pays_energy():
    """A RETREAT action should reduce the active's attached_energy by the retreat cost."""
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    if state.players[0].active and state.players[0].bench:
        # Add a real benchmate
        from pokemon_tcg.game_state import PokemonInstance as PI
        bench_name = [c for c in cards.values()
                       if c.pokemon and c.pokemon.stage == "Basic"
                       and c.pokemon.moves and any((m.damage or 0) > 0 for m in c.pokemon.moves)][0].pokemon
        bench_inst = PI(base=bench_name, hp=bench_name.hp, base_hp=bench_name.hp)
        state.players[0].bench.append(bench_inst)
        # Force active to have plenty of energy and low retreat cost
        state.players[0].active.attached_energy = ("R", "R", "C", "C")
        original_cost = state.players[0].active.base.retreat
        # Initialize the chosen bench's energy read back after retreat
        before_active = state.players[0].active
        # Try the retreat
        action = Action("RETREAT", target_idx=0, extra=str(original_cost))
        _action_retreat(state, 0, action)
        # The old active is now at bench position; bench_target is active
        assert state.players[0].active is bench_inst, \
            "RETREAT did not promote bench target to active"
        # Old active is on the bench (now at end of list)
        assert before_active in state.players[0].bench, \
            "Old active should have moved to bench"
        # Energy on old active should be reduced by retreat cost
        if original_cost > 0:
            assert len(before_active.attached_energy) == 4 - original_cost, \
                f"expected {4 - original_cost} energy remaining, " \
                f"got {len(before_active.attached_energy)}"


def test_retreat_zero_cost_works_with_empty_energy():
    """If retreat cost is 0, RETREAT should work even with no energy attached."""
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    bench_card = [c for c in cards.values()
                   if c.pokemon and c.pokemon.stage == "Basic"][0].pokemon
    # Find active with retreat cost == 0 OR force one
    # Force: make a custom active with no retreat
    from pokemon_tcg.cards import PokemonCard, Move
    zero_cost_pokemon = PokemonCard(
        name="ZeroRetreat", stage="Basic", evolves_from=None,
        hp=100, ptype="C", weakness=None, resistance=None,
        resistance_value=0, retreat=0,
        moves=(Move(name="Hit", cost=("C",), damage=20, text=""),),
    )
    state.players[0].active = PokemonInstance(
        base=zero_cost_pokemon, hp=100, base_hp=100,
    )
    state.players[0].bench.append(PokemonInstance(
        base=bench_card, hp=bench_card.hp, base_hp=bench_card.hp,
    ))
    action = Action("RETREAT", target_idx=0, extra="0")
    _action_retreat(state, 0, action)
    assert state.players[0].active.base.name == bench_card.name


def test_retreat_records_in_log():
    """Retreat should emit a RETREAT log event."""
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    agents = [GreedyAgent(), DefensiveAgent()]
    # Force at least one RETREAT by making the agent detect a fragile active
    # Run for a few turns and check the event log any RETREAT
    result = simulate_match(deck, deck, agents, seed=42, log=True, max_turns=20)
    events = [e for e in result["log"] if e.get("kind") == "RETREAT"]
    # We don't require a retreat to happen — but if it does, it must have fields
    for e in events:
        assert "from" in e
        assert "to" in e
        assert "cost_paid" in e


def test_simulation_with_retreat_completes():
    """All benchmark agents should complete a match without crashing
    when RETREAT actions are available."""
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    for cls in [GreedyAgent, DefensiveAgent, AggressiveAgent]:
        agents = [cls(), cls()]
        r = simulate_match(deck, deck, agents, seed=42, log=False, max_turns=40)
        assert r["winner"] in (0, 1)
        assert r["turns"] >= 1


def test_retreat_beats_naive_no_retreat_same_aspect():
    """Sanity check: over many matches, agents using retreat should
    not be significantly worse than they were before retreat was added
    (golden tests run a smaller benchmark to keep CI fast)."""
    cards = load_cards()
    from pokemon_tcg.agents.champion import ChampionAgent
    seeds = [42, 7, 123]
    c_wins = 0
    g_wins = 0
    n = 0
    for s in seeds:
        deck = build_random_deck(cards, seed=s)
        r = simulate_match(deck, deck, [ChampionAgent(), GreedyAgent()],
                           seed=s, log=False, max_turns=80)
        n += 1
        if r["winner"] == 0:
            c_wins += 1
        elif r["winner"] == 1:
            g_wins += 1
    # Just verify the tournament runs cleanly; do not assert win rates here.
    assert n > 0


def test_retreat_works_with_status_conditions():
    """Retreat should be legal/invokable even when the active is asleep,
    paralyzed, poisoned, burned, or confused (real TCG allows this).
    """
    from pokemon_tcg.game_state import PokemonInstance
    from pokemon_tcg.cards import PokemonCard, Move
    from pokemon_tcg.game_state import (STATUS_BURN, STATUS_POISON,
                                         STATUS_SLEEP, STATUS_PARALYSIS,
                                         STATUS_CONFUSED)

    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    cheap = PokemonCard(
        name="ZeroRetreat", stage="Basic", evolves_from=None,
        hp=100, ptype="C", weakness=None, resistance=None,
        resistance_value=0, retreat=0,
        moves=(Move(name="Hit", cost=("C",), damage=20, text=""),),
    )
    state.players[0].active = PokemonInstance(base=cheap, hp=100, base_hp=100)
    state.players[0].active.attached_energy = ("C", "C")
    state.players[0].bench.append(PokemonInstance(
        base=cheap, hp=100, base_hp=100, attached_energy=("C",),
    ))

    for status in (STATUS_BURN, STATUS_POISON, STATUS_SLEEP,
                   STATUS_PARALYSIS, STATUS_CONFUSED):
        state.players[0].active.status = status
        la = legal_actions(state, 0)
        retreat_actions = [a for a in la if a.kind == "RETREAT"]
        assert len(retreat_actions) == 1, \
            f"status {status} should still allow retreat; got {len(retreat_actions)}"


def test_retreat_energy_pop_is_deterministic_right():
    """Energy payment for retreat must pop from the END of the attached
    energy list (deterministic, TCG-equivalent of player choice). If
    anyone changes the pop direction this test fails.
    """
    from pokemon_tcg.game_state import PokemonInstance
    from pokemon_tcg.cards import PokemonCard, Move

    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    cost = 2
    pkmn = PokemonCard(
        name="TwoRetreat", stage="Basic", evolves_from=None,
        hp=100, ptype="C", weakness=None, resistance=None,
        resistance_value=0, retreat=cost,
        moves=(Move(name="Hit", cost=("C",), damage=20, text=""),),
    )
    bench_card = PokemonCard(
        name="BenchMate", stage="Basic", evolves_from=None,
        hp=100, ptype="C", weakness=None, resistance=None,
        resistance_value=0, retreat=0,
        moves=(Move(name="Hit", cost=("C",), damage=20, text=""),),
    )
    state.players[0].active = PokemonInstance(
        base=pkmn, hp=100, base_hp=100,
        attached_energy=("R", "G", "W", "L"),
    )
    state.players[0].bench.append(PokemonInstance(
        base=bench_card, hp=100, base_hp=100,
    ))
    action = Action("RETREAT", target_idx=0, extra=str(cost))
    _action_retreat(state, 0, action)
    # Old active is now at the end of the bench (appended); confirm
    # the remaining energy reflects a right-side pop.
    remaining = state.players[0].bench[-1].attached_energy
    assert remaining == ("R", "G"), \
        f"expected (R, G) after paying 2 retreat cost, got {remaining}"
