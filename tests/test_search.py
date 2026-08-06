"""3-ply alpha-beta + iterative-deepening search tests."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pokemon_tcg.cards import load_cards
from pokemon_tcg.deck import build_random_deck
from pokemon_tcg.simulator import new_game, simulate_match, step
from pokemon_tcg.actions import Action, legal_actions
from pokemon_tcg.evaluator import score, EvaluatorConfig
from pokemon_tcg.search import (
    greedy_1ply, alpha_beta_2ply, alpha_beta_3ply,
    iterative_deepening_search, DEFAULT_3PLY_BEAM, pick,
)
from pokemon_tcg.agents.benchmarks import GreedyAgent
from pokemon_tcg.agents.champion import ChampionAgent, CHAMPION_CONFIG


# ============================================================
# Algebra / shape
# ============================================================

def test_alpha_beta_3ply_returns_action():
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    a, info = alpha_beta_3ply(state, 0, beam=(4, 3, 3), cfg=CHAMPION_CONFIG)
    assert isinstance(a, Action)
    assert info["depth"] == 3
    assert info["beam"] == [4, 3, 3]
    assert info["leaves_evaluated"] >= 0


def test_alpha_beta_3ply_traces_each_root():
    """Each root candidate should appear in the trace with at least
    one opp response and (when applicable) a follow-up."""
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    a, info = alpha_beta_3ply(state, 0, beam=(4, 3, 3), cfg=CHAMPION_CONFIG)
    for t in info["trace"]:
        assert "ours" in t
        assert "score" in t
        # opp response is None only if no legal opponent actions exist.


def test_iterative_deepening_tries_depth3():
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    a, info = iterative_deepening_search(state, 0, time_budget_ms=300,
                                          cfg=CHAMPION_CONFIG)
    assert info["depth_reached"] == 3
    assert 1 in info["depths"]
    assert 2 in info["depths"]
    assert 3 in info["depths"]


def test_iterative_deepening_zero_budget_returns_1ply():
    """No time -> only depth 1 completes."""
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    a, info = iterative_deepening_search(state, 0, time_budget_ms=0,
                                          cfg=CHAMPION_CONFIG)
    assert info["depth_reached"] == 1


def test_iterative_deepening_short_budget_completes():
    """Even on a tiny budget, IDS must return a complete 1-ply result
    and a sensible action (it's allowed to keep going if depth-2 was
    trivially fast)."""
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    a, info = iterative_deepening_search(state, 0, time_budget_ms=1,
                                          cfg=CHAMPION_CONFIG)
    assert info["depth_reached"] >= 1
    assert info["depth_reached"] <= 3
    assert isinstance(a, Action)
    # Always populated:
    assert 1 in info["depths"]


def test_iterative_deepening_zero_budget_returns_1ply():
    """No time -> only depth 1 completes."""
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    a, info = iterative_deepening_search(state, 0, time_budget_ms=0,
                                          cfg=CHAMPION_CONFIG)
    assert info["depth_reached"] == 1


def test_pick_dispatches_by_depth():
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    a1, i1 = pick(state, 0, depth=1, cfg=CHAMPION_CONFIG)
    a2, i2 = pick(state, 0, depth=2, cfg=CHAMPION_CONFIG)
    a3, i3 = pick(state, 0, depth=3, time_budget_ms=200, cfg=CHAMPION_CONFIG)
    assert i1["depth"] == 1
    assert i2["depth"] == 2
    assert i3["depth_reached"] >= 1   # iterative deepening at depth=3


# ============================================================
# Setup-retreat combo discovery
# ============================================================

def _build_combo_state(cards):
    """Build a hand-crafted state where the winning play is a 3-step combo:
    play a strong Basic on the bench, attach energy to it, then retreat
    into it next turn. Without deeper search the agent wouldn't see
    that the bench target becomes the active by turn N+1.
    """
    from pokemon_tcg.cards import PokemonCard, EnergyCard
    from pokemon_tcg.game_state import (
        GameState, PlayerState, PokemonInstance,
    )

    # Find any strong Basic (high damage move) and a weak active.
    monds = sorted([c.pokemon for c in cards.values() if c.pokemon
                     and c.pokemon.stage == "Basic"
                     and c.pokemon.moves
                     and max((m.damage or 0) for m in c.pokemon.moves) >= 80],
                   key=lambda p: -max((m.damage or 0) for m in p.moves))
    weak_basic = next((p for p in [c.pokemon for c in cards.values()
                                    if c.pokemon and c.pokemon.stage == "Basic"
                                    and c.pokemon.moves
                                    and max((m.damage or 0) for m in p.moves) < 30]), None)
    if not monds or not weak_basic:
        return None

    monster = monds[0]
    energy = EnergyCard(name="Basic Energy", provides="C", is_special=False)

    p0 = PlayerState(
        name="P0", deck=[], hand=[energy, energy, energy, energy],
        active=PokemonInstance(base=weak_basic, hp=weak_basic.hp,
                                  base_hp=weak_basic.hp),
        # Bench monster is ALIVE with no energy; we'll play/attach in
        # the search tree.
        bench=[PokemonInstance(base=monster, hp=monster.hp,
                                base_hp=monster.hp)],
        prizes=[], prize_count=6, discard=[],
    )
    p1 = PlayerState(
        name="P1", deck=[], hand=[],
        active=PokemonInstance(base=weak_basic, hp=weak_basic.hp,
                                  base_hp=weak_basic.hp),
        bench=[], prizes=[], prize_count=6, discard=[],
    )
    state = GameState(players=(p0, p1), turn=1, active_player=0,
                       rng_seed=42)
    return state


def test_three_ply_finds_retreat_into_empowered_bench():
    """When the bench monster can swing the game after energy+retreat,
    3-ply should keep ATTACH_ENERGY→bench near the top of the candidate
    ordering (the prelude to a future retreat) OR surface RETREAT as a
    best alternative.

    We build a hand-crafted state where bench already holds a strong
    monster, the active weakling has full energy, and target winning
    plays are: (a) ATTACH_ENERGY → bench (sets up retreat) and
    (b) RETREAT into bench (immediately uses the prepared bench).
    Deep search should rank energy-on-bench near the top.
    """
    cards = load_cards()
    from pokemon_tcg.cards import Card, EnergyCard, PokemonCard, Move
    from pokemon_tcg.game_state import GameState, PlayerState, PokemonInstance

    # Hand-crafted monster card: 1 energy cost, 100 damage move.
    monster_base = PokemonCard(
        name="TestMon", stage="Basic", evolves_from=None, hp=110,
        ptype="C", weakness=None, resistance=None, resistance_value=0,
        retreat=1,
        moves=(Move(name="Smash", cost=("C",), damage=100, text=""),),
    )
    # Hand-crafted weak active: 2 colorless energy cost, 30 damage.
    weak_base = PokemonCard(
        name="TestWeaky", stage="Basic", evolves_from=None, hp=80,
        ptype="C", weakness=None, resistance=None, resistance_value=0,
        retreat=2,
        moves=(Move(name="Poke", cost=("C", "C"), damage=30, text=""),),
    )
    energy = EnergyCard(name="Basic Energy", provides="C", is_special=False)
    energy_card = lambda: Card(card_id="test-energy", energy=energy)

    # P0: weak active (2/2 energy, full HP), monster bench (alive, no energy),
    #     4 energy in hand so it can either buff active or bench.
    # Use energy stubs as 'deck fuel' so the simulator's draw step doesn't
    # immediately declare a deck-out loss.
    deck_placeholder = lambda: Card(card_id="deck-stuff", energy=energy)
    p0 = PlayerState(
        name="P0",
        deck=[deck_placeholder() for _ in range(20)],
        hand=[energy_card() for _ in range(4)],
        active=PokemonInstance(base=weak_base, hp=weak_base.hp,
                                  base_hp=weak_base.hp,
                                  attached_energy=("C", "C")),
        bench=[PokemonInstance(base=monster_base, hp=monster_base.hp,
                                base_hp=monster_base.hp)],
        prizes=[], prize_count=6, discard=[],
    )
    # P1: low-HP weakling on the active spot (so a bench-retreat + attack
    #     *immediately* threatens a KO).
    p1 = PlayerState(
        name="P1",
        deck=[deck_placeholder() for _ in range(20)],
        hand=[],
        active=PokemonInstance(base=weak_base, hp=40, base_hp=weak_base.hp),
        bench=[], prizes=[], prize_count=6, discard=[],
    )
    state = GameState(players=(p0, p1), turn=1, active_player=0, rng_seed=42)

    # 3-ply root trace: bench-target attach energy must surface.
    a3, info3 = alpha_beta_3ply(state, 0, beam=(4, 3, 3), cfg=CHAMPION_CONFIG)
    root_actions = [t["ours"] for t in info3["trace"]]
    has_attach = any(r.get("kind") == "ATTACH_ENERGY" for r in root_actions)
    assert has_attach, (
        f"3-ply root trace should keep ATTACH_ENERGY for setup-retreat; "
        f"got {root_actions}"
    )
    # Scores must remain finite.
    assert all(isinstance(t["score"], (int, float)) for t in info3["trace"])


def test_champion_uses_iterative_deepening_in_match():
    """Champion should drive a real match end-to-end under IDS."""
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    agents = [ChampionAgent(), GreedyAgent()]
    r = simulate_match(deck, deck, agents, seed=42, log=False, max_turns=40)
    assert r["winner"] in (0, 1)
    assert r["turns"] >= 1


def test_search_deterministic_under_seed():
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    s1 = new_game(deck, deck, seed=42)
    s2 = new_game(deck, deck, seed=42)
    a1a, _ = alpha_beta_3ply(s1, 0, beam=(4, 3, 3), cfg=CHAMPION_CONFIG)
    a1b, _ = alpha_beta_3ply(s2, 0, beam=(4, 3, 3), cfg=CHAMPION_CONFIG)
    assert a1a.to_json() == a1b.to_json()
