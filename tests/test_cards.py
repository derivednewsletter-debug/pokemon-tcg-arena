"""Card database tests."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pokemon_tcg.cards import (
    load_cards, parse_energy_cost, parse_damage, parse_retreat,
    POKEMON_TYPES, COLORLESS, Move, PokemonCard, EnergyCard, Card,
)


def test_parse_energy_basic():
    assert parse_energy_cost("{G}{G}{C}") == ["G", "G", "C"]
    assert parse_energy_cost("{R}") == ["R"]
    assert parse_energy_cost("n/a") == []
    assert parse_energy_cost("") == []
    assert parse_energy_cost("{any}") == ["C"]


def test_parse_energy_combined():
    assert parse_energy_cost("{R}{R}{C}") == ["R", "R", "C"]
    assert parse_energy_cost("{G}{C}") == ["G", "C"]
    assert parse_energy_cost("{L}{L}") == ["L", "L"]


def test_parse_damage():
    assert parse_damage("120") == 120
    assert parse_damage("0") == 0
    assert parse_damage("n/a") is None
    assert parse_damage("") is None
    assert parse_damage("30x") == 30


def test_parse_retreat():
    assert parse_retreat("2") == 2
    assert parse_retreat("0") == 0
    assert parse_retreat("n/a") == 0
    assert parse_retreat("") == 0


def test_cards_loaded():
    cards = load_cards()
    assert len(cards) > 900, f"expected >900 unique card names, got {len(cards)}"
    kinds = {"pokemon": 0, "energy": 0, "trainer": 0}
    for c in cards.values():
        kinds[c.kind] = kinds.get(c.kind, 0) + 1
    assert kinds["pokemon"] > 700
    assert kinds["energy"] >= 8
    assert kinds["trainer"] >= 100


def test_pokemon_stage_normalized():
    cards = load_cards()
    for c in cards.values():
        if c.pokemon:
            assert c.pokemon.stage in ("Basic", "Stage 1", "Stage 2"), \
                f"unexpected stage {c.pokemon.stage!r} for {c.pokemon.name}"


def test_pokemon_have_hp():
    cards = load_cards()
    sample = [c for c in cards.values() if c.pokemon and c.pokemon.stage == "Basic"][:30]
    for c in sample:
        assert c.pokemon.hp > 0, f"{c.pokemon.name} has no HP"


def test_pokemon_have_attacks_or_filtered():
    """Deck builder filters out Pokemon with no damage-dealing attacks; the
    raw dataset may include search-style attacks with damage=None or 0."""
    cards = load_cards()
    with_attacks = sum(1 for c in cards.values()
                        if c.pokemon and c.pokemon.moves
                        and any((m.damage or 0) > 0 for m in c.pokemon.moves))
    assert with_attacks > 400, f"too few Pokemon with attacks ({with_attacks})"


def test_move_can_play_simple():
    m = Move(name="X", cost=("R", "R"), damage=80)
    assert m.can_play(["R", "R"])
    assert not m.can_play(["R"])
    assert not m.can_play([])
    # Colorless as any
    m2 = Move(name="Y", cost=("C",), damage=30)
    assert m2.can_play(["R"])
    assert m2.can_play(["W"])
    assert not m2.can_play([])


def test_move_can_play_mixed():
    m = Move(name="Z", cost=("R", "C"), damage=120)
    assert m.can_play(["R", "R"])
    assert m.can_play(["R", "W"])
    assert not m.can_play(["R"])
    assert not m.can_play(["W"])


def test_basic_energies_present():
    cards = load_cards()
    have = {c.energy.provides for c in cards.values() if c.energy and not c.energy.is_special}
    assert have == {"G", "R", "W", "L", "P", "F", "D", "M"}, have


def test_special_energies_present():
    cards = load_cards()
    sp = [c for c in cards.values() if c.energy and c.energy.is_special]
    assert len(sp) >= 5, f"expected >=5 special energies, got {len(sp)}"


def test_pokemon_type_distribution():
    cards = load_cards()
    counts = {t: 0 for t in POKEMON_TYPES}
    for c in cards.values():
        if c.pokemon and c.pokemon.ptype in counts:
            counts[c.pokemon.ptype] += 1
    total = sum(counts.values())
    assert total > 700
    # Every type should have at least 30 Pokemon
    for t, n in counts.items():
        assert n >= 30, f"type {t} only has {n} pokemon"


def test_best_damage_handles_empty():
    """Both cards and PokemonInstance must return 0 for empty move lists."""
    from pokemon_tcg.game_state import PokemonInstance
    # Construct a PokemonCard with no moves
    p = PokemonCard(name="Test", stage="Basic", evolves_from=None,
                    hp=100, ptype="C", weakness=None, resistance=None,
                    resistance_value=0, retreat=0, moves=())
    assert p.best_damage() == 0
    assert p.best_usable_damage([]) == 0
    inst = PokemonInstance(base=p, hp=p.hp, base_hp=p.hp)
    assert inst.best_damage() == 0
    assert inst.best_usable_damage() == 0


def test_real_pokemon_sometimes_have_empty_moves():
    """Some real cards (e.g. Fossil cards) have no attacks; verify dataset."""
    cards = load_cards()
    no_move_pokes = [c for c in cards.values()
                      if c.pokemon and not c.pokemon.moves]
    if no_move_pokes:
        # If we found any, they should still have HP
        for c in no_move_pokes:
            assert c.pokemon.hp > 0
