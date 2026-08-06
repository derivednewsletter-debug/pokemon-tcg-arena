"""Card database backed by the official engine (``cg.api``).

Loads every card and attack from the engine's authoritative database once
per process and exposes helpers for the agent: attack usability, damage
estimation (weakness x2 / resistance -30, matching the engine's
``CalcDamage``), KO math, and prize-value lookups.

The engine's dataclasses are re-exported so the rest of the agent never
needs to import ``cg.api`` directly.
"""
from __future__ import annotations

import threading
from collections import Counter
from typing import Optional

from cg.api import (
    Attack, Card, CardData, CardType, EnergyType, Log, LogType,
    Observation, Option, OptionType, Pokemon, PlayerState, SelectContext,
    SelectData, SelectType, SpecialConditionType, State,
)

# Re-export engine enums/dataclasses used across the agent.
__all__ = [
    "Attack", "Card", "CardData", "CardType", "EnergyType", "Log", "LogType",
    "Observation", "Option", "OptionType", "Pokemon", "PlayerState",
    "SelectContext", "SelectData", "SelectType", "SpecialConditionType",
    "State",
]

# ---------------------------------------------------------------------------
# Database loading (cached)
# ---------------------------------------------------------------------------

_CARD: dict[int, CardData] = {}
_ATTACK: dict[int, Attack] = {}
_lock = threading.Lock()
_loaded = False


def ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    with _lock:
        if _loaded:
            return
        from cg.api import all_attack, all_card_data
        for c in all_card_data():
            _CARD[c.cardId] = c
        for a in all_attack():
            _ATTACK[a.attackId] = a
        _loaded = True


def card(card_id: int) -> Optional[CardData]:
    ensure_loaded()
    return _CARD.get(card_id)


def attack(attack_id: int) -> Optional[Attack]:
    ensure_loaded()
    return _ATTACK.get(attack_id)


def all_cards() -> list[CardData]:
    ensure_loaded()
    return list(_CARD.values())


# ---------------------------------------------------------------------------
# Energy helpers
# ---------------------------------------------------------------------------

def energy_count(pokemon: Pokemon) -> int:
    """Number of energy attached to a Pokemon (resolved energy list)."""
    if pokemon is None:
        return 0
    return len(pokemon.energies) if pokemon.energies else 0


def can_use_attack(pokemon: Pokemon, atk: Attack) -> bool:
    """True if the Pokemon's attached energy satisfies the attack cost."""
    if pokemon is None or atk is None:
        return False
    have: Counter = Counter(pokemon.energies or [])
    need = list(atk.energies or [])
    # Prefer paying colored costs with matching energy; colorless can be
    # paid with anything. Rainbow pays any colored slot.
    for t in need:
        if t == EnergyType.COLORLESS:
            if sum(have.values()) > 0:
                _take_any(have)
            else:
                return False
        else:
            if have.get(t, 0) > 0:
                have[t] -= 1
                if have[t] == 0:
                    del have[t]
            elif have.get(EnergyType.RAINBOW, 0) > 0:
                have[EnergyType.RAINBOW] -= 1
                if have[EnergyType.RAINBOW] == 0:
                    del have[EnergyType.RAINBOW]
            else:
                return False
    return True


def _take_any(have: Counter) -> None:
    for k, v in list(have.items()):
        if v > 0:
            if v == 1:
                del have[k]
            else:
                have[k] = v - 1
            return


def usable_attacks(pokemon: Pokemon, card_data: CardData) -> list[Attack]:
    if pokemon is None or card_data is None:
        return []
    out = []
    for aid in card_data.attacks or []:
        a = attack(aid)
        if a is not None and can_use_attack(pokemon, a):
            out.append(a)
    return out


def best_damage_attack(pokemon: Pokemon, card_data: CardData) -> Optional[Attack]:
    """Highest-damage attack the Pokemon can currently use."""
    best, bd = None, -1
    for a in usable_attacks(pokemon, card_data):
        d = a.damage or 0
        if d > bd:
            best, bd = a, d
    return best


def best_potential_attack(card_data: CardData) -> Optional[Attack]:
    """Highest-damage attack the card could ever use (ignoring energy)."""
    best, bd = None, -1
    for aid in card_data.attacks or []:
        a = attack(aid)
        if a is not None and (a.damage or 0) > bd:
            best, bd = a, a.damage or 0
    return best


# ---------------------------------------------------------------------------
# Damage / KO math  (mirrors engine CalcDamage: weakness x2, resist -30)
# ---------------------------------------------------------------------------

def _type_contains(container: Optional[EnergyType], t: EnergyType) -> bool:
    if container is None:
        return False
    if container == t:
        return True
    if container == EnergyType.RAINBOW:
        return True
    if container == EnergyType.TEAM_ROCKET and t in (EnergyType.PSYCHIC, EnergyType.DARKNESS):
        return True
    return False


def estimate_damage(attack: Attack, attacker: CardData, defender: CardData) -> int:
    """Estimate damage an attack deals to a defender (weakness x2, resist -30)."""
    if attack is None or attack.damage is None or attack.damage <= 0:
        return 0
    dmg = attack.damage
    text = attack.text or ""
    if "Don't apply Weakness and Resistance" in text or "isn't affected by Weakness" in text:
        return dmg
    atk_type = attacker.energyType if attacker else EnergyType.COLORLESS
    if defender is not None and _type_contains(defender.weakness, atk_type):
        dmg *= 2
    if defender is not None and _type_contains(defender.resistance, atk_type):
        dmg -= 30
        if dmg <= 0:
            return 0
    return dmg


def is_ko(attack: Attack, attacker: CardData, defender: CardData,
          defender_hp: int, margin: int = 0) -> bool:
    """True if the attack would knock out a defender at ``defender_hp``.

    ``margin`` adds safety so we don't over-rely on exact damage math
    (e.g. +20 to hedge against unmodelled attack-text modifiers).
    """
    if defender_hp <= 0:
        return True
    d = estimate_damage(attack, attacker, defender)
    return d >= defender_hp + margin


# ---------------------------------------------------------------------------
# Card classification
# ---------------------------------------------------------------------------

def prize_value(c: CardData) -> int:
    """Prize cards the opponent takes for KOing this Pokemon."""
    if c is None:
        return 1
    if c.megaEx:
        return 3
    if c.ex:
        return 2
    return 1


def is_basic_pokemon(c: CardData) -> bool:
    return c is not None and c.cardType == CardType.POKEMON and c.basic


def is_energy(c: CardData) -> bool:
    return c is not None and c.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)


def is_trainer(c: CardData) -> bool:
    return c is not None and c.cardType in (CardType.ITEM, CardType.TOOL,
                                            CardType.SUPPORTER, CardType.STADIUM)


def energy_type_letter(t: EnergyType) -> str:
    return {0: "C", 1: "G", 2: "R", 3: "W", 4: "L", 5: "P", 6: "F",
            7: "D", 8: "M", 9: "N", 10: "A", 11: "Rk"}.get(int(t), "?")


def describe(card_id: int) -> str:
    c = card(card_id)
    if c is None:
        return f"#{card_id}"
    s = f"#{card_id} {c.name}"
    if c.cardType == CardType.POKEMON:
        s += f" [{c.hp}hp basic={c.basic} s1={c.stage1} s2={c.stage2} ex={c.ex} mega={c.megaEx} retr={c.retreatCost}]"
    return s
