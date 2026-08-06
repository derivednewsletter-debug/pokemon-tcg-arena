"""Pokemon TCG card database.

Loads cards from the EN Card Data.csv and exposes a clean, typed model for
each card kind (Pokemon, Energy, Trainer). Each Pokemon's moves are stored
as a list of `Move` objects; multi-attack Pokemon (one CSV row per attack)
are merged into a single Card.

The data shape here is the simulator's source of truth — any change to
how status effects, energy costs, or damage formula work begins here.
"""
from __future__ import annotations

import csv
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable


# ========================================================================
# Constants
# ========================================================================

POKEMON_TYPES = ("G", "R", "W", "L", "P", "F", "D", "M")  # Grass, Fire, Water, Lightning, Psychic, Fighting, Dark, Metal
TYPE_NAMES = {"G": "Grass", "R": "Fire", "W": "Water", "L": "Lightning",
              "P": "Psychic", "F": "Fighting", "D": "Dark", "M": "Metal"}
COLORLESS = "C"

# Energy token parsed from "{G}{G}" etc. — supports "C", "any", and combos
ENERGY_RE = re.compile(r"\{([A-Za-z]+|any)\}")


def parse_energy_cost(cost: str) -> list[str]:
    """Convert "GGC" -> ["G","G","C"]; "C" alone -> ["C"]."""
    if not cost or cost == "n/a":
        return []
    out = ENERGY_RE.findall(cost)
    # Normalize: {any} is treated as Colorless for cost purposes (the agent
    # can pay it with any single energy from hand).
    return [COLORLESS if t.lower() == "any" else t for t in out]


def parse_damage(d: str) -> int | None:
    if not d or d == "n/a":
        return None
    m = re.match(r"\s*(\d+)", d)
    return int(m.group(1)) if m else None


def parse_retreat(r: str) -> int:
    if not r or r == "n/a":
        return 0
    m = re.match(r"\s*(\d+)", r)
    return int(m.group(1)) if m else 0


# ========================================================================
# Data classes
# ========================================================================

@dataclass(frozen=True)
class Move:
    name: str
    cost: tuple[str, ...]
    damage: int | None
    text: str = ""

    def can_play(self, attached: list[str]) -> bool:
        """Return True if `attached` (energy tokens on this Pokemon)
        satisfies the attack cost. Colorless `C` can be paid by any energy
        type; named types require an exact match in that slot.
        """
        have = list(attached)
        for need in self.cost:
            if need == COLORLESS:
                if not have:
                    return False
                have.pop()
                continue
            try:
                idx = have.index(need)
            except ValueError:
                return False
            have.pop(idx)
        return True

    def cost_total(self) -> int:
        return len(self.cost)

    def is_ko(self, target_hp: int) -> bool:
        """Cheap preview: pure damage comparison (ignores weakness/resist/text)."""
        return self.damage is not None and self.damage >= target_hp


@dataclass(frozen=True)
class PokemonCard:
    name: str
    stage: str  # "Basic", "Stage 1", "Stage 2"
    evolves_from: str | None
    hp: int
    ptype: str  # one of POKEMON_TYPES, COLORLESS, or "" for trainer-tagged
    weakness: str | None
    resistance: str | None
    resistance_value: int
    retreat: int
    moves: tuple[Move, ...]
    is_ex: bool = False
    is_trainer_pokemon: bool = False
    rule_tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def attack_count(self) -> int:
        return len(self.moves)

    def best_damage(self) -> int:
        if not self.moves:
            return 0
        return max((m.damage or 0) for m in self.moves)

    def best_usable_damage(self, attached: list[str]) -> int:
        u = [m for m in self.moves if m.can_play(attached)]
        return max((m.damage or 0) for m in u) if u else 0


@dataclass(frozen=True)
class EnergyCard:
    name: str
    provides: str  # the type it provides (one POKEMON_TYPES letter or COLORLESS)
    is_special: bool = False
    text: str = ""


@dataclass(frozen=True)
class TrainerCard:
    name: str
    category: str  # "Item", "Supporter", "Stadium", "Pokemon Tool"
    text: str = ""

    @property
    def is_supporter(self) -> bool:
        return self.category == "Supporter"

    @property
    def is_stadium(self) -> bool:
        return self.category == "Stadium"


@dataclass(frozen=True)
class Card:
    """Discriminated union wrapper."""
    card_id: str
    pokemon: PokemonCard | None = None
    energy: EnergyCard | None = None
    trainer: TrainerCard | None = None

    @property
    def kind(self) -> str:
        if self.pokemon is not None:
            return "pokemon"
        if self.energy is not None:
            return "energy"
        if self.trainer is not None:
            return "trainer"
        return "unknown"

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        if self.pokemon:
            return f"Card({self.pokemon.name}, {self.pokemon.stage}, {self.pokemon.hp}HP)"
        if self.energy:
            return f"Card({self.energy.name}, {self.energy.provides})"
        if self.trainer:
            return f"Card({self.trainer.name}, {self.trainer.category})"
        return "Card(unknown)"


# ========================================================================
# Parser
# ========================================================================

# Track pokemon by name so multi-attack rows merge into one Card
def _normalize_path(path: str) -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", path))


def load_cards(csv_path: str = "EN Card Data.csv") -> dict[str, Card]:
    """Parse the card CSV into a dict keyed by card name.

    Pokemon entries are *merged* across rows (same name, multiple attacks)
    so each Card holds a complete move list. Card ID is preserved for
    traceability but not used as the lookup key because the CSV repeats IDs
    for multi-attack Pokemon.
    """
    path = _normalize_path(csv_path)
    raw: dict[str, dict] = {}

    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stage = row["Stage (Pokémon)/Type (Energy and Trainer)"].strip()
            name = row["Card Name"].strip()
            card_id = row["Card ID"].strip()
            rule = (row.get("Rule") or "").strip()
            category = (row.get("Category") or "").strip()
            text = (row.get("Effect Explanation") or "").strip()

            entry = raw.setdefault(name, {
                "id": card_id, "stage": stage, "rule": rule,
                "category": category, "evolves": (row.get("Previos stage") or "").strip() or None,
                "hp": row.get("HP", ""), "ptype": (row.get("Type") or "").strip(),
                "weakness": (row.get("Weakness") or "").strip() or None,
                "resistance": (row.get("Resistance (Type)") or "").strip() or None,
                "retreat": row.get("Retreat", ""),
                "moves": [], "text": text,
            })
            # Capture every multi-move row (each attack becomes one Move)
            mv = (row.get("Move Name") or "").strip()
            if mv and stage.endswith("Pokémon"):
                cost = parse_energy_cost(row.get("Cost", ""))
                dmg = parse_damage(row.get("Damage", ""))
                if mv and mv not in [m["name"] for m in entry["moves"]]:
                    entry["moves"].append({"name": mv, "cost": cost, "damage": dmg, "text": text})

    cards: dict[str, Card] = {}
    for name, e in raw.items():
        stage = e["stage"]
        # Catch the "Basic Pokémon" pattern that ships through merged multi-move rows
        if stage.endswith("Pokémon") and not e["moves"]:
            # we already handled it in the energy/trainer block; if stages
            # got here with no moves, skip (will be picked up next time)
            for m_existing in e["moves"]:
                pass
        if "Energy" in stage:
            # Provide type = the parsed {X} token; default to COLORLESS
            provides = COLORLESS
            m = ENERGY_RE.match(e.get("text", "") + e.get("ptype", ""))
            if m:
                provides = COLORLESS if m.group(1).lower() == "any" else m.group(1).upper()
            if not provides:
                provides = COLORLESS
            cards[name] = Card(
                card_id=e["id"],
                energy=EnergyCard(name=name, provides=provides,
                                  is_special="Special" in stage, text=e["text"]),
            )
            continue
        if stage.endswith("Pokémon"):
            # Type extraction: ptype could be "{G}{G}" or "{G}" -> first letter
            ptype = (e.get("ptype") or "").strip()
            m = ENERGY_RE.search(ptype)
            type_letter = COLORLESS
            if m and m.group(1).lower() != "any":
                type_letter = m.group(1).upper()
            # Weakness / Resistance text parsing -> ("R", "+20")
            weakness = _extract_text_after(e.get("weakness") or "", prefix="")
            resistance = e.get("resistance")
            res_value = 0
            if resistance and resistance not in ("n/a", ""):
                rm = re.search(r"([A-Z])\s*([+-]?\d+)", resistance)
                if rm:
                    resistance = rm.group(1)
                    res_value = int(rm.group(2))
            else:
                resistance = None
            moves = tuple(Move(name=m["name"], cost=tuple(m["cost"]),
                               damage=m["damage"], text=m["text"] or "")
                          for m in e["moves"])
            # Normalize stage string: "Basic Pokémon" -> "Basic"
            stage_short = ("Basic" if stage == "Basic Pokémon"
                           else "Stage 1" if stage == "Stage 1 Pokémon"
                           else "Stage 2" if stage == "Stage 2 Pokémon"
                           else stage)
            cards[name] = Card(
                card_id=e["id"],
                pokemon=PokemonCard(
                    name=name, stage=stage_short,
                    evolves_from=e["evolves"],
                    hp=int(e["hp"]) if e["hp"] and e["hp"].isdigit() else 0,
                    ptype=type_letter,
                    weakness=weakness or None,
                    resistance=resistance,
                    resistance_value=res_value,
                    retreat=parse_retreat(e["retreat"]),
                    moves=moves,
                    is_ex="ex" in name.lower() or "ex" in (e["rule"] or "").lower(),
                    is_trainer_pokemon=bool(e["category"]) and category != "n/a",
                    rule_tags=tuple(t for t in re.split(r"[ ,]", e["rule"]) if t),
                ),
            )
            continue
        # Trainer: Item / Supporter / Stadium / Tool
        category_map = {
            "Item": "Item",
            "Pokémon Tool": "Pokemon Tool",
            "Supporter": "Supporter",
            "Stadium": "Stadium",
        }
        category = category_map.get(stage, "Item")
        cards[name] = Card(card_id=e["id"], trainer=TrainerCard(
            name=name, category=category, text=e["text"]))

    return cards


def _extract_text_after(s: str, prefix: str = "") -> str | None:
    if not s:
        return None
    m = ENERGY_RE.search(s)
    if m and m.group(1).lower() != "any":
        return m.group(1).upper()
    return None


def _ensure_legacy_keys(d: dict) -> dict:  # backwards compat for stage strings
    return d

def filter_pokemon(cards: dict[str, Card], require_attack: bool = True) -> list[PokemonCard]:
    """Return all Pokemon cards, optionally dropping those with no attacks."""
    out = []
    for c in cards.values():
        if c.pokemon is None:
            continue
        if require_attack and not c.pokemon.moves:
            continue
        out.append(c.pokemon)
    return out


def filter_by_type(cards: dict[str, Card], ptype: str) -> list[Card]:
    return [c for c in cards.values() if c.pokemon is not None and c.pokemon.ptype == ptype]


# ========================================================================
# CLI
# ========================================================================

if __name__ == "__main__":  # pragma: no cover - manual sanity check
    cards = load_cards()
    pokes = [c for c in cards.values() if c.pokemon is not None]
    energies = [c for c in cards.values() if c.energy is not None]
    trainers = [c for c in cards.values() if c.trainer is not None]
    print(f"cards: {len(cards)}  pokemon: {len(pokes)}  energy: {len(energies)}  trainer: {len(trainers)}")
    print("type breakdown:")
    for t in POKEMON_TYPES:
        n = sum(1 for c in pokes if c.pokemon.ptype == t)
        print(f"  {TYPE_NAMES[t]}: {n}")
    # Sample a few big attackers
    sample = sorted(pokes, key=lambda c: c.pokemon.best_damage(), reverse=True)[:5]
    for c in sample:
        print(f"  {c.pokemon.name} ({c.pokemon.hp}HP {c.pokemon.ptype}) best={c.pokemon.best_damage()}")
