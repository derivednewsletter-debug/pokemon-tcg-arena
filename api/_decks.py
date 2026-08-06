"""Deck registry for the arena.

Ten curated 60-card decks (single source of truth) plus validation for
player-built custom decks. The support skeleton is shared: Naveen /
Lillie's Determination / Carmine for draw, Hyper Aroma (ACE SPEC) /
Ultra Ball / Dusk Ball / Energy Search / Pokegear 3.0 for search,
Lucky Helmet + Roto-Stick for consistency, and Rare Candy in the
Stage-2 lines.
"""
from __future__ import annotations

import os
from collections import Counter

from card_db import CardType, card

# ---------------------------------------------------------------------------
# deck definitions
# ---------------------------------------------------------------------------
_SUPP = [(1239, 4), (1227, 4), (1192, 4)]                 # Naveen / Lillie / Carmine
_ITEMS_S1 = [(1082, 1), (1121, 4), (1102, 4), (1119, 4),  # Hyper Aroma ACE, Ultra, Dusk,
             (1122, 4), (1077, 4), (1156, 4)]             # Energy Search, Pokegear, Roto, Helmet
_ITEMS_S2 = [(1082, 1), (1121, 4), (1102, 4), (1119, 4),  # + Rare Candy for Stage-2 lines
             (1122, 4), (1079, 3), (1077, 3), (1156, 2)]

SPECS: list[dict] = [
    {
        "id": "mega_aboma", "type": "water",
        "name": "Mega Abomasnow Wall",
        "blurb": ("The AI's tuned tournament deck — a 350 HP wall that "
                  "swings for 200 while shielding itself. Slow to set up, "
                  "devastating when it lands."),
        "spec": [(722, 4), (723, 4), (721, 4)] + _SUPP + _ITEMS_S1 + [(3, 11)],
    },
    {
        "id": "palafin", "type": "water",
        "name": "Palafin Turbo",
        "blurb": ("Finizen evolves into Palafin, then Zero-to-Hero swaps in "
                  "Palafin ex — 250 damage for one energy. Fast and aggressive."),
        "spec": [(105, 4), (106, 4), (107, 4), (1086, 4)] + _SUPP +
                [(1082, 1), (1121, 4), (1119, 4), (1122, 4), (1156, 4), (1077, 3)] + [(3, 12)],
    },
    {
        "id": "mega_lucario", "type": "fighting",
        "name": "Mega Lucario Beatdown",
        "blurb": ("A pure Fighting beatdown: Riolu into Mega Lucario ex — "
                  "270 damage for two energy. No tricks, just damage."),
        "spec": [(677, 4), (678, 4), (721, 2)] + _SUPP + _ITEMS_S1 + [(6, 13)],
    },
    {
        "id": "mega_dragonite", "type": "dragon",
        "name": "Mega Dragonite ex",
        "blurb": ("Dratini → Dragonair → Mega Dragonite ex: 330 damage for "
                  "three energy on a 370 HP body. The biggest stick in the pool."),
        "spec": [(902, 4), (903, 4), (904, 4)] + _SUPP + _ITEMS_S2 + [(3, 6), (4, 5)],
    },
    {
        "id": "mega_emboar", "type": "fire",
        "name": "Mega Emboar ex",
        "blurb": ("Tepig → Pignite → Mega Emboar ex: 380 HP and 320 damage "
                  "Crimson Blast. Slow, but one shot wins games."),
        "spec": [(567, 4), (568, 4), (932, 4)] + _SUPP + _ITEMS_S2 + [(2, 11)],
    },
    {
        "id": "luxray", "type": "lightning",
        "name": "Luxray ex Volt",
        "blurb": ("Shinx → Luxio → Luxray ex: 250 damage for just two "
                  "energy — the fastest big-hitter in the arena."),
        "spec": [(1035, 4), (1036, 4), (954, 4)] + _SUPP + _ITEMS_S2 + [(4, 11)],
    },
    {
        "id": "mega_gengar", "type": "darkness",
        "name": "Mega Gengar ex",
        "blurb": ("Gastly → Haunter → Mega Gengar ex: a 350 HP wall that "
                  "hits 230 with Void Gale. Tricky, grindy, resilient."),
        "spec": [(59, 4), (60, 4), (772, 4)] + _SUPP + _ITEMS_S2 + [(7, 11)],
    },
    {
        "id": "mega_venusaur", "type": "grass",
        "name": "Mega Venusaur ex",
        "blurb": ("Bulbasaur → Ivysaur → Mega Venusaur ex: 380 HP and Jungle "
                  "Dump for 240. The tankiest deck in the arena."),
        "spec": [(650, 4), (651, 4), (652, 4)] + _SUPP + _ITEMS_S2 + [(1, 11)],
    },
    {
        "id": "greninja", "type": "water",
        "name": "Greninja ex Ninja",
        "blurb": ("Froakie → Frogadier → Greninja ex: 170 damage for a single "
                  "energy. Lightning-fast hit-and-run pressure."),
        "spec": [(33, 4), (34, 4), (40, 4)] + _SUPP + _ITEMS_S2 + [(3, 11)],
    },
    {
        "id": "hydreigon", "type": "darkness",
        "name": "Hydreigon ex Dark",
        "blurb": ("Deino → Zweilous → Hydreigon ex: 200 damage for two energy "
                  "on a 330 HP dragon. Reliable mid-range pressure."),
        "spec": [(227, 4), (228, 4), (229, 4)] + _SUPP + _ITEMS_S2 + [(7, 11)],
    },
]

_cache: dict[str, list[int]] = {}


def _build(spec: list) -> list[int]:
    out: list[int] = []
    for cid, cnt in spec:
        out.extend([cid] * cnt)
    assert len(out) == 60, f"deck has {len(out)} cards, expected 60"
    return out


def get_deck(deck_id: str) -> list[int]:
    spec = next((s for s in SPECS if s["id"] == deck_id), None)
    if spec is None:
        raise KeyError(f"unknown deck {deck_id}")
    if deck_id not in _cache:
        _cache[deck_id] = _build(spec["spec"])
    return list(_cache[deck_id])


def all_decks() -> list[dict]:
    out = []
    for s in SPECS:
        out.append({
            "id": s["id"], "name": s["name"], "blurb": s["blurb"],
            "type": s["type"], "cards": get_deck(s["id"]),
        })
    return out


def deck_summary(deck_id: str) -> dict:
    s = next((x for x in SPECS if x["id"] == deck_id), None) or {}
    return {"id": deck_id, "name": s.get("name", deck_id),
            "type": s.get("type", "")}


# ---------------------------------------------------------------------------
# custom deck validation
# ---------------------------------------------------------------------------
def validate_deck(ids) -> tuple[bool, str]:
    """Check a player-built deck is legal. Returns (ok, reason)."""
    if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
        return False, "deck must be a list of card IDs"
    if len(ids) != 60:
        return False, f"a deck must have exactly 60 cards (you have {len(ids)})"

    counts = Counter(ids)
    basic_pokemon = 0
    ace = 0
    for cid, n in counts.items():
        c = card(cid)
        if c is None:
            return False, f"unknown card id {cid}"
        if c.cardType == CardType.POKEMON and c.basic:
            basic_pokemon += n
        if c.cardType not in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY) and n > 4:
            return False, f"more than 4 copies of {c.name}"
        if getattr(c, "aceSpec", False):
            ace += n
    if basic_pokemon < 1:
        return False, "the deck needs at least 1 Basic Pokémon"
    if ace > 1:
        return False, "only 1 ACE SPEC card is allowed"
    return True, "ok"
