"""Curated decks the arena can use (both for the human and the AI)."""
from __future__ import annotations

import os

from _paths import repo_root

_SPECS = [
    {
        "id": "mega_aboma",
        "name": "Mega Abomasnow Wall",
        "blurb": ("The AI's tuned tournament deck — a 350 HP wall that "
                  "swings for 200 while shielding itself. Slow to set up, "
                  "devastating when it lands."),
        "source": "builtin",
    },
    {
        "id": "palafin",
        "name": "Palafin Turbo",
        "blurb": ("Finizen evolves into Palafin, then the Zero-to-Hero "
                  "ability swaps in Palafin ex — 250 damage for one "
                  "energy. Fast and aggressive."),
        "source": "file",
        "path": ["tools", "decks", "palafin.csv"],
    },
    {
        "id": "mega_lucario",
        "name": "Mega Lucario Beatdown",
        "blurb": ("A pure Fighting beatdown: Riolu into Mega Lucario ex, "
                  "270 damage for two energy. No tricks, just damage."),
        "source": "file",
        "path": ["tools", "decks", "mega_lucario.csv"],
    },
]

_cache: dict[str, list[int]] = {}


def _load_deck_ids(spec: dict) -> list[int]:
    if spec["source"] == "builtin":
        from deck import DECK
        return list(DECK)
    root = repo_root()
    path = os.path.join(root, *spec["path"])
    with open(path) as fh:
        ids = [int(x) for x in fh.read().split() if x.strip()]
    assert len(ids) == 60, f"deck {spec['id']} has {len(ids)} cards"
    return ids


def all_decks() -> list[dict]:
    out = []
    for spec in _SPECS:
        out.append({
            "id": spec["id"],
            "name": spec["name"],
            "blurb": spec["blurb"],
            "cards": get_deck(spec["id"]),
        })
    return out


def get_deck(deck_id: str) -> list[int]:
    spec = next((s for s in _SPECS if s["id"] == deck_id), None)
    if spec is None:
        raise KeyError(f"unknown deck {deck_id}")
    if deck_id not in _cache:
        _cache[deck_id] = _load_deck_ids(spec)
    return list(_cache[deck_id])
