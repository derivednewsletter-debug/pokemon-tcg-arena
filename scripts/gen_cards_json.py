"""Generate the frontend card data.

* ``public/cards.json``  — compact metadata for every card in the 10
  curated decks (used to render boards and hands in-game).
* ``public/catalog.json`` — the full ~1,200 card pool with a simple
  power rating (used by the deck builder).

Run from the repo root:
    python3 scripts/gen_cards_json.py
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "submission"))
sys.path.insert(0, os.path.join(ROOT, "api"))

from card_db import CardType, EnergyType, attack, card  # noqa: E402
from _decks import all_decks  # noqa: E402

CARD_TYPE_NAMES = {
    int(CardType.POKEMON): "pokemon",
    int(CardType.BASIC_ENERGY): "energy",
    int(CardType.SPECIAL_ENERGY): "energy",
    int(CardType.SUPPORTER): "supporter",
    int(CardType.ITEM): "item",
    int(CardType.TOOL): "tool",
    int(CardType.STADIUM): "stadium",
}


def card_entry(cid: int) -> dict | None:
    c = card(cid)
    if c is None:
        return None
    entry = {
        "id": c.cardId,
        "name": c.name,
        "type": CARD_TYPE_NAMES.get(int(c.cardType), "other"),
        "energyType": int(c.energyType) if c.energyType is not None else 0,
        "hp": c.hp,
        "stage": ("basic" if c.basic else
                  "stage1" if c.stage1 else
                  "stage2" if c.stage2 else ""),
        "ex": bool(c.ex),
        "mega": bool(c.megaEx),
        "evolvesFrom": c.evolvesFrom,
        "retreat": c.retreatCost or 0,
        "aceSpec": bool(getattr(c, "aceSpec", False)),
        "attacks": [],
        "text": "",
        "rating": 0,
    }
    best_dmg, best_cost = 0, 0
    for aid in (c.attacks or []):
        a = attack(aid)
        if a is None:
            continue
        dmg = a.damage or 0
        cost = len(a.energies or [])
        entry["attacks"].append({
            "name": a.name, "damage": dmg,
            "cost": [int(e) for e in (a.energies or [])],
            "text": (a.text or "")[:160],
        })
        if dmg > best_dmg:
            best_dmg, best_cost = dmg, cost
    for s in (c.skills or []):
        entry["text"] += (s.name + ": " if s.name else "") + (s.text or "") + " "
    entry["text"] = entry["text"].strip()[:220]

    # simple power rating for the builder's sort order
    if c.cardType == CardType.POKEMON:
        eff = best_dmg / max(1, best_cost) if best_dmg else 0
        entry["rating"] = round(eff * 1.15 + (c.hp or 0) * 0.16 +
                                (55 if c.basic else 90 if c.stage1 else 120), 1)
    else:
        low = entry["text"].lower()
        r = 0.0
        m = re.search(r"draw (\d+)", low)
        if m:
            r += min(int(m.group(1)), 6) * 3
        if "draw until" in low:
            r += 12
        if "search your deck" in low:
            r += 8
        if c.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
            r = 10
        entry["rating"] = round(r, 1)
    return entry


def main() -> None:
    # cards.json — union of all curated decks
    ids: set[int] = set()
    for d in all_decks():
        ids.update(d["cards"])
    cards = {}
    for cid in sorted(ids):
        e = card_entry(cid)
        if e:
            cards[str(cid)] = e

    public = os.path.join(ROOT, "public")
    os.makedirs(public, exist_ok=True)
    with open(os.path.join(public, "cards.json"), "w") as fh:
        json.dump({"cards": cards, "energy_names": {
            int(e): e.name.lower() for e in EnergyType}}, fh, indent=1)
    print(f"cards.json: {len(cards)} cards (curated deck pool)")

    # catalog.json — full pool for the deck builder
    from card_db import all_cards
    catalog = {}
    for c in all_cards():
        e = card_entry(c.cardId)
        if e:
            catalog[str(c.cardId)] = e
    with open(os.path.join(public, "catalog.json"), "w") as fh:
        json.dump({"cards": catalog, "energy_names": {
            int(e): e.name.lower() for e in EnergyType}}, fh)
    size = os.path.getsize(os.path.join(public, "catalog.json")) // 1024
    print(f"catalog.json: {len(catalog)} cards ({size} KiB)")


if __name__ == "__main__":
    main()
