"""Generate ``public/cards.json`` — compact metadata for every card the
arena can show (all cards across the curated decks).

Run from the repo root:
    python3 scripts/gen_cards_json.py
"""
from __future__ import annotations

import json
import os
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


def main() -> None:
    ids: set[int] = set()
    for d in all_decks():
        ids.update(d["cards"])
    ids = sorted(ids)

    cards = {}
    for cid in ids:
        c = card(cid)
        if c is None:
            continue
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
            "aceSpec": bool(c.aceSpec),
            "attacks": [],
            "text": "",
        }
        for aid in (c.attacks or []):
            a = attack(aid)
            if a is None:
                continue
            entry["attacks"].append({
                "name": a.name,
                "damage": a.damage or 0,
                "cost": [int(e) for e in (a.energies or [])],
                "text": (a.text or "")[:160],
            })
        # ability / rule text from skills
        for s in (c.skills or []):
            entry["text"] += (s.name + ": " if s.name else "") + (s.text or "") + " "
        entry["text"] = entry["text"].strip()[:220]
        cards[str(cid)] = entry

    out_path = os.path.join(ROOT, "public", "cards.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({"cards": cards, "energy_names": {
            int(e): e.name.lower() for e in EnergyType}}, fh, indent=1)
    print(f"wrote {len(cards)} cards -> {out_path}")


if __name__ == "__main__":
    main()
