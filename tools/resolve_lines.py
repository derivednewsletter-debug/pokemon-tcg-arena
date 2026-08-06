"""Resolve card IDs + energy IDs for the 10 curated decks."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submission"))

from card_db import all_cards, card, CardType  # noqa: E402

NAMES = [
    "Dratini", "Dragonair", "Mega Dragonite ex",
    "Tepig", "Pignite", "Mega Emboar ex",
    "Shinx", "Luxio", "Luxray ex",
    "Gastly", "Haunter", "Mega Gengar ex",
    "Bulbasaur", "Ivysaur", "Mega Venusaur ex",
    "Froakie", "Frogadier", "Greninja ex",
    "Deino", "Zweilous", "Hydreigon ex",
]

by_name = {}
for c in all_cards():
    if c.cardType != CardType.POKEMON:
        continue
    by_name.setdefault(c.name, []).append(c)

for n in NAMES:
    lst = by_name.get(n) or []
    c = lst[0] if lst else None
    if c is None:
        print(f"???  {n}")
        continue
    ba = None
    for aid in (c.attacks or []):
        from card_db import attack
        a = attack(aid)
        if a and (ba is None or (a.damage or 0) > (ba.damage or 0)):
            ba = a
    print(f"#{c.cardId:5d} {c.name:22s} hp={c.hp:4d} stage1={int(c.stage1)} stage2={int(c.stage2)} "
          f"ex={int(c.ex)} basic={int(c.basic)} evoFrom={c.evolvesFrom} "
          f"best={(ba.name + ' ' + str(ba.damage)) if ba else '-'}")

print("\nEnergy cards:")
for c in all_cards():
    if c.cardType in (CardType.BASIC_ENERGY,) and c.name:
        print(f"#{c.cardId:5d} {c.name}")
