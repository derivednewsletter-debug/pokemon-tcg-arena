"""Dump full engine details for specific card IDs."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "submission"))
from card_db import card, attack, all_cards

IDS = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [311, 602, 676, 210, 328, 971, 216]

for cid in IDS:
    c = card(cid)
    if c is None:
        print(f"#{cid}: NOT FOUND"); continue
    print(f"#{cid} {c.name} | type={c.cardType} hp={c.hp} retr={c.retreatCost} "
          f"basic={c.basic} s1={c.stage1} s2={c.stage2} ex={c.ex} megaEx={c.megaEx} "
          f"tera={c.tera} aceSpec={c.aceSpec} energyType={c.energyType} "
          f"weak={c.weakness} resist={c.resistance} evolvesFrom={c.evolvesFrom}")
    for s in (c.skills or []):
        print(f"    SKILL: {s.name}: {s.text[:160]}")
    for aid in (c.attacks or []):
        a = attack(aid)
        if a:
            print(f"    ATK #{aid} {a.name} dmg={a.damage} cost={[int(e) for e in (a.energies or [])]} :: {a.text[:150]}")
    print()

print("=== BASIC ENERGIES ===")
for c in all_cards():
    if c.cardType == 5:
        print(f"#{c.cardId} {c.name} energyType={c.energyType}")
