import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "submission"))
from card_db import all_cards, attack, card, CardType

# Find Finizen and pre-evolutions
for c in all_cards():
    if c.cardType != CardType.POKEMON:
        continue
    n = (c.name or "").lower()
    if any(k in n for k in ("finizen", "slakoth", "vigoroth", "froakie", "frogadier",
                            "scorbunny", "raboot", "cramorant", "pikachu ex", "lunatone")):
        print(f"#{c.cardId:5d} {c.name[:40]:42s} hp={c.hp:4d} retr={c.retreatCost} "
              f"basic={int(c.basic)} s1={int(c.stage1)} s2={int(c.stage2)} ex={int(c.ex)} "
              f"mega={int(c.megaEx)} tera={int(c.tera)} type={c.energyType} evoFrom={c.evolvesFrom}")
        for aid in (c.attacks or []):
            a = attack(aid)
            if a:
                print(f"      ATK {a.name}: {a.damage}dmg cost={[int(e) for e in (a.energies or [])]} :: {(a.text or '')[:80]}")

print()
for cid in [1249, 1169, 1121, 1125, 1082, 1086, 1119, 1100, 1101, 1115, 1156, 1096]:
    c = card(cid)
    if c is None:
        print(f"#{cid} NOT FOUND"); continue
    print(f"#{cid} {c.name} type={c.cardType} ace={int(c.aceSpec)}")
    for s in (c.skills or []):
        print(f"   {s.text}")
