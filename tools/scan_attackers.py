"""Scan the pool for reliable, unconditional attackers.

Prints basics and evolution lines whose best attack has no conditional
language (no "does nothing", coin flips, or setup requirements) and good
damage-per-energy.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "submission"))
from card_db import all_cards, attack, CardType

BAD_WORDS = ["does nothing", "coin", "flip", "if you don't", "if your opponent",
             "same name", "in any way", "if you have", "exactly", "unless",
             "if this pokémon", "if this pokemon"]

def is_conditional(text):
    low = (text or "").lower()
    return any(w in low for w in BAD_WORDS)

def best_unconditional(c, allow_self_damage=True):
    best, bd, bc = None, 0, 99
    for aid in (c.attacks or []):
        a = attack(aid)
        if a is None or not a.damage or a.damage <= 0:
            continue
        if is_conditional(a.text or ""):
            continue
        cost = len(a.energies or [])
        d = a.damage
        if d > bd or (d == bd and cost < bc):
            best, bd, bc = a, d, cost
    return best

print("=== BASIC POKEMON with unconditional attack >= 90 dmg, <= 2 energy ===")
rows = []
for c in all_cards():
    if c.cardType != CardType.POKEMON or not c.basic:
        continue
    a = best_unconditional(c)
    if a is None or a.damage < 90 or len(a.energies or []) > 2:
        continue
    hp = c.hp or 0
    if hp < 90:
        continue
    cost = len(a.energies or [])
    eff = a.damage / max(cost, 1)
    score = hp * 0.5 + a.damage * 1.0 - cost * 30 - (c.retreatCost or 0) * 10 - (80 if c.ex else 0) - (160 if c.megaEx else 0)
    rows.append((score, c, a, hp, cost))
rows.sort(key=lambda r: -r[0])
for score, c, a, hp, cost in rows[:40]:
    txt = (a.text or "")[:60].replace("\n", " ")
    print(f"{score:7.1f} #{c.cardId:5d} {c.name[:38]:40s} hp={hp:3d} retr={c.retreatCost} "
          f"ex={int(c.ex)} type={c.energyType} | {a.name}: {a.damage}dmg cost={cost} :: {txt}")

print("\n=== TOP STAGE1/2 with unconditional attack >= 120 dmg, <= 3 energy ===")
rows = []
for c in all_cards():
    if c.cardType != CardType.POKEMON or c.basic:
        continue
    a = best_unconditional(c)
    if a is None or a.damage < 120 or len(a.energies or []) > 3:
        continue
    hp = c.hp or 0
    cost = len(a.energies or [])
    eff = a.damage / max(cost, 1)
    score = hp * 0.5 + a.damage * 1.0 - cost * 30 - (c.retreatCost or 0) * 10 - (80 if c.ex else 0) - (160 if c.megaEx else 0)
    rows.append((score, c, a, hp, cost))
rows.sort(key=lambda r: -r[0])
for score, c, a, hp, cost in rows[:30]:
    txt = (a.text or "")[:60].replace("\n", " ")
    print(f"{score:7.1f} #{c.cardId:5d} {c.name[:38]:40s} hp={hp:3d} retr={c.retreatCost} "
          f"ex={int(c.ex)} evoFrom={c.evolvesFrom} | {a.name}: {a.damage}dmg cost={cost} :: {txt}")
