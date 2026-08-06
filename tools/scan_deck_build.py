"""Scan full evolution lines + trainers for deck building."""
import sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "submission"))
from card_db import all_cards, attack, CardType

def dump(c, tag=""):
    print(f"  #{c.cardId:5d} {c.name[:42]:44s} hp={c.hp:4d} retr={c.retreatCost} "
          f"basic={int(c.basic)} s1={int(c.stage1)} s2={int(c.stage2)} ex={int(c.ex)} "
          f"mega={int(c.megaEx)} tera={int(c.tera)} ace={int(c.aceSpec)} type={c.energyType} evoFrom={c.evolvesFrom} {tag}")
    for aid in (c.attacks or []):
        a = attack(aid)
        if a:
            print(f"      ATK {a.name}: {a.damage}dmg cost={[int(e) for e in (a.energies or [])]} :: {(a.text or '')[:90]}")

print("=== EVOLUTION LINES ===")
TARGETS = ["Palafin", "Rotom", "Greninja", "Cinderace", "Mega Dragonite", "Garchomp", "Slaking"]
by_id = {c.cardId: c for c in all_cards()}
by_name = {c.name: c for c in all_cards()}
for t in TARGETS:
    matches = [c for c in all_cards() if c.cardType == CardType.POKEMON and t.lower() in (c.name or "").lower()]
    for c in matches:
        dump(c)

print("\n=== ALL SUPPORTERS with draw/search (full text) ===")
supps = []
for c in all_cards():
    if c.cardType != CardType.SUPPORTER:
        continue
    text = c.skills[0].text if c.skills else ""
    low = text.lower()
    if any(k in low for k in ("draw", "search", "shuffle your hand", "prize")):
        supps.append((c, text))
def draw_n(text):
    m = re.search(r"draw (\d+)", text.lower())
    return int(m.group(1)) if m else (10 if "draw until" in text.lower() or "draw cards until" in text.lower() else 0)
supps.sort(key=lambda x: -draw_n(x[1]))
for c, text in supps[:25]:
    first_line = text.split("\n")[0]
    print(f"  #{c.cardId:5d} {c.name[:38]:40s} :: {first_line[:95]}")

print("\n=== ITEMS with search/draw (full text) ===")
items = []
for c in all_cards():
    if c.cardType not in (CardType.ITEM, CardType.TOOL, CardType.STADIUM):
        continue
    text = c.skills[0].text if c.skills else ""
    low = text.lower()
    if any(k in low for k in ("search", "draw", "deck")):
        items.append((c, text))
for c, text in items:
    first_line = text.split("\n")[0]
    print(f"  #{c.cardId:5d} {c.name[:38]:40s} :: {first_line[:95]}")
