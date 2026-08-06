"""Scan the full card pool for strong evolution lines (for deck building).

For each Pokémon that can be a deck's "finisher" (stage1/stage2/ex/mega),
report: chain (basic -> stage1 -> stage2), HP, best attack damage + cost,
retreat, and a rough power score. Run from repo root:
    python3 tools/scan_lines.py
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submission"))

from card_db import all_cards, attack, CardType, EnergyType  # noqa: E402

ENERGY = {int(e): e.name.lower() for e in EnergyType}

cards = [c for c in all_cards() if c.cardType == CardType.POKEMON]
by_name: dict[str, list] = {}
for c in cards:
    by_name.setdefault((c.name or "").lower(), []).append(c)


def card_by_name(n: str):
    lst = by_name.get((n or "").lower()) or []
    return lst[0] if lst else None


def best_attack(c):
    best = None
    for aid in (c.attacks or []):
        a = attack(aid)
        if a is None:
            continue
        d = a.damage or 0
        if best is None or d > best[0]:
            best = (d, a)
    return best


def chain_of(c):
    """basic -> stage1 -> stage2 names for a card."""
    out = [c.name]
    cur = c
    seen = set()
    while cur and cur.evolvesFrom and cur.evolvesFrom not in seen:
        seen.add(cur.evolvesFrom)
        out.insert(0, cur.evolvesFrom)
        cur = card_by_name(cur.evolvesFrom)
        if cur is None:
            break
    return out


rows = []
for c in cards:
    if not (c.stage1 or c.stage2 or c.ex or c.megaEx):
        continue
    ba = best_attack(c)
    if ba is None:
        continue
    dmg, a = ba
    hp = c.hp or 0
    # filter: strong wall or strong attacker
    if hp < 230 and dmg < 140:
        continue
    cost = len(a.energies or []) if a.energies else 0
    score = dmg * 0.9 + hp * 0.4 - cost * 40
    chain = chain_of(c)
    rows.append({
        "id": c.cardId, "name": c.name, "hp": hp, "dmg": dmg,
        "cost": cost, "cost_types": [ENERGY.get(int(e), "?") for e in (a.energies or [])],
        "chain": chain, "ex": bool(c.ex), "mega": bool(c.megaEx),
        "score": score, "retreat": c.retreatCost or 0,
    })

rows.sort(key=lambda r: -r["score"])
print(f"{'id':>5} {'name':32s} {'HP':>4} {'dmg':>4} {'cost':>4}  chain")
for r in rows[:40]:
    print(f"{r['id']:>5} {r['name'][:32]:32s} {r['hp']:>4} {r['dmg']:>4} "
          f"{'/'.join(r['cost_types']):>4}  {' -> '.join(r['chain'])}")
