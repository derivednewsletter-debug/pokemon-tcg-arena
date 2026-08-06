"""Deck analysis over the full engine card pool.

Scores the 1267 cards to surface strong, *simple-to-pilot* attackers,
draw/search supporters, and utility items — then validates candidate
60-card decks against the real engine (``battle_start`` must not error).

Usage (from repo root, python 3.11+):
    python3 tools/deck_analysis.py --attackers 15 --supports 12
"""
from __future__ import annotations

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "submission"))

from card_db import (  # noqa: E402
    all_cards, attack, best_potential_attack, CardType, EnergyType,
    estimate_damage, is_basic_pokemon, prize_value,
)

TEXTLESS_DRAW = 0.0


def _text_hint(c) -> str:
    return " ".join(a.text or "" for a in
                    [attack(x) for x in (c.attacks or []) if attack(x)])


def _draw_score(c) -> float:
    """Heuristic: how much draw/search power this trainer provides."""
    if c.cardType == CardType.SUPPORTER:
        t = (c.skills[0].text if c.skills and c.skills[0].text else "") + " " + _text_hint(c)
        low = t.lower()
        score = 0.0
        if "draw" in low and "card" in low:
            import re
            m = re.search(r"draw (\d+)", low)
            if m:
                score = min(int(m.group(1)) * 2.0, 12.0)
            elif "draw until" in low or "draw cards" in low:
                score = 10.0
        if "search" in low:
            score += 6.0
        if "deck" in low and ("your deck" in low) and "search" in low:
            score += 4.0
        if "discard" in low and "hand" in low and ("draw" in low or "switch" in low):
            score += 3.0
        return score
    if c.cardType in (CardType.ITEM, CardType.TOOL, CardType.STADIUM):
        low = (_text_hint(c) or "").lower()
        score = 0.0
        if "search" in low:
            score += 5.0
        if "draw" in low:
            score += 4.0
        if "energy" in low and ("attach" in low or "retriev" in low):
            score += 4.0
        return score
    return 0.0


def attacker_score(c) -> float:
    """Score a Basic Pokemon as a standalone attacker for a heuristic bot.

    Rewards: high HP, cheap high damage, low retreat, non-ex (1 prize).
    Penalises: coin flips in the attack text, self-damage, multi-energy
    requirements, ex/mega (2-3 prizes), 2-energy+ costs with weak payoff.
    """
    if not is_basic_pokemon(c):
        return -1e9
    best = best_potential_attack(c)
    if best is None or not best.damage:
        return -1e9
    text = (best.text or "").lower()
    if "coin" in text or "flip" in text:
        pass  # mild penalty below
    hp = c.hp or 0
    dmg = best.damage or 0
    cost = len(best.energies or [])
    if cost == 0:
        cost = 1
    eff = dmg / cost
    score = hp * 1.0 + eff * 45.0
    score -= (c.retreatCost or 0) * 8.0
    if c.ex:
        score -= 60.0
    if c.megaEx:
        score -= 120.0
    if "coin" in text or "flip" in text:
        score -= 25.0
    if "discard" in text and "energy" in text:
        score -= 20.0
    if cost >= 3 and dmg < 120:
        score -= 30.0
    return score


def top_attackers(n=20):
    out = []
    for c in all_cards():
        s = attacker_score(c)
        if s > -1e8:
            best = best_potential_attack(c)
            out.append((s, c, best))
    out.sort(key=lambda x: -x[0])
    return out[:n]


def top_supporters(n=15):
    out = []
    for c in all_cards():
        if c.cardType != CardType.SUPPORTER:
            continue
        out.append((_draw_score(c), c))
    out.sort(key=lambda x: -x[0])
    return out[:n]


def top_items(n=15):
    out = []
    for c in all_cards():
        if c.cardType not in (CardType.ITEM, CardType.TOOL, CardType.STADIUM):
            continue
        out.append((_draw_score(c), c))
    out.sort(key=lambda x: -x[0])
    return out[:n]


def validate_deck(deck, quiet=True):
    """Validate a 60-card deck against the engine."""
    from cg.game import battle_start, battle_finish
    if len(deck) != 60:
        return "length != 60"
    deck2 = list(deck)
    obs, sd = battle_start(deck, deck2)
    battle_finish()
    if sd.errorType != 0:
        return f"engine error player={sd.errorPlayer} type={sd.errorType}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attackers", type=int, default=15)
    ap.add_argument("--supports", type=int, default=12)
    args = ap.parse_args()

    print("=== TOP BASIC ATTACKERS (heuristic-bot friendly) ===")
    for s, c, best in top_attackers(args.attackers):
        print(f"{s:8.1f}  #{c.cardId:5d} {c.name[:40]:42s} hp={c.hp:4d} "
              f"retr={c.retreatCost} ex={int(c.ex)} mega={int(c.megaEx)} "
              f"best={best.name} dmg={best.damage} cost={len(best.energies or [])}")

    print("\n=== TOP SUPPORTERS (draw/search) ===")
    for s, c in top_supporters(args.supports):
        t = (c.skills[0].text if c.skills else "")[:70].replace("\n", " ")
        print(f"{s:8.1f}  #{c.cardId:5d} {c.name[:40]:42s} {t}")

    print("\n=== TOP ITEMS/TOOLS/STADIUMS (search/draw/energy) ===")
    for s, c in top_items(args.supports):
        t = _text_hint(c)[:70].replace("\n", " ")
        print(f"{s:8.1f}  #{c.cardId:5d} {c.name[:40]:42s} {t}")


if __name__ == "__main__":
    main()
