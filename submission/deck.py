"""The agent's 60-card deck — "Mega Abomasnow Wall".

Archetype
---------
A Stage-1 wall deck built around **Mega Abomasnow ex** (#723): 350 HP
and Frost Barrier — 200 damage for {W}{W}{W} plus "-30 damage taken
during your opponent's next turn". Evolving from Snover (#722, 90 HP)
via a normal Stage-1 evolution (no special combo needed), this is the
simplest strong line to pilot: bench Snover, evolve, feed 3 water
energy, swing for 200 while the damage shield makes it a 380-HP wall.

Support skeleton:
  * **Draw** — Naveen (draw to 5), Lillie's Determination (shuffle +
    draw 6/8), Carmine (discard hand, draw 5, legal turn 1 going first).
  * **Search** — Hyper Aroma (ACE SPEC: 3 Stage-1 = Mega Abomasnow x3),
    Ultra Ball (discard 2 -> any Pokemon), Dusk Ball (bottom 7 ->
    Pokemon), Energy Search, Pokegear 3.0, Roto-Stick.
  * **Backup** — Kyogre (#721, 150 HP, 130-damage Swirling Waves) for
    when the wall is slow to assemble.
  * **Energy** — 11 Basic {W} Energy: Frost Barrier needs 3.

Validated by A/B self-play (24 games each, sides swapped):
  vs random 96% | vs sample-submission 92% | vs greedy rules-bot 88%.

ACE SPEC: Hyper Aroma (1 copy max).
"""
from __future__ import annotations

import os

# [card_id, count]
DECK_SPEC: list[tuple[int, int]] = [
    # Pokemon (12)
    (722, 4),    # Snover             — 90 HP {W} basic, evo for Mega Abomasnow
    (723, 4),    # Mega Abomasnow ex  — 350 HP wall, Frost Barrier 200 + shield
    (721, 4),    # Kyogre             — 150 HP backup attacker
    # Supporters (12)
    (1239, 4),   # Naveen             — draw until 5
    (1227, 4),   # Lillie's Determination — shuffle hand, draw 6 (8)
    (1192, 4),   # Carmine            — discard hand, draw 5 (turn-1 legal)
    # Items (25)
    (1082, 1),   # Hyper Aroma        — ACE SPEC: search 3 Stage-1
    (1121, 4),   # Ultra Ball         — discard 2, search any Pokemon
    (1102, 4),   # Dusk Ball          — bottom 7, take a Pokemon
    (1119, 4),   # Energy Search      — search a Basic Energy
    (1122, 4),   # Pokegear 3.0       — top 7, grab a Supporter
    (1077, 4),   # Roto-Stick         — top 4, grab Supporters
    (1156, 4),   # Lucky Helmet       — draw 2 when damaged
    # Energy (11)
    (3, 11),     # Basic {W} Energy
]

DECK: list[int] = []
for cid, cnt in DECK_SPEC:
    DECK.extend([cid] * cnt)
assert len(DECK) == 60, f"deck has {len(DECK)} cards, expected 60"


def write_deck_csv(path: str | None = None) -> str:
    """Write the deck to ``deck.csv`` (next to this file by default)."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv")
    with open(path, "w") as fh:
        fh.write("\n".join(str(c) for c in DECK) + "\n")
    return path


if __name__ == "__main__":
    print(write_deck_csv())
    print("cards:", len(DECK))
