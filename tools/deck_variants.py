"""Deck variants for A/B testing.

Writes each variant to ``kaggle_pokemon/tools/decks/<name>.csv`` and
validates it against the real engine (``battle_start`` must not error).
Use with the selfplay harness via ``--deck-a`` / ``--deck-b``.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "submission"))
sys.path.insert(0, os.path.join(HERE, "..", "submission", "cg"))

VARIANTS = {
    # Palafin Turbo (current default) — 340 HP wall, Giga Impact 250 for 1 {W}
    "palafin": [
        (105, 4), (106, 4), (107, 4),
        (1239, 4), (1227, 4), (1192, 4),
        (1082, 1), (1121, 4), (1086, 4), (1119, 4), (1122, 4), (1077, 3),
        (1156, 4),
        (3, 12),
    ],
    # Mega Abomasnow ex — 350 HP wall, Frost Barrier 200 for 3 {W}
    "mega_aboma": [
        (722, 4), (723, 4), (721, 4),
        (1239, 4), (1227, 4), (1192, 4),
        (1082, 1), (1121, 4), (1119, 4), (1122, 4), (1077, 4), (1102, 4),
        (1156, 4),
        (3, 11),
    ],
    # Mega Lucario ex — 340 HP, Mega Brave 270 for 2 {F}
    "mega_lucario": [
        (677, 4), (678, 4), (721, 2),
        (1239, 4), (1227, 4), (1192, 4),
        (1082, 1), (1121, 4), (1119, 4), (1122, 4), (1077, 4), (1102, 4),
        (1156, 4),
        (6, 13),
    ],
}


def build(name: str) -> list[int]:
    spec = VARIANTS[name]
    deck = []
    for cid, cnt in spec:
        deck.extend([cid] * cnt)
    assert len(deck) == 60, f"{name}: {len(deck)} cards, expected 60"
    return deck


def validate(deck: list[int]) -> str | None:
    """Return an error string if the engine rejects the deck, else None."""
    from cg.game import battle_finish, battle_start
    try:
        obs, sd = battle_start(deck, deck)
        battle_finish()
    except Exception as e:  # noqa: BLE001
        return f"exception: {e}"
    if sd.errorType != 0:
        return f"engine error player={sd.errorPlayer} type={sd.errorType}"
    return None


def main():
    out_dir = os.path.join(HERE, "decks")
    os.makedirs(out_dir, exist_ok=True)
    for name in VARIANTS:
        deck = build(name)
        err = validate(deck)
        path = os.path.join(out_dir, name + ".csv")
        with open(path, "w") as fh:
            fh.write("\n".join(str(c) for c in deck) + "\n")
        status = "OK" if err is None else f"INVALID: {err}"
        print(f"{name}: {len(deck)} cards -> {status} -> {path}")


if __name__ == "__main__":
    main()
