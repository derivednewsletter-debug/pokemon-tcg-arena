"""Dump all curated decks to tools/decks/*.csv and sanity-check them.

Usage:
    python3 tools/gen_decks.py          # write CSVs + battle_start check
    python3 tools/gen_decks.py --play  # + quick self-play vs random
"""
from __future__ import annotations

import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "api"))
sys.path.insert(0, os.path.join(ROOT, "submission"))

from _decks import all_decks, get_deck, validate_deck  # noqa: E402


def main() -> None:
    out_dir = os.path.join(ROOT, "tools", "decks")
    os.makedirs(out_dir, exist_ok=True)
    decks = all_decks()

    for d in decks:
        ok, reason = validate_deck(d["cards"])
        assert ok, f"{d['id']}: {reason}"
        path = os.path.join(out_dir, d["id"] + ".csv")
        with open(path, "w") as fh:
            fh.write("\n".join(str(c) for c in d["cards"]) + "\n")
        print(f"{d['id']:16s} 60 cards OK -> {path}")

    if "--play" not in sys.argv:
        return

    from cg.game import battle_start, battle_select, battle_finish

    def random_pick(sel):
        n = len(sel["option"])
        k = max(sel["minCount"], min(sel["maxCount"], n))
        return random.sample(list(range(n)), k) if k else []

    for d in decks:
        deck = d["cards"]
        obs, sd = battle_start(deck, deck)
        assert obs is not None, f"{d['id']} battle_start failed: {sd.errorPlayer} {sd.errorType}"
        wins, errors = 0, 0
        for _ in range(4):
            obs, _ = battle_start(deck, deck)
            steps = 0
            winner = None
            while steps < 2000:
                sel, cur = obs["select"], obs["current"]
                if sel is None or cur is None or cur["result"] != -1:
                    winner = cur["result"] if cur else None
                    break
                picks = random_pick(sel)
                try:
                    obs = battle_select(picks)
                except Exception as e:
                    errors += 1
                    break
                steps += 1
            if winner == 0:
                wins += 1
            battle_finish()
        print(f"{d['id']:16s} vs-random {wins}/4, errors={errors}")


if __name__ == "__main__":
    main()
