"""Local end-to-end test of the arena API (plays a full game vs the AI).

Run from the repo root:
    python3 scripts/smoke_web.py [games]

Uses Flask's test client so it exercises the exact HTTP surface Vercel
serves. A naive heuristic "human" picks setup/go-first correctly and
otherwise attacks when possible, else plays/ends.
"""
from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import _paths  # noqa: F401  (sys.path bootstrap)
from api.game import app
from api.learn import app as learn_app


def pick_for_menu(menu: dict) -> list[int]:
    items = menu["items"]
    if not items:
        return []
    mn = menu["minCount"] or 1
    if menu["type"] == "IS_FIRST":
        return [0]  # go first
    if menu["type"] == "SETUP":
        return [i["index"] for i in items[:mn]]
    # MAIN: prefer attack, else pokemon play, else first non-end, else end
    for it in items:
        if it["kind"] == "attack":
            return [it["index"]]
    for it in items:
        if it["kind"] == "play":
            return [it["index"]]
    for it in items:
        if it["kind"] != "end":
            return [it["index"]]
    return [items[0]["index"]]


def play_one(client, human_deck: str, ai_deck: str, difficulty: str, label: str) -> dict:
    r = client.post("/api/game/new", json={
        "human_deck": human_deck, "ai_deck": ai_deck, "difficulty": difficulty})
    assert r.status_code == 200, f"new failed: {r.status_code} {r.data[:300]}"
    view = r.get_json()
    steps = 0
    while not view.get("over"):
        menu = view.get("menu")
        assert menu is not None, "no menu and game not over"
        picks = pick_for_menu(menu)
        r = client.post("/api/game/act", json={"game_id": view["game_id"], "picks": picks})
        assert r.status_code == 200, f"act failed: {r.status_code} {r.data[:300]}"
        view = r.get_json()
        steps += 1
        assert steps < 400, "game did not terminate"
    assert view["winner"] in (0, 1), f"bad winner {view['winner']}"
    return view


def play_custom(client, deck_ids, ai_deck, difficulty):
    """Play a game where the human uses a fully custom deck list."""
    r = client.post("/api/game/new", json={
        "custom_deck": deck_ids, "ai_deck": ai_deck, "difficulty": difficulty})
    assert r.status_code == 200, f"custom new failed: {r.status_code} {r.data[:300]}"
    view = r.get_json()
    steps = 0
    while not view.get("over"):
        menu = view.get("menu")
        assert menu is not None, "no menu and not over"
        picks = pick_for_menu(menu)
        r = client.post("/api/game/act", json={"game_id": view["game_id"], "picks": picks})
        assert r.status_code == 200, f"custom act failed: {r.status_code} {r.data[:300]}"
        view = r.get_json()
        steps += 1
        assert steps < 400, "custom game did not terminate"
    assert view["decks"]["human"] == "custom", "custom deck id not recorded"
    return view


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    client = app.test_client()
    wins = 0
    total_turns = 0
    deck_ids = ["mega_aboma", "palafin", "mega_lucario", "mega_dragonite",
                "mega_emboar", "luxray", "mega_gengar", "mega_venusaur",
                "greninja", "hydreigon"]
    for g in range(games):
        deck = deck_ids[g % len(deck_ids)]
        view = play_one(client, deck, "mega_aboma", "medium", f"game {g}")
        wins += 1 if view["winner"] == 1 else 0
        total_turns += view["turn"]
        print(f"game {g} ({deck}): winner={view['winner']} turns={view['turn']}")

    # custom deck: play one full game with a hand-built deck list
    from api._decks import get_deck
    custom = get_deck("luxray")
    cv = play_custom(client, custom, "palafin", "easy")
    print(f"custom deck game: winner={cv['winner']} turns={cv['turn']} "
          f"(human deck recorded as {cv['decks']['human']})")

    # invalid decks must be rejected with 400
    for bad in (["1"] * 60, [], list(range(60)), [3] * 60):
        r = client.post("/api/game/new", json={"custom_deck": bad, "ai_deck": "mega_aboma"})
        assert r.status_code == 400, f"invalid deck {bad[:3]}... not rejected ({r.status_code})"
    print("invalid custom decks rejected (400) ✓")

    # learning endpoint
    r = learn_app.test_client().get("/api/learn")
    assert r.status_code == 200, "learn GET failed"
    data = r.get_json()
    prof = data["profile"]
    print("learning profile:", json.dumps({k: prof[k] for k in
          ("n_games", "agent_win_rate", "aggression", "retreat_freq",
           "supporter_first", "openers")}, indent=1))
    assert prof["n_games"] >= games, "records not shared between modules"
    print("\nSMOKE TEST OK")


if __name__ == "__main__":
    main()
