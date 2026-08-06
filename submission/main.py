"""Kaggle submission entry point — "Palafin Turbo" agent.

The platform calls ``agent(obs_dict)`` for every selection. The first
call (``obs.select is None``) must return the 60-card deck; every later
call returns option indices.

Agent summary
-------------
* **Deck**: Finizen -> Palafin ex wall (340 HP, 250 dmg for 1 {W}) with
  a heavy draw/search skeleton (Naveen, Lillie's Determination, Carmine,
  Hyper Aroma, Ultra Ball, Buddy-Buddy Poffin) and Cramorant backup.
* **Policy**: a position evaluator over prize lead / board / KO threat /
  resources, driving every MAIN decision, plus **engine-search
  lookahead** (``search_begin``/``search_step``) that plays each
  candidate action and the opponent's reply through the real rules
  engine under predicted hidden information before choosing.
* **Safety**: every selection path validates against the engine's
  min/max counts and falls back to a legal random pick on any error.
"""
from __future__ import annotations

import os

from agent import Agent

# ---- deck -----------------------------------------------------------------
def _load_deck() -> list[int]:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "deck.csv")
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/" + os.path.basename(path)
    if os.path.exists(path):
        with open(path) as fh:
            ids = [int(x) for x in fh.read().split() if x.strip()]
        if len(ids) == 60:
            return ids
    from deck import DECK
    return DECK# Tunable policy knobs (set via environment for local experiments).
GO_FIRST = os.environ.get("PTCG_GO_FIRST", "1") != "0"
LOOKAHEAD_MS = float(os.environ.get("PTCG_LOOKAHEAD_MS", "110"))
USE_LOOKAHEAD = os.environ.get("PTCG_NO_LOOKAHEAD", "0") == "0"
WORLDS = int(os.environ.get("PTCG_WORLDS", "2"))
ROUNDS = int(os.environ.get("PTCG_ROUNDS", "2"))


_AGENT: Agent | None = None


def _get_agent() -> Agent:
    global _AGENT
    if _AGENT is None:
        _AGENT = Agent(_load_deck(), go_first=GO_FIRST,
                       lookahead_budget_ms=LOOKAHEAD_MS,
                       use_lookahead=USE_LOOKAHEAD,
                       worlds=WORLDS, rounds=ROUNDS)
    return _AGENT


def agent(obs_dict: dict) -> list[int]:
    """Main competition entry point."""
    return _get_agent().choose(obs_dict)


if __name__ == "__main__":
    # Local smoke test: play one game against a random opponent.
    import random
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cg.game import battle_start, battle_select, battle_finish

    deck = _load_deck()
    obs, sd = battle_start(deck, deck)
    print("battle_start err:", sd.errorPlayer, sd.errorType)
    steps = 0
    while True:
        sel = obs["select"]
        cur = obs["current"]
        if sel is None or cur is None or cur["result"] != -1:
            break
        picks = agent(obs)
        assert isinstance(picks, list) and len(picks) >= sel["minCount"] and \
            len(picks) <= sel["maxCount"], f"bad picks {picks} for {sel['type']}"
        obs = battle_select(picks)
        steps += 1
        if steps > 2000:
            print("too many steps")
            break
    print("game ended at step", steps, "result:", cur["result"] if cur else "?")
    battle_finish()
