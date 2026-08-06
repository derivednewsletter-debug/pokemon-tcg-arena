"""Aggregate human-play records into an *opponent profile*.

Every completed web game produces a record: who won, which decks were
used, who went first, and — crucially — a flat list of the *human's*
MAIN-action types (``attack``, ``play_supporter``, ``play_item``,
``play_pokemon``, ``attach``, ``evolve``, ``retreat``, ``end``) in the
order the player made them.

``build_profile`` boils those records down into a handful of scalar
tendencies. The AI agent consumes this profile as ``opp_profile``:
its engine-search lookahead then simulates the opponent's replies using
the learned human behavior instead of generic optimal play — so the AI
*adapts to how people actually play against it* (e.g. it no longer
wastes search budget expecting retreats humans never make, and it values
moves that punish attack-first humans).
"""
from __future__ import annotations

from collections import Counter

ACTION_LABELS = {
    "attack": "Attacks",
    "play_supporter": "Supporters",
    "play_item": "Items",
    "play_pokemon": "Pokémon",
    "attach": "Energy attach",
    "evolve": "Evolutions",
    "retreat": "Retreats",
    "ability": "Abilities",
    "end": "Ends turn",
}


def empty_profile() -> dict:
    return {
        "n_games": 0,
        "agent_win_rate": None,
        "human_win_rate": None,
        "aggression": 0.5,
        "supporter_first": 0.0,
        "retreat_freq": 0.5,
        "avg_turns": None,
        "by_deck": {},
        "action_dist": {},
        "openers": [],
    }


def build_profile(records: list[dict]) -> dict:
    """Turn a list of finished-game records into a tendency profile."""
    prof = empty_profile()
    n = len(records)
    if n == 0:
        return prof
    prof["n_games"] = n

    agent_wins = sum(1 for r in records if r.get("winner") == 1)
    prof["agent_win_rate"] = agent_wins / n
    prof["human_win_rate"] = sum(1 for r in records if r.get("winner") == 0) / n
    prof["avg_turns"] = round(
        sum(r.get("turns") or 0 for r in records) / n, 1)

    # deck preferences
    by_deck = Counter(r.get("human_deck") or "unknown" for r in records)
    prof["by_deck"] = dict(by_deck.most_common())

    # action tendencies across all human decisions
    all_actions: list[str] = []
    first_actions: list[str] = []
    for r in records:
        acts = r.get("human_actions") or []
        all_actions.extend(acts)
        if acts:
            first_actions.append(acts[0])

    if all_actions:
        dist = Counter(all_actions)
        total = len(all_actions)
        prof["action_dist"] = {
            ACTION_LABELS.get(k, k): round(v / total, 3)
            for k, v in dist.most_common()
        }
        prof["aggression"] = round(dist.get("attack", 0) / total, 3)
        prof["retreat_freq"] = round(dist.get("retreat", 0) / total, 3)

    if first_actions:
        opener = Counter(first_actions)
        prof["openers"] = [
            {"action": ACTION_LABELS.get(k, k), "count": v}
            for k, v in opener.most_common(5)
        ]
        prof["supporter_first"] = round(opener.get("play_supporter", 0) / len(first_actions), 3)

    return prof
