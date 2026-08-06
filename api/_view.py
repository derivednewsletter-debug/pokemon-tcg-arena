"""Translate engine observations into JSON the frontend renders.

Two jobs:

1. **Board view** — the visible game state: both players' Active/Bench
   Pokémon (with current HP, attached energy, tools, status), the
   human's hand, prize/deck counts. The AI's hand is never revealed.
2. **Action menu** — the human's current choices as labelled buttons:
   MAIN actions ("Attack: Frost Barrier (200)", "Play Snover",
   "End turn"), setup picks, and the go-first decision.
"""
from __future__ import annotations

from cg.api import SelectContext, SelectType
from card_db import EnergyType, attack, card
from strategy import (
    OPT_ABILITY, OPT_ATTACH, OPT_ATTACK, OPT_DISCARD, OPT_END,
    OPT_EVOLVE, OPT_PLAY, OPT_RETREAT,
)

ENERGY_NAMES = {int(e): e.name.lower() for e in EnergyType}

STATUS_FLAGS = ("poisoned", "burned", "asleep", "paralyzed", "confused")

SETUP_PROMPTS = {
    SelectContext.SETUP_ACTIVE_POKEMON: "Choose your Active Pokémon",
    SelectContext.SETUP_BENCH_POKEMON: "Choose your Bench Pokémon",
}


# ---------------------------------------------------------------------------
# small card/pokemon views
# ---------------------------------------------------------------------------
def _card_name(cid: int) -> str:
    c = card(cid)
    return c.name if c else f"Card #{cid}"


def pokemon_view(p: dict) -> dict:
    return {
        "id": p.get("id"),
        "name": _card_name(p.get("id") or 0),
        "hp": p.get("hp"),
        "maxHp": p.get("maxHp"),
        "energies": [int(e) for e in (p.get("energies") or [])],
        "tools": [_card_name(t["id"]) for t in (p.get("tools") or []) if t],
    }


def player_view(ps: dict, reveal_hand: bool) -> dict:
    active = ps.get("active") or []
    act = active[0] if active and active[0] else None
    return {
        "active": pokemon_view(act) if act else None,
        "bench": [pokemon_view(p) for p in (ps.get("bench") or []) if p],
        "hand": [
            {"id": c.get("id"), "name": _card_name(c.get("id") or 0),
             "serial": c.get("serial")}
            for c in (ps.get("hand") or [])
        ] if reveal_hand else None,
        "handCount": ps.get("handCount"),
        "deckCount": ps.get("deckCount"),
        "prizeCount": len(ps.get("prize") or []),
        "discardCount": len(ps.get("discard") or []),
        "status": {k: bool(ps.get(k)) for k in STATUS_FLAGS},
    }


# ---------------------------------------------------------------------------
# option resolution / labels
# ---------------------------------------------------------------------------
def option_card_id(sel: dict, o: dict, state: dict, who: int) -> int | None:
    """Best-effort card ID an option refers to (deck / hand / prize)."""
    if o.get("cardId"):
        return o["cardId"]
    idx = o.get("index")
    if idx is None:
        return None
    deck = sel.get("deck")
    if deck is not None and o.get("area") == 1 and 0 <= idx < len(deck):
        return deck[idx].get("id")
    hand = (state["players"][who].get("hand") or [])
    if 0 <= idx < len(hand):
        return hand[idx].get("id")
    return None


def _pokemon_at(state: dict, who: int, o: dict) -> dict | None:
    area, ipos = o.get("inPlayArea"), o.get("inPlayIndex")
    ps = state["players"][who]
    if area == 4:
        active = ps.get("active") or []
        return active[0] if active and active[0] else None
    if area == 5:
        bench = ps.get("bench") or []
        if ipos is not None and 0 <= ipos < len(bench):
            return bench[ipos]
    return None


def _target_name(target: dict | None) -> str:
    """Name of an in-play pokemon dict (raw engine dicts have no name)."""
    if not target:
        return "a Pokémon"
    return _card_name(target.get("id") or 0)


def describe_main_option(state: dict, who: int, o: dict, hand: list[dict]) -> str:
    """Human-readable label for a MAIN option."""
    t = o.get("type")
    idx = o.get("index")

    if t == OPT_PLAY:
        cid = hand[idx]["id"] if (idx is not None and 0 <= idx < len(hand)) else None
        return f"Play {_card_name(cid)}" if cid else "Play a card"

    if t == OPT_ATTACH:
        cid = hand[idx]["id"] if (idx is not None and 0 <= idx < len(hand)) else None
        ename = _card_name(cid) if cid else "Energy"
        target = _pokemon_at(state, who, o)
        return f"Attach {ename} to {_target_name(target)}"

    if t == OPT_EVOLVE:
        cid = hand[idx]["id"] if (idx is not None and 0 <= idx < len(hand)) else None
        target = _pokemon_at(state, who, o)
        return (f"Evolve {_target_name(target)} into {_card_name(cid)}"
                if cid else "Evolve a Pokémon")

    if t == OPT_RETREAT:
        target = _pokemon_at(state, who, o)
        return "Retreat " + (_target_name(target) if target else "the Active")

    if t == OPT_ATTACK:
        a = attack(o.get("attackId"))
        if a:
            cost = " ".join(ENERGY_NAMES.get(int(e), "?") for e in (a.energies or []))
            return f"Attack: {a.name} — {a.damage or 0} dmg [{cost}]"
        return "Attack"

    if t == OPT_ABILITY:
        return "Use an ability"

    if t == OPT_DISCARD:
        cid = hand[idx]["id"] if (idx is not None and 0 <= idx < len(hand)) else None
        return f"Discard {_card_name(cid)}" if cid else "Discard"

    if t == OPT_END:
        return "End turn"

    return f"Action {idx}"


# ---------------------------------------------------------------------------
# menus
# ---------------------------------------------------------------------------
def build_menu(obs: dict, who: int) -> dict | None:
    """Return the human's current choice menu (None when nothing to pick)."""
    sel = obs["select"]
    if sel is None:
        return None
    t = sel["type"]
    state = obs["current"]

    if t == SelectType.MAIN:
        hand = state["players"][who].get("hand") or []
        items = []
        for i, o in enumerate(sel["option"]):
            items.append({
                "index": i,  # option position — what the client sends back
                "kind": _kind(o.get("type")),
                "label": describe_main_option(state, who, o, hand),
            })
        return {
            "type": "MAIN",
            "prompt": "Choose your action",
            "items": items,
            "minCount": sel["minCount"],
            "maxCount": sel["maxCount"],
        }

    if t == SelectType.YES_NO and sel.get("context") == SelectContext.IS_FIRST:
        return {
            "type": "IS_FIRST",
            "prompt": "Who goes first?",
            "items": [
                {"index": 0, "kind": "first", "label": "You go first"},
                {"index": 1, "kind": "second", "label": "You go second"},
            ],
            "minCount": 1, "maxCount": 1,
        }

    if t == SelectType.CARD and sel.get("context") in SETUP_PROMPTS:
        hand = state["players"][who].get("hand") or []
        items = []
        for i, o in enumerate(sel["option"]):
            cid = option_card_id(sel, o, state, who)
            items.append({
                "index": i,  # option position
                "cardId": cid,
                "kind": "card",
                "label": _card_name(cid) if cid else "?",
            })
        return {
            "type": "SETUP",
            "prompt": SETUP_PROMPTS[sel.get("context")],
            "items": items,
            "minCount": sel["minCount"],
            "maxCount": sel["maxCount"],
        }

    return None


def _kind(t: int | None) -> str:
    return {
        OPT_PLAY: "play", OPT_ATTACH: "attach", OPT_EVOLVE: "evolve",
        OPT_RETREAT: "retreat", OPT_ATTACK: "attack", OPT_ABILITY: "ability",
        OPT_DISCARD: "discard", OPT_END: "end",
    }.get(t, "other")
