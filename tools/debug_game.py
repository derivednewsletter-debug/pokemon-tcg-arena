"""Debug one game: print our MAIN decisions with card names, attacks, evolves."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "submission"))

from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class
from agent import Agent
from deck import DECK
from strategy import choose_main_action, _main_candidates
from card_db import card

agent = Agent(DECK, lookahead_budget_ms=200.0)
deck = DECK
obs, sd = battle_start(deck, deck)
print("start err", sd.errorPlayer, sd.errorType)
steps = 0
my_mains = 0
while True:
    sel = obs["select"]
    cur = obs["current"]
    if sel is None or cur is None or cur["result"] != -1:
        break
    p = cur["yourIndex"]
    if sel["type"] == 0 and sel["context"] == 0 and p == 0:
        my_mains += 1
        oo = to_observation_class(obs)
        who = oo.current.yourIndex
        opts = oo.select.option
        hand = oo.current.players[who].hand or []
        cands = _main_candidates(oo, who, opts)
        desc = []
        for c in cands:
            o = opts[c]
            nm = ""
            if o.type == 7:  # PLAY
                cid = o.cardId or (hand[o.index].id if o.area == 2 and o.index is not None and o.index < len(hand) else 0)
                cd = card(cid)
                nm = cd.name if cd else f"?{cid}"
            elif o.type == 8:  # ATTACH
                nm = f"energy->{o.inPlayArea}/{o.inPlayIndex}"
            elif o.type == 9:  # EVOLVE
                nm = f"evolve hand[{o.index}]->{o.inPlayArea}/{o.inPlayIndex}"
            elif o.type == 13:
                nm = f"atk#{o.attackId}"
            elif o.type == 14:
                nm = "END"
            desc.append((c, o.type, nm))
        t0 = time.time()
        idx = choose_main_action(oo, agent.tracker, agent.lookahead)
        dt = (time.time() - t0) * 1000
        chosen = opts[idx]
        chn = ""
        if chosen.type == 7:
            cid = chosen.cardId or (hand[chosen.index].id if chosen.area == 2 and chosen.index is not None and chosen.index < len(hand) else 0)
            cd = card(cid)
            chn = cd.name if cd else f"?{cid}"
        elif chosen.type == 13:
            chn = f"atk#{chosen.attackId}"
        elif chosen.type == 14:
            chn = "END"
        elif chosen.type == 9:
            chn = "EVOLVE"
        print(f"[M{my_mains:3d}] turn={cur['turn']} hand={len(hand)} actAct={cur['turnActionCount']} "
              f"cands={desc} -> {idx}({chosen.type}:{chn}) dt={dt:.0f}ms")
        obs = battle_select([idx])
        steps += 1
        if my_mains > 40:
            break
        continue
    picks = agent.choose(obs)
    obs = battle_select(picks)
    steps += 1
    if steps > 2000:
        print("MAX STEPS"); break
print("END steps", steps, "result", cur["result"] if cur else "?", "my_mains", my_mains)
battle_finish()
