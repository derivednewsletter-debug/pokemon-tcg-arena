"""Trace our agent as P1 vs a random P0."""
import os, sys, time, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "submission"))
from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class
from agent import Agent
from deck import DECK
from strategy import choose_main_action, _main_candidates, evaluate
from card_db import card

rng = random.Random(5)

def random_agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return DECK
    sel = obs.select
    n = len(sel.option)
    if n == 0:
        return []
    k = rng.randint(sel.minCount, sel.maxCount) if sel.maxCount > 0 else 0
    return rng.sample(range(n), k) if k else []

agent = Agent(DECK, lookahead_budget_ms=150.0)
obs, sd = battle_start(DECK, DECK)
steps = 0
mains = 0
while True:
    sel = obs["select"]
    cur = obs["current"]
    if sel is None or cur is None or cur["result"] != -1:
        break
    p = cur["yourIndex"]
    if p == 1 and sel["type"] == 0 and sel["context"] == 0:
        mains += 1
        oo = to_observation_class(obs)
        who = 1
        opts = oo.select.option
        hand = oo.current.players[1].hand or []
        cands = _main_candidates(oo, who, opts)
        desc = []
        for c in cands:
            o = opts[c]
            nm = ""
            if o.type == 7:
                cid = o.cardId or (hand[o.index].id if o.index is not None and o.index < len(hand) else 0)
                cd = card(cid)
                nm = cd.name if cd else f"?{cid}"
            elif o.type == 8:
                nm = f"energy->{o.inPlayArea}/{o.inPlayIndex}"
            elif o.type == 9:
                nm = f"evolve hand[{o.index}]->{o.inPlayArea}/{o.inPlayIndex}"
            elif o.type == 13:
                from card_db import attack
                a = attack(o.attackId)
                nm = (a.name + f"({a.damage})") if a else f"atk#{o.attackId}"
            elif o.type == 14:
                nm = "END"
            elif o.type == 12:
                nm = "RETREAT"
            desc.append((c, o.type, nm))
        t0 = time.time()
        idx = choose_main_action(oo, agent.tracker, agent.lookahead)
        dt = (time.time() - t0) * 1000
        o = opts[idx]
        chn = ""
        if o.type == 7:
            cid = o.cardId or (hand[o.index].id if o.index is not None and o.index < len(hand) else 0)
            cd = card(cid)
            chn = cd.name if cd else f"?{cid}"
        elif o.type == 13:
            from card_db import attack
            a = attack(o.attackId)
            chn = (a.name + f"({a.damage})") if a else f"atk#{o.attackId}"
        elif o.type == 14:
            chn = "END"
        elif o.type == 12:
            chn = "RETREAT"
        elif o.type == 9:
            chn = "EVOLVE"
        elif o.type == 8:
            chn = "ATTACH"
        bench = [card(b.id).name for b in (oo.current.players[1].bench or [])]
        act = oo.current.players[1].active
        actn = card(act[0].id).name if act and act[0] else "None"
        print(f"[M{mains:3d}] turn={cur['turn']} active={actn} bench={bench} "
              f"cands={desc} -> {idx}({o.type}:{chn}) dt={dt:.0f}ms")
        obs = battle_select([idx])
        steps += 1
        if mains > 25:
            break
        continue
    picks = random_agent(obs) if p == 0 else agent.choose(obs)
    obs = battle_select(picks)
    steps += 1
    if steps > 1000:
        break
print("END", steps, "result", cur["result"] if cur else "?")
battle_finish()
