"""Dump raw MAIN option payloads for one game (player 0), printing the
full dict of each option so we can decode PLAY/EVOLVE/ATTACH/ATTACK."""
import sys, os, json, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "submission"))
from cg.game import battle_start, battle_select, battle_finish
from deck import DECK
from card_db import card, attack as atk

deck = DECK
obs, sd = battle_start(deck, deck)
steps = 0
mains = 0
rng = random.Random(7)
while True:
    sel = obs["select"]
    cur = obs["current"]
    if sel is None or cur is None or cur["result"] != -1:
        break
    p = cur["yourIndex"]
    if sel["type"] == 0 and sel["context"] == 0 and p == 0:
        mains += 1
        hand = cur["players"][p]["hand"]
        print(f"\n=== MAIN #{mains} turn={cur['turn']} actAct={cur['turnActionCount']} hand={[c['id'] for c in hand]} ===")
        for i, o in enumerate(sel["option"]):
            payload = {k: v for k, v in o.items() if v is not None and k not in ("serial",)}
            t = o["type"]
            nm = ""
            if t == 7 and o.get("index") is not None and o["index"] < len(hand):
                nm = card(hand[o["index"]]["id"]).name
            elif t == 13:
                a = atk(o.get("attackId"))
                nm = a.name if a else f"atk{o.get('attackId')}"
            print(f"  [{i}] type={t} {nm} payload={payload}")
        if mains >= 3:
            break
        k = 1
        picks = rng.sample(range(len(sel["option"])), 1) if len(sel["option"]) > 1 else [0]
        # prefer attacking to see combat; else play cards
        atk_opts = [i for i, o in enumerate(sel["option"]) if o["type"] == 13]
        if atk_opts and mains > 0:
            picks = [atk_opts[0]]
        obs = battle_select(picks)
        steps += 1
        continue
    n = len(sel["option"])
    k = rng.randint(sel["minCount"], sel["maxCount"]) if sel["maxCount"] > 0 else 0
    obs = battle_select(rng.sample(range(n), k) if k else [])
    steps += 1
    if steps > 500:
        break
print("done", steps)
battle_finish()
