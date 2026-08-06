"""Trace when EVOLVE options appear: play our deck with a simple policy that
puts Finizen on bench and keeps Palafin ex in hand, dumping the board."""
import sys, os, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "submission"))
from cg.game import battle_start, battle_select, battle_finish
from deck import DECK
from card_db import card

deck = DECK
obs, sd = battle_start(deck, deck)
rng = random.Random(3)
steps = 0
mains = 0
while True:
    sel = obs["select"]
    cur = obs["current"]
    if sel is None or cur is None or cur["result"] != -1:
        break
    p = cur["yourIndex"]
    ps = cur["players"][p]
    if sel["type"] == 0 and sel["context"] == 0 and p == 0:
        mains += 1
        hand_ids = [c["id"] for c in ps["hand"]]
        act = ps["active"][0]
        bench = ps["bench"]
        act_desc = f"{card(act['id']).name}(hp{act['hp']})" if act else "None"
        bench_desc = [f"{card(b['id']).name}(hp{b['hp']})" for b in bench]
        types = sorted(set(o["type"] for o in sel["option"]))
        evo_opts = [(i, o) for i, o in enumerate(sel["option"]) if o["type"] == 9]
        play_cards = []
        for o in sel["option"]:
            if o["type"] == 7 and o.get("index") is not None and o["index"] < len(ps["hand"]):
                play_cards.append(card(hand_ids[o["index"]]).name)
        print(f"[M{mains}] turn={cur['turn']} actAct={cur['turnActionCount']} "
              f"active={act_desc} bench={bench_desc} hand={[card(h).name for h in hand_ids]}")
        print(f"    optTypes={types} evo={len(evo_opts)} play={play_cards} "
              f"supporterPlayed={cur['supporterPlayed']} energyAttached={cur['energyAttached']}")
        if mains >= 12:
            break
        # pick: evolve if offered; else play Finizen if in hand; else attack; else end
        opts = sel["option"]
        idx = None
        for i, o in enumerate(opts):
            if o["type"] == 9:
                idx = i; break
        if idx is None:
            for i, o in enumerate(opts):
                if o["type"] == 7 and o.get("index") is not None and o["index"] < len(ps["hand"]) and hand_ids[o["index"]] == 105:
                    idx = i; break
        if idx is None:
            for i, o in enumerate(opts):
                if o["type"] == 8:
                    idx = i; break
        if idx is None:
            for i, o in enumerate(opts):
                if o["type"] == 13:
                    idx = i; break
        if idx is None:
            for i, o in enumerate(opts):
                if o["type"] == 14:
                    idx = i; break
        if idx is None:
            idx = 0
        obs = battle_select([idx])
        steps += 1
        continue
    n = len(sel["option"])
    k = rng.randint(sel["minCount"], sel["maxCount"]) if sel["maxCount"] > 0 else 0
    obs = battle_select(rng.sample(range(n), k) if k else [])
    steps += 1
    if steps > 600:
        break
print("done", steps)
battle_finish()
