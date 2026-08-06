"""Trace the Palafin Zero-to-Hero combo selections end-to-end."""
import sys, os, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "submission"))
from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class
from deck import DECK
from card_db import card

deck = DECK
obs, sd = battle_start(deck, deck)
rng = random.Random(11)
steps = 0
EVOLVED = False
SWAPPED = False
while True:
    sel = obs["select"]
    cur = obs["current"]
    if sel is None or cur is None or cur["result"] != -1:
        break
    p = cur["yourIndex"]
    ps = cur["players"][p]
    hand_ids = [c["id"] for c in ps["hand"]]
    opts = sel["option"]
    act = (ps["active"][0] if ps["active"] else None)
    bench = ps["bench"]
    act_name = card(act["id"]).name if act else "None"
    bench_names = [card(b["id"]).name for b in bench]
    my_turn = (p == 0)
    if sel["type"] != 0 or sel["context"] != 0 or not my_turn:
        # nested or opponent: dump first 12 interesting ones
        if my_turn and steps < 200:
            print(f"  [nested] step={steps} type={sel['type']} ctx={sel['context']} "
                  f"min={sel['minCount']} max={sel['maxCount']} opt={len(opts)}")
            for i, o in enumerate(opts[:6]):
                print(f"      opt{i}: { {k: v for k, v in o.items() if v is not None and k != 'serial'} }")
            if sel["type"] == 9:
                picks = [0]
            else:
                n = len(opts)
                k = max(sel["minCount"], min(sel["maxCount"], n))
                picks = list(range(k))
            obs = battle_select(picks)
            steps += 1
            continue
        n = len(opts)
        k = rng.randint(sel["minCount"], sel["maxCount"]) if sel["maxCount"] > 0 else 0
        obs = battle_select(rng.sample(range(n), k) if k else [])
        steps += 1
        continue
    # OUR MAIN
    print(f"[M] turn={cur['turn']} active={act_name}(hp{act['hp'] if act else '-'}) "
          f"bench={bench_names} hand={[card(h).name for h in hand_ids]}")
    idx = None
    # 1. evolve if offered
    for i, o in enumerate(opts):
        if o["type"] == 9:
            idx = i; break
    # 2. play Finizen if in hand
    if idx is None:
        for i, o in enumerate(opts):
            if o["type"] == 7 and o.get("index") is not None and o["index"] < len(hand_ids) and hand_ids[o["index"]] == 105:
                idx = i; break
    # 3. attach energy to active (or bench evolver)
    if idx is None:
        for i, o in enumerate(opts):
            if o["type"] == 8 and o.get("inPlayArea") == 4:
                idx = i; break
    # 4. retreat
    if idx is None:
        for i, o in enumerate(opts):
            if o["type"] == 12:
                idx = i; break
    # 5. attack
    if idx is None:
        for i, o in enumerate(opts):
            if o["type"] == 13:
                idx = i; break
    # 6. end
    if idx is None:
        for i, o in enumerate(opts):
            if o["type"] == 14:
                idx = i; break
    if idx is None:
        idx = 0
    t = opts[idx]["type"]
    print(f"    -> pick {idx} type={t}")
    obs = battle_select([idx])
    steps += 1
    if steps > 400:
        break
print("done", steps, "result", cur["result"] if cur else "?")
battle_finish()
