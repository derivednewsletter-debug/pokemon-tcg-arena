"""Per-game stats: winner, end reason, Palafin ex usage, attack quality."""
import os, sys, time, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "submission"))
sys.path.insert(0, os.path.join(HERE, "..", "submission", "cg"))

from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class, LogType
from deck import DECK
from card_db import card, attack

def make_random(deck, seed):
    rng = random.Random(seed)
    def fn(obs_dict):
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return deck
        sel = obs.select
        n = len(sel.option)
        if n == 0:
            return []
        k = rng.randint(sel.minCount, sel.maxCount) if sel.maxCount > 0 else 0
        return rng.sample(range(n), k) if k else []
    return fn

def make_ours(deck, budget):
    from agent import Agent
    a = Agent(deck, lookahead_budget_ms=budget)
    def fn(obs_dict):
        return a.choose(obs_dict)
    return fn

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0
ours = make_ours(DECK, BUDGET)

wins = 0
for g in range(N):
    fn0 = ours
    fn1 = make_random(DECK, 100 + g)
    obs, sd = battle_start(DECK, DECK)
    steps = 0
    our_attacks = []
    palafin_ex_seen = False
    reason = None
    while True:
        sel = obs["select"]
        cur = obs["current"]
        if sel is None or cur is None:
            break
        if cur["result"] != -1:
            reason = cur.get("resultReason") or obs.get("result")
            for lg in obs.get("logs", []):
                if lg.get("type") == 23:
                    reason = lg.get("reason")
            break
        p = cur["yourIndex"]
        # track our (P0) attack picks
        if p == 0 and sel["type"] == 0:
            opts = sel["option"]
            for o in opts:
                if o.get("type") == 13:
                    pass
            picks = fn0(obs) if p == 0 else fn1(obs)
            # find which attack we picked
            if picks and opts[picks[0]].get("type") == 13:
                aid = opts[picks[0]].get("attackId")
                a = attack(aid)
                our_attacks.append(a.damage if a else 0)
        else:
            picks = fn0(obs) if p == 0 else fn1(obs)
        # track palafin ex in play for P0
        for pi in (0,):
            ps = cur["players"][pi]
            for pok in ([ps["active"][0]] if ps["active"] and ps["active"][0] else []) + ps["bench"]:
                if pok and pok["id"] == 107:
                    palafin_ex_seen = True
        # sanitize
        n = len(sel["option"])
        picks = [i for i in picks if isinstance(i, int) and 0 <= i < n]
        seen = set()
        picks = [i for i in picks if not (i in seen or seen.add(i))]
        if len(picks) < sel["minCount"]:
            for i in range(n):
                if len(picks) >= sel["minCount"]:
                    break
                if i not in picks:
                    picks.append(i)
        if len(picks) > sel["maxCount"]:
            picks = picks[: sel["maxCount"]]
        obs = battle_select(picks)
        steps += 1
        if steps > 2000:
            break
    winner = cur["result"] if cur else -1
    if winner == 0:
        wins += 1
    max_atk = max(our_attacks) if our_attacks else 0
    print(f"game {g}: winner={winner} reason={reason} steps={steps} "
          f"our_attacks={len(our_attacks)} max_dmg={max_atk} avg={sum(our_attacks)/len(our_attacks) if our_attacks else 0:.0f} "
          f"palafin_ex_in_play={palafin_ex_seen}")
    battle_finish()
print(f"WINS: {wins}/{N}")
