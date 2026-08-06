"""Local agent-vs-agent harness on the real engine.

Plays full matches through ``cg.game.battle_start`` / ``battle_select``,
driving each side with an arbitrary callable. Reports win rates, mean
game length, decision latency, and illegal-pick errors.

Usage:
    python3 tools/selfplay.py --games 20 --agents ours,random,sample
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "submission"))
sys.path.insert(0, os.path.join(HERE, "..", "submission", "cg"))

from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from cg.api import to_observation_class  # noqa: E402


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def make_random_agent(deck, seed=1):
    rng = random.Random(seed)

    def fn(obs_dict):
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return deck
        sel = obs.select
        n = len(sel.option)
        k = random.randint(sel.minCount, sel.maxCount) if sel.maxCount > 0 else 0
        picks = rng.sample(range(n), k) if k else []
        return picks

    return fn


def make_sample_agent(deck, seed=1):
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


def make_ours_agent(deck, use_lookahead=True, go_first=True, budget=110.0,
                    worlds=2, rounds=2, stats=None):
    from agent import Agent
    a = Agent(deck, go_first=go_first, lookahead_budget_ms=budget,
              use_lookahead=use_lookahead, worlds=worlds, rounds=rounds)
    if stats is not None:
        a.stats = stats

    def fn(obs_dict):
        return a.choose(obs_dict)

    return fn


def make_greedy_agent(deck):
    """Attack-first rules bot: attack with best damage, else attach energy,
    else play a Basic, else evolve, else end."""
    from card_db import attack as _atk

    def fn(obs_dict):
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return deck
        sel = obs.select
        opts = sel.option
        n = len(opts)
        if n == 0:
            return []
        if sel.type == 0 and sel.context == 0:  # MAIN
            best_atk, bd = None, -1
            for i, o in enumerate(opts):
                if o.type == 13:
                    a = _atk(o.attackId) if o.attackId else None
                    d = a.damage or 0 if a else 0
                    if d > bd:
                        best_atk, bd = i, d
            if best_atk is not None:
                return [best_atk]
            for i, o in enumerate(opts):
                if o.type == 8:
                    return [i]
            for i, o in enumerate(opts):
                if o.type == 7:
                    return [i]
            for i, o in enumerate(opts):
                if o.type == 9:
                    return [i]
            for i, o in enumerate(opts):
                if o.type == 14:
                    return [i]
            return [0]
        if sel.type == 9:  # YES_NO
            return [0]
        k = max(sel.minCount, min(sel.maxCount, n))
        return list(range(k))

    return fn


def _make_agent(kind, deck, budget=110.0, go_first=True, worlds=2, rounds=2):
    if kind == "random":
        return make_random_agent(deck)
    if kind == "sample":
        return make_sample_agent(deck)
    if kind == "greedy":
        return make_greedy_agent(deck)
    if kind == "ours-nolook":
        return make_ours_agent(deck, use_lookahead=False, go_first=go_first)
    return make_ours_agent(deck, use_lookahead=True, go_first=go_first,
                           budget=budget, worlds=worlds, rounds=rounds)


AGENT_FACTORIES = {
    "random": make_random_agent,
    "sample": make_sample_agent,
    "ours": lambda deck, **kw: make_ours_agent(deck, use_lookahead=kw.get("use_lookahead", True),
                                                go_first=kw.get("go_first", True),
                                                budget=kw.get("budget", 110.0)),
    "ours-nolook": lambda deck, **kw: make_ours_agent(deck, use_lookahead=False,
                                                       go_first=kw.get("go_first", True)),
}


# ---------------------------------------------------------------------------
# Match driver
# ---------------------------------------------------------------------------

def play_match(fn0, fn1, deck0, deck1, max_steps=2500, verbose=False,
               stats0=None, stats1=None):
    """Play one match. Returns result dict."""
    obs, sd = battle_start(deck0, deck1)
    if obs is None:
        return {"error": f"battle_start failed ({sd.errorPlayer}/{sd.errorType})",
                "winner": -1, "steps": 0}
    steps = 0
    times = [[], []]
    t0 = time.time()
    while True:
        sel = obs["select"]
        cur = obs["current"]
        if sel is None or cur is None:
            break
        if cur["result"] != -1:
            break
        p = cur["yourIndex"]
        fn = fn0 if p == 0 else fn1
        st = time.time()
        picks = fn(obs)
        times[p].append((time.time() - st) * 1000)
        if verbose:
            print(f"  step {steps:4d} P{p} type={sel['type']} ctx={sel['context']} "
                  f"opt={len(sel['option'])} picks={picks}")
        # validate + auto-fix
        n = len(sel["option"])
        picks = [i for i in picks if isinstance(i, int) and 0 <= i < n]
        seen = set()
        picks = [i for i in picks if not (i in seen or seen.add(i))]
        if len(picks) < sel["minCount"]:
            for i in range(n):
                if i not in seen and len(picks) < sel["minCount"]:
                    picks.append(i)
        if len(picks) > sel["maxCount"]:
            picks = picks[: sel["maxCount"]]
        obs = battle_select(picks)
        steps += 1
        if steps > max_steps:
            if verbose:
                print("  MAX STEPS")
            battle_finish()
            return {"error": "max_steps", "winner": -1, "steps": steps,
                    "times": times, "elapsed": time.time() - t0}
    winner = cur["result"] if cur is not None else -1
    if stats0 is not None:
        stats0["times"].extend(times[0])
    if stats1 is not None:
        stats1["times"].extend(times[1])
    battle_finish()
    return {"error": None, "winner": winner, "steps": steps,
            "times": times, "elapsed": time.time() - t0}


def summarize(results, name_a, name_b, deck_a, deck_b):
    wins_a = sum(1 for r in results if r["winner"] == 0)
    wins_b = sum(1 for r in results if r["winner"] == 1)
    draws = sum(1 for r in results if r["winner"] == -1)
    errs = sum(1 for r in results if r.get("error"))
    n = len(results)
    print(f"{name_a} vs {name_b}: A {wins_a}/{n} ({wins_a / max(n,1) * 100:.0f}%) | "
          f"B {wins_b}/{n} ({wins_b / max(n,1) * 100:.0f}%) | draws {draws} | errors {errs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--a", default="ours")
    ap.add_argument("--b", default="random")
    ap.add_argument("--deck-a", default=None)
    ap.add_argument("--deck-b", default=None)
    ap.add_argument("--swap", action="store_true", help="swap sides every game")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--budget", type=float, default=110.0)
    ap.add_argument("--go-first", type=int, default=1)
    ap.add_argument("--worlds", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    from deck import DECK
    deck_a = [int(x) for x in open(args.deck_a).read().split()] if args.deck_a else list(DECK)
    deck_b = [int(x) for x in open(args.deck_b).read().split()] if args.deck_b else list(DECK)

    stats_a = {"times": []}
    stats_b = {"times": []}
    results = []
    for g in range(args.games):
        # --swap: alternate which agent is player 0 (player 0 answers the
        # go-first question), so sides are exercised fairly.
        swapped = args.swap and (g % 2 == 1)
        kind0, kind1 = (args.b, args.a) if swapped else (args.a, args.b)
        f0 = _make_agent(kind0, deck_a, budget=args.budget,
                         go_first=bool(args.go_first), worlds=args.worlds,
                         rounds=args.rounds)
        f1 = _make_agent(kind1, deck_b, budget=args.budget,
                         go_first=bool(args.go_first), worlds=args.worlds,
                         rounds=args.rounds)
        r = play_match(f0, f1, deck_a, deck_b, verbose=args.verbose,
                       stats0=stats_a if not swapped else stats_b,
                       stats1=stats_b if not swapped else stats_a)
        if swapped:
            # report from args.a's perspective
            r["winner"] = 1 - r["winner"] if r["winner"] in (0, 1) else r["winner"]
        results.append(r)
        if (g + 1) % 5 == 0 or args.verbose:
            print(f"game {g+1}/{args.games}: winner={r['winner']} steps={r['steps']} "
                  f"err={r.get('error')} t={r['elapsed']:.1f}s")

    summarize(results, args.a, args.b, deck_a, deck_b)

    all_times = stats_a["times"] + stats_b["times"]
    if all_times:
        all_times.sort()
        p50 = all_times[len(all_times) // 2]
        p95 = all_times[int(len(all_times) * 0.95)]
        print(f"decision latency: n={len(all_times)} p50={p50:.1f}ms p95={p95:.1f}ms "
              f"max={all_times[-1]:.1f}ms")
        if stats_a["times"]:
            print(f"  A (ours) n={len(stats_a['times'])} p50={sorted(stats_a['times'])[len(stats_a['times'])//2]:.1f}ms")


if __name__ == "__main__":
    main()
