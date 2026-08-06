"""Tests for the real-engine competition agent (``submission/``).

Run with:
    python3 tests/test_submission_agent.py
(the engine dylib/so must be loadable — see submission/cg/sim.py).

Covers: deck validity, option decoding, damage estimation vs the engine
formula, tracker predictions, evaluate() sanity, and end-to-end
robustness (many games vs random baselines, zero crashes, latency cap).
"""
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "submission"))
sys.path.insert(0, os.path.join(ROOT, "submission", "cg"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from cg.api import to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

from deck import DECK, DECK_SPEC  # noqa: E402

PASS = []
FAIL = []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  ok  {name}")
    else:
        FAIL.append((name, detail))
        print(f"FAIL  {name}  {detail}")


def test_deck_valid():
    obs, sd = battle_start(DECK, DECK)
    battle_finish()
    check("deck length == 60", len(DECK) == 60, str(len(DECK)))
    check("deck valid per engine", sd.errorType == 0, f"{sd.errorPlayer}/{sd.errorType}")
    total = sum(c for _, c in DECK_SPEC)
    check("deck spec sums to 60", total == 60, str(total))
    from collections import Counter
    from card_db import card as _card, CardType
    c = Counter(DECK)
    non_energy = {cid: n for cid, n in c.items()
                  if (cd := _card(cid)) is None or cd.cardType != CardType.BASIC_ENERGY}
    check("no non-energy card more than 4x", max(non_energy.values()) <= 4,
          str(max(non_energy.values())))


def test_option_decoding():
    from strategy import option_card_id
    from card_db import Card, Pokemon, PlayerState
    from cg.api import Option, SelectData, State, AreaType

    # hand option (PLAY carries only index)
    hand = [Card(id=1239, serial=1, playerIndex=0), Card(id=3, serial=2, playerIndex=0)]
    me = PlayerState(active=[], bench=[], benchMax=5, deckCount=40, discard=[],
                     prize=[], handCount=2, hand=hand, poisoned=False,
                     burned=False, asleep=False, paralyzed=False, confused=False)
    state = State(turn=3, turnActionCount=1, yourIndex=0, firstPlayer=0,
                  supporterPlayed=False, stadiumPlayed=False, energyAttached=False,
                  retreated=False, result=-1, stadium=[], looking=None,
                  players=[me, me])
    sel = SelectData(type=1, context=7, minCount=1, maxCount=1,
                     remainDamageCounter=0, remainEnergyCost=0, option=[],
                     deck=None, contextCard=None, effect=None)
    o = Option(type=7, index=1, playerIndex=0)
    check("PLAY option resolves hand card", option_card_id(sel, o, state) == 3)

    # deck option (TO_HAND from a search: area=DECK, index into sel.deck)
    sel2 = SelectData(type=1, context=7, minCount=1, maxCount=1,
                      remainDamageCounter=0, remainEnergyCost=0, option=[],
                      deck=[Card(id=107, serial=9, playerIndex=0)],
                      contextCard=None, effect=None)
    o2 = Option(type=3, area=1, index=0, playerIndex=0)
    check("deck option resolves via sel.deck", option_card_id(sel2, o2, state) == 107)


def test_damage_math():
    from card_db import attack, card, estimate_damage, is_ko
    from cg.api import CardData, Attack, EnergyType, CardType, Skill

    def mk(cid, etype, weak, resist):
        return CardData(cardId=cid, name="x", cardType=CardType.POKEMON,
                        retreatCost=1, hp=100, weakness=weak, resistance=resist,
                        energyType=etype, basic=True, stage1=False, stage2=False,
                        ex=False, megaEx=False, tera=False, aceSpec=False,
                        evolvesFrom=None, skills=[], attacks=[])

    atk = Attack(attackId=1, name="a", text="", damage=80,
                 energies=[EnergyType.WATER])
    att = mk(1, EnergyType.WATER, None, None)
    defd = mk(2, EnergyType.GRASS, EnergyType.WATER, None)   # weak to water
    check("weakness x2", estimate_damage(atk, att, defd) == 160)
    defr = mk(3, EnergyType.FIRE, None, EnergyType.WATER)    # resist water
    check("resistance -30", estimate_damage(atk, att, defr) == 50)
    defp = mk(4, EnergyType.FIRE, None, None)
    check("plain damage", estimate_damage(atk, att, defp) == 80)
    check("KO check", is_ko(atk, att, defp, 80))
    check("no KO", not is_ko(atk, att, defp, 81))


def test_tracker_predictions():
    from tracker import Tracker
    from card_db import Card, PlayerState
    from cg.api import State, Card as ApiCard

    from cg.api import Observation
    tr = Tracker(DECK, seed=42)
    hand = [ApiCard(id=DECK[0], serial=1, playerIndex=0)]
    me = PlayerState(active=[], bench=[], benchMax=5, deckCount=47, discard=[],
                     prize=[None] * 6, handCount=1, hand=hand, poisoned=False,
                     burned=False, asleep=False, paralyzed=False, confused=False)
    state = State(turn=3, turnActionCount=1, yourIndex=0, firstPlayer=0,
                  supporterPlayed=False, stadiumPlayed=False, energyAttached=False,
                  retreated=False, result=-1, stadium=[], looking=None,
                  players=[me, me])
    obs = Observation(select=None, logs=[], current=state, search_begin_input=None)
    w = tr.predictions(obs)
    check("your_deck count", len(w["your_deck"]) == 47, str(len(w["your_deck"])))
    check("your_prize count", len(w["your_prize"]) == 6)
    check("opponent_deck count", len(w["opponent_deck"]) == 47)
    check("opponent_hand count", len(w["opponent_hand"]) == 1)
    w2 = tr.predictions(obs)
    check("predictions resample (2 worlds differ)", w["your_deck"] != w2["your_deck"])


def test_evaluate_sanity():
    from strategy import evaluate, WIN, LOSE
    from card_db import Card, PlayerState
    from cg.api import State, Card as ApiCard

    from cg.api import Observation

    def mk_state(winner):
        ps = PlayerState(active=[], bench=[], benchMax=5, deckCount=40, discard=[],
                         prize=[None] * 6, handCount=0, hand=[], poisoned=False,
                         burned=False, asleep=False, paralyzed=False, confused=False)
        st = State(turn=5, turnActionCount=1, yourIndex=0, firstPlayer=0,
                   supporterPlayed=False, stadiumPlayed=False, energyAttached=False,
                   retreated=False, result=winner, stadium=[], looking=None,
                   players=[ps, ps])
        return Observation(select=None, logs=[], current=st, search_begin_input=None)

    check("terminal win +1e6", evaluate(mk_state(0), 0) == WIN)
    check("terminal loss -1e6", evaluate(mk_state(0), 1) == LOSE)


def test_robustness_games(n=30, budget=90.0):
    from selfplay import make_random_agent, make_ours_agent
    rng = random.Random(7)
    wins = 0
    errors = 0
    times = []
    max_steps = 0
    for g in range(n):
        f0 = make_ours_agent(DECK, budget=budget)
        f1 = make_random_agent(DECK, seed=100 + g)
        obs, sd = battle_start(DECK, DECK)
        if obs is None:
            errors += 1
            continue
        steps = 0
        while True:
            sel, cur = obs["select"], obs["current"]
            if sel is None or cur is None or cur["result"] != -1:
                break
            p = cur["yourIndex"]
            fn = f0 if p == 0 else f1
            t0 = time.time()
            picks = fn(obs)
            times.append((time.time() - t0) * 1000)
            n_opt = len(sel["option"])
            picks = [i for i in picks if isinstance(i, int) and 0 <= i < n_opt]
            seen = set()
            picks = [i for i in picks if not (i in seen or seen.add(i))]
            if len(picks) < sel["minCount"]:
                for i in range(n_opt):
                    if len(picks) >= sel["minCount"]:
                        break
                    if i not in picks:
                        picks.append(i)
            if len(picks) > sel["maxCount"]:
                picks = picks[: sel["maxCount"]]
            try:
                obs = battle_select(picks)
            except Exception as e:  # noqa: BLE001
                errors += 1
                break
            steps += 1
            if steps > 2000:
                errors += 1
                break
        if cur is not None and cur["result"] == 0:
            wins += 1
        max_steps = max(max_steps, steps)
        battle_finish()
    times.sort()
    p95 = times[int(len(times) * 0.95)] if times else 0
    check(f"robustness: {n} games, 0 crashes", errors == 0, f"errors={errors}")
    check(f"robustness: win rate vs random >= 70%", wins / max(n, 1) >= 0.7,
          f"{wins}/{n}")
    check("robustness: p95 decision latency < 250ms", p95 < 250.0, f"{p95:.0f}ms")
    check("robustness: games terminate < 2000 steps", max_steps < 2000, str(max_steps))
    print(f"  (win rate {wins}/{n}, p95 {p95:.0f}ms, max steps {max_steps})")


def main():
    print("== submission agent tests ==")
    test_deck_valid()
    test_option_decoding()
    test_damage_math()
    test_tracker_predictions()
    test_evaluate_sanity()
    test_robustness_games()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for name, detail in FAIL:
        print(f"  FAILED: {name} {detail}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
