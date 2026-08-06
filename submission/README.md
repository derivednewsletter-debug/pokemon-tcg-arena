# Pokémon TCG AI Battle — Competition Submission

This folder is a **self-contained Kaggle submission**. Zip it and upload:

```bash
cd kaggle_pokemon/submission
zip -r ../submission.zip . -x "*/__pycache__/*" "*.pyc"
```

The competition harness imports `main.agent` and calls it once per
selection; the first call (no selection) must return the 60-card deck.

## Layout

| File | Purpose |
|------|---------|
| `main.py`    | Entry point — `agent(obs_dict) -> list[int]` |
| `agent.py`   | Selection dispatcher: every SelectType/context -> a decision, with safe fallback |
| `strategy.py`| Position evaluator + MAIN action policy (prize/board/KO/resource heuristics) |
| `lookahead.py`| Engine-search lookahead: plays each candidate + opponent reply through the real rules engine under predicted hidden info |
| `tracker.py` | Hidden-information tracker: seen cards, deck composition, per-decision predicted worlds |
| `card_db.py` | Card/attack database from the engine + damage/KO math (weakness x2, resist -30) |
| `deck.py` / `deck.csv` | The 60-card deck |
| `cg/`       | Official engine bindings (do not edit) |

## The agent

* **Deck — "Mega Abomasnow Wall"**: Snover -> Mega Abomasnow ex (350 HP,
  Frost Barrier 200 damage for {W}{W}{W} with a -30-damage shield), with
  a Naveen / Lillie's Determination / Carmine draw engine and Hyper
  Aroma / Ultra Ball / Dusk Ball search. Kyogre backs it up.
* **Brain — heuristic + forward search**: every MAIN decision scores its
  candidate actions with a position evaluator, then verifies the top
  candidates by simulating each action (and the opponent's full reply)
  through the **official engine's search API** across 2 predicted hidden
  worlds and 2 rounds of play. Decisions average ~10 ms (p95 < 50 ms).
* **Adaptability**: the policy adapts to prize lead, KO threat on either
  side, energy needs, and bench setup; the search opponent model plays
  our own heuristic so candidate actions are judged against a
  reasonable reply, not a passive one.
* **Safety**: every selection is validated against the engine's min/max
  counts and falls back to a legal pick on any unexpected state.

## Local validation

From `kaggle_pokemon/`:

```bash
python3 tests/test_submission_agent.py                 # 22 tests vs the real engine
python3 tools/selfplay.py --games 24 --a ours --b random --swap
python3 tools/selfplay.py --games 24 --a ours --b greedy --swap
```

Reference results (24 games each, sides swapped):
random 96% | sample-submission 92% | greedy rules-bot 88%.
