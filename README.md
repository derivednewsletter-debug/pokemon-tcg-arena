# Pokémon TCG Arena — AI agent + playable web arena

> Repo: https://github.com/derivednewsletter-debug/pokemon-tcg-arena

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fderivednewsletter-debug%2Fpokemon-tcg-arena)

A complete Pokémon TCG AI: a competitive **engine-search agent** (built for
the Kaggle *Pokémon TCG AI Battle* competition) plus a **playable web arena**
where humans battle the AI — and every game teaches it how humans play.

## What's in this repo

| Path | What it is |
|------|-----------|
| `submission/` | The **competition agent**: plays the *real* official rules engine (`cg`) with engine-search lookahead. This folder is what you'd zip for Kaggle. |
| `api/` | **Vercel serverless backend**: game lifecycle (`/api/game/*`) + learning telemetry (`/api/learn`). |
| `public/` | **Frontend**: the playable arena + the "AI Brain" learning dashboard (vanilla JS, no build step). |
| `learning/` | Aggregates human game records into the opponent profile the AI plays with. |
| `scripts/` | Dev tools: `smoke_web.py` (end-to-end test), `gen_cards_json.py`, self-play benchmarks. |
| `pokemon_tcg/` | Legacy research framework (hand-rolled simulator + search agents) — see the section at the bottom. |
| `tools/` | A/B self-play harness + deck analysis used to tune the agent and decks. |
| `tests/` | Regression suite incl. 22 real-engine tests for the submission agent. |

## How the agent thinks

* **Value function** (`submission/strategy.py`) — scores positions by prize
  lead, board HP, KO threat both ways, attack power, bench setup, resources.
* **Engine-search lookahead** (`submission/lookahead.py`) — every MAIN
  decision plays candidate actions *and the opponent's reply* through the
  real rules engine (`search_begin`/`search_step`) under predicted hidden
  information, then picks the best-scoring line.
* **Hidden-info tracker** (`submission/tracker.py`) — models what's left in
  each deck, samples multiple possible worlds per decision.
* **Deck** — Mega Abomasnow ex wall: 350 HP, Frost Barrier 200. A/B
  validated at **96% vs random, 96% vs the official sample deck, 88% vs a
  greedy rules-bot** (24 games each, sides swapped).

## Run the arena locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# API (Flask dev servers)
.venv/bin/python api/game.py   &   # :5001 — /api/game/*
.venv/bin/python api/learn.py  &   # :5002 — /api/learn

# Serve the frontend (any static server)
python3 -m http.server 8080 --directory public
# open http://localhost:8080
```

Or just run the end-to-end smoke test (plays full games through the HTTP
surface):

```bash
.venv/bin/python scripts/smoke_web.py 3
```

## Deploy to Vercel

**One click:**

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fderivednewsletter-debug%2Fpokemon-tcg-arena)

The button imports this repo into Vercel and deploys it — no local setup
needed. Or from the CLI:

```bash
vercel login        # one-time
vercel --prod       # from the repo root
```

1. Push this repo to GitHub, then **Import** it in Vercel (root directory =
   repo root, framework = Other). `vercel.json` wires `api/*.py` as Python
   functions and `public/` is served at `/` automatically.
2. The engine binaries ship in `submission/cg/` — `libcg.so` (Linux x86-64)
   loads on Vercel, `libcg-arm64.so` for ARM, `libcg.dylib` for macOS local
   dev. No build step.
3. **Optional but recommended — persistent learning:** create a Vercel KV
   store and add `KV_REST_API_URL` + `KV_REST_API_TOKEN` to the project env.
   Without it, learning data lives in the function's memory (per instance,
   reset on cold starts); with KV, every finished game is persisted and
   shared across instances.
4. Deploy. Open the URL, play a few games, then check **AI Brain**
   (`/stats`) — it fills in as games finish.

> ⚠️ **Why GitHub alone won't run the game:** the arena needs the Python
> backend (`/api/game/*` — real rules engine, lookahead, learning). GitHub
> Pages serves static files only, so the page loads but every API call 404s
> (you'll see `Failed to load resource: 404` in the console and the deck
> pickers stay empty). Deploy on Vercel (button above) or run locally
> (below) and it works.

> Serverless note: a live game lives in the memory of one server instance,
> so it can expire if the instance goes to sleep mid-match. Finished games
> are recorded regardless. The UI handles this gracefully ("server went to
> sleep — start a new battle").

## How the AI learns from players

Every MAIN action a human takes is recorded — anonymized, just action
classes (`attack`, `play_supporter`, `retreat`, …), deck choice, and who
won. `learning/profiles.py` aggregates these into an **opponent profile**:
aggression (attack rate), retreat frequency, first-move habits, deck
preferences. The agent's lookahead (`submission/strategy.py` +
`lookahead.py`, `opp_profile=`) then simulates the *opponent* using those
learned human tendencies — so the more people play, the better the AI plans
against humans. The **AI Brain** page shows exactly what it has learned.

## The Kaggle submission

`submission/` is self-contained (entry point `main.py`, agent modules, and
the official `cg/` engine package). Zip its contents and submit:

```bash
cd submission && zip -r ../submission.zip . -x '*__pycache__*'
```

Full agent docs: `submission/README.md` and `docs/competition_agent.md`.

---

# Legacy research framework (pokemon_tcg/)

---
# Pokémon TCG AI Battle Challenge — Competitive Training Agent

_A modular Python framework for building, benchmarking, and experimenting
with Pokémon Trading Card Game AI agents._

## What this is

A competitive Pokémon TCG AI agent that:

- Plays fully autonomous matches.
- Decides every action from a structured game state.
- Logs every game for offline analysis.
- Runs head-to-head tournaments across many deck seeds.
- Reports metrics like win rate, prize differential, decision latency.
- Identifies failure patterns (POOR_SETUP, ENERGY_STARVED, etc.) per
  matchup.

The system mirrors the Kaggriculture engineering guidelines:

| Layer | Purpose |
|-------|---------|
| **cards**       | Card database parsed from `EN Card Data.csv` |
| **deck**        | Deck construction (random + themed) |
| **game_state**  | Immutable per-player state + global match state |
| **actions**     | Legal action enumeration |
| **simulator**   | Rules engine (the only mutator of state) |
| **evaluator**   | Position evaluation (heuristic scoring) |
| **search**      | 1-ply greedy + 2-ply alpha-beta + 3-ply alpha-beta + iterative deepening + rollouts |
| **value head**  | Pluggable value function: hand-tuned heuristic OR small MLP trained from self-play |
| **retreat model** | Logistic-regression retreat predictor trained from self-play (v0.5) |
| **agents**      | Agent interface + benchmark strategies |
| **experiments** | Batch runner + round-robin tournament |
| **evaluation**  | Metrics + leaderboard |
| **logging_utils** | Structured game-log writer |
| **analysis**    | Failure categorization |

## Quickstart

```bash
# Run the full test suite
python3 -m pokemon_tcg tests

# Run a single match in verbose mode
python3 -m pokemon_tcg simulate --a Greedy --b Defensive --seed 42 --turns 40

# Benchmark Greedy against three opponents
python3 -m pokemon_tcg benchmark --agent Greedy \
    --opponents Defensive,Aggressive,SearchAgent \
    --seeds 42,7,123,999 --games 4

# Round-robin tournament
python3 -m pokemon_tcg tournament --agents Champion,SearchAgent,Greedy \
    --seeds 42,7,123,2024 --games 2 --pool themed
```

## Architecture

### Data flow

```
   csv input       Loaded Card database     Decks (60-card lists)
        │                  │                        │
        v                  v                        v
    cards.py     →      deck.py     →      experiments/deck_pool.py
                                                  │
                                            Shared pool of
                                            reproducible decks
                                                  │
                                                  v
                              ┌─────────────────────────────────┐
                              │   agents (heuristic / search)   │
                              └────────────────┬────────────────┘
                                               │
            ┌──────────────────────┐           │
            │   simulator.py       │  step(action)  ◄──── action
            │   (only mutator)     │           │
            └─────────┬────────────┘           │
                      │                        │
                      v                        │
                game_state.py ─────────────────┘ read-only views
                                              │
                                              v
                                       evaluator.py + search.py
                                              │
                                              v
                                Action chosen + structured log
```

### Module map

| File | Lines | Lines of doc | Notes |
|------|-------|--------------|-------|
| `pokemon_tcg/cards.py`          | ~330 | ~50 | Card types + CSV parser |
| `pokemon_tcg/deck.py`           | ~140 | ~30 | Random + themed deck construction |
| `pokemon_tcg/game_state.py`     | ~250 | ~40 | Per-player + global state |
| `pokemon_tcg/actions.py`        | ~140 | ~30 | Legal action enumeration |
| `pokemon_tcg/simulator.py`      | ~330 | ~50 | Rules engine (only state mutator) |
| `pokemon_tcg/evaluator.py`      | ~150 | ~40 | Position evaluation |
| `pokemon_tcg/search.py`         | ~330 | ~80 | 1-ply + 2-ply + 3-ply + iterative deepening |
| `pokemon_tcg/agents/base.py`     | ~50  | ~20 | Agent interface |
| `pokemon_tcg/agents/benchmarks.py` | ~250 | ~50 | Heuristic strategies |
| `pokemon_tcg/agents/champion.py` | ~110 | ~30 | Best-of-everything strategy |
| `pokemon_tcg/experiments/runner.py`    | ~110 | ~30 | Single-vs-many batch |
| `pokemon_tcg/experiments/tournament.py` | ~120 | ~30 | Round-robin + Elo |
| `pokemon_tcg/experiments/deck_pool.py`  | ~80  | ~20 | Shared deck pool |
| `pokemon_tcg/evaluation/metrics.py`     | ~120 | ~30 | Match-level metrics |
| `pokemon_tcg/evaluation/leaderboard.py` | ~110 | ~30 | Multi-strategy comparison |
| `pokemon_tcg/logging_utils/game_log.py` | ~80  | ~20 | JSON serialization |
| `pokemon_tcg/analysis/failure.py`       | ~110 | ~30 | Loss categorization |

## Strategies

### Greedy

1-ply search: score every legal action by immediate evaluator delta,
pick the highest. Fast (<5ms per turn).

### SearchAgent

2-ply alpha-beta with beam pruning. Sees the opponent's best
response before picking. ~30ms per turn.

**Tip:** Recommended for `tournament` runs but slows significantly
at higher beam widths. `beam=8` is a good default.

### Aggressive / Defensive

Heuristically-weighted Greedy variants. Both are useful as opponents
to verify your agent plays correctly across play styles.

- **Aggressive:** prize value + KO potential weighted up. Plays for
  knockouts aggressively.
- **Defensive:** active HP weighted up; doesn't over-commit.

### EnergyRamp / BenchBuffer

Single-axis strategies that test how well your agent handles trivial
heuristics.

- **EnergyRamp:** Attach energy first; only attack when active reaches
  ≥120 damage potential.
- **BenchBuffer:** Bench a Basic first; reach `min_bench` size before
  doing anything else.

### Champion

Combined: **iterative-deepening 3-ply alpha-beta** (1-ply → 2-ply → 3-ply
under a 250 ms time budget) with the champion evaluator. Setup turns
(1-5) use a heuristic that prioritizes energy attaching and bench
building; afterwards, deep lookahead drives every move. The 3-ply
search explicitly considers our second action after an opponent's
reply, exposing the canonical setup-then-retreat-then-attack combo:

    T1 (us):    PLAY_POKEMON(monster on bench)
    T1 (opp):   PASS / KO something
    T2 (us):    ATTACH_ENERGY(monster) → RETREAT into monster → ATTACK

The candidate-ordering bonus (`search._setup_aware_order_bonus`) keeps
setup actions near the top of every level of the search tree so the
deepest tier can verify the combo end-to-end. Tuned to win ~60% of
head-to-heads against Greedy on the standard random-deck benchmark
(14 seeds × 2 games × sides swapped = 28 matches; see
"Pluggable value head" below).

**Retreat-aware:** since v0.2 the Champion performs `RETREAT` actions
when the active Pokemon is in danger and the bench target has a
substantially stronger attack profile. The agent's biased score
penalises retreats by 6× the retreat cost to discourage energy waste
on marginal swaps. **In v0.5** this hand-coded buffer heuristic can
be replaced by a learned :class:`RetreatWinPredictor`; see
"Retreat-model integration" below.

**Pluggable value head:** every Champion decision goes through a
`ValueFunction` (defined in `evaluator.py`). The default
`HeuristicValueFn(CHAMPION_CONFIG)` is the hand-tuned score; replacing
it requires no code changes elsewhere — drop in a `LearnedValueFn`
trained from self-play and the entire search stack will use it:

```python
from pokemon_tcg.evaluator import LearnedValueFn
from pokemon_tcg.agents.champion import ChampionAgent

champ = ChampionAgent(config={
    "value_fn": LearnedValueFn(path="results/value_head_v2.npz"),
})
```

### Pluggable value head (heuristic vs MLP)

Both `HeuristicValueFn` and `LearnedValueFn` implement the same
`ValueFunction` Protocol (`__call__(state, who) -> float` plus
`.kind` and `.score(...)`). The MLP variant uses a 32 -> 16 -> 1
feed-forward network trained with mini-batch SGD against self-play
game outcomes (`+1` for the winner's perspective, `-1` for the
loser's, `0` for a draw).

To train one::

    python3 -m pokemon_tcg train-value --games 64 --epochs 60 \\
        --spawn-a Champion --spawn-b Greedy \\
        --out results/value_head_v2.npz

The trainer captures a feature ``(features_from_state)`` at every
turn of every game (`simulate_match` is monkey-patched for capture),
pairs each snapshot with the eventual game outcome, and fits a
scaled target. Loss is MSE on ``tanh(prev_output)`` so the model's
scalar lives in [-1, 1] which keeps gradients well-behaved across
hundreds of epochs.

### Retreat-model integration (added v0.5)

Per-turn retreat decisions can also route through a learned logistic
regression that predicts *the probability that the player wins the
match from this state, given that a retreat is legal.* `EvaluatorConfig`
gains two new fields: ``retreat_win_prob_threshold`` (default 0.5) and
``retreat_win_prob_margin`` (default 0.05). Champion retreats only when
``predict_proba(state, who) >= threshold + margin``.

Train a 12-feature logistic predictor from self-play:

    python3 -m pokemon_tcg train-retreat-threshold \\
        --games 32 --epochs 60 --lr 0.05 \\
        --spawn-a Champion --spawn-b Greedy \\
        --out results/retreat_champion.npz \\
        --threshold-out results/retreat_threshold_champion.json

The trainer writes the recommended threshold (and a per-threshold
accuracy grid) to JSON, so Champion can adopt it without re-tuning::

    from pokemon_tcg.analysis.retreat_model import RetreatWinPredictor
    predictor = RetreatWinPredictor.load("results/retreat_champion.npz")
    champ = ChampionAgent(config={"retreat_predictor": predictor})

#### Honest finding — 14-seed A/B

| Variant                                  | Wins | Win rate |
|------------------------------------------|-----:|---------:|
| `Champion(heuristic, no predictor)`      | 18/28 | 64.3%   |
| `Champion(predictor, t=0.50)`            | 14/28 | 50.0%   |
| `Champion(predictor, t=0.40..0.95)`     | 14/28 | 50.0%   |

The trainer's recommended threshold (0.50, accuracy 87.6% on held-out
data) does **not** outperform the magic-buffer heuristic. The
logistic model has learned a *positional* signal — retreat-eligible
turns are concentrated in winning positions — rather than a
*causal* signal. Improving this requires either (a) ~10x more
self-play games, or (b) a counterfactual trainer that records the
outcome conditional on whether the agent actually retreated. The
plumbing (predictor, threshold recommender, JSON output,
Champion integration, 20+ tests) is in place and ready to absorb
either improvement.

### A/B benchmark — Champion (heuristic) vs Champion (MLP)

`Champion` (default, `HeuristicValueFn`) vs `Champion(LearnedValueFn,
trained)` vs Greedy on **14 seeds × 2 games × sides swapped = 28
matches**. We also ran head-to-head `Champion(MLP) vs Champion(Heur)`
matched pairs to measure whether the trained value head beats the
hand-tuned one.

| Value head                       | Wins | Win rate | vs Greedy |
|----------------------------------|-----:|---------:|----------:|
| `HeuristicValueFn` (default)     | 17/28 | 60.7% | baseline |
| `LearnedValueFn` — 8 self-play games  | 14/28 | 50.0% | -10.7% vs baseline |
| `LearnedValueFn` — 24 self-play games | 12/28 | 42.9% | -17.8% vs baseline |
| `LearnedValueFn` — 64 self-play games | 14/28 | 50.0% | -10.7% vs baseline |

**Head-to-head wins (matched pairs, 28 games):**

| Matchup | Wins | Win rate |
|---------|-----:|---------:|
| `Champion(MLP, 64 games)` vs `Champion(Heur)` |  13/28 | 46.4% |
| `Champion(Heur)`            vs `Champion(MLP, 64 games)` | 15/28 | 53.6% |

**Interpretation.** The pluggable value head works end-to-end —
Champion with `LearnedValueFn` runs end-to-end games without crashing
and the same `iterative_deepening_search` runs against it. After 64
self-play training games the MLP closes most of the gap to the
hand-tuned heuristic (46% vs 54% in matched pairs) but does not
strictly *outperform* it. Two reasons live inside this measurement:

1. The tanh output bound keeps the MLP in [-1, 1] while the heuristic
   can output values around ±10^6 for terminal positions; search's
   min/max arithmetic is more decisive with the larger dynamic range.
2. Training on `±1` game outcomes (binary) provides less learning
   signal than regression on the continuous heuristic score. A
   distillation trainer (target = heuristic score, not game result)
   or 100× more games would close that gap further.

The 14-seed A/B shows the heuristic Champion's **17/28** win rate
over Greedy is the same baseline the agent has had for v0.2; the
MLP variant lands within ±3 wins of that baseline, well inside the
noise floor for 28 matches.

### GreedyNoRetreat

Identical to `Greedy` but `RETREAT` actions are filtered out before
scoring. Used as a baseline in the matched pairs A/B benchmark below
to measure whether retreat mechanics improve performance.

## Decision engine

The decision engine is intentionally pluggable:

- DecisionEngine → call(observation) → Action.

Plug in new agents by subclassing `Agent` or registering them in
`pokemon_tcg/agents/benchmarks.py: BENCHMARKS`.

## Experiment framework

### Runner

`pokemon_tcg.experiments.runner.ExperimentRunner.run(agent_a, agent_b, decks, seeds, games_per_pair)`

Runs N games between two agents and emits `ExperimentResult` JSON with
metrics, all match records, and per-game timing.

### Tournament

`pokemon_tcg.experiments.tournament.run_tournament(agents, seeds, games_per_pair)`

Round-robin every-vs-every and accumulates Elo + matchup table.
Output is human-readable Markdown.

### Metrics

- Win rate (per player index).
- Mean game length on win vs loss.
- Mean prize differential.
- Decision latency (p50, p95).
- Failure category breakdown (via `analysis/failure.py`).

## Logging

Every match produces a structured event log:

```json
[
  {"kind": "SETUP", "active_player": 0},
  {"kind": "STEP_START", "turn": 1, "player": 0, "hand_size": 6, "active": "Pikachu ex"},
  {"kind": "ACTION", "action": {"kind": "ATTACK", "extra": "Thunderbolt"}},
  {"kind": "PRIZE_TAKEN"/"PRIZE_LOST", "remaining": 5},
  {"kind": "ATTACK", "move": "Thunderbolt", "damage": 200, "target": "Charizard"},
  {"kind": "GAME_OVER", "winner": 0, "turns": 42}
]
```

`MatchLogger` persists these to `results/` for offline analysis.

## Reproducibility

Every experiment is deterministic given the seed:

- Decks are constructed from a seed.
- `new_game` shuffles with `random.Random(seed)`.
- Status flips use a derivation of the game seed.

Two runs of `python3 -m pokemon_tcg benchmark --agent Greedy --seeds 42,7`
produce **identical** results on the same machine.

## Retreat mechanics (added v0.2)

The simulator implements real-TCG retreat semantics:

| Step | Real TCG | Our simulator |
|------|----------|---------------|
| 1 | Pay retreat cost (discard that many energy) | Pop `retreat` tokens from active's `attached_energy` |
| 2 | Old active moves to the bench | Append `me.active` to `me.bench` |
| 3 | Chosen benched Pokemon becomes active | Set `me.active = me.bench[target_idx]` |
| 4 | End turn | Opponent draws, turn flips |

`legal_actions(state, who)` emits one `RETREAT` per alive bench Pokemon
when the active has enough energy to pay the retreat cost. The
`_action_retreat` simulator function logs every retreat with
`{from, to, cost_paid, energy_discarded, remaining_energy}` fields.

### Retreat benchmark results

`Champion` vs `Greedy` (matched pairs, 14 seeds, 14 games each, same
decks, sides swapped for fairness):

| Variant | Win rate vs Greedy |
|---------|-------------------|
| Champion with retreat enabled | 9/14 (64%) |
| Champion with retreat filtered out | 9/14 (64%) |

The new mechanic is correctly used by Champion's evaluator (roughly
1 in 6 turns has a retreat when the heuristic decides a swap is
justified) but conservatively scored so it doesn't burn energy on
marginal swaps. **The head-to-head A/B shows no measurable win-rate
delta** within this sample size; the mechanic is added primarily for
*game completeness* (real TCG allows voluntary active swaps). The
benefit of stronger retreat heuristics is unlocked with the 3-ply
+ iterative deepening search described above (12/20 vs Champion 2-ply
in matched pairs).

### 3-ply + iterative deepening benchmark results

`Champion` (default, with 3-ply IDS, 250 ms budget) vs `Champion`
(`max_depth=2`, classical 2-ply), 10 seeds × 2 games per seed, sides
swapped for fairness:

| Variant | Wins | Win rate |
|---------|-----:|---------:|
| Champion **with** 3-ply IDS | **12** | **60%** |
| Champion without 3-ply (2-ply only) | 8 | 40% |

The deeper search adds ~10–30 ms per turn in mid-game positions; the
per-game wall-clock overhead is modest (12.8 s vs ~8 s for the 2-ply
counterpart across 20 games). The win-rate lift comes chiefly from
the third ply: 3-ply explicitly verifies that a setup-and-retreat
sequence retains a positive position *before* we commit the swap,
so Champion stops occasional energy-waste retreats it would have
taken under 2-ply alone.

## Possible extensions

The framework deliberately separates concerns so the following
extensions are local changes:

- **Neural evaluator:** Replace `EvaluatorConfig` + `_ohko_potential`
  with a learned value estimator. *(Shipped as ``LearnedValueFn`` in
  v0.4 — see the "Pluggable value head" section for the A/B benchmark.)*
- **Stronger search:** Add 4-ply alpha-beta or MCTS pushdown for
  even deeper lookahead (3-ply IDS is already shipped).
- **MCTS:** Replace `greedy_1ply` with MCTS rollout-based planning.
- **RL fine-tuning:** Treat the champion evaluator as a reward
  function and run self-play to learn a policy network.
- **Deck-aware eval:** Different evaluator configs per matchup
  (currently the runner uses one config).
- **Full TCG:** Add Trainer card effects (search, recovery, draw N).

## Files

    pokemon_tcg/
        __init__.py
        __main__.py       CLI entrypoint
        cards.py          CSV → typed Card objects
        deck.py           Random + themed deck builders
        game_state.py     Per-player + global state
        actions.py        Legal action enumeration
        simulator.py      Rules engine (only state mutator)
        evaluator.py      Heuristic position scoring
        search.py         1-ply + 2-ply + 3-ply + iterative deepening
        agents/
            base.py
            benchmarks.py
            champion.py
        experiments/
            deck_pool.py
            runner.py
            tournament.py
        evaluation/
            metrics.py
            leaderboard.py
        logging_utils/
            game_log.py
        analysis/
            failure.py
    tests/
        __init__.py
        test_cards.py
        test_deck.py
        test_simulator.py
        test_evaluator.py
        test_agents.py
        test_champion.py
        test_experiments.py
        test_retreat.py
        test_search.py
        test_value_nn.py   MLP value head + self-play tests
    docs/
        architecture.md
    results/             Per-match JSON logs

## Testing

```bash
# Run all 98 tests
python3 -m pokemon_tcg tests

# Filter
python3 -m pokemon_tcg tests --filter search
```

## Web dashboard

```bash
python3 -m pokemon_tcg web --port 5055
```

Three pages, three JSON endpoints, no JavaScript build step:

* `/`         — Elo leaderboard with win-rate bars + matchup summary
* `/matrix`   — pair-wise head-to-head heatmap
* `/replay`   — turn-by-turn replay: seed-driven stepper with a board
                reconstruction and failure-analyzer annotations

The first request triggers a small lazy tournament (`Champion`,
`SearchAgent`, `Greedy`, `GreedyNoRetreat`, `Aggressive`, `Defensive`
across 3 seeds × 1 game) that takes ~8 s; cached responses are <5 ms.
The cache TTL is 10 minutes and `Refresh` re-runs it on demand.

JSON API for external tooling:

* `GET /api/leaderboard`               — Elo + wins/games per agent
* `GET /api/matrix`                    — matchup matrix `{a: {b: wins/games}}`
* `GET /api/replay?p0=A&p1=B&seed=N`   — single annotated match
* `GET /api/runs`                      — recent tournament runs
* `GET /api/health`                    — liveness probe

## License note

Card data (`EN Card Data.csv`) is provided by the challenge. Card text
and stats remain the property of their respective authors.

## Acknowledgements

Built atop the Kaggriculture engineering patterns for consistency with
the existing research platform. The replay/benchmark/log style closely
mirrors `kaggriculture/main.py`.
