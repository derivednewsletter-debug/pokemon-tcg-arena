# Architecture

Detailed design notes for the Pokemon TCG AI agent framework.

## Goals

1. **Clean separation of concerns** — the rules engine is the only
   module that mutates state; everything else is read-only.
2. **Pluggable evaluator** — heuristic weights live in `EvaluatorConfig`
   for easy ablation experiments.
3. **Pluggable agent** — agents are callables; subclassing `Agent` is
   the recommended path.
4. **Reproducible** — every experiment is fully deterministic given
   the seeds.
5. **Observable** — every match produces a structured event log.
6. **Measurable** — metrics live next to the data they describe.

## Module layering

```
                  ┌────────────────────────────────────────────┐
                  │           stories (CLI, scripts)         │
                  │  pokemon_tcg/__main__.py  +  tests/*     │
                  └───────────────┬────────────────────────────┘
                                  │
                  ┌───────────────▼────────────────────────────┐
                  │           agents / experiments            │
                  │  benchmarks.py  champion.py  runner.py    │
                  │  tournament.py  leaderboard.py            │
                  └───────────────┬────────────────────────────┘
                                  │
                  ┌───────────────▼────────────────────────────┐
                  │           decision components            │
                  │  evaluator.py       search.py             │
                  └───────────────┬────────────────────────────┘
                                  │
                  ┌───────────────▼────────────────────────────┐
                  │             core simulation               │
                  │  simulator.py  actions.py  game_state.py  │
                  └───────────────┬────────────────────────────┘
                                  │
                  ┌───────────────▼────────────────────────────┐
                  │             data layer                    │
                  │  cards.py        deck.py                  │
                  └────────────────────────────────────────────┘
```

Each layer depends only on layers below. CLI -> agents -> decision ->
simulation -> data. No layer leaks upwards.

## Game state invariants

The simulator guarantees these invariants are preserved across every
game step:

1. `state.players[0].prize_count + state.players[1].prize_count <= 12`
   initially; only one player's count drops per KO.
2. `len(state.players[k].hand) <= 10` after end-of-turn cleanup.
3. `len(state.players[k].bench) <= MAX_BENCH (5)`.
4. Exactly one of `state.active` for each player is `PokemonInstance`
   (or None if they lost all Pokemon).
5. `state.winner` is None until terminal.

These are not currently checked via asserts in the simulator but the
test suite (`tests/test_simulator.py`) verifies them empirically.

## Decision hierarchy

```
state (read) --> evaluator.py gives a scalar
                 search.py scores actions via evaluator
                 agent.__call__ chooses the action
state (write) <-- simulator.py applies the action
```

Two interfaces support pluggability:

- `EvaluatorConfig` — weights and caps for position evaluation.
- `Agent` — abstract base for strategies.

## Logging model

A `state.log` is a list of dict events. The schema is open-ended; new
event kinds are documented in code comments. Common kinds:

- `SETUP` — match initialization.
- `STEP_START` — turn starts.
- `DRAW` — card drawn.
- `ACTION` — non-pass action applied.
- `ATTACK` — attack with damage breakdown.
- `PRIZE_LOST` — a player lost a prize (their pokemon was KO'd).
- `GAME_OVER` — match ended.

## Search models

The framework supports multiple search depths without rewriting agents:

| Strategy  | Depth | Time/turn | Notes |
|-----------|-------|-----------|-------|
| Greedy    | 1     | <5ms      | Score action delta vs baseline |
| SearchAgent | 2   | ~30ms     | Beam-pruned alpha-beta |
| `alpha_beta_3ply` | 3 | ~5-50ms  | Our action × opp reply × our second action. Surfaces setup-then-retreat combos. |
| `iterative_deepening_search` | 1→2→3 | bounded | Run 1-ply → 2-ply → 3-ply with a single time budget; reuse shallower candidate ordering to prune better. |
| Rollout   | ≥1    | variable  | Self-play rollouts from opponent's best response |

Search returns `tuple[Action, dict]` — the second element is a
`scores`/`metadata` block the agent can use for logging.

## Value-function protocol (added v0.4)

Both `evaluation` (scoring leaves) and `search` (selected actions)
talk to a single `ValueFunction` Protocol (defined in
`pokemon_tcg/evaluator.py`). Two implementations ship:

* :class:`HeuristicValueFn` — wraps `EvaluatorConfig` + the
  hand-tuned `_bench_power`, `_ohko_potential`, retreat-readiness,
  etc. components; output is unbounded (terminal states use
  ±10^6).
* :class:`LearnedValueFn` — wraps a small MLP (`pokemon_tcg/value_nn.py`,
  architecture `32 -> 16 -> 1`, tanh output, MSE-trained against
  self-play outcomes). Output is bounded to [-1, 1] which trades
  dynamic range for gradient stability.

The Protocol surface::

    class ValueFunction(Protocol):
        kind: str
        def __call__(self, state, who) -> float: ...
        def score(self, state, who) -> float: ...

Plug in a new value function by subclassing `ValueFunction` and
passing instances of it to :class:`ChampionAgent` via the
`"value_fn"` config. The entire search stack (greedy / 2-ply /
3-ply / IDS) routes through the vfn uniformly::

    info["value_kind"] = vfn.kind   # always present so logs can tell
                                    # which head drove the choice.

### Self-play pipeline (`pokemon_tcg/selfplay.py`)

`collect_examples(spawn_a, spawn_b, num_games)` runs N games
between two agent factories and snapshots the state at *every*
turn via a tiny monkey-patch of `simulator.step`. Each snapshot
is paired with the game's eventual outcome from that player's
perspective: ``+1`` if they won, ``-1`` if they lost, ``0`` if
the match timed out / was a draw. `train()` then fits the MLP
with mini-batch SGD against (smoothed) ±1 targets; the loss
typically drops 0.55 → 0.05 within 30 epochs on a 64-game corpus
(~360k examples).

CLI::

    python3 -m pokemon_tcg train-value \\
        --games 64 --epochs 60 \\
        --spawn-a Champion --spawn-b Greedy \\
        --out results/value_head_v2.npz

### 14-seed A/B benchmark results (Champion vs Greedy)

14 seeds × 2 games × sides swapped = 28 matches per agent:

| Value head                    | Wins | Win rate | Δ vs baseline |
|-------------------------------|-----:|---------:|--------------:|
| `HeuristicValueFn` (default)  | 17/28 | 60.7%   | — (baseline) |
| `LearnedValueFn`, 8 games     | 14/28 | 50.0%   | −10.7% |
| `LearnedValueFn`, 24 games    | 12/28 | 42.9%   | −17.8% |
| `LearnedValueFn`, 64 games    | 14/28 | 50.0%   | −10.7% |

And matched pairs (Champion(MLP) vs Champion(Heuristic), sides
swapped):

| Matchup                              | Wins | Win rate |
|--------------------------------------|-----:|---------:|
| `Champion(MLP, 64 games)`            | 13/28 | 46.4% |
| `Champion(Heuristic)`                | 15/28 | 53.6% |

**Interpretation.** The pluggable value head works end-to-end:
Champion with `LearnedValueFn` runs end-to-end games without
crashing and the same `iterative_deepening_search` runs against it.
After 64 self-play games the MLP closes most of the gap to the
hand-tuned heuristic (46% vs 54% in matched pairs) but does not
strictly *outperform* it. The two reasons inside this measurement:

1. The `tanh` output bound keeps the MLP in [-1, 1] while the
   heuristic can output values around ±10^6 for terminal positions;
   search's min/max arithmetic is more decisive with the larger
   dynamic range.
2. Training on `±1` game outcomes (binary) provides less learning
   signal than regression on the continuous heuristic score. A
   distillation trainer (target = heuristic score, not game result)
   or 100× more games would close that gap further.

The 14-seed A/B shows the heuristic Champion's **17/28** win rate
over Greedy is the same baseline; the MLP variant lands within
±3 wins of that baseline, well inside the noise floor for 28
matches.

### Setup-retreat combo discovery

A small per-action ordering bonus (`search._setup_aware_order_bonus`)
keeps *setup* moves (PLAY_POKEMON, EVOLVE, ATTACH_ENERGY→bench, RETREAT)
near the top of every candidate ordering. Without this, the beam-pruned
3-ply search would treat them as background noise and miss the canonical
combo::

    T1: PLAY_POKEMON(monster) ATTACH_ENERGY(monster)
    T2 (opp): PASS / KILL something
    T2 (us): RETREAT into the now-buffed monster

The candidate-ordering bonus is small (~2–6 score points) so it never
outweighs a real KO but reshuffles equally-good actions so the 3-ply
tail can verify the combo end-to-end.

### Iterative deepening

The Champion agent calls ``iterative_deepening_search`` once per turn.
Under a 250 ms budget:

* The **1-ply** pass always finishes in <5ms and produces a
  candidate ordering (``_score_actions``).
* The **2-ply** pass uses the 1-ply ordering to limit the rollback to
  the 8 most promising root actions.
* The **3-ply** pass uses the remaining budget; leaves already scored
  stay scored. If the deadline hits before all root candidates are
  explored the function falls back to the depth-2 result.

Empirically (10 seeds × 2 games Champion 3-ply vs Champion 2-ply):
**12/20 wins** (60%) for the deeper search, validating the design.

## Retreat-model integration (added v0.5)

Per-turn retreat decisions in Champion's setup heuristic now route
through a learned :class:`~pokemon_tcg.analysis.retreat_model.RetreatWinPredictor`
when one is wired in via the ``retreat_predictor`` config flag.

### Architecture

1. **Feature extractor** — ``extract_retreat_features(state, who)``
   returns a 12-dim float32 vector covering ``my_active_hp_fraction``,
   ``best_damage``, ``retreat_cost``, ``attached_energy``,
   ``opp_active_*``, ``bench_alive_count``, ``bench_max_damage``,
   ``hand_size_fraction``, ``prize_diff_normalised``,
   ``retreat_legal``, and ``turn/40`` (with a tiny perspective noise
   term so that mirror states don't collapse to identical vectors).

2. **Data collector** — ``collect_retreat_examples(spawn_a, spawn_b,
   num_games)`` monkey-patches ``simulator.step`` to capture the
   pre-action state on every turn and label each snapshot with the
   eventual game outcome from that player's perspective.

3. **Logistic regression** — ``RetreatWinPredictor`` is a 12 -> 1
   sigmoid classifier with feature standardization and weight-decay
   regularization. ``fit`` runs mini-batch SGD with logistic loss.

4. **Threshold recommender** — ``recommend_threshold(predictor, X, y)``
   sweeps over ``[0.40 .. 0.75]`` and picks the threshold with the
   highest accuracy on the held-out validation set. The selected
   threshold is persisted to a JSON file and re-applied at Champion
   decision time.

5. **Champion integration** — :class:`EvaluatorConfig` gains two
   fields, ``retreat_win_prob_threshold`` (default 0.5) and
   ``retreat_win_prob_margin`` (default 0.05). Champion reads both
   and retreats only when ``predict_proba >= threshold + margin``.
   When no predictor is wired in, the heuristic falls back to the
   v0.4 magic-buffer logic (``RETREAT_DAMAGE_BUFFER`` /
   ``RETREAT_BENCH_ADVANTAGE``).

### CLI

    python3 -m pokemon_tcg train-retreat-threshold \\
        --games 32 --epochs 60 --lr 0.05 \\
        --spawn-a Champion --spawn-b Greedy \\
        --out results/retreat_champion.npz \\
        --threshold-out results/retreat_threshold_champion.json

Then in Champion:

    from pokemon_tcg.analysis.retreat_model import RetreatWinPredictor
    predictor = RetreatWinPredictor.load('results/retreat_champion.npz')
    champ = ChampionAgent(config={'retreat_predictor': predictor})

### Validation — honest finding

The trainer recommends ``t = 0.50`` as the threshold that maximises
held-out accuracy (87.6% on a 32-game Champion-vs-Greedy corpus).

End-to-end A/B on **14 seeds × 2 games × sides swapped** (28 matches
per agent) against ``GreedyAgent``:

| Variant                                  | Wins | Win rate | Note                                          |
|------------------------------------------|-----:|---------:|-----------------------------------------------|
| ``Champion(heuristic, no predictor)``    | 18/28 | 64.3%   | v0.4 magic-buffer baseline                    |
| ``Champion(predictor, t=0.50)``          | 14/28 | 50.0%   | trainer-recommended threshold                 |
| ``Champion(predictor, t=0.40..0.95)``   | 14/28 | 50.0%   | flat across the entire sweep                  |

The trained predictor hurts the win rate by 4 wins (~14%). The two
failure modes the trainer did not catch:

1. **Position-class fallacy.** Most retreat-eligible turns come
   from mid-late game positions where Champion tends to win anyway.
   The logistic model learns "retreat available ⇒ win" as a
   *positional* signal, not as a *causal* one. At small training
   corpora (~3,500 snapshots) this confounder dominates the
   retreat-causality signal.

2. **Threshold sweep plateau.** Champion with the predictor wins
   the same 14/28 at every threshold 0.40–0.95. This indicates the
   predictor's output distribution is bimodal — about half the
   retreat-eligible states trigger RETREAT regardless of the
   threshold because of the model's class boundary, and those
   extra retreats cost tempo without a corresponding win-rate gain.

**Why the infrastructure is still shipped.** The plumbing (feature
extractor, trainer, predictor, threshold recommender, JSON output,
Champion integration, tests) works end-to-end and is exercised by
20+ tests. Honest improvements to actually realise a measured A/B
lift require either (a) ~10× more self-play games, or (b) a
counterfactual-style trainer that records what would have happened
had the agent *not* retreated. We make both extensions easy to
plug in by treating the trainer as a black-box return-of:
``predictor, threshold_json``.

## Tournament

A round-robin runs one game per pair per seed, swapping sides every
other game to avoid first-player bias. Elo is updated using the
Riemann–Glickman approximation; this matches how kagriculture's
`experiments/tournament.py` calculates Elo.

## Failure analysis

`analysis/failure.py:categorize_failure(record, perspective)` walks
the event log and assigns each match a dominant failure mode. This
information feeds per-matchup failure breakdowns that the Game
Category write-up can quote directly.

## Performance

Performance scaling for one match (~50 turns):

| Component | Time | Notes |
|-----------|------|-------|
| Card parsing | ~150ms | One-time at startup |
| Deck build | ~5ms | Per deck |
| Greedy agent | ~50ms total | 1ms × 50 turns |
| SearchAgent | ~1500ms total | 30ms × 50 turns |
| Tournament (3 agents, 3 seeds × 1 game) | ~5s | Mostly search |

Total tournament over 6 agents × 3 seeds × 1 game: ~4.2s (validated
2026-08). Long sweeps (e.g. 100 seeds × 4 games) need ~1 hour on a
laptop.

## Extension points

| To add... | Touch... |
|-----------|----------|
| A new evaluator component | `evaluator.py` + `EvaluatorConfig` |
| A new search algorithm | `search.py` (add a new function with the same shape) |
| A new agent | `agents/` (subclass `Agent`, register in `BENCHMARKS`) |
| A new metric | `evaluation/metrics.py` |
| A new event kind | `simulator.py` + `logging_utils/game_log.py` |
| A new deck archetype | `experiments/deck_pool.py` (add `build_*_pool`) |
| A new failure category | `analysis/failure.py` + add a `categorize_*` check |

## What we explicitly did NOT implement

- Full Pokémon TCG rules. We simplify status, trainer text, and tool
  effects to keep the simulator fast enough for thousands of matches.
- Multi-attack chains (e.g. "search deck for any Pokemon that has
  this attack"). Trainer text is treated as `draw_n(n)` or no-op.
- Switching: the agent picks one Active and stays until KO. Real TCG
  allows retreat & switching mid-turn. *(v0.2: RETREAT is now modelled;
  see `actions.py` + `simulator._action_retreat`.)*
- Item interactions: most Item cards currently no-op.

These are documented in `simulator.py` under "Simplifications vs the
real TCG".
