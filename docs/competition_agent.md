# Competition agent — design & results

This document describes `submission/`, the agent built for the **Kaggle
"Pokémon TCG AI Battle"** competition against the official `cabt`
engine, as opposed to the simplified simulator in `pokemon_tcg/`.

## The competition

* The agent is a pure function `agent(obs_dict) -> list[int]` returning
  option indices (or the 60-card deck on the first call).
* The engine (`submission/cg/`) is an official binary implementing the
  full TCG rules; it validates every pick.
* Key engine rules discovered from its source:
  * First player **cannot attack** and **cannot play Supporters** on
    turn 1 (Carmine is explicitly legal turn 1 going first).
  * **Evolution is banned on turns 1-2** (`state.turn <= 2` check).
  * Damage: **weakness x2, resistance -30** (`CalcDamage`).
  * The setup `DRAW_COUNT` selection is the **opponent-mulligan bonus**
    — you choose how many cards to draw (0..N); draw the max.
  * Some Pokemon enter play only via abilities (e.g. Palafin ex's
    "Zero to Hero": evolve Finizen -> Palafin, retreat it, search-swap).
* The engine exposes `search_begin`/`search_step` — a **perfect-information
  forward simulator** (~0.1 ms/step) that accepts *predicted* hidden info
  (your deck order, prizes, opponent hand/deck/active).

## Architecture

```
agent() → Agent.choose(obs)
           ├─ obs.select is None → return deck
           ├─ tracker.observe(obs)         # accumulate logs, reconcile piles
           └─ _handle(obs)
                ├─ MAIN  → choose_main_action
                │          ├─ _main_candidates  (dedup + prune to ~8 actions)
                │          ├─ lookahead.score_candidate × worlds  (engine search)
                │          │    play candidate → resolve nested picks →
                │          │    our heuristic continuation → opponent heuristic
                │          │    turn → repeat for `rounds` → evaluate state
                │          └─ heuristic fallback (evaluate delta)
                └─ nested (CARD/ENERGY/ATTACK/COUNT/YES_NO/…) → rule pickers
                     setup actives, searches (TO_HAND/TO_FIELD), damage
                     placement (finish KOs), retreat costs, mulligan draws,
                     go-first, ability activation, coin flips…
```

### Value function (`strategy.evaluate`)

Score from a player's perspective over the live engine state:

* Prize lead (dominant), active/bench HP, usable attack damage
* KO threat both ways (OHKO/2HKO with engine damage math)
* Bench **potential** damage (future attackers), hand size, energy,
  deck-out risk, status effects, first-player parity

### Hidden-information model (`tracker`)

* Tracks every revealed card (both players' discards, in-play pokemon +
  attached energy/tools, revealed prizes) against the 60-card list.
* The opponent's unseen remainder is sampled from the mirror deck minus
  revealed cards (a prior that improves as info is revealed).
* Per decision, `sample_worlds` draws 2 independent hidden worlds; every
  candidate action is scored under the **same** worlds for a fair
  comparison (robust to prediction error).

### Search lookahead (`lookahead`)

For each candidate MAIN action: `search_begin` with the predicted world,
`search_step` the action, then resolve every subsequent selection with
rule policies (our own heuristic for our turns, the same heuristic for
the opponent's turns — a real adversarial reply, not a pass), for
`rounds=2` full rounds or until a step budget, then evaluate. This
catches multi-turn sequences: play-draw-attack, attach-then-swing,
setup-retreat combos, and walking into a KO.

## Results (local, sides swapped each game)

| Matchup (our agent)                 | Games | Win rate |
|-------------------------------------|------:|---------:|
| vs random (our deck)                | 24    | 96%      |
| vs sample submission (our deck)     | 24    | 92%      |
| vs greedy rules-bot (our deck)      | 24    | 88%      |
| vs greedy (Palafin deck)            | 20    | 70%      |
| vs greedy (Mega Lucario deck)       | 20    | 70%      |
| mirror (ours vs ours)               | 24    | 50%      |
| with lookahead vs without           | 20    | 65%      |

Decision latency: p50 ~10 ms, p95 < 50 ms, max ~85 ms (safe under the
engine's per-player time limit).

Deck A/B showed the **Mega Abomasnow ex** wall (350 HP, 200-damage
shielded attack, normal evolution) out-performs the Palafin ex combo
deck (340 HP, 250 for 1 {W}) against the greedy bot: 88% vs 70%,
because the simpler line races better under pressure.

## Adaptability notes

* The policy is **state-adaptive**: it reacts to prize lead (finish
  KOs, avoid feeding 2-prize attackers when behind), incoming KO threat
  (retreat/sacrifice decisions), energy starvation (attach first), and
  bench construction (spread bodies).
* Deck-search selections pick the **best** card for the position
  (attackers > combo pieces > energy > support) and optional discards
  keep cards unless the effect demands otherwise.
* Every path is exception-safe: any unexpected selection shape degrades
  to a legal random pick, so the agent never crashes a match.

## Extension points

* **Stronger hidden model**: infer the opponent's deck theme from the
  card pool instead of the mirror prior (see `tracker.opp_deck_composition`).
* **Deeper search**: raise `rounds`/`worlds` (see `main.py` env knobs
  `PTCG_ROUNDS`, `PTCG_WORLDS`, `PTCG_LOOKAHEAD_MS`); diminishing
  returns observed beyond 2/2 at current budgets.
* **Learned value head**: replace `evaluate` with an MLP trained from
  self-play outcomes (the simplified-sim `pokemon_tcg` framework already
  prototypes this).
* **New decks**: `tools/deck_variants.py` builds + engine-validates
  variants; `tools/selfplay.py` A/Bs them.
