"""Predict the per-turn probability of winning when a retreat is available.

This module trains a small logistic regression on the structured
event log produced by ``selfplay.py`` / ``simulate_match``. Features
are the **retreat-relevant subset** of the GameState (active HP, bench
strength, opp's attack, hand size, prize diff, turn) — exactly what's
needed to decide whether to swap the active Pokemon.

The trained model exposes :meth:`RetreatWinPredictor.predict_proba`,
which returns a single probability in ``[0.0, 1.0]``. The Champion
agent reads that probability and retreats only when
``p >= config.retreat_win_prob_threshold + config.retreat_win_prob_margin``.

Why logistic regression and not the MLP value head? The MLP trains
on the **outcome** of the entire game; here we want a **step-local**
estimate — at this exact state, given that retreat is on the table,
what's the expected match win rate? Logistic regression on a few
thousand retreat-eligible snapshots converges in seconds and the
resulting ``p`` is monotone in the obvious win/don't-win features,
so the produced threshold is interpretable.

Outputs
=======

* ``extract_retreat_features(state, who)`` — 12-dim feature vector.
* ``RetreatExample`` — a single (features, label, meta) tuple.
* ``RetreatDataCollector.collect(...)`` — run N games, dump examples.
* ``RetreatWinPredictor`` — fit / predict / save / load.
* ``recommend_threshold(predictor, examples)`` — sweep over a small
  grid of thresholds and report the one giving highest accuracy on
  the held-out half.
"""
from __future__ import annotations

import dataclasses
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np

from ..actions import Action
from ..game_state import GameState, PlayerState, PokemonInstance
from ..simulator import simulate_match
from ..value_nn import FEATURE_DIM


REATREAT_FEATURE_DIM = 12


def extract_retreat_features(state: GameState, who: int) -> np.ndarray:
    """Return a 12-dim float32 vector describing the retreat decision.

    Feature layout (in this exact order — Trainer / Champion
    modules that read this array should not rely on the order
    but should rely on :data:`REATREAT_FEATURE_DIM`)::

        0  my_active_hp_fraction (0..1)
        1  my_active_best_usable_damage (0..300)
        2  my_active_retreat_cost (0..4)
        3  my_active_attached_energy_count (0..10)
        4  opp_active_hp_fraction (0..1)
        5  opp_active_best_usable_damage (0..300)
        6  my_bench_alive_count (0..5 + active = 6)
        7  my_bench_max_damage (0..300)
        8  hand_size_fraction (hand / 7)
        9  prize_diff_normalised (opp.prize_cnt - my.prize_cnt, max +/-6)
       10  retreat_available (1.0 if RETREAT legal, else 0.0)
       11  turn / 40 (clipped to [0, 1])
    """
    me = state.me(who)
    opp = state.opp(who)
    f = np.zeros(REATREAT_FEATURE_DIM, dtype=np.float32)

    a = me.active
    if a is not None and a.hp > 0:
        f[0] = a.hp / max(a.base_hp, 1)
        # Potential max damage (i.e. once the active is fully energised)
        # rather than the currently usable damage — retreat decisions
        # are about *future* attacker potential, not the present.
        f[1] = a.best_damage() / 300.0
        f[2] = a.base.retreat
        f[3] = len(a.attached_energy) / 4.0
    if opp.active is not None and opp.active.hp > 0:
        f[4] = opp.active.hp / max(opp.active.base_hp, 1)
        # Same: use potential max, not currently usable. The retreat
        # decision is partly about absorbing an incoming OHKO.
        f[5] = opp.active.best_damage() / 300.0
    f[6] = (1 if (me.active and me.active.hp > 0) else 0) \
        + sum(1 for p in me.bench if p.hp > 0)
    f[7] = max((p.best_damage() for p in me.bench if p.hp > 0),
               default=0) / 300.0
    f[8] = min(len(me.hand), 7) / 7.0
    f[9] = (opp.prize_count - me.prize_count) / 6.0
    # 10: retreat currently legal in this state?
    f[10] = 1.0 if _retreat_legal(state, who) else 0.0
    # 11: turn/40 plus a tiny perspective indicator so that double-mirror
    # game states (where every state slot is identical between P0 and P1)
    # don't collapse to numerically equal feature vectors. The
    # perturbation is < 1e-3, well below the regression's signal-noise
    # threshold.
    f[11] = min(state.turn, 40) / 40.0 + 1e-3 * (1.0 if who == 0 else -1.0)

    # Light scaling — keep features in [0, 1] for logistic regression.
    f = np.clip(f, -1.5, 1.5)
    return f.astype(np.float32)


def _retreat_legal(state: GameState, who: int) -> bool:
    me = state.me(who)
    if me.active is None or me.active.hp <= 0:
        return False
    cost = me.active.base.retreat
    if len(me.active.attached_energy) < cost:
        return False
    return any(p.hp > 0 for p in me.bench)


# ========================================================================
# Training examples
# ========================================================================

@dataclass
class RetreatExample:
    features: np.ndarray            # shape (12,)
    label: float                   # 1.0 if the player eventually won, else 0.0
    turn: int
    player: int
    game_seed: int
    had_retreat_legal: bool

    def to_np(self) -> tuple[np.ndarray, float]:
        return self.features, self.label


@dataclass
class RetreatDataResult:
    examples: List[RetreatExample] = field(default_factory=list)
    games_played: int = 0
    elapsed_sec: float = 0.0

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.examples:
            return (np.zeros((0, REATREAT_FEATURE_DIM), dtype=np.float32),
                    np.zeros((0, 1), dtype=np.float32))
        x = np.stack([e.features for e in self.examples], axis=0)
        y = np.array([e.label for e in self.examples],
                     dtype=np.float32).reshape(-1, 1)
        return x, y


# ========================================================================
# Data collection (mirrors selfplay.collect_examples but retreat-targeted)
# ========================================================================

AgentFactory = Any   # callable returning something with __call__(state, who) -> Action


def collect_retreat_examples(spawn_a: AgentFactory,
                              spawn_b: AgentFactory,
                              num_games: int = 16,
                              base_seeds: Optional[Iterable[int]] = None
                              ) -> RetreatDataResult:
    """Run ``num_games`` of self-play; emit one example per retreat-eligible
    turn where a RETREAT action was in the action set (regardless of whether
    the player actually retreated). Target = eventual game outcome from
    that player's perspective.
    """
    if base_seeds is None:
        base_seeds = [42, 7, 123, 999, 2024, 31, 17, 88, 256, 31415,
                      5, 64, 13, 21, 47, 11]
    seeds = list(base_seeds)[:num_games]

    result = RetreatDataResult()
    t0 = time.time()

    from ..cards import load_cards
    from ..deck import build_random_deck

    for game_idx, seed in enumerate(seeds):
        cards = load_cards()
        deck_a = build_random_deck(cards, seed=seed)
        deck_b = build_random_deck(cards, seed=seed)
        agent_a = spawn_a()
        agent_b = spawn_b()

        snapshots: List[tuple[int, GameState]] = []  # (player, pre-action state)

        # Patch step to capture pre-action state and immediately re-resolve
        # whether a retreat was legal at that point.
        from .. import simulator as _sim
        original_step = _sim.step

        def capturing_step(state: GameState, action: Action) -> GameState:
            snapshots.append((state.active_player, state.deepcopy()))
            return original_step(state, action)

        try:
            _sim.step = capturing_step
            match_out = simulate_match(deck_a, deck_b, [agent_a, agent_b],
                                         seed=seed * 10000 + game_idx,
                                         log=False, max_turns=80)
        finally:
            _sim.step = original_step

        winner = match_out.get("winner")
        # Build lookup so we can determine per-snapshot winner without
        # re-running the simulator.
        for player, snap in snapshots:
            # Determine if a RETREAT action was legal at this state.
            if _retreat_legal(snap, player):
                if winner is None:
                    label = 0.0
                else:
                    label = 1.0 if winner == player else 0.0
                feats = extract_retreat_features(snap, player)
                result.examples.append(RetreatExample(
                    features=feats, label=label, player=player,
                    game_seed=seed, turn=snap.turn,
                    had_retreat_legal=True,
                ))
        result.games_played += 1

    result.elapsed_sec = time.time() - t0
    return result


# ========================================================================
# Logistic regression
# ========================================================================

class RetreatWinPredictor:
    """Tiny logistic-regression model on 12 retreat features.

    Forward:
        p = sigmoid( (x - mean) / std @ w + b )

    Training: plain SGD with a logistic loss and L2 weight decay.
    """

    def __init__(self):
        self.weights = np.zeros((REATREAT_FEATURE_DIM, 1), dtype=np.float32)
        self.bias = np.zeros((1,), dtype=np.float32)
        self.feature_mean = np.zeros((REATREAT_FEATURE_DIM,), dtype=np.float32)
        self.feature_std = np.ones((REATREAT_FEATURE_DIM,), dtype=np.float32)
        self._fitted = False

    # ---- Forward -------------------------------------------------------

    def predict_proba(self, features: np.ndarray) -> float:
        """``features``: shape ``(12,)``. Returns scalar in [0, 1]."""
        x = self._normalize(features.reshape(1, -1))
        with np.errstate(over="ignore", invalid="ignore"):
            z = float(x @ self.weights + self.bias)
            # Clip z to avoid sigmoid overflow on degenerate inputs.
            z = max(-20.0, min(20.0, z))
            return float(1.0 / (1.0 + np.exp(-z)))

    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        x = self._normalize(features)
        with np.errstate(over="ignore", invalid="ignore"):
            z = x @ self.weights + self.bias
            z = np.clip(z, -20.0, 20.0)
            return 1.0 / (1.0 + np.exp(-z))

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        # Avoid divide-by-zero on degenerate features.
        std = np.maximum(self.feature_std, 1e-6)
        return (x - self.feature_mean) / std

    # ---- Training ------------------------------------------------------

    def fit(self, x: np.ndarray, y: np.ndarray,
            epochs: int = 50, batch_size: int = 32,
            lr: float = 0.05, weight_decay: float = 1e-4,
            rng_seed: int = 7,
            standardize: bool = True) -> list[float]:
        """Mini-batch SGD on the logistic loss.

        Returns the per-epoch loss curve so callers can sanity-check
        that training actually converged.
        """
        n = x.shape[0]
        if n == 0:
            return []

        if standardize:
            self.feature_mean = x.mean(axis=0)
            self.feature_std = x.std(axis=0)
        x_norm = self._normalize(x.astype(np.float32))
        y32 = y.astype(np.float32)

        rng = np.random.default_rng(rng_seed)
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            perm = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                xb = x_norm[idx]
                yb = y32[idx]
                # Forward
                z = xb @ self.weights + self.bias
                # Numerical stability: clip z to avoid exp overflow
                z = np.clip(z, -20.0, 20.0)
                p = 1.0 / (1.0 + np.exp(-z))
                eps = 1e-7
                loss = -np.mean(yb * np.log(p + eps) + (1.0 - yb) * np.log(1.0 - p + eps))
                epoch_loss += float(loss) * len(idx)
                # Gradient: dL/dz = (p - y).mean dL/dW = x.T @ (p - y).mean / m
                m = max(len(idx), 1)
                grad_z = (p - yb) / m
                grad_w = xb.T @ grad_z + weight_decay * self.weights
                grad_b = grad_z.sum(axis=0)
                self.weights -= (lr * grad_w).astype(np.float32)
                self.bias -= (lr * grad_b).astype(np.float32)
            losses.append(epoch_loss / n)
        self._fitted = True
        return losses

    # ---- Persistence ---------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path,
                 weights=self.weights,
                 bias=self.bias,
                 feature_mean=self.feature_mean,
                 feature_std=self.feature_std,
                 fitted=np.array([1 if self._fitted else 0], dtype=np.int32))

    @classmethod
    def load(cls, path: str) -> "RetreatWinPredictor":
        d = np.load(path)
        m = cls()
        m.weights = d["weights"].astype(np.float32)
        m.bias = d["bias"].astype(np.float32)
        m.feature_mean = d["feature_mean"].astype(np.float32)
        m.feature_std = d["feature_std"].astype(np.float32)
        m._fitted = bool(int(d["fitted"][0]))
        return m

    # ---- Dunder --------------------------------------------------------

    def __repr__(self) -> str:
        return f"RetreatWinPredictor(fitted={self._fitted}, n_features={REATREAT_FEATURE_DIM})"


# ========================================================================
# Threshold recommender
# ========================================================================

def recommend_threshold(predictor: RetreatWinPredictor,
                         x: np.ndarray, y: np.ndarray,
                         thresholds: Optional[Sequence[float]] = None,
                         ) -> dict:
    """Pick the threshold maximising accuracy on the supplied examples.

    For each candidate threshold ``t``:
        * Predict ``p = sigmoid(x @ w + b)`` for every example.
        * ``predicted_win = p >= t``.
        * Accuracy = fraction of examples where predicted_win == y.
    Returns ``{"threshold": <best>, "accuracy": <best_acc>, "grid": {...}}``.
    """
    if thresholds is None:
        thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    if x.shape[0] == 0:
        return {"threshold": 0.5, "accuracy": 0.0, "grid": {}}

    p = predictor.predict_batch(x).flatten()
    grid = {}
    best_t = 0.5
    best_acc = -1.0
    for t in thresholds:
        pred = (p >= t).astype(np.float32)
        acc = float(((pred == y.flatten()).mean()))
        grid[round(float(t), 4)] = round(acc, 4)
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return {"threshold": best_t, "accuracy": best_acc, "grid": grid}


# ========================================================================
# Convenience end-to-end pipeline
# ========================================================================

def train_and_pick_threshold(num_games: int = 16,
                              spawn_a: Optional[AgentFactory] = None,
                              spawn_b: Optional[AgentFactory] = None,
                              epochs: int = 60,
                              lr: float = 0.05,
                              out_path: str = "results/retreat_model.npz",
                              threshold_out: str = "results/retreat_threshold.json",
                              verbose: bool = True) -> RetreatWinPredictor:
    """Run the full trainer + threshold recommender in one call."""
    if spawn_a is None:
        from ..agents.champion import ChampionAgent as A
        spawn_a = A
    if spawn_b is None:
        from ..agents.benchmarks import GreedyAgent as B
        spawn_b = B

    if verbose:
        print(f"[retreat] collecting examples from {num_games} self-play games...")
    data = collect_retreat_examples(spawn_a, spawn_b, num_games=num_games)
    if verbose:
        print(f"[retreat] {data.games_played} games done in {data.elapsed_sec:.1f}s "
              f"— collected {len(data.examples)} retreat-eligible turns")

    if not data.examples:
        raise RuntimeError("no retreat-eligible training examples collected")

    x, y = data.to_arrays()
    if verbose:
        wins = int((y == 1).sum())
        losses = int((y == 0).sum())
        print(f"[retreat] examples={x.shape[0]} wins={wins} losses={losses}")

    # Hold out 25% for the threshold recommender.
    n = x.shape[0]
    rng = np.random.default_rng(43)
    perm = rng.permutation(n)
    cut = int(n * 0.75)
    train_x, train_y = x[perm[:cut]], y[perm[:cut]]
    val_x, val_y = x[perm[cut:]], y[perm[cut:]]

    predictor = RetreatWinPredictor()
    losses = predictor.fit(train_x, train_y, epochs=epochs,
                            lr=lr, batch_size=min(64, max(n // 4, 1)))
    if verbose:
        print(f"[retreat] SGD finished: loss[0]={losses[0]:.4f} loss[-1]={losses[-1]:.4f}")

    # Sanity check on val set
    val_acc = float(((predictor.predict_batch(val_x).flatten() >= 0.5).astype(np.float32)
                     == val_y.flatten()).mean())
    if verbose:
        print(f"[retreat] val acc at p>=0.5: {val_acc:.2%}")

    summary = recommend_threshold(predictor, val_x, val_y)
    if verbose:
        print(f"[retreat] best threshold={summary['threshold']:.3f} "
              f"(val acc={summary['accuracy']:.2%})")
        for t, acc in summary['grid'].items():
            print(f"[retreat]   t={t:.2f}: acc={acc:.4f}")

    predictor.save(out_path)
    if verbose:
        print(f"[retreat] saved weights -> {out_path}")

    # Persist the threshold + summary as JSON so Champion can pick it up.
    import json
    with open(threshold_out, "w") as fh:
        json.dump({
            "threshold": summary["threshold"],
            "accuracy": summary["accuracy"],
            "grid": summary["grid"],
            "n_examples": int(x.shape[0]),
            "n_train": int(train_x.shape[0]),
            "n_val": int(val_x.shape[0]),
        }, fh, indent=2)
    if verbose:
        print(f"[retreat] saved threshold summary -> {threshold_out}")

    return predictor
