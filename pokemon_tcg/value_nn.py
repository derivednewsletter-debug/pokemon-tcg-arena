"""Tiny MLP value head — pure numpy, no autograd dependency.

Two pieces:

* ``features_from_state(state, who)`` — extract a fixed-length
  numerical feature vector from a ``GameState``. This is the
  input to the MLP.
* ``MLP`` — a 2-3 layer feed-forward MLP with stored weights,
  forward / backward / SGD update / save / load.

Why numpy-only? The repository avoids third-party ML dependencies
(torch, jax, sklearn) so the trainer + value head are pure-stdlib
plus numpy (already on requirements for the dashboard). This keeps
the runnable surface area small and the trainer inspectable.

Architectural notes
====================

* Input dim is determined at runtime from the first call to
  ``forward``. Default architecture is 32 -> 16 -> 1 (3-layer
  total weights, 2 layers with ReLU + 1 linear output).
* Loss is mean-squared error between the linear output and the
  target (typically a +1/-1/0 game outcome from self-play).
* Training uses plain SGD; this is more than enough to learn a
  ~32-feature linear-ish function over a few thousand examples.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .game_state import GameState, PlayerState, PokemonInstance, WIN_PRIZES


# Feature vector size — also used by ``selfplay.py`` to gate the trainer.
FEATURE_DIM = 32


def features_from_state(state: GameState, who: int) -> np.ndarray:
    """Extract a fixed-length feature vector from ``state`` for player ``who``.

    The 32 features (in this exact order) are::

        0  my prize-count remaining (lower = winning more)
        1  opp prize-count remaining
        2  active HP (mine)
        3  active base HP (mine, max)
        4  active HP fraction (mine)
        5  active number of attached energy
        6  active best-usable damage
        7  opp active HP
        8  opp active base HP
        9  opp active HP fraction
       10  opp active best-usable damage
       11  bench count (mine)
       12  bench count (opp)
       13  bench strength — sum of best-usable damage (mine)
       14  bench strength (opp)
       15  hand size (mine)
       16  hand size (opp)
       17  energy cards in hand (mine)
       18  energy cards in hand (opp)
       19  trainer count in hand (mine)
       20  trainer count in hand (opp)
       21  pokemon Basics in hand (mine)
       22  KO threat — 1.0 if any of our attackers can OHKO opp active
       23  opp KO threat
       24  bench energy headroom above retreat cost (mine)
       25  bench energy headroom above retreat cost (opp)
       26  num bench pokemons with HP > 0 that have at least 1 energy
       27  num bench pokemons that are Stage 1 / 2 (mine)
       28  deck size (mine)
       29  deck size (opp)
       30  turn number (1..MAX_TURNS)
       31  +1.0 if active_player == who else -1.0 (am I on move?)

    All values are clipped/scaled to roughly [-5, 5] so the network
    starts in a sane range (no need for batch normalization during
    quick experiments).
    """
    me = state.me(who)
    opp = state.opp(who)
    f = np.zeros(FEATURE_DIM, dtype=np.float32)

    # Prize diff
    f[0] = WIN_PRIZES - me.prize_count      # prizes we've taken (more = winning)
    f[1] = WIN_PRIZES - opp.prize_count     # prizes opp has taken

    # Active HP / energy / damage
    if me.active is not None and me.active.hp > 0:
        f[2] = me.active.hp
        f[3] = me.active.base_hp
        f[4] = me.active.hp / max(me.active.base_hp, 1)
        f[5] = len(me.active.attached_energy)
        f[6] = me.active.best_usable_damage()
    if opp.active is not None and opp.active.hp > 0:
        f[7] = opp.active.hp
        f[8] = opp.active.base_hp
        f[9] = opp.active.hp / max(opp.active.base_hp, 1)
        f[10] = opp.active.best_usable_damage()

    # Bench tallies + strength
    f[11] = _alive_bench_count(me)
    f[12] = _alive_bench_count(opp)
    f[13] = _bench_damage_sum(me)
    f[14] = _bench_damage_sum(opp)

    # Hand tallies
    f[15] = len(me.hand)
    f[16] = len(opp.hand)
    f[17] = _energy_in_hand(me)
    f[18] = _energy_in_hand(opp)
    f[19] = _trainer_in_hand(me)
    f[20] = _trainer_in_hand(opp)
    f[21] = _pokemon_basics_in_hand(me)

    # KO threat (1.0 / 0.0)
    if opp.active is not None:
        f[22] = 1.0 if _can_ohko(me, opp.active) else 0.0
    if me.active is not None:
        f[23] = 1.0 if _can_ohko(opp, me.active) else 0.0

    # Retreat headroom
    if me.active is not None:
        cost = me.active.base.retreat
        f[24] = max(0, len(me.active.attached_energy) - cost) if cost > 0 else 1.0
    if opp.active is not None:
        cost = opp.active.base.retreat
        f[25] = max(0, len(opp.active.attached_energy) - cost) if cost > 0 else 1.0

    # Bench charged / evolved
    f[26] = sum(1 for p in me.bench if p.hp > 0 and len(p.attached_energy) >= 1)
    f[27] = sum(1 for p in me.bench if p.hp > 0 and p.base.stage != "Basic")

    # Deck sizes + turn
    f[28] = len(me.deck)
    f[29] = len(opp.deck)
    f[30] = state.turn
    f[31] = 1.0 if state.active_player == who else -1.0

    # Compress to a sane range (light normalization)
    return _clip_and_scale(f)


def _clip_and_scale(f: np.ndarray) -> np.ndarray:
    """Light per-feature scaling — values stay in [-5, 5] without
    needing batched normalization. The HeuristicValueFn produces
    numbers in the same range, which makes (vfn_heur - vfn_mlp)
    comparisons at integration time meaningful."""
    out = f.copy()
    # Counts / sizes — clip at 30
    out = np.clip(out, -30, 30)
    # Divide by 5 to bring into [-6, 6] range used by activations.
    out = out / 5.0
    return out.astype(np.float32)


def _alive_bench_count(me: PlayerState) -> int:
    return sum(1 for p in me.bench if p.hp > 0) + (1 if me.active and me.active.hp > 0 else 0)


def _bench_damage_sum(me: PlayerState) -> int:
    total = 0
    if me.active:
        total += me.active.best_usable_damage()
    for p in me.bench:
        if p.hp > 0:
            total += p.best_usable_damage()
    return total


def _energy_in_hand(me: PlayerState) -> int:
    return sum(1 for c in me.hand if c.energy)


def _trainer_in_hand(me: PlayerState) -> int:
    return sum(1 for c in me.hand if c.trainer)


def _pokemon_basics_in_hand(me: PlayerState) -> int:
    return sum(1 for c in me.hand if c.pokemon and c.pokemon.stage == "Basic")


def _can_ohko(me: PlayerState, target: PokemonInstance) -> bool:
    attackers = []
    if me.active and me.active.hp > 0:
        attackers.append(me.active)
    attackers.extend(p for p in me.bench if p.hp > 0)
    for atk in attackers:
        for mv in atk.usable_moves():
            if mv.damage and mv.damage >= target.hp:
                return True
    return False


# ========================================================================
# MLP
# ========================================================================

class MLP:
    """A tiny fully-connected MLP with stored weights + biases.

    Defaults to 32 → 16 → 1 (single regression output). All layers are
    ReLU except the output (linear). The output is interpreted as a
    heuristic-style scalar; we don't squash it because the trainer
    uses targets in roughly [-1, 1].
    """

    def __init__(self, layer_sizes: Sequence[int] = (32, 16, 1), seed: int = 17):
        rng = np.random.default_rng(seed)
        self.layer_sizes = list(layer_sizes)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            # He initialization for ReLU layers
            scale = np.sqrt(2.0 / max(in_dim, 1))
            self.weights.append(rng.normal(0.0, scale, (in_dim, out_dim)).astype(np.float32))
            self.biases.append(np.zeros(out_dim, dtype=np.float32))

    # ---- Forward --------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. ``x`` is either shape ``(D,)`` or ``(N, D)``.

        Returns activations at every layer; the final layer is the
        output. Output uses ``tanh`` to bound predictions to [-1, 1]
        so the network can be trained against self-play win/lose
        targets (-1 / +1) without an output explosion.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        cur = x
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = cur @ W + b
            if i < len(self.weights) - 1:
                z = np.maximum(0.0, z)
            else:
                z = np.tanh(z)
            cur = z
        return cur

    # ---- Backward / SGD step --------------------------------------------

    def sgd_step(self, x: np.ndarray, y: np.ndarray, lr: float = 0.01,
                 grad_clip: float = 1.0) -> float:
        """Plain SGD step on a single batch (or element-wise on vectors).

        ``x`` — shape ``(N, D_in)`` input features.
        ``y`` — shape ``(N, D_out)`` regression targets in [-1, 1].
        ``grad_clip`` — L2 clip threshold applied to ``grad`` at every
        layer; prevents exploding gradients at early epochs.

        Returns the MSE loss used for the step.
        """
        # Forward, saving intermediate activations for backprop
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        n = max(int(y.shape[0]), 1)
        layer_inputs = [x]
        layer_outputs = [x]
        cur = x
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = cur @ W + b
            if i < len(self.weights) - 1:
                z = np.maximum(0.0, z)
            layer_inputs.append(z)
            layer_outputs.append(z)
            cur = z

        # Forward + tanh on output (so yhat is in [-1, 1]).
        yhat_z = cur
        yhat = np.tanh(yhat_z)
        # Sanitize — replace inf/nan in prediction with 0 so loss stays finite
        # (e.g. when an early explosion happens BEFORE the slope_aware backprop
        # but happens *only after* a few steps we are still safe).
        yhat = np.where(np.isfinite(yhat), yhat, 0.0).astype(np.float32)
        diff = (yhat - y).astype(np.float32)
        # MSE clipped to a finite positive value
        loss = float(np.mean(diff ** 2))

        # tanh derivative: 1 - tanh^2
        d_tanh = (1.0 - yhat ** 2).astype(np.float32)
        grad = (diff * d_tanh / n).astype(np.float32)

        for i in reversed(range(len(self.weights))):
            cur_in = layer_outputs[i]
            d_pre = grad
            if i < len(self.weights) - 1:
                # ReLU derivative on layer i+1 pre-activation.
                d_pre = d_pre * (layer_inputs[i + 1] > 0)
            # Global L2 clip on the propagated gradient to keep things tame.
            if grad_clip > 0:
                gn = float(np.linalg.norm(d_pre))
                if gn > grad_clip:
                    d_pre = d_pre * (grad_clip / gn)
            dW = (cur_in.T @ d_pre).astype(np.float32)
            db = d_pre.sum(axis=0).astype(np.float32)
            # Per-row clip on dW as belt-and-braces.
            if grad_clip > 0:
                norms = np.linalg.norm(dW, axis=1, keepdims=True)
                scale = np.minimum(1.0, grad_clip / np.maximum(norms, 1e-8))
                dW = dW * scale
            self.weights[i] -= lr * dW
            self.biases[i] -= lr * db
            grad = (d_pre @ self.weights[i].T).astype(np.float32)

        # Sanitize: replace NaN/Inf parameters with a small reset value.
        for i in range(len(self.weights)):
            bad = ~np.isfinite(self.weights[i])
            if bad.any():
                self.weights[i] = np.where(bad, 0.01, self.weights[i])
            bad = ~np.isfinite(self.biases[i])
            if bad.any():
                self.biases[i] = np.where(bad, 0.0, self.biases[i])

        return loss

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 200,
            batch_size: int = 32, lr: float = 0.01,
            rng_seed: int = 17) -> list[float]:
        """Mini-batch SGD loop. Returns the per-epoch loss curve."""
        rng = np.random.default_rng(rng_seed)
        n = x.shape[0]
        losses = []
        for epoch in range(epochs):
            perm = rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                loss = self.sgd_step(x[idx], y[idx], lr=lr)
                epoch_loss += loss * len(idx)
            losses.append(epoch_loss / n)
        return losses

    # ---- Persistence ----------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path,
                 **{f"W{i}": w for i, w in enumerate(self.weights)},
                 **{f"b{i}": b for i, b in enumerate(self.biases)},
                 layer_sizes=np.array(self.layer_sizes, dtype=np.int32))

    def load(self, path: str) -> None:
        data = np.load(path)
        layer_sizes = list(data["layer_sizes"].astype(int))
        if layer_sizes != self.layer_sizes:
            self.layer_sizes = layer_sizes
            self.weights = []
            self.biases = []
            for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
                self.weights.append(np.zeros((in_dim, out_dim), dtype=np.float32))
                self.biases.append(np.zeros(out_dim, dtype=np.float32))
        for i in range(len(self.weights)):
            self.weights[i] = data[f"W{i}"]
            self.biases[i] = data[f"b{i}"]

    def __repr__(self) -> str:
        sizes = "->".join(str(n) for n in self.layer_sizes)
        return f"MLP({sizes})"
