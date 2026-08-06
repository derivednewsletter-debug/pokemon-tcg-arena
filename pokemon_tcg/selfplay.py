"""Self-play data generation for the value head.

This module connects the simulator + agents to a stream of
``(features, target)`` examples that the :class:`MLP` value head
trains on.

Workflow
========

1. ``collect_examples(spawn_a, spawn_b, num_games, ...)`` — runs N
   games between two agent factories, snapshots each player's
   ``GameState`` once per turn, and pairs every snapshot with the
   game's eventual outcome from that player's perspective
   (``+1`` if they won, ``-1`` if they lost, ``0`` for a draw).
2. ``train(...)`` — fits an :class:`MLP` against the collected
   examples using mini-batch SGD.
3. ``save`` / ``load`` — round-trip the trained weights to disk so
   they can be reloaded by the Champion without retraining.

The trainer is intentionally simple (no PPO, no replay buffer, no
target network); the goal is to demonstrate the *pluggability* of
the value head, not to win a research benchmark. In practice, even
a few hundred self-play games against a Champion baseline give the
MLP enough signal to roughly track the heuristic on silent positions.

CLI
===

The trainer is also wired into the ``pokemon_tcg`` CLI::

    python3 -m pokemon_tcg train-value --games 20 --epochs 30 \\
        --out results/value_head.npz

Picks default seeds so the run is deterministic.
"""
from __future__ import annotations

import dataclasses
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np

from .actions import Action
from .game_state import GameState
from .simulator import new_game, simulate_match
from .value_nn import FEATURE_DIM, MLP, features_from_state
from .evaluator import HeuristicValueFn, LearnedValueFn


AgentFactory = Callable[[], Any]


@dataclass
class TrainingExample:
    """One snapshot from self-play with the eventual outcome attached."""
    features: np.ndarray         # shape (FEATURE_DIM,)
    target: float                # -1.0 (loss), 0.0 (draw), +1.0 (win)
    player: int                  # 0 or 1
    game_seed: int
    turn: int

    def to_np(self) -> Tuple[np.ndarray, float]:
        return self.features, self.target


@dataclass
class SelfPlayResult:
    examples: List[TrainingExample] = field(default_factory=list)
    games_played: int = 0
    wins_p0: int = 0
    wins_p1: int = 0
    draws: int = 0
    elapsed_sec: float = 0.0

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        x = np.stack([e.features for e in self.examples], axis=0) if self.examples \
            else np.zeros((0, FEATURE_DIM), dtype=np.float32)
        y = np.array([e.target for e in self.examples], dtype=np.float32) \
            .reshape(-1, 1) if self.examples else np.zeros((0, 1), dtype=np.float32)
        return x, y


def _make_snapshots(state: GameState, seed: int) -> List[Tuple[int, np.ndarray]]:
    """Snapshot features for both players at the current state.

    Returns ``[(player_idx, features)]``. Used by the simulator hook
    below; we expose this separately so the trainer can be unit
    tested without a full game loop.
    """
    return [(0, features_from_state(state, 0)),
            (1, features_from_state(state, 1))]


def collect_examples(spawn_a: AgentFactory, spawn_b: AgentFactory,
                     num_games: int = 16,
                     base_seeds: Optional[Iterable[int]] = None,
                     include_log: bool = False,
                     max_turns: int = 80) -> SelfPlayResult:
    """Run ``num_games`` of self-play and return paired (features, target).

    Both ``spawn_a`` and ``spawn_b`` are agent factories: callables
    that produce a fresh agent instance per game. We re-instantiate
    each game so any internal RNG carried by the agent (e.g. Champion
    iterative deepening) cannot leak between games.

    Each game's per-turn snapshots are taken via a tiny monkey-patch
    of ``simulator.step`` so we don't need to fork the simulator.
    The patch captures the state immediately BEFORE the agent's
    action, which is the canonical moment to attach a value label.

    Examples are stored *as observed* — the trainer is responsible
    for converting them into (features, targets) arrays and shuffling.
    """
    if base_seeds is None:
        base_seeds = [42, 7, 123, 999, 2024, 31, 17, 88, 256, 31415, 5, 64, 13, 21, 47, 11]
    seeds = list(base_seeds)[:num_games]

    result = SelfPlayResult()
    t0 = time.time()

    # Read cards lazily so each module can be imported cheaply.
    from .cards import load_cards
    from .deck import build_random_deck

    for game_idx, seed in enumerate(seeds):
        cards = load_cards()
        deck_a = build_random_deck(cards, seed=seed)
        deck_b = build_random_deck(cards, seed=seed)
        agent_a = spawn_a()
        agent_b = spawn_b()

        # Storage for per-turn snapshots
        snapshots: List[Tuple[int, np.ndarray, int]] = []  # (player, features, turn)

        # State we want to inject snapshots into
        wrapper = {"snapshots": snapshots}

        # Patch `simulator.step` so each pre-action state is captured.
        from . import simulator as _sim
        original_step = _sim.step

        def capturing_step(state: GameState, action: Action) -> GameState:
            which = state.active_player
            features = features_from_state(state, which)
            wrapper["snapshots"].append((which, features, state.turn))
            return original_step(state, action)

        # Run a real match with the patched step
        try:
            _sim.step = capturing_step
            match_out = simulate_match(deck_a, deck_b, [agent_a, agent_b],
                                         seed=seed * 10000 + game_idx,
                                         log=False, max_turns=max_turns)
        finally:
            _sim.step = original_step

        winner = match_out.get("winner")
        if winner == 0:
            result.wins_p0 += 1
        elif winner == 1:
            result.wins_p1 += 1
        else:
            result.draws += 1
        result.games_played += 1

        # Pair snapshots with outcome
        for player, features, turn in wrapper["snapshots"]:
            if winner is None:
                t = 0.0
            else:
                t = 1.0 if winner == player else -1.0
            result.examples.append(TrainingExample(
                features=features, target=t, player=player,
                game_seed=seed, turn=turn,
            ))

    result.elapsed_sec = time.time() - t0
    return result


def train(sp_result: SelfPlayResult, epochs: int = 30,
          batch_size: int = 32, lr: float = 0.01,
          model: Optional[MLP] = None,
          verbose: bool = True,
          rng_seed: int = 17) -> MLP:
    """Fit ``model`` (or a fresh :class:`MLP`) against the examples.

    Returns the trained MLP.
    """
    if model is None:
        model = MLP([FEATURE_DIM, 16, 1])
    x, y = sp_result.to_arrays()
    if x.shape[0] == 0:
        # No data — best we can do is return the freshly initialised net.
        return model

    # Mix in a small amount of label smoothing (-1, 0, +1) -> (-0.95 etc.)
    # to avoid over-confident outputs.
    y_smooth = y * 0.95

    losses = model.fit(x, y_smooth, epochs=epochs,
                       batch_size=min(batch_size, max(x.shape[0], 1)),
                       lr=lr, rng_seed=rng_seed)
    if verbose:
        print(f"[train] examples={x.shape[0]} epochs={epochs} "
              f"loss[0]={losses[0]:.4f} loss[-1]={losses[-1]:.4f}")
    return model


def save_model(model: MLP, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model.save(path)


def load_model(path: str) -> MLP:
    model = MLP([FEATURE_DIM, 16, 1])
    model.load(path)
    return model


def make_learned_value_fn(model_path: Optional[str] = None
                           ) -> LearnedValueFn:
    """Build a :class:`LearnedValueFn`, optionally loading weights."""
    fn = LearnedValueFn()
    if model_path and os.path.exists(model_path):
        fn.load(model_path)
    return fn


# ========================================================================
# Convenience: perform one end-to-end training run.
# ========================================================================

def run_selfplay_and_train(num_games: int = 16,
                            spawn_a: Optional[AgentFactory] = None,
                            spawn_b: Optional[AgentFactory] = None,
                            epochs: int = 30, lr: float = 0.01,
                            out_path: str = "results/value_head.npz",
                            verbose: bool = True) -> MLP:
    """Run self-play + train + save.

    Defaults to ``ChampionAgent`` vs ``GreedyAgent`` so the
    trainer is self-contained.
    """
    if spawn_a is None:
        from .agents.champion import ChampionAgent
        spawn_a = ChampionAgent
    if spawn_b is None:
        from .agents.benchmarks import GreedyAgent
        spawn_b = GreedyAgent

    if verbose:
        print(f"[selfplay] collecting {num_games} games ...")
    sp = collect_examples(spawn_a, spawn_b, num_games=num_games)
    if verbose:
        print(f"[selfplay] {sp.games_played} games done in {sp.elapsed_sec:.1f}s "
              f"— collected {len(sp.examples)} examples "
              f"(wins_p0={sp.wins_p0}, wins_p1={sp.wins_p1})")
    if verbose:
        print(f"[train] fitting MLP ({FEATURE_DIM} -> 16 -> 1) on "
              f"{len(sp.examples)} examples ...")
    model = train(sp, epochs=epochs, lr=lr, verbose=verbose)
    save_model(model, out_path)
    if verbose:
        print(f"[save]  weight file -> {out_path}")
    return model
