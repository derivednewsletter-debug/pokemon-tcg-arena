"""Benchmark agent strategies.

Each strategy is a deterministic wrapper around the action-selection
heuristics. They share the same shared selection pipeline (action
filtering + scoring) but differ in weights and priority shaping — the
goal is to provide enough diversity for head-to-head tournament
comparison AND give the search agent something to beat.

Agents
======
* ``Greedy``              score every legal action; pick the highest
* ``SearchAgent``         2-ply alpha-beta with `beam` pruning
* ``Aggressive``          maximize prize-taking and OHKO plays
* ``Defensive``           stay alive longer; late attacks
* ``EnergyRamp``          attach energy before attacking
* ``BenchBuffer``         play Basic Pokemon before doing anything else
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..actions import Action, legal_actions
from ..evaluator import score, DEFAULT_CONFIG, EvaluatorConfig
from ..game_state import GameState, PlayerState
from ..search import greedy_1ply, alpha_beta_2ply
from .base import Agent, observe


# ========================================================================
# Action scoring core (shared by heuristic agents)
# ========================================================================

@dataclass
class ActionScorer:
    """Per-priority heuristic — extends the evaluator delta with
    configurable weights so each strategy can shape the search."""
    cfg: EvaluatorConfig = field(default_factory=lambda: DEFAULT_CONFIG)

    def base_score(self, state: GameState, who: int) -> float:
        return score(state, who, self.cfg)

    def score_action(self, state: GameState, who: int, action: Action,
                     next_state: GameState) -> float:
        return score(next_state, who, self.cfg) - self.base_score(state, who)


# ========================================================================
# GreedyAgent — picks the highest evaluator delta
# ========================================================================

class GreedyAgent(Agent):
    """Score every legal action by immediate evaluator delta; pick best."""
    name = "Greedy"

    DEFAULT = dict(search_depth=1, evaluator_config={})

    def __init__(self, config=None):
        super().__init__(config)
        self._cfg = {**self.DEFAULT, **self.config}

    def __call__(self, state: GameState, who: int) -> Action:
        a, _ = greedy_1ply(state, who)
        return a


class GreedyNoRetreatAgent(Agent):
    """Same as Greedy but RETREAT actions are filtered out, mirroring
    pre-retreat behavior. Used as a baseline for A/B benchmarks."""
    name = "GreedyNoRetreat"

    DEFAULT = dict(search_depth=1, evaluator_config={})

    def __init__(self, config=None):
        super().__init__(config)
        self._cfg = {**self.DEFAULT, **self.config}

    def __call__(self, state: GameState, who: int) -> Action:
        actions = legal_actions(state, who)
        # Strip RETREAT actions to compare to the pre-retreat behavior.
        actions = [a for a in actions if a.kind != "RETREAT"]
        base = score(state, who)
        scored = []
        for a in actions:
            nxt = state.deepcopy()
            from .benchmarks import _apply
            nxt = _apply(nxt, a, who)
            s = score(nxt, who) - base
            key = (a.kind, a.source_idx or -1,
                   a.target_idx if a.target_idx is not None else -2,
                   str(a.extra or ""))
            scored.append((s, key, a))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[0][2] if scored else Action("PASS")


def _apply(state, action, who):
    """Apply a single action to state (used by GreedyNoRetreat)."""
    from ..simulator import step
    nxt = state.deepcopy()
    return step(nxt, action)


class SearchAgent(Agent):
    """2-ply alpha-beta with beam pruning. Strong but slower."""
    name = "SearchAgent"

    DEFAULT = dict(beam=8, evaluator_config={})

    def __init__(self, config=None):
        super().__init__(config)
        self._cfg = {**self.DEFAULT, **self.config}

    def __call__(self, state: GameState, who: int) -> Action:
        a, _ = alpha_beta_2ply(state, who, beam=int(self._cfg["beam"]))
        return a


# ========================================================================
# Heuristic policies — biased Greedy
# ========================================================================

class AggressiveAgent(Agent):
    """Heavily weight prizes and OHKO potential; play maximally offensive."""
    name = "Aggressive"

    DEFAULT = dict(
        prize_value=180.0, active_hp_factor=0.4,
        bench_strength_factor=1.4, ko_threat_factor=2.0,
    )

    def __init__(self, config=None):
        super().__init__(config)
        self._cfg = {**self.DEFAULT, **self.config}

    def __call__(self, state: GameState, who: int) -> Action:
        cfg = _build_evaluator(self._cfg)
        return _biased_greedy(state, who, cfg)


class DefensiveAgent(Agent):
    """Preserve HP, prep bench, attack only when forced."""
    name = "Defensive"

    DEFAULT = dict(
        prize_value=80.0, active_hp_factor=1.2,
        bench_strength_factor=0.7, ko_threat_factor=0.5,
    )

    def __init__(self, config=None):
        super().__init__(config)
        self._cfg = {**self.DEFAULT, **self.config}

    def __call__(self, state: GameState, who: int) -> Action:
        cfg = _build_evaluator(self._cfg)
        return _biased_greedy(state, who, cfg)


class EnergyRampAgent(Agent):
    """Attach energy first; only attack when active reach >=120 damage."""
    name = "EnergyRamp"

    DEFAULT = dict(min_attack_damage=120, evaluator_config={})

    def __init__(self, config=None):
        super().__init__(config)
        self._cfg = {**self.DEFAULT, **self.config}

    def __call__(self, state: GameState, who: int) -> Action:
        return _energy_ramp_pick(state, who, int(self._cfg["min_attack_damage"]))


class BenchBufferAgent(Agent):
    """Always bench a Basic Pokemon if the bench has fewer than 5."""
    name = "BenchBuffer"

    DEFAULT = dict(min_bench=4, evaluator_config={})

    def __init__(self, config=None):
        super().__init__(config)
        self._cfg = {**self.DEFAULT, **self.config}

    def __call__(self, state: GameState, who: int) -> Action:
        return _bench_first_pick(state, who, int(self._cfg["min_bench"]))


# ========================================================================
# Helpers
# ========================================================================

def _build_evaluator(overrides: dict) -> EvaluatorConfig:
    cfg = EvaluatorConfig(
        prize_value=float(overrides.get("prize_value", 100)),
        active_hp_factor=float(overrides.get("active_hp_factor", 0.5)),
        bench_strength_factor=float(overrides.get("bench_strength_factor", 1.2)),
        ko_threat_factor=float(overrides.get("ko_threat_factor", 0.8)),
        status_self_penalty=float(overrides.get("status_self_penalty", 20)),
        status_opp_bonus=float(overrides.get("status_opp_bonus", 30)),
    )
    return cfg


def _biased_greedy(state: GameState, who: int, cfg: EvaluatorConfig) -> Action:
    """Standard greedy with custom evaluator."""
    actions = legal_actions(state, who)
    base = score(state, who, cfg)
    best = (base, Action("PASS"))
    for a in actions:
        next_state = state.deepcopy()
        from ..simulator import step
        try:
            next_state = step(next_state, a)
        except Exception:
            continue
        s = score(next_state, who, cfg) - base
        if s > best[0]:
            best = (s, a)
    return best[1]


def _energy_ramp_pick(state: GameState, who: int, min_dmg: int) -> Action:
    """Attach energy if active damage is below `min_dmg`; otherwise attack."""
    me = state.me(who)
    if me.active and me.active.best_usable_damage() < min_dmg \
            and not me.energy_attached_this_turn:
        for a in legal_actions(state, who):
            if a.kind == "ATTACH_ENERGY":
                return a
    # Otherwise pick max-damage attack
    return _max_damage(state, who)


def _bench_first_pick(state: GameState, who: int, min_bench: int) -> Action:
    """Bench a Basic if bench count < min_bench."""
    me = state.me(who)
    if len(me.bench) < min_bench and len(me.bench) < 5:
        for a in legal_actions(state, who):
            if a.kind == "PLAY_POKEMON":
                return a
    return _max_damage(state, who)


def _max_damage(state: GameState, who: int) -> Action:
    """Pick the attack with the highest base damage (ties broken by energy cost)."""
    me = state.me(who)
    if me.active is None:
        return Action("PASS")
    best = None
    for m in me.active.base.moves:
        if not m.can_play(list(me.active.attached_energy)):
            continue
        if best is None or (m.damage or 0) > (best[1].damage or 0):
            best = (m.name, m)
    if best is None:
        return Action("PASS")
    return Action("ATTACK", extra=best[0])


BENCHMARKS: dict[str, type] = {
    "Greedy": GreedyAgent,
    "GreedyNoRetreat": GreedyNoRetreatAgent,
    "SearchAgent": SearchAgent,
    "Aggressive": AggressiveAgent,
    "Defensive": DefensiveAgent,
    "EnergyRamp": EnergyRampAgent,
    "BenchBuffer": BenchBufferAgent,
}

# Lazy Champion import — avoids forcing evaluation imports on this module
try:
    from .champion import ChampionAgent
    BENCHMARKS["Champion"] = ChampionAgent
except Exception:  # pragma: no cover - lazy guard
    pass
