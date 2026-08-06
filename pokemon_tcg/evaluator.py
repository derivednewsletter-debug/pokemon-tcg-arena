"""Position evaluator — score a state from `who`'s perspective.

The evaluator is the heuristic backbone of the agent. It assigns a scalar
to every (state, who) pair so search can pick +max actions and +min
opponent actions.

Two interchangeable interfaces live in this module:

* ``score(state, who, cfg=...)`` — module-level function (legacy / terse).
* ``ValueFunction`` protocol + ``HeuristicValueFn`` / ``LearnedValueFn``
  implementations — class-style interface used by ``search.py`` since v0.3.

The class interface is what you should wire into a new agent. It plugs in
both a hand-tuned heuristic (just pass ``HeuristicValueFn(CHAMPION_CONFIG)``)
and a learned value head (``LearnedValueFn(model)`` with a trained MLP).

Component weights are tunable via `EvaluatorConfig`. Components:

  * Prize differential: dominant signal; each prize taken by `who`
    is worth PRIZE_VALUE; each lost prize subtracts the same.
  * Active HP advantage: bigger HP surplus on our Active vs theirs.
  * Bench strength: sum of expected damage our bench can deal
    given attached energy (best usable move, capped at 300).
  * Energy in hand (resource conversion): energy not yet played.
  * Hand size: with diminishing returns past 7 cards.
  * KO threat: how close we are to one-shotting their Active.
  * Status exposure: status on our active is bad; status on theirs is good.

Components are summed; final score is `sum(weights[i] * values[i])`.
Lower scores mean worse for `who`. Search uses `score(state, me)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

from .game_state import GameState, PlayerState, PokemonInstance, WIN_PRIZES, STATUS_BURN, STATUS_POISON


# Tunable weights (empirically tweaked; validated in tests/test_evaluator.py)
@dataclass
class EvaluatorConfig:
    prize_value: float = 100.0
    active_hp_factor: float = 0.5
    bench_strength_factor: float = 1.2
    bench_damage_cap: float = 300.0
    energy_in_hand: float = 0.5
    hand_size_value: float = 0.4
    hand_size_cap: int = 7
    ko_threat_factor: float = 0.8
    status_self_penalty: float = 20.0
    status_opp_bonus: float = 30.0
    # Retreat mechanic weights (added v0.2)
    # `retreat_readiness_weight` rewards having a viable retreat swap in
    # the action set. Champion-style configs may turn this up to bias the
    # search toward considering retreats as part of the plan.
    retreat_readiness_weight: float = 1.0
    # Retreat-model threshold (added v0.5): when a trained regression
    # model (`RetreatWinPredictor`) is wired up, the Champion agent
    # only retreats when ``predict_proba(state) >= threshold``. The
    # ``retreat_win_prob_margin`` is a safety buffer subtracted from
    # the prediction so we require a strict margin above the threshold.
    # Tuned defaults: **0.5** = "more likely to win than lose on retreat";
    # margin **0.05** keeps the gate slightly conservative.
    retreat_win_prob_threshold: float = 0.5
    retreat_win_prob_margin: float = 0.05


DEFAULT_CONFIG = EvaluatorConfig()


# ========================================================================
# ValueFunction protocol (pluggable)
# ========================================================================

@runtime_checkable
class ValueFunction(Protocol):
    """Pluggable value-function interface used by `search.py`.

    A ``ValueFunction`` is anything callable as ``vfn(state, who) -> float``
    where higher is better for ``who``. Two implementations are provided:

    * :class:`HeuristicValueFn` — the hand-tuned score (default).
    * :class:`LearnedValueFn` — a small MLP regressed from self-play.

    Implementations must additionally provide:

    * ``.kind`` — a short tag ("heuristic" / "mlp") for logging.
    * ``.score(state, who) -> float`` — same callable shape.

    Search is agnostic to which one is plugged in.
    """

    kind: str

    def __call__(self, state: GameState, who: int) -> float: ...
    def score(self, state: GameState, who: int) -> float: ...


class HeuristicValueFn:
    """Value function backed by the hand-tuned :func:`score`.

    This is the default ValueFunction and is what existing agents
    (``Greedy``, ``SearchAgent``, ``Champion``) use. We expose it as a
    class so the search signature is uniform regardless of which value
    function is plugged in.
    """

    kind = "heuristic"

    def __init__(self, cfg: EvaluatorConfig | None = None):
        self.cfg = cfg or DEFAULT_CONFIG

    def __call__(self, state: GameState, who: int) -> float:
        return score(state, who, self.cfg)

    def score(self, state: GameState, who: int) -> float:
        return self(state, who)


class LearnedValueFn:
    """Value function backed by a small MLP regressed from self-play.

    Wraps the ``MLP`` defined in :mod:`pokemon_tcg.value_nn` and exposes
    the same callable contract as :class:`HeuristicValueFn`. The MLP
    takes a fixed-length feature vector (extracted by
    :func:`features_from_state`) and returns a single scalar score.

    The trainer (built in :mod:`pokemon_tcg.selfplay`) loads pre-trained
    weights from disk by default; the constructor accepts an optional
    ``path`` argument that loads ``npz`` weights during init.
    """

    kind = "mlp"

    def __init__(self, model: Optional[Any] = None, path: Optional[str] = None):
        # Avoid circular imports: the MLP class lives in value_nn.
        from .value_nn import MLP, features_from_state
        if model is None:
            # Default architecture; matches `pokemon_tcg.selfplay` training.
            self.model = MLP([32, 16, 1])
        else:
            self.model = model
        self._features = features_from_state
        if path is not None:
            self.load(path)

    def score(self, state: GameState, who: int) -> float:
        x = self._features(state, who)
        return float(self.model.forward(x.reshape(1, -1))[0, 0])

    def __call__(self, state: GameState, who: int) -> float:
        return self.score(state, who)

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model.load(path)


def score(state: GameState, who: int, cfg: EvaluatorConfig | None = None) -> float:
    """Static evaluation: higher is better for `who`.

    Kept as a module-level function for `tests/test_evaluator.py` and
    for terse use in benchmarks. Production agents should use
    :class:`HeuristicValueFn` (or :class:`LearnedValueFn`) so the
    search signature is uniform.
    """
    cfg = cfg or DEFAULT_CONFIG
    if state.is_terminal():
        return 1_000_000.0 if state.winner == who else -1_000_000.0
    me = state.me(who)
    opp = state.opp(who)
    s = 0.0
    # Prize diff: opponent has 6 → 0, we have 6 → 0. Player who has FEWER
    # remaining prizes is WINNING (they've taken more).
    s += cfg.prize_value * ((6 - opp.prize_count) - (6 - me.prize_count))
    # Active HP
    my_hp = me.active.hp if me.active and me.active.hp > 0 else 0
    their_hp = opp.active.hp if opp.active and opp.active.hp > 0 else 0
    s += cfg.active_hp_factor * (my_hp - their_hp)
    # Bench strength — best usable attack by each Pokemon
    s += cfg.bench_strength_factor * _bench_power(me)
    s -= cfg.bench_strength_factor * _bench_power(opp)
    # KO threat — if any of our bench+active can OHKO their active in one hit
    if opp.active is not None:
        s += cfg.ko_threat_factor * _ohko_potential(me, opp.active)
    # Energy in hand
    s += cfg.energy_in_hand * _energy_count(me.hand)
    # Retreat readiness — bonus for having enough energy to retreat if needed
    s += _retreat_readiness(me, cfg)
    # Hand size (diminishing)
    hand_bonus = cfg.hand_size_value * min(len(me.hand), cfg.hand_size_cap) \
        + cfg.hand_size_value * max(0, cfg.hand_size_cap - len(me.hand)) * 0.4
    s += hand_bonus
    # Status
    if me.active is not None and me.active.status:
        s -= cfg.status_self_penalty
    if opp.active is not None and opp.active.status:
        s += cfg.status_opp_bonus
    return s


def _retreat_readiness(me: PlayerState, cfg: EvaluatorConfig) -> float:
    """Bonus for being able to retreat (i.e. having energy to swap actives).

    Larger bench + available energy = more tactical options. Default
    weight is small so a non-retreat-heavy agent isn't pulled toward
    gratuitous retreat picks.
    """
    if me.active is None:
        return 0.0
    cost = me.active.base.retreat
    have = len(me.active.attached_energy)
    bench_alive = sum(1 for p in me.bench if p.hp > 0)
    if cost <= 0:
        # Free retreat every turn — strong but not overwhelming.
        return cfg.retreat_readiness_weight * bench_alive * 25.0
    # Energy above retreat cost = headroom
    headroom = max(0, have - cost)
    return cfg.retreat_readiness_weight * (bench_alive * 4.0 + headroom * 1.5)


def _bench_power(me: PlayerState) -> float:
    """Sum of best-usable damage across our board, capped."""
    total = 0.0
    cap = DEFAULT_CONFIG.bench_damage_cap
    if me.active is not None:
        total += me.active.best_usable_damage()
    for p in me.bench:
        if p.hp <= 0:
            continue
        total += p.best_usable_damage()
    return min(total, cap * 5)  # 5 Pokemon * cap


def _energy_count(hand) -> int:
    n = 0
    for c in hand:
        if c.energy:
            n += 1
    return n


def _ohko_potential(me: PlayerState, target: PokemonInstance) -> float:
    """200 if we can OHKO their active, 50 if we can 2HKO."""
    # All attackers (active + bench)
    attackers = []
    if me.active is not None:
        attackers.append(me.active)
    attackers.extend(p for p in me.bench if p.hp > 0)
    can_ohko = False
    can_twohko = False
    for atk in attackers:
        for mv in atk.usable_moves():
            if not mv.damage:
                continue
            d = mv.damage
            if d >= target.hp:
                can_ohko = True
            elif d * 2 >= target.hp:
                can_twohko = True
    if can_ohko:
        return 200.0
    if can_twohko:
        return 50.0
    return 0.0
