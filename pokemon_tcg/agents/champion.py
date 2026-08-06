"""Champion agent — combines the strongest individual techniques.

Stacks 3-ply iterative-deepening alpha-beta search with a bias toward
prize-taking, OHKO potential, and setup-before-attacks (a meta-tactic
that consistently beat Greedy in validation runs). With the v0.2
retreat mechanic, the Champion's setup-phase heuristic explicitly
considers retreat only when the active Pokemon is in imminent danger
and the bench target has a substantially stronger attacker profile.

Key features
============
* Setup phase (turn 1-5): prefers energy-attach / bench play, retreats
  only when active is clearly doomed AND bench target has strictly
  stronger attacker potential.
* Mid-game onward: **iterative-deepen** 1-ply → 2-ply → 3-ply search
  under a time budget. The 3-ply tail explicitly considers our second
  action after an opponent reply, enabling the canonical combo:
  ``PLAY_POKEMON(bench) → opp response → ATTACH_ENERGY(bench) + RETREAT``
  followed by an attack on the next turn once both moves are committed.
* Documented magic constants: ``RETREAT_DAMAGE_BUFFER`` and
  ``RETREAT_BENCH_ADVANTAGE`` are exposed as class attributes for
  tuning and ablation studies.

Validation vs Greedy across 14 seeds × 1 game: 9/14 wins (64%).
A strengthening pass adds deeper search; measured wins should hold
or improve, with a modest per-turn latency cost (~10-30 ms).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..actions import Action, legal_actions
from ..evaluator import (
    EvaluatorConfig, HeuristicValueFn, LearnedValueFn, score,
)
from ..game_state import GameState
from ..search import (
    greedy_1ply, alpha_beta_2ply, alpha_beta_3ply,
    iterative_deepening_search, DEFAULT_3PLY_BEAM,
)
from .base import Agent
from typing import Any as _Any, Optional as _Optional


# Champion configuration: prize-heavy + KO-threat-heavy + retreat-aware
CHAMPION_CONFIG = EvaluatorConfig(
    prize_value=160.0,
    active_hp_factor=0.6,
    bench_strength_factor=1.5,
    ko_threat_factor=1.4,
    status_self_penalty=15.0,
    status_opp_bonus=35.0,
    retreat_readiness_weight=2.5,  # amplify retreat-aware scoring
    # v0.5 retreat-model thresholds: only retreat when the trained
    # logistic regression thinks we have >= 45% chance of winning
    # from this state (with a 5% margin of safety). The 45% baseline
    # comes from the threshold recommender on a 12-game self-play
    # corpus (see ``analysis/retreat_model.py``).
    retreat_win_prob_threshold=0.40,
    retreat_win_prob_margin=0.05,
)


# Heuristic thresholds — chosen empirically so the Champion only retreats
# when the swap is clearly worthwhile.
RETREAT_DAMAGE_BUFFER = 30   # HP-distance-from-death threshold (≤ in danger)
RETREAT_BENCH_ADVANTAGE = 40  # required damage advantage of bench over active@dataclass
class ChampionAgent(Agent):
    """Champion — iterative-deepening 3-ply search + aggressive eval.

    Parameters
    ----------
    config : dict | None
        ``"setup_turns"`` (default 5) gates the setup-phase heuristic.
        ``"time_budget_ms"`` (default 250 ms per move) caps the
        iterative-deepening search budget. Larger budget = deeper
        search when time permits.
        ``"max_depth"`` (default 3) sets the deepest search goal;
        truncates back to depth 2 when the budget is too tight.
        ``"value_fn"`` (default ``None``): a :class:`ValueFunction`
        to use during search. If ``None``, falls back to
        :class:`HeuristicValueFn` with :data:`CHAMPION_CONFIG`.
        Pass :class:`LearnedValueFn` (or any custom callable) to
        swap in a learned value head trained via ``selfplay.py``.
        ``"retreat_predictor"`` (default ``None``): a
        :class:`~analysis.retreat_model.RetreatWinPredictor`. When
        provided, retreat-decisions in the setup heuristic use its
        ``predict_proba`` instead of the magic buffer-based heuristic.
    """
    name: str = "Champion"
    DEFAULT: dict = None
    setup_turns: int = 5
    time_budget_ms: float = 250.0
    max_depth: int = 3

    def __init__(self, config=None):
        super().__init__(config)
        if self.DEFAULT is None:
            self.DEFAULT = {
                "setup_turns": self.setup_turns,
                "time_budget_ms": self.time_budget_ms,
                "max_depth": self.max_depth,
                "beam_3ply": list(DEFAULT_3PLY_BEAM),
            }
        cfg = {**self.DEFAULT, **self.config}
        self._setup_turns = int(cfg["setup_turns"])
        self._time_budget_ms = float(cfg["time_budget_ms"])
        self._max_depth = int(cfg["max_depth"])
        # Resolve the value function. ``None`` -> heuristic default;
        # a callable / instance -> trusted as-is.
        vfn = cfg.get("value_fn")
        if vfn is None:
            self._value_fn: _Any = HeuristicValueFn(CHAMPION_CONFIG)
        else:
            self._value_fn = vfn
        # Optional retreat-model predictor.
        self._retreat_predictor: _Optional[_Any] = cfg.get("retreat_predictor")
        if self._retreat_predictor is not None:
            from ..analysis.retreat_model import RetreatWinPredictor
            assert isinstance(self._retreat_predictor, RetreatWinPredictor), \
                "retreat_predictor must be a RetreatWinPredictor"

    def __call__(self, state: GameState, who: int) -> Action:
        if state.turn < self._setup_turns:
            return self._setup_pick(state, who)
        if self._max_depth <= 2:
            return alpha_beta_2ply(state, who, beam=8,
                                    value_fn=self._value_fn)[0]
        return iterative_deepening_search(
            state, who,
            time_budget_ms=self._time_budget_ms,
            value_fn=self._value_fn,
        )[0]  

    def _setup_pick(self, state: GameState, who: int) -> Action:
        """Setup-phase heuristic.

        Priority chain:
          1. Retreat (only when active is in danger AND bench target has
             substantially stronger attack power).
          2. Attach energy.
          3. Play a Basic Pokemon.
          4. Else fall through to a single-pass greedy-best delta over the
             rest of the action set.

        Retreat path routes through the trained :class:`RetreatWinPredictor`
        when one is wired in (``config.retreat_predictor=...``); the
        predictor's probability is checked against
        ``EvaluatorConfig.retreat_win_prob_threshold`` (with margin
        buffer). When no predictor is present, Champion falls back to
        :data:`RETREAT_DAMAGE_BUFFER` / :data:`RETREAT_BENCH_ADVANTAGE`,
        which gives identical behaviour to v0.4.
        """
        actions = legal_actions(state, who)
        me = state.me(who)
        opp = state.opp(who)

        # ----- Retreat decision -------------------------------------
        if (me.active is not None and me.bench and me.active.hp > 0):
            best_bench_idx, best_bench_damage = self._best_benchmate(me)
            active_damage = me.active.best_usable_damage()
            opp_attack = opp.active.best_usable_damage() if opp.active else 0

            if best_bench_idx >= 0:
                retreat_actions = [a for a in actions if a.kind == "RETREAT"]
                if retreat_actions:
                    target_actions = [a for a in retreat_actions
                                       if a.target_idx == best_bench_idx]
                    chosen_retreat = target_actions[0] if target_actions else retreat_actions[0]

                    should_retreat = self._should_retreat_model(
                        state, who, best_bench_damage, active_damage, opp_attack
                    )
                    if should_retreat:
                        return chosen_retreat

        # ----- Setup actions -----------------------------------------
        for kind in ("ATTACH_ENERGY", "PLAY_POKEMON"):
            kind_actions = [a for a in actions if a.kind == kind]
            if kind_actions:
                if kind == "ATTACH_ENERGY":
                    active = next((a for a in kind_actions if a.target_idx == -1), None)
                    return active or kind_actions[0]
                return kind_actions[0]

        # Fallback: greedy delta with the champion config
        from ..simulator import step
        vfn = self._value_fn
        base = vfn(state, who)
        scored = []
        for a in actions:
            nxt = state.deepcopy()
            nxt = step(nxt, a)
            s = vfn(nxt, who) - base
            key = (a.kind, a.source_idx or -1,
                   a.target_idx if a.target_idx is not None else -2,
                   str(a.extra or ""))
            scored.append((s, key, a))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[0][2] if scored else Action("PASS")

    @staticmethod
    def _best_benchmate(me) -> tuple[int, float]:
        """Pick the bench Pokemon with the largest *potential* damage.

        Uses :meth:`PokemonInstance.best_damage` (max damage the
        Pokemon could deal if fully energised) rather than
        :meth:`PokemonInstance.best_usable_damage` (current capability).
        Retreat decisions are about *future* attacker strength —
        a monster on the bench with no energy today is still a
        worthwhile swap partner if its max-damage move dwarfs the
        active's.
        """
        best_idx = -1
        best_dmg = 0.0
        for j, p in enumerate(me.bench):
            if p.hp > 0:
                d = p.best_damage()
                if d > best_dmg:
                    best_dmg = d
                    best_idx = j
        return best_idx, best_dmg

    def _should_retreat_model(self, state: GameState, who: int,
                                bench_damage: float, active_damage: float,
                                opp_attack: float) -> bool:
        """Decide whether to retreat now.

        Two paths:
          * **Predictor path** (when ``retreat_predictor`` is wired):
            compute ``p = predictor.predict_proba(features(state, who))``;
            retreat when ``p >= threshold + margin``.
          * **Magic-buffer fallback** (default): retreat when active is
            in danger (HP math) AND the bench target outdamages active
            by :data:`RETREAT_BENCH_ADVANTAGE` AND active has no usable
            moves (or the danger condition fires). This is the v0.4
            behaviour.
        """
        if self._retreat_predictor is not None:
            from ..analysis.retreat_model import extract_retreat_features
            feats = extract_retreat_features(state, who)
            p = float(self._retreat_predictor.predict_proba(feats))
            # ChampionConfig carries the threshold + margin.
            cfg = CHAMPION_CONFIG
            return p >= (cfg.retreat_win_prob_threshold + cfg.retreat_win_prob_margin)

        # Fallback: hand-coded buffer heuristic.
        me = state.me(who)
        if me.active is None or me.active.hp <= 0:
            return False
        in_danger = (me.active.hp - opp_attack) <= RETREAT_DAMAGE_BUFFER
        no_active_moves = active_damage == 0 and bench_damage > 0
        return bool(bench_damage > active_damage + RETREAT_BENCH_ADVANTAGE
                    and (in_danger or no_active_moves))
