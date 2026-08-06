"""Lookahead search over Pokemon TCG states.

Three search modes live here:

* ``greedy_1ply``         — pick the action with the largest evaluator delta.
* ``alpha_beta_2ply``     — our action × opponent's best (pessimistic) reply.
* ``alpha_beta_3ply``     — our action × opp reply × our second action; the
                             depth where setup-then-RETREAT-then-attack
                             combos become visible.
* ``iterative_deepening_search`` — run depth 1 → 2 → 3, deepening only as
                             the time budget permits. Each deeper pass
                             reuses the shallower one's candidate ordering to
                             prune more aggressively.

The interface is deliberately small so an agent can switch search modes
without code changes::

    act, info = pick(state, who)
    act, info = pick(state, who, depth=3, time_budget_ms=250)
"""
from __future__ import annotations

import time
from typing import Any, Iterable, Optional, Sequence

from .actions import Action, legal_actions
from .evaluator import (
    EvaluatorConfig, HeuristicValueFn, LearnedValueFn, ValueFunction, score
)
from .game_state import GameState


ValueFn = Any  # duck-typed: callable (state, who) -> float


def _resolve_value_fn(value_fn: Optional[ValueFn], cfg: Optional[EvaluatorConfig]
                      ) -> ValueFunction:
    """Return a callable ``(state, who) -> float``.

    Resolution rule:

    * If ``value_fn`` is given, use whichever happens to be passed
      (works for :class:`HeuristicValueFn`, :class:`LearnedValueFn`,
      or a raw lambda). We sanity-check the interface.
    * Otherwise build a fresh :class:`HeuristicValueFn(cfg)` so the
      legacy ``cfg=...`` keyword keeps working.

    The returned object exposes a ``.kind`` attribute for logging.
    """
    if value_fn is None:
        return HeuristicValueFn(cfg)
    if isinstance(value_fn, HeuristicValueFn) or isinstance(value_fn, LearnedValueFn):
        return value_fn
    # Trust the caller: any callable with a ``.kind`` is fine.
    if hasattr(value_fn, "kind"):
        return value_fn
    # Wrap raw callables uniformly. We attach ``.kind`` for compatibility.
    wrapped = value_fn
    try:
        wrapped.kind = getattr(value_fn, "kind", "custom")
    except Exception:
        pass
    return wrapped


# ========================================================================
# Helpers
# ========================================================================

# Aiming for: 1-ply = pick best evaluator delta;
#             2-ply = best response to opponent's best reply;
#             3-ply = best setup-line through one opp reply.

def _action_key(a: Action) -> tuple:
    """Deterministic tiebreak key so search is reproducible."""
    return (a.kind, a.source_idx or -1,
            a.target_idx if a.target_idx is not None else -2,
            str(a.extra or ""))


def _apply_clone(state: GameState, action: Action) -> GameState:
    """Apply a single action to a deep-copy of state.

    Re-uses the simulator's `step` so search mirrors the real game (each
    step flips the active player, draws a card, etc.). Search depth is
    expressed in *our* steps, so deeper plies span multiple full turns.
    """
    from .simulator import step as _step
    nxt = state.deepcopy()
    return _step(nxt, action)


def _setup_aware_order_bonus(a: Action, who: int) -> float:
    """Tiny bias to keep setup actions near the top of the candidate list.

    PLAY_POKEMON and ATTACH_ENERGY→bench are deliberately ordered before
    a 2-ly opponent reply so the 3-ply tail can find the
    PLAY_POKEMON → RETREAT → ATTACK combo (the user's stated goal).

    Values are small compared to a KO delta (~80 score points) so they
    never outweigh a real KO but reorder search-equivalent candidates.
    """
    if a.kind == "PLAY_POKEMON":
        return 6.0   # grow bench
    if a.kind == "EVOLVE":
        return 4.0   # evolve something
    if a.kind == "ATTACH_ENERGY":
        # Bench targets (not active) get a small bonus because they
        # foreshadow a RETREAT-then-ATTACK combo.
        return 3.0 if (a.target_idx is not None and a.target_idx >= 0) else 1.0
    if a.kind == "RETREAT":
        return 2.0
    return 0.0


def _score_actions(state: GameState, who: int,
                   cfg: EvaluatorConfig | None = None,
                   value_fn: ValueFn = None) -> list[tuple[float, tuple, Action]]:
    """Score every legal action by 1-ply delta + setup-order bonus.

    Returns a list of ``(delta_with_bonus, key, action)`` triples sorted
    descending by score (deterministic secondary tiebreak).

    Either ``cfg`` (legacy heuristic-config path) or ``value_fn``
    (pluggable value function) is consulted. ``value_fn`` wins if both
    are provided.
    """
    vfn = _resolve_value_fn(value_fn, cfg)
    actions = legal_actions(state, who)
    base = vfn(state, who)
    scored = []
    for a in actions:
        delta = vfn(_apply_clone(state, a), who) - base
        score_with_bonus = delta + _setup_aware_order_bonus(a, who)
        scored.append((score_with_bonus, _action_key(a), a))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored


# ========================================================================
# One-ply greedy
# ========================================================================

def greedy_1ply(state: GameState, who: int,
                cfg: EvaluatorConfig | None = None,
                value_fn: ValueFn = None) -> tuple[Action, dict]:
    """Score every legal action by immediate evaluator delta.

    Either ``cfg`` (legacy heuristic-config path) or ``value_fn``
    (pluggable value function) is consulted. ``value_fn`` wins if
    both are provided.
    """
    scored = _score_actions(state, who, cfg=cfg, value_fn=value_fn)
    if not scored:
        return Action("PASS"), {"candidates": 0, "best_delta": 0.0, "scores": [],
                                "value_kind": _resolve_value_fn(value_fn, cfg).kind}
    best_score, _, best_action = scored[0]
    vfn = _resolve_value_fn(value_fn, cfg)
    info = {
        "scores": [{"delta": d, "action": a.to_json()} for d, _, a in scored[:10]],
        "best_delta": best_score,
        "candidates": len(scored),
        "depth": 1,
        "value_kind": vfn.kind,
    }
    return best_action, info


# ========================================================================
# Two-ply alpha-beta
# ========================================================================

def alpha_beta_2ply(state: GameState, who: int, beam: int = 8,
                    cfg: EvaluatorConfig | None = None,
                    value_fn: ValueFn = None) -> tuple[Action, dict]:
    """Our top-`beam` actions × opponent's best response (1-ply greedy).

    Returns the action that maximizes our worst-case score.
    """
    vfn = _resolve_value_fn(value_fn, cfg)
    scored = _score_actions(state, who, cfg=cfg, value_fn=value_fn)
    ours_top = scored[:beam]
    best = None
    info = {"candidates": len(scored), "beam": beam, "depth": 2, "scores": [],
            "value_kind": vfn.kind}
    for our_delta, _key, our_a in ours_top:
        next_state = _apply_clone(state, our_a)
        opp_idx = 1 - who
        opp_a, _ = greedy_1ply(next_state, opp_idx, value_fn=value_fn)
        result = _apply_clone(next_state, opp_a)
        worst_case = vfn(result, who)
        info["scores"].append({
            "ours": our_a.to_json(),
            "ours_delta": our_delta,
            "opp_response": opp_a.to_json(),
            "score": worst_case,
        })
        if best is None or worst_case > best[0]:
            best = (worst_case, our_a)
    return (best[1] if best else ours_top[0][2]), info


# ========================================================================
# Three-ply alpha-beta (with setup retreat combo bias)
# ========================================================================

# beam3 == (K0, K1, K2) — top-K candidates at each ply.
# K0: how many of OUR actions to explore at root
# K1: how many of OPP's responses per root
# K2: how many of OUR second actions per opp response
# Total leaves <= K0 * K1 * K2; default 6*5*5 = 150 leaves, well under 1s.

DEFAULT_3PLY_BEAM = (6, 5, 5)


def alpha_beta_3ply(state: GameState, who: int,
                    beam: tuple[int, int, int] = DEFAULT_3PLY_BEAM,
                    cfg: EvaluatorConfig | None = None,
                    time_budget_s: float | None = None,
                    value_fn: ValueFn = None) -> tuple[Action, dict]:
    """Three-ply alpha-beta search.

    Pattern::

        max_{a0 ∈ top-K0 ours}   min_{a1 ∈ top-K1 opp}   max_{a2 ∈ top-K2 ours}   score(s_2, who)

    Returns the *first-stage* action ``a0`` that maximises our worst-case
    position after the opponent's reply and our follow-up.

    The setup-aware ordering bonus (``_setup_aware_order_bonus``) keeps
    RETREAT candidates alive in K2 whenever our recent actions grew
    the bench, surfacing the canonical "PLAY → ATTACH → RETREAT → ATTACK"
    combo the user requested.

    A ``time_budget_s`` cutoff prunes the search early. Leaves already
    scored stay scored; remaining root candidates are skipped.
    """
    vfn = _resolve_value_fn(value_fn, cfg)
    beam_a, beam_b, beam_c = beam
    scored = _score_actions(state, who, cfg=cfg, value_fn=value_fn)
    ours_top = scored[:beam_a]

    best = None
    info = {
        "candidates": len(scored),
        "beam": list(beam),
        "depth": 3,
        "leaves_evaluated": 0,
        "nodes_evaluated": 0,
        "best_score": None,
        "truncated_by_time": False,
        "trace": [],   # per root action: opp response + best follow-up + score
        "value_kind": vfn.kind,
    }
    deadline = (time.monotonic() + time_budget_s) if time_budget_s else None

    for our_score_bonus, _key, our_a in ours_top:
        if deadline is not None and time.monotonic() > deadline:
            info["truncated_by_time"] = True
            break

        s_after_a = _apply_clone(state, our_a)
        opp_idx = 1 - who

        # --- Ply 1: opponent's turn. Take opponent's TOP-K1 responses.
        opp_scored = _score_actions(s_after_a, opp_idx, cfg=cfg, value_fn=value_fn)[:beam_b]
        worst_across_opp = None
        opp_response_record = None

        for opp_delta, _opp_key, opp_a in opp_scored:
            if deadline is not None and time.monotonic() > deadline:
                info["truncated_by_time"] = True
                break

            s_after_b = _apply_clone(s_after_a, opp_a)

            # --- Ply 2: our second turn. Take our TOP-K2 follow-ups.
            our_followups = _score_actions(s_after_b, who, cfg=cfg, value_fn=value_fn)
            info["nodes_evaluated"] += len(our_followups)

            if our_followups:
                # Take the BEST follow-up in this leaf path.
                best_follow = our_followups[0]
                leaf_state = _apply_clone(s_after_b, best_follow[2])
                leaf_score = vfn(leaf_state, who)
            else:
                leaf_state = s_after_b
                leaf_score = vfn(leaf_state, who)

            info["leaves_evaluated"] += 1

            if (worst_across_opp is None) or (leaf_score < worst_across_opp):
                worst_across_opp = leaf_score
                opp_response_record = (opp_a.to_json(), our_followups[0][2].to_json()
                                        if our_followups else None,
                                        leaf_score - vfn(s_after_a, who))

        info["trace"].append({
            "ours": our_a.to_json(),
            "ours_bonus": our_score_bonus,
            "opp_top": opp_response_record[0] if opp_response_record else None,
            "ours_follow": opp_response_record[1] if opp_response_record else None,
            "score": worst_across_opp,
        })

        if best is None or (worst_across_opp is not None and worst_across_opp > best[0]):
            best = (worst_across_opp, our_a)
            info["best_score"] = worst_across_opp

    # Recover the action straight from best (do not give up if time-truncated).
    chosen = best[1] if best else ours_top[0][2]
    return chosen, info


# ========================================================================
# Iterative deepening (depth 1 -> 2 -> 3 with deadline)
# ========================================================================

DEFAULT_IDS_BEAMS = {"depth_1_width": 16,
                     "depth_2_beam": 8,
                     "depth_3_beam": DEFAULT_3PLY_BEAM}


def iterative_deepening_search(state: GameState, who: int,
                                time_budget_ms: float = 250,
                                beams: dict | None = None,
                                cfg: EvaluatorConfig | None = None,
                                value_fn: ValueFn = None,
                                ) -> tuple[Action, dict]:
    """Run 1-ply → 2-ply → 3-ply, deepening only as the deadline permits.

    Honoured time budget: returns whatever depth was completed successfully.
    The 3-ply tail gets the *remainder* of the budget so shallower depths
    always have a chance to finish first.

    Returns ``(action, info)`` where ``info["depth_reached"]`` reports
    the deepest fully-completed depth.
    """
    beams = {**DEFAULT_IDS_BEAMS, **(beams or {})}
    info: dict = {
        "depth_reached": 0,
        "elapsed_ms": 0.0,
        "time_budget_ms": time_budget_ms,
        "depths": {},
        "value_kind": _resolve_value_fn(value_fn, cfg).kind,
    }
    deadline = time.monotonic() + time_budget_ms / 1000.0

    # ---- Depth 1 ----
    t0 = time.monotonic()
    a1, info1 = greedy_1ply(state, who, cfg=cfg, value_fn=value_fn)
    info["depths"][1] = info1 | {"ms": round((time.monotonic() - t0) * 1000, 2)}
    chosen = a1
    info["depth_reached"] = 1
    leftover = deadline - time.monotonic()
    if leftover <= 0:
        info["elapsed_ms"] = round((time.monotonic() - t0) * 1000, 2)
        return chosen, info

    # ---- Depth 2 ----
    t0 = time.monotonic()
    a2, info2 = alpha_beta_2ply(state, who,
                                 beam=beams["depth_2_beam"],
                                 cfg=cfg, value_fn=value_fn)
    info["depths"][2] = info2 | {"ms": round((time.monotonic() - t0) * 1000, 2)}
    chosen = a2
    info["depth_reached"] = 2
    leftover = deadline - time.monotonic()
    if leftover <= 0:
        info["elapsed_ms"] = round((deadline - t0) * 1000, 2)
        return chosen, info

    # ---- Depth 3 (use the *remaining* budget) ----
    t0 = time.monotonic()
    a3, info3 = alpha_beta_3ply(state, who,
                                 beam=beams["depth_3_beam"], cfg=cfg,
                                 time_budget_s=max(0.0, leftover),
                                 value_fn=value_fn)
    info["depths"][3] = info3 | {"ms": round((time.monotonic() - t0) * 1000, 2)}
    if info3.get("truncated_by_time"):
        # Time cut us off — keep depth-2 result.
        info["depth_reached"] = 2
        info["elapsed_ms"] = round((deadline - t0) * 1000, 2)
        return chosen, info
    chosen = a3
    info["depth_reached"] = 3
    info["elapsed_ms"] = round((deadline - t0) * 1000, 2)
    return chosen, info


# ========================================================================
# Rollout search (kept for completeness)
# ========================================================================

def rollout_pick(state: GameState, who: int, search_cls,
                 n_rollouts: int = 4, seed_offset: int = 1000) -> tuple[Action, dict]:
    """For each of OUR top-K candidates, run a short self-play rollout
    and pick the action with the highest winrate."""
    actions = legal_actions(state, who)
    base = score(state, who)
    scored = []
    for a in actions:
        s = score(_apply_clone(state, a), who) - base
        key = _action_key(a)
        scored.append((s, key, a))
    scored.sort(key=lambda x: (-x[0], x[1]))
    scored = scored[:max(4, n_rollouts)]
    best = None
    info = {"candidates": len(actions), "depth": "rollout", "results": []}
    for delta, _key, a in scored:
        wins = 0
        for r in range(n_rollouts):
            try:
                outcome = _quick_rollout(_apply_clone(state, a), who, search_cls,
                                          seed_offset + r * 31)
                if outcome == "win":
                    wins += 1
            except Exception:
                pass
        info["results"].append({"action": a.to_json(), "wins": wins})
        if best is None or wins > best[0]:
            best = (wins, a)
    return best[1] if best is not None else scored[0][2], info


def _quick_rollout(state: GameState, who: int, search_cls,
                   seed: int) -> str:
    """Pessimistic opponent rollout — alternate greedy 1-ply actions
    for both players until the game ends. Cheap; returns "win" if `who`
    eventually wins the match."""
    cur = state
    safety = 30
    while not cur.is_terminal() and safety > 0:
        a, _ = greedy_1ply(cur, cur.active_player)
        cur = _apply_clone(cur, a)
        cur.active_player = 1 - cur.active_player
        cur.turn += 1
        safety -= 1
    if cur.winner == who:
        return "win"
    if cur.winner == 1 - who:
        return "lose"
    return "draw"


# ========================================================================
# Convenience entry-point
# ========================================================================

def pick(state: GameState, who: int,
         depth: int = 3, time_budget_ms: float = 250,
         cfg: EvaluatorConfig | None = None,
         value_fn: ValueFn = None) -> tuple[Action, dict]:
    """One-call wrapper used by agents.

    * ``depth == 1`` -> single greedy step.
    * ``depth == 2`` -> 2-ply alpha-beta.
    * ``depth >= 3`` -> iterative deepening (time-budgeted).

    Pluggable value functions are passed via ``value_fn=``. ``cfg`` is
    retained for backward compatibility; if both are given, ``value_fn``
    wins.
    """
    if depth <= 1:
        return greedy_1ply(state, who, cfg=cfg, value_fn=value_fn)
    if depth == 2:
        # Allow ~all of the budget for the 2-ply pass
        return alpha_beta_2ply(state, who, beam=8,
                                cfg=cfg, value_fn=value_fn)
    return iterative_deepening_search(state, who,
                                       time_budget_ms=time_budget_ms,
                                       cfg=cfg, value_fn=value_fn)
