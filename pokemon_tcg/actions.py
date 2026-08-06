"""Legal-action enumeration.

A turn in Pokemon TCG is structured as: draw -> [optional play actions] ->
attack -> end. The legal actions available depend on the player's current
board and the cards in hand.

We enumerate a rich set of atomic actions (rather than free-form dicts) so
the agent's decision engine can score them uniformly and the simulator
can apply them deterministically:

  * PLAY_POKEMON        put a Basic from hand onto Bench
  * EVOLVE              evolve a Pokemon in play using a card from hand
  * RETREAT             swap active for a benched Pokemon, paying retreat-energy cost
  * ATTACH_ENERGY       attach an energy from hand to a Pokemon in play
  * ATTACK              use a Pokemon move
  * PLAY_TRAINER        play a Trainer (Item / Supporter / Stadium / Tool)
  * PASS                end the turn (must fire one Attack or PASS)
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

from .cards import Card, PokemonCard, EnergyCard, TrainerCard, COLORLESS, POKEMON_TYPES
from .game_state import GameState, PlayerState, PokemonInstance, MAX_BENCH


@dataclass(frozen=True)
class Action:
    """An atomic, legal action.

    All actions are flat dicts serializable to JSON for logging. Agents can
    add new action kinds by extending the simulator's `apply` switch.
    """
    kind: str                       # see module docstring
    source_idx: Optional[int] = None    # index into hand
    target_idx: Optional[int] = None    # target slot (Active = -1, bench = 0..4)
    extra: Optional[str] = None         # attack name, energy type, etc.

    def to_json(self) -> dict:
        return {"kind": self.kind, "source_idx": self.source_idx,
                "target_idx": self.target_idx, "extra": self.extra}

    @staticmethod
    def from_json(d: dict) -> "Action":
        return Action(kind=d["kind"], source_idx=d.get("source_idx"),
                      target_idx=d.get("target_idx"), extra=d.get("extra"))


# ========================================================================
# Enumeration
# ========================================================================

def legal_actions(state: GameState, who: int) -> list[Action]:
    """Return every legal action for `who` on the current turn."""
    out: list[Action] = []
    me: PlayerState = state.me(who)
    opp: PlayerState = state.opp(who)

    # 0. Already a winner? only end turn (idempotent)
    if state.is_terminal():
        return [Action("PASS")]

    # 1. Play a Basic Pokemon from hand to bench
    bench_open = len(me.bench) < MAX_BENCH
    if bench_open:
        for i, c in enumerate(me.hand):
            if c.pokemon and c.pokemon.stage == "Basic":
                out.append(Action("PLAY_POKEMON", source_idx=i,
                                  target_idx=len(me.bench)))

    # 2. Evolve any active/bench Pokemon
    evolve_targets = _evolve_targets(me)
    for i, c in enumerate(me.hand):
        if not c.pokemon:
            continue
        for slot_idx, current in evolve_targets:
            if c.pokemon.evolves_from == current.base.name:
                out.append(Action("EVOLVE", source_idx=i,
                                  target_idx=slot_idx))

    # 3. Attach an energy (once per turn)
    if not me.energy_attached_this_turn:
        for i, c in enumerate(me.hand):
            if c.energy:
                # Eligible targets: active + bench with a Pokemon
                targets: list[int] = []
                if me.active is not None:
                    targets.append(-1)
                for j, p in enumerate(me.bench):
                    if p.hp > 0:
                        targets.append(j)
                for t in targets:
                    out.append(Action("ATTACH_ENERGY", source_idx=i, target_idx=t))

    # 3b. Retreat: swap active with a benched Pokemon, paying retreat cost.
    # Cost = active.base.retreat (any energy type OK, one per cost step).
    if me.active is not None and me.active.hp > 0 \
            and len(me.active.attached_energy) >= me.active.base.retreat:
        for j, p in enumerate(me.bench):
            if p.hp > 0:
                out.append(Action("RETREAT", target_idx=j,
                                  extra=str(me.active.base.retreat)))

    # 4. Play a Trainer card
    for i, c in enumerate(me.hand):
        if not c.trainer:
            continue
        cat = c.trainer.category
        if cat == "Supporter" and me.supporter_played_this_turn:
            continue
        if cat in ("Item", "Supporter", "Stadium", "Pokemon Tool"):
            out.append(Action("PLAY_TRAINER", source_idx=i,
                              extra=cat))

    # 5. Attack with active (or PASS if none)
    if me.active is not None and me.active.hp > 0:
        for move in me.active.usable_moves():
            out.append(Action("ATTACK", extra=move.name))
    out.append(Action("PASS"))

    return out


def _evolve_targets(me: PlayerState) -> list[tuple[int, PokemonInstance]]:
    out: list[tuple[int, PokemonInstance]] = []
    if me.active is not None:
        out.append((-1, me.active))
    for j, p in enumerate(me.bench):
        if p.hp > 0:
            out.append((j, p))
    return out


def describe(actions: list[Action], me: PlayerState) -> list[dict]:
    """Render actions as {kind, hand_card, target, info} for debugging."""
    out = []
    for a in actions:
        info = a.to_json()
        if a.source_idx is not None and 0 <= a.source_idx < len(me.hand):
            c = me.hand[a.source_idx]
            info["hand_card"] = c.pokemon.name if c.pokemon else c.energy.name \
                if c.energy else c.trainer.name if c.trainer else "?"
        out.append(info)
    return out
