"""Game state models for the Pokemon TCG simulator.

A `GameState` owns the global match metadata (turn, seed, winner) plus
two immutable-side `PlayerState`s. Mutations only ever go through the
simulator — agents and evaluators treat these objects as read-only.

This file also defines the `Observation` shape that agents see (a strict
subset relevant to the acting player) so the Game/Agent boundary is sharp.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Iterable

from .cards import Card, Move, PokemonCard, EnergyCard, TrainerCard, COLORLESS


# Status conditions (battle state)
STATUS_BURN = "BURN"
STATUS_POISON = "POISON"
STATUS_SLEEP = "SLEEP"
STATUS_PARALYSIS = "PARALYSIS"
STATUS_CONFUSED = "CONFUSED"
ALL_STATUS = (STATUS_BURN, STATUS_POISON, STATUS_SLEEP, STATUS_PARALYSIS, STATUS_CONFUSED)

# Win/loss constants
WIN_PRIZES = 6
MAX_BENCH = 5
MAX_HAND_SIZE = 10  # standard end-of-turn discard cap


# ========================================================================
# Board representation
# ========================================================================

@dataclass
class PokemonInstance:
    """A Pokemon as it sits on the field: the card template plus gameplay
    state (HP, energy, status, etc.)."""
    base: PokemonCard
    hp: int                # remaining damage counters of HP
    base_hp: int           # starting HP (snapshot of base.hp)
    attached_energy: tuple[str, ...] = ()  # energy tokens, each = energy type
    status: str | None = None
    tool_attached: bool = False
    is_active: bool = False  # True if on Active spot

    def can_evolve_into(self, target: PokemonCard) -> bool:
        return target.evolves_from == self.base.name

    # Convenience views --------------------------------------------------
    def usable_moves(self) -> list[Move]:
        return [m for m in self.base.moves if m.can_play(list(self.attached_energy))]

    def best_damage(self) -> int:
        if not self.base.moves:
            return 0
        return max((m.damage or 0) for m in self.base.moves)

    def best_usable_damage(self) -> int:
        u = self.usable_moves()
        return max((m.damage or 0) for m in u) if u else 0

    def deepcopy(self) -> "PokemonInstance":
        return PokemonInstance(
            base=self.base,
            hp=self.hp,
            base_hp=self.base_hp,
            attached_energy=tuple(self.attached_energy),
            status=self.status,
            tool_attached=self.tool_attached,
            is_active=self.is_active,
        )

    def to_dict(self) -> dict:
        return {
            "name": self.base.name,
            "stage": self.base.stage,
            "type": self.base.ptype,
            "hp": self.hp,
            "max_hp": self.base_hp,
            "energy": list(self.attached_energy),
            "status": self.status,
            "active": self.is_active,
            "moves": [{"name": m.name, "cost": list(m.cost), "damage": m.damage}
                      for m in self.base.moves],
        }


# ========================================================================
# Player state
# ========================================================================

@dataclass
class HandEntry:
    """Streaming hand representation — we don't track every card identity
    in the hand by default (that's expensive to render) but we do keep the
    set of {Pokemon, Energy, Trainer}-kind counts for legal-action
    enumeration."""
    pokemon: tuple[PokemonCard, ...] = ()
    energy: tuple[EnergyCard, ...] = ()
    trainer: tuple[TrainerCard, ...] = ()


@dataclass
class PlayerState:
    name: str
    deck: list[Card]
    hand: list[Card]
    active: PokemonInstance | None
    bench: list[PokemonInstance]
    prizes: list[Card]            # last 6 (revealed once opponent takes one)
    prize_count: int
    discard: list[Card]
    supporter_played_this_turn: bool = False
    energy_attached_this_turn: bool = False
    # Cards revealed publicly (e.g. by Supporter search): other player can see
    public_reveals: list[Card] = field(default_factory=list)

    def alive_pokemon(self) -> list[PokemonInstance]:
        out = []
        if self.active is not None:
            out.append(self.active)
        out.extend(p for p in self.bench if p.hp > 0)
        return out

    def deck_size(self) -> int:
        return len(self.deck)

    def hand_size(self) -> int:
        return len(self.hand)

    def total_pokemon_on_field(self) -> int:
        return (1 if self.active else 0) + sum(1 for p in self.bench if p.hp > 0)

    def deepcopy(self) -> "PlayerState":
        return PlayerState(
            name=self.name,
            deck=list(self.deck),
            hand=list(self.hand),
            active=self.active.deepcopy() if self.active is not None else None,
            bench=[p.deepcopy() for p in self.bench],
            prizes=list(self.prizes),
            prize_count=self.prize_count,
            discard=list(self.discard),
            supporter_played_this_turn=self.supporter_played_this_turn,
            energy_attached_this_turn=self.energy_attached_this_turn,
            public_reveals=list(self.public_reveals),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "active": self.active.to_dict() if self.active else None,
            "bench": [p.to_dict() for p in self.bench],
            "bench_count": len(self.bench),
            "hand_size": len(self.hand),
            "deck_size": len(self.deck),
            "prize_count": self.prize_count,
            "discard_size": len(self.discard),
        }


# ========================================================================
# Game state
# ========================================================================

@dataclass
class GameState:
    """Top-level match state. Players alternate making actions every turn."""
    players: tuple[PlayerState, PlayerState]
    turn: int
    active_player: int          # index into players (0 or 1)
    rng_seed: int
    log: list[dict] = field(default_factory=list)
    winner: int | None = None    # 0, 1, or None until terminal

    def me(self, who: int) -> PlayerState:
        return self.players[who]

    def opp(self, who: int) -> PlayerState:
        return self.players[1 - who]

    def is_terminal(self) -> bool:
        return self.winner is not None

    def deepcopy(self) -> "GameState":
        return GameState(
            players=(self.players[0].deepcopy(), self.players[1].deepcopy()),
            turn=self.turn, active_player=self.active_player, rng_seed=self.rng_seed,
            log=list(self.log), winner=self.winner,
        )


# ========================================================================
# Observation (what the agent sees)
# ========================================================================

def make_observation(state: GameState, who: int) -> dict:
    """A trimmed dict view of state for `who`'s agent. We expose the
    acting player's hand implicitly via a summary; identity of opponent's
    hand is hidden, but board state is fully visible."""
    me = state.me(who)
    opp = state.opp(who)
    obs = {
        "turn": state.turn,
        "active_player": state.active_player,
        "rng_seed": state.rng_seed,
        "me": {
            "active": me.active.to_dict() if me.active else None,
            "bench": [p.to_dict() for p in me.bench],
            "hand": [_hand_summary(c) for c in me.hand],
            "hand_size": len(me.hand),
            "deck_size": len(me.deck),
            "prize_count": me.prize_count,
            "discard_size": len(me.discard),
        },
        "opp": {
            "active": opp.active.to_dict() if opp.active else None,
            "bench": [p.to_dict() for p in opp.bench],
            "bench_count": len(opp.bench),
            "hand_size": len(opp.hand),
            "deck_size": len(opp.deck),
            "prize_count": opp.prize_count,
            "discard_size": len(opp.discard),
        },
    }
    return obs


def _hand_summary(c: Card) -> dict:
    """Public card identity tag for the agent's own hand."""
    if c.pokemon:
        return {"kind": "pokemon", "name": c.pokemon.name, "hp": c.pokemon.hp,
                "stage": c.pokemon.stage, "type": c.pokemon.ptype,
                "attacks": [{"name": m.name, "damage": m.damage,
                             "cost": sum(1 for _ in m.cost)} for m in c.pokemon.moves]}
    if c.energy:
        return {"kind": "energy", "name": c.energy.name, "provides": c.energy.provides,
                "special": c.energy.is_special}
    if c.trainer:
        return {"kind": "trainer", "name": c.trainer.name, "category": c.trainer.category}
    return {"kind": "unknown"}


def hand_kinds(hand: list[Card]) -> dict[str, list[int]]:
    """Counts of each card kind in hand (used for action enumeration)."""
    counts = {"pokemon_basic": [], "pokemon_stage1": [], "pokemon_stage2": [],
              "energy_basic": [], "energy_special": [],
              "trainer_item": [], "trainer_supporter": [], "trainer_stadium": [],
              "trainer_tool": []}
    for c in hand:
        if c.pokemon:
            tag = "pokemon_basic" if c.pokemon.stage == "Basic" else "pokemon_stage1" \
                if c.pokemon.stage == "Stage 1" else "pokemon_stage2"
            counts[tag].append(0)
        elif c.energy:
            counts["energy_special" if c.energy.is_special else "energy_basic"].append(0)
        elif c.trainer:
            cat = c.trainer.category
            counts[{
                "Item": "trainer_item", "Supporter": "trainer_supporter",
                "Stadium": "trainer_stadium", "Pokemon Tool": "trainer_tool",
            }.get(cat, "trainer_item")].append(0)
    return counts
