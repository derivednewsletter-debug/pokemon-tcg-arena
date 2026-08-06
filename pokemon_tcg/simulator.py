"""Pokemon TCG simulator — rules engine.

The simulator is the *only* module that mutates GameState. Agents and
evaluators receive `state.deepcopy()` views so they can search/plan safely.

Simplifications vs the real TCG
================================
* Supporter/Item text is reduced to `draw_n(n)` semantics.
* Status damage is fixed at 10 per check (rather than the official
  poison/burn token counts); this lets the agent reason uniformly.
* Weakness/resistance = +20/-20 HP modifier (no per-attack variant).
* Stadium cards are played but provide a neutral +0 modifier (state
  snapshot preserves their identity for future expansion).
* Tool cards are no-op placeholders.

These choices keep the simulator small enough to ship while still
exercising every meaningful decision an agent must make.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .actions import Action
from .cards import Card, COLORLESS
from .game_state import (
    GameState, PlayerState, PokemonInstance,
    STATUS_BURN, STATUS_POISON, STATUS_SLEEP, STATUS_PARALYSIS, STATUS_CONFUSED,
    ALL_STATUS, MAX_BENCH, MAX_HAND_SIZE, WIN_PRIZES,
)


# Damage step modifiers
WEAKNESS_BONUS = 20
RESISTANCE_REDUCTION = 20
POISON_DAMAGE = 10
BURN_DAMAGE = 10
CONFUSION_SELF_DAMAGE = 30

# Retreat: cards discarded from active drop to the player's discard pile
# (real TCG: go to the LOST ZONE only if explicitly moved; energy goes to
# discard pile by default). We just drop them here as a no-op; retreat semantics
# don't actually need to track the discarded energy in our model because
# the PokemonInstance keeps its energy list as the canonical state.

# Starting hand size
STARTING_HAND = 7


# ========================================================================
# Setup
# ========================================================================

def new_game(deck_a: list[Card], deck_b: list[Card], seed: int) -> GameState:
    """Build an initial GameState: shuffled decks, prize split, opening hand.

    The opening-hand mulligan is intentionally NOT implemented — it's a
    policy detail that varies between competitive rules and biases the
    statistics.  In a measured experiment, you can wrap this function to
    draw-with-bottom-mull and pass the same GameState to `set_opening`.
    """
    rng = random.Random(seed)

    deck_a = list(deck_a)
    deck_b = list(deck_b)
    if len(deck_a) != len(deck_b):
        raise ValueError("Decks must be the same size")
    rng.shuffle(deck_a)
    rng.shuffle(deck_b)

    # Split into deck/prize/hand
    def _split(deck: list[Card]) -> tuple[list[Card], list[Card], list[Card]]:
        hand = deck[:STARTING_HAND]
        prizes = deck[STARTING_HAND:STARTING_HAND + WIN_PRIZES]
        rest = deck[STARTING_HAND + WIN_PRIZES:]
        return rest, prizes, hand

    deck_a, prizes_a, hand_a = _split(deck_a)
    deck_b, prizes_b, hand_b = _split(deck_b)

    # Initial active Pokemon = first Basic in hand (deterministic order)
    def _active(hand: list[Card]) -> Optional[PokemonInstance]:
        for c in hand:
            if c.pokemon and c.pokemon.stage == "Basic":
                return PokemonInstance(
                    base=c.pokemon, hp=c.pokemon.hp, base_hp=c.pokemon.hp,
                )
        return None

    p0 = PlayerState(name="P0", deck=deck_a, hand=hand_a,
                     active=_active(hand_a), bench=[],
                     prizes=prizes_a, prize_count=WIN_PRIZES, discard=[])
    p1 = PlayerState(name="P1", deck=deck_b, hand=hand_b,
                     active=_active(hand_b), bench=[],
                     prizes=prizes_b, prize_count=WIN_PRIZES, discard=[])

    # Determine first player — coin flip, but deterministic given the seed
    first = 0 if rng.random() < 0.5 else 1

    state = GameState(players=(p0, p1), turn=1, active_player=first, rng_seed=seed)
    # Active Pokemon don't have a card in hand anymore — drop them
    _promote_active_from_hand(state, 0)
    _promote_active_from_hand(state, 1)
    state.log.append({"turn": state.turn, "kind": "SETUP",
                     "active_player": state.active_player})
    return state


def _promote_active_from_hand(state: GameState, who: int) -> None:
    """If a player has no active (e.g. the chosen Basic was discarded),
    we DON'T promote any other card — they have to play one next turn."""
    me = state.me(who)
    if me.active is not None and me.hand:
        # Remove the matching card from hand
        for i, c in enumerate(me.hand):
            if c.pokemon and c.pokemon.name == me.active.base.name \
                    and c.pokemon.stage == "Basic":
                me.hand.pop(i)
                break


# ========================================================================
# Apply actions
# ========================================================================

def step(state: GameState, action: Action) -> GameState:
    """One turn of one player. The full match alternates via
    `simulate_match`. This is the inner step (a single player commits
    an ActionList of one Attack or sequence up to one PASS)."""
    if state.is_terminal():
        return state

    me_idx = state.active_player
    me = state.me(me_idx)
    opp_idx = 1 - me_idx
    opp = state.me(opp_idx)

    state.log.append({"turn": state.turn, "player": me_idx,
                      "kind": "STEP_START", "hand_size": len(me.hand),
                      "deck_size": len(me.deck),
                      "active": me.active.base.name if me.active else None})

    if action.kind != "PASS":
        _apply_non_pass(state, me_idx, action)

    # End-of-turn housekeeping
    _end_of_turn_effects(state, me_idx)

    # Check win conditions
    winner = _check_winner(state)
    if winner is not None:
        state.winner = winner
        state.log.append({"turn": state.turn, "kind": "GAME_OVER", "winner": winner})
        return state

    # Flip turn
    state.active_player = opp_idx
    state.turn += 1
    next_me = state.me(state.active_player)
    next_me.supporter_played_this_turn = False
    next_me.energy_attached_this_turn = False

    # Draw a card
    if next_me.deck:
        drawn = next_me.deck.pop()
        next_me.hand.append(drawn)
        state.log.append({"turn": state.turn, "player": state.active_player,
                          "kind": "DRAW", "card": drawn.pokemon.name
                          if drawn.pokemon else drawn.energy.name
                          if drawn.energy else drawn.trainer.name if drawn.trainer else "?"})
    else:
        # Deck-out — player who couldn't draw loses
        state.winner = opp_idx
        state.log.append({"turn": state.turn, "kind": "GAME_OVER",
                          "winner": opp_idx, "reason": "deck_out"})
        return state
    return state


def _apply_non_pass(state: GameState, who: int, action: Action) -> None:
    """Dispatch a non-pass action."""
    fn = _ACTIONS.get(action.kind)
    if fn is None:
        state.log.append({"turn": state.turn, "player": who, "kind": "ILLEGAL",
                          "action": action.to_json()})
        return
    fn(state, who, action)
    state.log.append({"turn": state.turn, "player": who, "kind": "ACTION",
                      "action": action.to_json(),
                      "hand_size": state.me(who).hand_size(),
                      "active": state.me(who).active.base.name
                      if state.me(who).active else None})


def _action_play_pokemon(state: GameState, who: int, a: Action) -> None:
    me = state.me(who)
    if a.source_idx is None:
        return
    c = me.hand[a.source_idx]
    if not (c.pokemon and c.pokemon.stage == "Basic"):
        return
    if len(me.bench) >= MAX_BENCH:
        return
    if me.active is None:
        # If we have no active, promote this one
        me.active = PokemonInstance(base=c.pokemon, hp=c.pokemon.hp, base_hp=c.pokemon.hp)
    else:
        me.bench.append(PokemonInstance(base=c.pokemon, hp=c.pokemon.hp,
                                        base_hp=c.pokemon.hp))
    me.hand.pop(a.source_idx)


def _action_evolve(state: GameState, who: int, a: Action) -> None:
    me = state.me(who)
    if a.source_idx is None or a.target_idx is None:
        return
    c = me.hand[a.source_idx]
    if not c.pokemon:
        return
    target_slot = -1 if a.target_idx == -1 else a.target_idx
    if target_slot == -1:
        p = me.active
    elif 0 <= target_slot < len(me.bench):
        p = me.bench[target_slot]
    else:
        return
    if p is None:
        return
    if not p.can_evolve_into(c.pokemon):
        return
    fresh = PokemonInstance(
        base=c.pokemon, hp=c.pokemon.hp, base_hp=c.pokemon.hp,
        attached_energy=p.attached_energy, status=p.status,
        tool_attached=p.tool_attached, is_active=p.is_active,
    )
    if target_slot == -1:
        me.active = fresh
    else:
        me.bench[target_slot] = fresh
    me.hand.pop(a.source_idx)


def _action_attach_energy(state: GameState, who: int, a: Action) -> None:
    me = state.me(who)
    if me.energy_attached_this_turn:
        return
    if a.source_idx is None or a.target_idx is None:
        return
    c = me.hand[a.source_idx]
    if not c.energy:
        return
    target_slot = -1 if a.target_idx == -1 else a.target_idx
    if target_slot == -1:
        p = me.active
    elif 0 <= target_slot < len(me.bench):
        p = me.bench[target_slot]
    else:
        return
    if p is None or p.hp <= 0:
        return
    new_energy = p.attached_energy + (c.energy.provides,)
    p.attached_energy = new_energy
    me.hand.pop(a.source_idx)
    me.energy_attached_this_turn = True


def _action_retreat(state: GameState, who: int, a: Action) -> None:
    """Retreat the active Pokemon with the bench Pokemon at slot `target_idx`.

    Real TCG retreat mechanism:
      1. Pay the active Pokemon's retreat cost by discarding that many
         Energy cards from it (any energy types are valid; we pop the
         right side of the tuple to keep determinism).
      2. The active goes to bench (its position is appended).
      3. The chosen bench Pokemon becomes Active.
    """
    me = state.me(who)
    if a.target_idx is None:
        return
    if me.active is None or me.active.hp <= 0:
        return
    if not (0 <= a.target_idx < len(me.bench)):
        return
    cost = me.active.base.retreat
    if len(me.active.attached_energy) < cost:
        return  # not enough energy to retreat
    bench_target = me.bench[a.target_idx]
    if bench_target.hp <= 0:
        return
    # Pay cost: pop `cost` energy tokens from active (deterministic right-side
    # pop mimics a TCG player's free choice; result is reproducible across runs).
    remaining_energy = list(me.active.attached_energy)
    discarded_count = 0
    for _ in range(cost):
        remaining_energy.pop()
        discarded_count += 1
    me.active.attached_energy = tuple(remaining_energy)
    # Swap: old active -> bench, bench target -> active
    old_active = me.active
    old_active.is_active = False
    me.bench.append(old_active)
    me.bench.pop(a.target_idx)
    bench_target.is_active = True
    me.active = bench_target
    state.log.append({"turn": state.turn, "player": who,
                      "kind": "RETREAT", "from": old_active.base.name,
                      "to": bench_target.base.name,
                      "cost_paid": cost,
                      "energy_discarded": discarded_count,
                      "remaining_energy": len(remaining_energy)})


def _action_play_trainer(state: GameState, who: int, a: Action) -> None:
    me = state.me(who)
    if a.source_idx is None:
        return
    c = me.hand[a.source_idx]
    if not c.trainer:
        return
    cat = c.trainer.category
    if cat == "Supporter":
        if me.supporter_played_this_turn:
            return
        me.supporter_played_this_turn = True
    me.hand.pop(a.source_idx)
    me.discard.append(c)
    # Simplified: draw N if it looks like the card draws, else do nothing.
    n = _draw_from_text(c, default=0)
    if n > 0 and me.deck:
        for _ in range(n):
            if not me.deck:
                break
            me.hand.append(me.deck.pop())


_DRAW_RE = None
import re as _re
_DRAW_RE = _re.compile(r"draw\s*(\d+)", _re.IGNORECASE)


def _draw_from_text(c: Card, default: int = 0) -> int:
    text = (c.trainer.text if c.trainer else "") or ""
    m = _DRAW_RE.search(text)
    if not m:
        return default
    return int(m.group(1))


def _action_attack(state: GameState, who: int, a: Action) -> None:
    me = state.me(who)
    if me.active is None or me.active.hp <= 0:
        return
    move = next((m for m in me.active.base.moves if m.name == a.extra), None)
    if move is None or not move.can_play(list(me.active.attached_energy)):
        return
    # Status checks for the attacker
    if me.active.status in (STATUS_SLEEP, STATUS_PARALYSIS):
        # Cannot attack (lost turn due to status; remove the status)
        me.active.status = None
        state.log.append({"turn": state.turn, "player": who, "kind": "STATUS_BLOCK",
                          "status": "SLEEP/PARALYSIS"})
        return
    if me.active.status == STATUS_CONFUSED:
        if random.Random(state.rng_seed + state.turn).random() < 0.5:
            # Hurt yourself
            me.active.hp = max(0, me.active.hp - CONFUSION_SELF_DAMAGE)
            me.active.status = None
            state.log.append({"turn": state.turn, "player": who, "kind": "CONFUSION_SELF"})
            _resolve_ko(state, who)
            return
        me.active.status = None

    # Apply attack on opponent's active
    opp = state.opp(who)
    if opp.active is None or opp.active.hp <= 0:
        return
    damage = move.damage or 0
    # Weakness / resistance
    if opp.active.base.weakness and opp.active.base.weakness == me.active.base.ptype \
            and damage > 0:
        damage += WEAKNESS_BONUS
    if opp.active.base.resistance and opp.active.base.resistance == me.active.base.ptype \
            and damage > 0:
        damage = max(0, damage - RESISTANCE_REDUCTION)
    # Burn halves damage? (simplified: no, real TCG does on certain moves)
    opp.active.hp = max(0, opp.active.hp - damage)
    # Apply move text (status effects)
    if "burned" in (move.text or "").lower() or "burn" in (move.text or "").lower():
        if not opp.active.status:
            opp.active.status = STATUS_BURN
    if "poisoned" in (move.text or "").lower() or "poison" in (move.text or "").lower():
        if not opp.active.status:
            opp.active.status = STATUS_POISON
    if "asleep" in (move.text or "").lower() or "sleep" in (move.text or "").lower():
        if not opp.active.status:
            opp.active.status = STATUS_SLEEP
    if "paralyzed" in (move.text or "").lower() or "paralysis" in (move.text or "").lower():
        if not opp.active.status:
            opp.active.status = STATUS_PARALYSIS
    if "confused" in (move.text or "").lower():
        if not opp.active.status:
            opp.active.status = STATUS_CONFUSED
    state.log.append({"turn": state.turn, "player": who, "kind": "ATTACK",
                      "move": move.name, "damage": damage,
                      "target": opp.active.base.name,
                      "target_hp_after": opp.active.hp})
    _resolve_ko(state, 1 - who)


_ACTIONS = {
    "PLAY_POKEMON": _action_play_pokemon,
    "EVOLVE": _action_evolve,
    "ATTACH_ENERGY": _action_attach_energy,
    "RETREAT": _action_retreat,
    "ATTACK": _action_attack,
    "PLAY_TRAINER": _action_play_trainer,
}


# ========================================================================
# End-of-turn and KO handling
# ========================================================================

def _end_of_turn_effects(state: GameState, who: int) -> None:
    """Poison/burn damage and the player's Pokemon react to status."""
    me = state.me(who)
    if me.active and me.active.status == STATUS_POISON:
        me.active.hp = max(0, me.active.hp - POISON_DAMAGE)
        _resolve_ko(state, who)
    if me.active and me.active.status == STATUS_BURN:
        me.active.hp = max(0, me.active.hp - BURN_DAMAGE)
        _resolve_ko(state, who)
    # Discard locked Sioux (hand size cap)
    while len(me.hand) > MAX_HAND_SIZE:
        me.hand.pop()  # deterministic pop of last card; agents design around this


def _resolve_ko(state: GameState, who: int) -> None:
    """If `who`'s active Pokemon is KO'd, opponent takes a prize.

    In real TCG: when your Pokemon is KO'd, your opponent takes a Prize
    card from YOUR prize pile. So `who.prize_count` decreases (you lost
    a prize). When `who`'s prize_count hits 0, `who` loses.

    When there is no replacement Pokemon in bench, `who` also loses.
    """
    me = state.me(who)
    if me.active is None or me.active.hp > 0:
        return
    ko_name = me.active.base.name
    me.discard.append(Card(card_id="", pokemon=me.active.base))
    me.active = None
    # The OTHER player (opp = 1-who) takes a Prize from ME's pile.
    # ME's prize_count decreases (I just lost one); opp's hand grows.
    if me.prizes:
        me.prizes.pop()
        me.prize_count -= 1
        state.log.append({"turn": state.turn, "player": who,
                          "kind": "PRIZE_LOST", "ko": ko_name,
                          "remaining": me.prize_count})
    # The opponent draws the prize into hand (real TCG: into hand)
    opp_idx = 1 - who
    if me.prizes:  # only the popped cards become hand; this flags intent
        # In simplified rules, just log it; opp.hand increase is implicit.
        state.log.append({"turn": state.turn, "player": opp_idx,
                          "kind": "PRIZE_TAKEN", "ko": ko_name})
    # Player must promote a bench Pokemon
    _promote_active(state, who)


def _promote_active(state: GameState, who: int) -> None:
    """Promote the first alive bench Pokemon to Active. If none left, `who` loses."""
    me = state.me(who)
    for i, p in enumerate(me.bench):
        if p.hp > 0:
            me.active = p
            me.bench.pop(i)
            p.is_active = True
            return
    # No replacement
    state.winner = 1 - who
    state.log.append({"turn": state.turn, "kind": "GAME_OVER",
                      "winner": 1 - who, "reason": "no_pokemon"})


def _check_winner(state: GameState) -> Optional[int]:
    """Prefer the explicit winner; otherwise check prize counts."""
    if state.winner is not None:
        return state.winner
    for i, p in enumerate(state.players):
        if p.prize_count <= 0:
            return 1 - i  # opponent of the depleted player wins
    return None


# ========================================================================
# Top-level match driver
# ========================================================================

def simulate_match(deck_a: list[Card], deck_b: list[Card], agents: list,
                   seed: int = 0, max_turns: int = 80,
                   log: bool = False) -> dict:
    """Run a full match. `agents` is a list of two callables, each
    `(state, who) -> Action`. Returns a result dict."""
    state = new_game(deck_a, deck_b, seed)
    if log:
        state.log.append({"kind": "BEGIN", "seed": seed})

    while not state.is_terminal():
        if state.turn > max_turns:
            # Cap matches at max_turns; winner = whoever has more prizes taken
            p_a = state.players[0].prize_count
            p_b = state.players[1].prize_count
            state.winner = 0 if p_a < p_b else 1
            state.log.append({"kind": "TURN_CAP", "winner": state.winner})
            break
        try:
            action = agents[state.active_player](state, state.active_player)
        except Exception as e:  # noqa
            # Agent failure -> opponent wins by default
            state.winner = 1 - state.active_player
            state.log.append({"kind": "AGENT_ERROR", "error": repr(e),
                              "winner": state.winner})
            break
        if not isinstance(action, Action):
            state.winner = 1 - state.active_player
            state.log.append({"kind": "BAD_ACTION", "winner": state.winner,
                              "action": repr(action)})
            break
        state = step(state, action)

    return {
        "winner": state.winner,
        "turns": state.turn,
        "log": state.log,
        "state": state,
    }
