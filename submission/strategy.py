"""Position evaluation + the MAIN selection policy.

Everything here works directly on the engine's observation dataclasses
(see :mod:`card_db` for re-exports). The value function is the agent's
"brain"; the MAIN policy uses it (optionally with engine-search
lookahead from :mod:`lookahead`) to pick the next action.
"""
from __future__ import annotations

import re
import time
from collections import Counter

from card_db import (
    Attack, CardData, CardType, EnergyType, Observation, Option, OptionType,
    Pokemon, PlayerState, SelectContext, SelectData, State,
    best_damage_attack, best_potential_attack, card, estimate_damage,
    is_ko, prize_value, usable_attacks,
)

# Option types in a MAIN selection.
OPT_PLAY = 7
OPT_ATTACH = 8
OPT_EVOLVE = 9
OPT_ABILITY = 10
OPT_DISCARD = 11
OPT_RETREAT = 12
OPT_ATTACK = 13
OPT_END = 14

WIN = 1_000_000.0
LOSE = -1_000_000.0


# ---------------------------------------------------------------------------
# Position evaluation
# ---------------------------------------------------------------------------

def evaluate(obs: Observation, who: int) -> float:
    """Score the current state from ``who``'s perspective. Higher = better."""
    state: State = obs.current
    if state is None:
        return 0.0
    if state.result != -1:
        return WIN if state.result == who else LOSE

    me: PlayerState = state.players[who]
    opp: PlayerState = state.players[1 - who]

    s = 0.0

    # --- Prize lead (dominant signal) ---
    my_prizes = len(me.prize or [])
    opp_prizes = len(opp.prize or [])
    s += 320.0 * (opp_prizes - my_prizes)

    # --- Board ---
    my_act, my_bench = _split(me)
    op_act, op_bench = _split(opp)

    if my_act is not None:
        s += _hp_value(my_act) * 1.0
    for p in my_bench:
        s += _hp_value(p) * 0.35
    if op_act is not None:
        s -= _hp_value(op_act) * 1.0
    for p in op_bench:
        s -= _hp_value(p) * 0.35

    # --- Attack power ---
    s += _board_damage(me, my_act, my_bench) * 0.9
    s -= _board_damage(opp, op_act, op_bench) * 0.9

    # --- Future attacker setup (bench potential) ---
    s += _bench_potential(my_bench) * 0.12
    s -= _bench_potential(op_bench) * 0.12

    # --- KO threat: can I OHKO their active? can they OHKO mine? ---
    if op_act is not None:
        ohko = _any_ohko(me, op_act)
        if ohko:
            s += 240.0
        else:
            twohko = _any_2hko(me, op_act)
            s += 60.0 if twohko else 0.0
    if my_act is not None:
        if _any_ohko(opp, my_act):
            s -= 260.0
        elif _any_2hko(opp, my_act):
            s -= 70.0

    # --- Resources ---
    n_hand = me.handCount or 0
    s += 4.0 * min(n_hand, 8)
    energies_in_hand = _energy_in_hand(me)
    s += 6.0 * energies_in_hand
    # bench fill (more bodies = more options)
    s += 10.0 * len(my_bench)
    # deck-out risk
    if (me.deckCount or 0) <= 2:
        s -= 200.0 * (3 - (me.deckCount or 0))

    # --- Status ---
    for flag in ("poisoned", "burned", "asleep", "paralyzed", "confused"):
        if getattr(me, flag, False):
            s -= 40.0
        if getattr(opp, flag, False):
            s += 40.0

    # --- Tempo: first player parity ---
    if state.turn >= 2:
        s += 5.0 if state.yourIndex == state.firstPlayer else -5.0

    return s


def _split(ps: PlayerState):
    act = ps.active[0] if ps.active and ps.active[0] is not None else None
    return act, [p for p in (ps.bench or []) if p is not None]


def _hp_value(p: Pokemon) -> float:
    return float(p.hp)


def _board_damage(ps: PlayerState, act: Pokemon | None, bench: list[Pokemon]) -> float:
    total = 0.0
    if act is not None:
        a = best_damage_attack(act, card(act.id))
        total += (a.damage or 0) if a else 0.0
    for p in bench:
        if p.hp <= 0:
            continue
        a = best_damage_attack(p, card(p.id))
        if a:
            total += min((a.damage or 0), 180) * 0.5
    return min(total, 400.0)


def _bench_potential(bench: list[Pokemon]) -> float:
    """Sum of best-possible attack damage across bench (future threat)."""
    total = 0.0
    for p in bench:
        if p is None or p.hp <= 0:
            continue
        cd = card(p.id)
        best = best_potential_attack(cd) if cd else None
        if best and best.damage:
            total += min(best.damage, 260)
    return total


def _any_ohko(ps: PlayerState, target: Pokemon) -> bool:
    tdata = card(target.id)
    for p in _all_pokemon(ps):
        for a in usable_attacks(p, card(p.id)):
            if a.damage and estimate_damage(a, card(p.id), tdata) >= target.hp:
                return True
    return False


def _any_2hko(ps: PlayerState, target: Pokemon) -> bool:
    tdata = card(target.id)
    for p in _all_pokemon(ps):
        for a in usable_attacks(p, card(p.id)):
            if a.damage and 2 * estimate_damage(a, card(p.id), tdata) >= target.hp:
                return True
    return False


def _all_pokemon(ps: PlayerState) -> list[Pokemon]:
    out = []
    if ps.active and ps.active[0] is not None:
        out.append(ps.active[0])
    out.extend(p for p in (ps.bench or []) if p is not None)
    return out


def _energy_in_hand(ps: PlayerState) -> int:
    if not ps.hand:
        return 0
    return sum(1 for c in ps.hand if _is_energy_id(c.id))


def _is_energy_id(cid: int) -> bool:
    c = card(cid)
    return c is not None and c.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)


# ---------------------------------------------------------------------------
# Card-value helpers (used by both the MAIN policy and nested pickers)
# ---------------------------------------------------------------------------

def rank_cards(ids: list[int]) -> list[tuple[int, float]]:
    """Rank card IDs from most to least useful (for search/discard picks)."""
    scored = []
    for cid in ids:
        c = card(cid)
        if c is None:
            scored.append((cid, 0.0))
            continue
        if c.cardType == CardType.POKEMON:
            if c.basic:
                best = best_potential_attack(c)
                dmg = (best.damage or 0) if best else 0
                cost = len(best.energies or []) if best else 3
                v = c.hp * 0.4 + dmg * 0.7 - cost * 15.0 - (40 if c.ex else 0)
            else:
                best = best_potential_attack(c)
                dmg = (best.damage or 0) if best else 0
                v = c.hp * 0.5 + dmg * 0.8
            # Zero-to-Hero combo pieces are precious to keep
            if c.name and ("Palafin" in c.name or "Finizen" in c.name):
                v += 60.0
        elif c.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
            v = 18.0
        elif c.cardType == CardType.SUPPORTER:
            v = 14.0 + _draw_text_value(c)
        elif c.cardType == CardType.ITEM:
            v = 10.0 + _item_text_value(c)
        elif c.cardType == CardType.TOOL:
            v = 8.0
        else:
            v = 6.0
        scored.append((cid, v))
    scored.sort(key=lambda x: -x[1])
    return scored


def _draw_text_value(c: CardData) -> float:
    text = " ".join(s.text or "" for s in (c.skills or []))
    low = text.lower()
    v = 0.0
    m = re.search(r"draw (\d+)", low)
    if m:
        v += min(int(m.group(1)), 6) * 2.0
    if "draw until" in low or "draw cards until" in low:
        v += 8.0
    if "search your deck" in low:
        v += 4.0
    if "shuffle your hand" in low:
        v += 3.0
    if "go first" in low:
        v += 2.0
    return v


def _item_text_value(c: CardData) -> float:
    text = " ".join(s.text or "" for s in (c.skills or []))
    low = text.lower()
    v = 0.0
    if "search your deck" in low:
        v += 6.0
    if "discard 2 other cards" in low:
        v += 3.0
    if "draw" in low:
        v += 3.0
    if "energy" in low:
        v += 2.0
    return v


def card_play_value(cid: int, obs: Observation, who: int) -> float:
    """Value of playing card ``cid`` from hand right now."""
    c = card(cid)
    if c is None:
        return 0.0
    state: State = obs.current
    me = state.players[who]
    bench_n = len(me.bench or [])

    if c.cardType == CardType.POKEMON:
        if c.basic:
            if bench_n >= (me.benchMax or 5):
                return -100.0
            best = best_potential_attack(c)
            dmg = (best.damage or 0) if best else 0
            cost = len(best.energies or []) if best else 3
            v = c.hp * 0.35 + dmg * 0.6 - cost * 12.0 - (35 if c.ex else 0)
            if c.name and "Finizen" in c.name:
                v += 55.0
            return v
        # non-basic in hand is played via EVOLVE options, not PLAY
        return 0.0
    if c.cardType == CardType.SUPPORTER:
        # a supporter is worth its draw value; using it consumes the
        # once-per-turn supporter slot, so slightly discount
        v = 4.0 + _draw_text_value(c)
        if c.name == "Carmine" and _is_first_player_turn_1(obs, who):
            v += 10.0  # Carmine is legal turn 1 even going first
        return v
    if c.cardType == CardType.ITEM:
        return _item_text_value(c)
    if c.cardType == CardType.TOOL:
        return 6.0  # equipping a tool (e.g. Lucky Helmet)
    if c.cardType == CardType.STADIUM:
        return 8.0
    return 2.0


def _is_first_player_turn_1(obs: Observation, who: int) -> bool:
    state = obs.current
    return (state is not None and state.turn == 1 and
            state.firstPlayer == who)


# ---------------------------------------------------------------------------
# MAIN policy
# ---------------------------------------------------------------------------

def main_option_key(opt: Option) -> tuple:
    """Stable key for deduplicating MAIN options."""
    return (opt.type, opt.area, opt.index, opt.inPlayArea, opt.inPlayIndex,
            opt.attackId, opt.cardId, opt.serial)


def choose_main_action(obs: Observation, tracker=None,
                       lookahead=None, budget_ms: float = 140.0,
                       profile=None) -> int:
    """Return the option index to pick in a MAIN selection.

    Priority: engine-search lookahead over a short candidate list (when
    ``lookahead`` is provided), else pure heuristic scoring. The hidden
    worlds for lookahead are sampled **once** per decision so every
    candidate is compared under the same set of predictions.

    ``profile`` optionally carries learned opponent tendencies (see
    ``learning.profiles``) used to bias the *heuristic* play when this
    policy stands in for the opponent inside engine-search lookahead —
    so the AI plans against how humans actually play.
    """
    sel: SelectData = obs.select
    who = obs.current.yourIndex
    opts: list[Option] = sel.option

    cands = _main_candidates(obs, who, opts, profile)
    if not cands:
        # nothing useful -> END if offered, else option 0
        for i, o in enumerate(opts):
            if o.type == OPT_END:
                return i
        return 0

    if lookahead is not None:
        worlds = None
        if tracker is not None:
            n_worlds = getattr(lookahead, "worlds", 2)
            try:
                worlds = tracker.sample_worlds(obs, n_worlds)
            except Exception:
                worlds = None
        start = time.time()
        best_idx, best_score = None, None
        for idx in cands:
            if time.time() - start > budget_ms:
                break
            try:
                sc = lookahead.score_candidate(obs, idx, who, worlds=worlds)
            except Exception:
                sc = None
            if sc is None:
                continue
            if best_score is None or sc > best_score:
                best_score, best_idx = sc, idx
        if best_idx is not None:
            return best_idx

    # heuristic fallback
    base = evaluate(obs, who)
    best_idx, best_score = None, None
    for idx in cands:
        sc = _heuristic_delta(obs, who, opts[idx], base, profile)
        if best_score is None or sc > best_score:
            best_score, best_idx = sc, idx
    if best_idx is None:
        return 0
    return best_idx


def option_card_id(sel: SelectData, o: Option, state: State) -> int:
    """Resolve the card ID an option refers to, across all areas.

    Engine options only sometimes carry ``cardId``; deck-search options
    point into ``sel.deck``, hand options into ``players[].hand`` and
    prize options into ``players[].prize`` (which may be facedown).
    """
    if o.cardId and o.cardId != 0:
        return o.cardId
    if o.area == 1 and sel.deck is not None and o.index is not None \
            and 0 <= o.index < len(sel.deck):
        return sel.deck[o.index].id
    if (o.area == 2 or o.area is None) and o.playerIndex is not None \
            and 0 <= o.playerIndex <= 1 and o.index is not None:
        # MAIN PLAY options carry no area; hand index still identifies the card
        ps = state.players[o.playerIndex]
        if ps.hand and 0 <= o.index < len(ps.hand):
            return ps.hand[o.index].id
    if o.area == 6 and o.playerIndex is not None and 0 <= o.playerIndex <= 1 \
            and o.index is not None:
        ps = state.players[o.playerIndex]
        pr = ps.prize or []
        if 0 <= o.index < len(pr) and pr[o.index] is not None:
            return pr[o.index].id
    return 0


def _main_candidates(obs: Observation, who: int, opts: list[Option],
                     profile=None) -> list[int]:
    """Shortlist MAIN options for consideration (dedup, prune junk)."""
    prof = profile or {}
    seen: set = set()
    cands: list[int] = []
    state: State = obs.current
    me = state.players[who]
    opp = state.players[1 - who]
    my_hand = me.hand or []

    attack_opts = []
    play_opts = []
    attach_targets: dict = {}
    evolve_opts = []
    retreat_opts = []
    end_idx = None

    for i, o in enumerate(opts):
        if o.type == OPT_END:
            end_idx = i
            continue
        key = main_option_key(o)
        if key in seen:
            continue
        seen.add(key)
        if o.type == OPT_ATTACK:
            attack_opts.append(i)
        elif o.type == OPT_PLAY:
            play_opts.append(i)
        elif o.type == OPT_ATTACH:
            tkey = (o.inPlayArea, o.inPlayIndex)
            attach_targets.setdefault(tkey, []).append(i)
        elif o.type == OPT_EVOLVE:
            evolve_opts.append(i)
        elif o.type == OPT_RETREAT:
            retreat_opts.append(i)
        elif o.type in (OPT_ABILITY, OPT_DISCARD):
            cands.append(i)  # rarely useful; keep 1
        else:
            cands.append(i)

    # --- Attacks: all of them (usually 1-2) ---
    cands.extend(attack_opts)

    # --- Attach: best 2 targets by need (active gets a small bonus) ---
    scored_targets = []
    for tkey, idxs in attach_targets.items():
        area, ipos = tkey
        pok = _pokemon_at(me, area, ipos)
        if pok is None:
            continue
        v = _attach_need(pok) + (12.0 if area == 4 else 0.0)
        scored_targets.append((v, idxs))
    scored_targets.sort(key=lambda x: -x[0])
    for _, idxs in scored_targets[:2]:
        # prefer the energy in hand that this target wants; take first
        cands.append(idxs[0])

    # --- Plays: top 3 by card value ---
    play_scored = []
    for i in play_opts:
        o = opts[i]
        cid = _option_card_id(o, my_hand)
        v = card_play_value(cid, obs, who) if cid else -5.0
        cd = card(cid)
        if cd is not None and cd.cardType == CardType.SUPPORTER:
            # learned: humans lean on supporters early
            v += prof.get('supporter_first', 0.0) * 8.0
        play_scored.append((v, i))
    play_scored.sort(key=lambda x: -x[0])
    cands.extend(i for _, i in play_scored[:3])

    # --- Evolve: prefer evolving the Active; score by evolved potential ---
    evo_scored = []
    for i in evolve_opts:
        o = opts[i]
        cid = _option_card_id(o, my_hand)
        cd = card(cid) if cid else None
        best = best_potential_attack(cd) if cd else None
        d = (best.damage or 0) if best else 0
        v = d * 0.5 + (35.0 if o.inPlayArea == 4 else 0.0)
        if cid and card(cid) and _has_bench_swap_ability_for(cd):
            v += 25.0  # enables the Zero-to-Hero style combo
        evo_scored.append((v, i))
    evo_scored.sort(key=lambda x: -x[0])
    cands.extend(i for _, i in evo_scored[:2])

    # --- Retreat: 1 if clearly good (rare if humans never retreat) ---
    if retreat_opts and prof.get('retreat_freq', 0.5) >= 0.15:
        if _retreat_worthwhile(me, opp):
            cands.append(retreat_opts[0])

    # --- END as a baseline ---
    if end_idx is not None:
        cands.append(end_idx)

    return cands


def _option_card_id(o: Option, hand: list) -> int | None:
    if o.cardId and o.cardId != 0:
        return o.cardId
    # PLAY options carry only ``index`` (position in hand); ATTACH also
    # uses ``index`` for the energy card's hand position. There is no
    # ``area`` field on PLAY payloads.
    if o.index is not None and 0 <= o.index < len(hand):
        return hand[o.index].id
    return None


def _pokemon_at(ps: PlayerState, area, ipos: int | None) -> Pokemon | None:
    if area == 4:  # ACTIVE
        if ps.active and ps.active[0] is not None:
            return ps.active[0]
        return None
    if area == 5:  # BENCH
        bench = ps.bench or []
        if ipos is not None and 0 <= ipos < len(bench):
            return bench[ipos]
        return None
    return None


def _attach_need(pok: Pokemon) -> float:
    """How valuable is one more energy on this Pokemon right now?"""
    cd = card(pok.id)
    if cd is None:
        return 0.0
    usable = usable_attacks(pok, cd)
    best_usable = max((a.damage or 0) for a in usable) if usable else 0
    best_pot = best_potential_attack(cd)
    pot = (best_pot.damage or 0) if best_pot else 0
    if pot <= best_usable:
        return 5.0  # nothing to enable
    need = len(best_pot.energies or []) - len(pok.energies or [])
    if need <= 0:
        return 5.0
    # big payoff attacks are worth pursuing
    return 12.0 + (pot - best_usable) * 0.15 / need + (30.0 if need == 1 else 0.0)


def _retreat_worthwhile(me: PlayerState, opp: PlayerState) -> bool:
    act = me.active[0] if me.active and me.active[0] is not None else None
    if act is None:
        return False
    bench = [p for p in (me.bench or []) if p is not None]
    if not bench:
        return False
    # Combo trigger: the active has an ability that fires when it moves
    # from the Active Spot to the Bench (e.g. Palafin's "Zero to Hero")
    # — retreating IS the play.
    if _has_bench_swap_ability(act):
        return True
    op_act = opp.active[0] if opp.active and opp.active[0] is not None else None
    opp_dmg = 0
    if op_act is not None:
        a = best_damage_attack(op_act, card(op_act.id))
        opp_dmg = (a.damage or 0) if a else 0
    act_best = best_damage_attack(act, card(act.id))
    act_dmg = (act_best.damage or 0) if act_best else 0
    for p in bench:
        b = best_damage_attack(p, card(p.id))
        bd = (b.damage or 0) if b else 0
        if bd > act_dmg + 30 and (act.hp - opp_dmg <= 40 or act_dmg == 0):
            return True
        # massive upgrade: a clearly stronger attacker on the bench
        if bd > act_dmg + 90:
            return True
    return False


def _has_bench_swap_ability_for(cd: CardData | None) -> bool:
    """True if a card's ability triggers when it moves Active -> Bench."""
    if cd is None or not cd.skills:
        return False
    for s in cd.skills:
        t = (s.text or "").lower()
        if "moves from the active spot to the bench" in t:
            return True
    return False


def _has_bench_swap_ability(pok: Pokemon) -> bool:
    return _has_bench_swap_ability_for(card(pok.id))


def _heuristic_delta(obs: Observation, who: int, o: Option, base: float,
                     profile=None) -> float:
    """Cheap action-quality estimate (used when search is unavailable)."""
    prof = profile or {}
    state: State = obs.current
    me = state.players[who]
    opp = state.players[1 - who]
    my_hand = me.hand or []

    if o.type == OPT_ATTACK:
        atk = _attack_by_id(o.attackId)
        if atk is None:
            return -10.0
        attacker = me.active[0] if me.active and me.active[0] is not None else None
        target = opp.active[0] if opp.active and opp.active[0] is not None else None
        if attacker is None or target is None:
            return 0.0
        acd = card(attacker.id)
        tcd = card(target.id)
        dmg = estimate_damage(atk, acd, tcd)
        v = dmg * 0.8
        if dmg >= target.hp:
            v += 240.0 + prize_value(tcd) * 40.0
        elif dmg * 2 >= target.hp:
            v += 60.0
        # learned: aggressive humans attack early -> attack-first opponent
        v += prof.get('aggression', 0.5) * 12.0
        return v
    if o.type == OPT_ATTACH:
        return 14.0
    if o.type == OPT_PLAY:
        cid = _option_card_id(o, my_hand)
        v = card_play_value(cid, obs, who) if cid else 2.0
        cd = card(cid)
        if cd is not None and cd.cardType == CardType.SUPPORTER:
            v += prof.get('supporter_first', 0.0) * 6.0
        return v
    if o.type == OPT_EVOLVE:
        return 22.0
    if o.type == OPT_RETREAT:
        act = me.active[0] if me.active and me.active[0] is not None else None
        if act is not None and _has_bench_swap_ability(act):
            return 30.0  # Zero-to-Hero style combo retreat
        return 12.0 + prof.get('retreat_freq', 0.5) * 10.0
    if o.type == OPT_END:
        return 0.0
    return -5.0


# ---------------------------------------------------------------------------
# Nested-selection pickers (used by agent.py and lookahead.py)
# ---------------------------------------------------------------------------

_attack_cache = {}


def _attack_by_id(aid: int | None):
    if aid is None:
        return None
    if aid in _attack_cache:
        return _attack_cache[aid]
    from card_db import attack as _atk
    a = _atk(aid)
    _attack_cache[aid] = a
    return a
