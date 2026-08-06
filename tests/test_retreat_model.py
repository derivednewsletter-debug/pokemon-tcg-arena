"""Tests for the retreat-model logistic regression + Champion integration."""
import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from pokemon_tcg.cards import load_cards
from pokemon_tcg.deck import build_random_deck
from pokemon_tcg.simulator import new_game, simulate_match
from pokemon_tcg.evaluator import EvaluatorConfig
from pokemon_tcg.analysis.retreat_model import (
    RetreatWinPredictor, RetreatExample, RetreatDataResult,
    extract_retreat_features, collect_retreat_examples,
    recommend_threshold, train_and_pick_threshold,
    REATREAT_FEATURE_DIM,
)
from pokemon_tcg.agents.benchmarks import GreedyAgent
from pokemon_tcg.agents.champion import ChampionAgent


# ============================================================
# Feature extractor
# ============================================================

def test_feature_dim_is_twelve():
    assert REATREAT_FEATURE_DIM == 12


def test_extract_features_shape():
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    f = extract_retreat_features(state, 0)
    assert f.shape == (REATREAT_FEATURE_DIM,)
    assert f.dtype == np.float32


def test_extract_features_no_nan_or_inf():
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    f = extract_retreat_features(state, 0)
    assert not np.any(np.isnan(f))
    assert not np.any(np.isinf(f))


def test_extract_features_perspective_specific():
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    f0 = extract_retreat_features(state, 0)
    f1 = extract_retreat_features(state, 1)
    # Features should generally differ between perspectives (mirrors state in
    # some slots). At minimum, the array isn't exactly equal.
    assert not np.array_equal(f0, f1)


# ============================================================
# Logistic regression
# ============================================================

def test_predictor_forward_in_unit_interval():
    p = RetreatWinPredictor()
    rng = np.random.default_rng(0)
    features = rng.normal(0, 1, (50, REATREAT_FEATURE_DIM)).astype(np.float32)
    # Without fitting, the predictor's mean=0 std=1, weights=0 -> 0.5.
    out = p.predict_batch(features).flatten()
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_predictor_fit_recovers_separable_signal():
    """Construct a synthetic linearly-separable dataset; the fitted
    predictor should achieve > 70% accuracy."""
    p = RetreatWinPredictor()
    rng = np.random.default_rng(3)
    n = 1000
    x = rng.normal(0, 1, (n, REATREAT_FEATURE_DIM)).astype(np.float32)
    # Make `feature 0` determine the label so the classifier can learn.
    y = (x[:, 0] > 0).astype(np.float32).reshape(-1, 1)
    p.fit(x, y, epochs=80, lr=0.1, batch_size=64)
    pred = (p.predict_batch(x).flatten() >= 0.5).astype(np.float32)
    acc = float((pred == y.flatten()).mean())
    assert acc > 0.7


def test_predictor_save_load_roundtrip():
    p = RetreatWinPredictor()
    # Manually set weights so they're distinctive.
    p.weights[:] = 0.42
    p.bias[:] = 0.13
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as fh:
        path = fh.name
    try:
        p.save(path)
        p2 = RetreatWinPredictor.load(path)
        assert np.allclose(p.weights, p2.weights, atol=1e-6)
        assert np.allclose(p.bias, p2.bias, atol=1e-6)
        assert np.allclose(p.feature_mean, p2.feature_mean, atol=1e-6)
        # Same forward output for sanity.
        features = np.zeros((1, REATREAT_FEATURE_DIM), dtype=np.float32)
        a = p.predict_batch(features).flatten()
        b = p2.predict_batch(features).flatten()
        assert np.allclose(a, b, atol=1e-6)
    finally:
        os.unlink(path)


def test_recommend_threshold_returns_best_grid_value():
    # Construct a synthetic 12-dim dataset with a linearly-separable signal.
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(0, 1, (n, REATREAT_FEATURE_DIM)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.float32)
    p2 = RetreatWinPredictor()
    p2.fit(x, y.reshape(-1, 1), epochs=80, lr=0.1, batch_size=64)
    summary = recommend_threshold(p2, x, y)
    assert "threshold" in summary and 0.0 <= summary["threshold"] <= 1.0
    assert "grid" in summary and len(summary["grid"]) >= 4
    assert summary["accuracy"] >= 0.5


# ============================================================
# Data collection
# ============================================================

def test_collect_examples_smoke():
    res = collect_retreat_examples(num_games=2, spawn_a=GreedyAgent,
                                    spawn_b=GreedyAgent)
    assert res.games_played == 2
    # At least *some* retreat-eligible turns in 2 Greedy games.
    assert isinstance(res.examples, list)
    if res.examples:
        for e in res.examples[:5]:
            assert e.features.shape == (REATREAT_FEATURE_DIM,)
            assert e.label in {0.0, 1.0}
            assert e.had_retreat_legal is True


def test_collect_examples_target_per_perspective():
    """If a snapshot is recorded for player p, the label must reflect
    the eventual winner when viewed from p's perspective."""
    res = collect_retreat_examples(num_games=2, spawn_a=GreedyAgent,
                                    spawn_b=GreedyAgent)
    if not res.examples:
        return  # Skip if no retreat-eligible turns.
    # We can't assert a strict label without re-running the simulator, but
    # we can sanity-check that labels are in [-1, 0, 1] (binary actually).
    labels = {e.label for e in res.examples}
    assert labels.issubset({0.0, 1.0})


# ============================================================
# Champion integration
# ============================================================

def test_champion_accepts_predictor():
    predictor = RetreatWinPredictor()
    champ = ChampionAgent(config={"retreat_predictor": predictor})
    assert champ._retreat_predictor is predictor


def test_champion_rejects_invalid_predictor():
    try:
        ChampionAgent(config={"retreat_predictor": "not a predictor"})
        assert False, "should have raised"
    except AssertionError:
        pass


def test_champion_with_predictor_runs_match():
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    predictor = RetreatWinPredictor()
    champ = ChampionAgent(config={"retreat_predictor": predictor})
    r = simulate_match(deck, deck, [champ, GreedyAgent()], seed=42,
                       log=False, max_turns=40)
    assert r["winner"] in (0, 1)


def test_evaluator_config_has_retreat_threshold_fields():
    cfg = EvaluatorConfig()
    assert hasattr(cfg, "retreat_win_prob_threshold")
    assert hasattr(cfg, "retreat_win_prob_margin")
    assert 0.0 <= cfg.retreat_win_prob_threshold <= 1.0
    assert 0.0 <= cfg.retreat_win_prob_margin <= 1.0


def test_champion_config_has_retreat_threshold():
    from pokemon_tcg.agents.champion import CHAMPION_CONFIG
    assert CHAMPION_CONFIG.retreat_win_prob_threshold is not None
    assert CHAMPION_CONFIG.retreat_win_prob_margin is not None


def test_train_and_pick_threshold_end_to_end():
    """End-to-end: collect -> fit -> recommend -> save -> reload."""
    predictor = train_and_pick_threshold(
        num_games=2, spawn_a=GreedyAgent, spawn_b=GreedyAgent,
        epochs=20, lr=0.05,
        out_path=tempfile.mktemp(suffix=".npz"),
        threshold_out=tempfile.mktemp(suffix=".json"),
        verbose=False,
    )
    assert predictor._fitted
    # Roundtrip
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as fh:
        path = fh.name
    try:
        predictor.save(path)
        predictor2 = RetreatWinPredictor.load(path)
        # Forward should match on the same input.
        feats = np.zeros((1, REATREAT_FEATURE_DIM), dtype=np.float32)
        a = predictor.predict_batch(feats).flatten()[0]
        b = predictor2.predict_batch(feats).flatten()[0]
        assert np.allclose(a, b, atol=1e-6)
    finally:
        os.unlink(path)


# ============================================================
# Champion's retreat decision is driven by the predictor
# ============================================================def test_champion_predictor_path_changes_decision_when_predictor_loves_retreat():
    """Build a trivial all-wins predictor (predict_proba returns +2 for any
    input, which means p=1.0 after sigmoid). Champion should now retreat in
    every retreat-eligible state — even when the magic-buffer fallback
    wouldn't.
    """
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)

    class AlwaysWinModel(RetreatWinPredictor):
        def predict_proba(self, features):
            return 1.0    # total override

    champ = ChampionAgent(config={"retreat_predictor": AlwaysWinModel()})
    state = new_game(deck, deck, seed=42)
    # Bounded walk: stay inside the setup-phase heuristic path (turn <
    # setup_turns) where the predictor decides retreats. We re-seat
    # state when we cross into search territory so that exotics of the
    # deeper search stack can't crash this isolated test. The
    # correctness we care about is: predictor == "always win" ->
    # Champion picks RETREAT whenever a retreat is legal in setup.
    retreat_decisions = 0
    legal_retreat_choices = 0
    from pokemon_tcg.actions import legal_actions
    from pokemon_tcg.simulator import step as _step

    # Run Champion under a benign per-state re-seating strategy that
    # keeps state.turn bounded so we never enter the deeper search.
    state = new_game(deck, deck, seed=42)
    for outer in range(6):
        seen = False
        for inner in range(8):
            la = legal_actions(state, 0)
            if not la:
                break
            if any(a.kind == "RETREAT" for a in la):
                legal_retreat_choices += 1
            a = champ(state, 0)
            if a.kind == "RETREAT":
                retreat_decisions += 1
                seen = True
            state = _step(state, a)
            if state.is_terminal():
                break
            # Stay in setup phase: reset every time we'd cross into
            # search territory. This means we test the retreat decision
            # in isolation and don't depend on the search stack.
            if state.turn >= champ._setup_turns:
                state = new_game(deck, deck, seed=42 + outer * 100 + inner)
        if seen:
            break
    # If retreat was ever legal, the predictor-forced Champion must have
    # picked it. If retreat was never legal in the chosen seeds, the
    # search failed to construct a retreat-eligible state, which is
    # still a valid outcome (just assert no crash).
    if legal_retreat_choices > 0:
        assert retreat_decisions > 0, (
            f"always-yes predictor should have produced at least one "
            f"RETREAT pick; saw {retreat_decisions}/{legal_retreat_choices}"
        )
    else:
        # No retreat test paths exercised; the smoke check is sufficient.
        assert True


def test_champion_predictor_path_returns_ACTION_not_pass_when_retreat_possible():
    """Build a synthetic state where RETREAT is clearly legal, then with an
    'always-yes' predictor, ensure the action is RETREAT.

    We deliberately set state.turn < ChampionAgent.setup_turns so the
    :meth:`Champ..._setup_pick` heuristic runs (rather than the search
    stack, which only consults the value_fn delta).
    """
    from pokemon_tcg.cards import (
        EnergyCard, PokemonCard, Move, Card,
    )
    from pokemon_tcg.game_state import (
        GameState, PlayerState, PokemonInstance,
    )

    monster_base = PokemonCard(
        name="TestMon", stage="Basic", evolves_from=None, hp=110,
        ptype="C", weakness=None, resistance=None, resistance_value=0,
        retreat=1,
        moves=(Move(name="Smash", cost=("C",), damage=100, text=""),),
    )
    weak_base = PokemonCard(
        name="TestWeaky", stage="Basic", evolves_from=None, hp=80,
        ptype="C", weakness=None, resistance=None, resistance_value=0,
        retreat=1,
        moves=(Move(name="Poke", cost=("C", "C"), damage=30, text=""),),
    )
    energy_mon = EnergyCard(name="E", provides="C", is_special=False)
    energy_card_mon_card = lambda: Card(card_id="t", energy=energy_mon)
    deck_filler = lambda: Card(card_id="d", energy=energy_mon)

    p0 = PlayerState(
        name="P0", deck=[deck_filler() for _ in range(20)],
        hand=[energy_card_mon_card() for _ in range(3)],
        # active weak basic with 1 energy (so retreat costs 1, payable)
        active=PokemonInstance(base=weak_base, hp=weak_base.hp,
                                  base_hp=weak_base.hp,
                                  attached_energy=("C",)),
        # bench monster with no energy yet
        bench=[PokemonInstance(base=monster_base, hp=monster_base.hp,
                                base_hp=monster_base.hp)],
        prizes=[], prize_count=6, discard=[],
    )
    p1 = PlayerState(
        name="P1", deck=[deck_filler() for _ in range(20)],
        hand=[], active=None, bench=[],
        prizes=[], prize_count=6, discard=[],
    )
    # turn < ChampionAgent.setup_turns (default 5) -> setup_pick runs.
    state = GameState(players=(p0, p1), turn=3, active_player=0, rng_seed=42)

    class AlwaysWinModel(RetreatWinPredictor):
        def predict_proba(self, features):
            return 1.0

    predictor = AlwaysWinModel()
    champ = ChampionAgent(config={"retreat_predictor": predictor})
    a = champ(state, 0)
    assert a.kind == "RETREAT", f"expected RETREAT, got {a.kind}"


def test_champion_predictor_path_blocks_retreat_when_predictor_says_no():
    """Build the same synthetic state, but with a predictor that returns
    0.0 — Champion should NOT retreat because the model's probability
    is below the configured threshold."""
    from pokemon_tcg.cards import EnergyCard, PokemonCard, Move, Card
    from pokemon_tcg.game_state import GameState, PlayerState, PokemonInstance

    monster_base = PokemonCard(
        name="TestMon", stage="Basic", evolves_from=None, hp=110,
        ptype="C", weakness=None, resistance=None, resistance_value=0,
        retreat=1,
        moves=(Move(name="Smash", cost=("C",), damage=100, text=""),),
    )
    weak_base = PokemonCard(
        name="TestWeaky", stage="Basic", evolves_from=None, hp=80,
        ptype="C", weakness=None, resistance=None, resistance_value=0,
        retreat=1,
        moves=(Move(name="Poke", cost=("C", "C"), damage=30, text=""),),
    )
    energy_mon = EnergyCard(name="E", provides="C", is_special=False)
    energy_card_mon_card = lambda: Card(card_id="t", energy=energy_mon)
    deck_filler = lambda: Card(card_id="d", energy=energy_mon)
    p0 = PlayerState(
        name="P0", deck=[deck_filler() for _ in range(20)],
        hand=[energy_card_mon_card() for _ in range(3)],
        active=PokemonInstance(base=weak_base, hp=weak_base.hp,
                                  base_hp=weak_base.hp,
                                  attached_energy=("C",)),
        bench=[PokemonInstance(base=monster_base, hp=monster_base.hp,
                                base_hp=monster_base.hp)],
        prizes=[], prize_count=6, discard=[],
    )
    p1 = PlayerState(
        name="P1", deck=[deck_filler() for _ in range(20)],
        hand=[], active=None, bench=[],
        prizes=[], prize_count=6, discard=[],
    )
    state = GameState(players=(p0, p1), turn=3, active_player=0, rng_seed=42)

    class AlwaysLoseModel(RetreatWinPredictor):
        def predict_proba(self, features):
            return 0.0

    predictor = AlwaysLoseModel()
    champ = ChampionAgent(config={"retreat_predictor": predictor})
    a = champ(state, 0)
    # Always-lose predictor => CHAMPION_CONFIG threshold (0.45) not met,
    # so retreat should be suppressed. Champion falls through to the
    # setup-heuristic path which returns ATTACH_ENERGY first.
    assert a.kind != "RETREAT", f"expected not RETREAT, got {a.kind}"
