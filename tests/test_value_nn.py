"""Tests for the pluggable value head (MLP) + self-play pipeline."""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from pokemon_tcg.cards import load_cards
from pokemon_tcg.deck import build_random_deck
from pokemon_tcg.simulator import new_game, simulate_match
from pokemon_tcg.game_state import GameState, PlayerState, PokemonInstance
from pokemon_tcg.cards import PokemonCard, Move, EnergyCard, Card
from pokemon_tcg.value_nn import (
    MLP, FEATURE_DIM, features_from_state,
)
from pokemon_tcg.evaluator import (
    ValueFunction, HeuristicValueFn, LearnedValueFn,
)
from pokemon_tcg.selfplay import (
    TrainingExample, SelfPlayResult,
    collect_examples, train, save_model, load_model,
)
from pokemon_tcg.agents.benchmarks import GreedyAgent
from pokemon_tcg.agents.champion import ChampionAgent
from pokemon_tcg.search import (
    greedy_1ply, alpha_beta_2ply, iterative_deepening_search,
)


# ============================================================
# Feature extractor
# ============================================================

def test_feature_dim_is_documented():
    assert FEATURE_DIM == 32


def test_features_shape_and_range():
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    f = features_from_state(state, 0)
    assert f.shape == (FEATURE_DIM,)
    assert f.dtype == np.float32
    # Features are scaled to approximately [-6, 6]
    assert f.max() <= 6.5
    assert f.min() >= -6.5


def test_features_handle_each_player():
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    f0 = features_from_state(state, 0)
    f1 = features_from_state(state, 1)
    # Player 0 sees their HP / bench as positives and opp's as different
    # feature slots. Both arrays should be valid (no NaN, no Inf).
    assert not np.any(np.isnan(f0))
    assert not np.any(np.isnan(f1))
    assert not np.any(np.isinf(f0))
    assert not np.any(np.isinf(f1))


def test_features_depend_on_player():
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    f0 = features_from_state(state, 0)
    f1 = features_from_state(state, 1)
    # Different perspectives of the same state yield different features
    # (slot 31 = on_move flips sign).
    assert not np.array_equal(f0, f1)


# ============================================================
# MLP
# ============================================================

def test_mlp_forward_output_shape():
    m = MLP([8, 4, 1])
    x = np.random.default_rng(0).normal(0, 1, (3, 8)).astype(np.float32)
    y = m.forward(x)
    assert y.shape == (3, 1)


def test_mlp_output_is_tanh_bounded():
    """Tanh on the output layer — predictions stay in [-1, 1]."""
    m = MLP([8, 4, 1])
    x = np.random.default_rng(0).normal(0, 5, (50, 8)).astype(np.float32)
    y = m.forward(x)
    assert y.max() <= 1.0 + 1e-5
    assert y.min() >= -1.0 - 1e-5


def test_mlp_sgd_step_finite():
    m = MLP([4, 4, 1], seed=7)
    x = np.random.default_rng(0).normal(0, 1, (8, 4)).astype(np.float32)
    y = np.random.default_rng(1).uniform(-1, 1, (8, 1)).astype(np.float32)
    loss = m.sgd_step(x, y, lr=0.01)
    assert np.isfinite(loss)


def test_mlp_sgd_decorrelates_after_toy_training():
    """Train a tiny linear-ish task; loss should drop."""
    m = MLP([4, 4, 1], seed=11)
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (200, 4)).astype(np.float32)
    W_target = rng.normal(0, 0.5, (4, 1)).astype(np.float32)
    y = np.tanh(x @ W_target + rng.normal(0, 0.05, (200, 1)).astype(np.float32))
    before = float(np.mean((m.forward(x) - y) ** 2))
    losses = m.fit(x, y, epochs=50, batch_size=16, lr=0.02)
    after = float(np.mean((m.forward(x) - y) ** 2))
    assert after < before
    assert np.isfinite(losses[-1])


def test_mlp_save_load_roundtrip():
    m = MLP([FEATURE_DIM, 16, 1])
    m.weights[0] = np.full_like(m.weights[0], 0.123)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as fh:
        path = fh.name
    try:
        m.save(path)
        m2 = MLP([FEATURE_DIM, 16, 1])
        m2.load(path)
        assert np.array_equal(m.weights[0], m2.weights[0])
        assert np.array_equal(m.biases[1], m2.biases[1])
    finally:
        os.unlink(path)


def test_mlp_reload_preserves_loss_for_untrained_model():
    """A freshly-loaded model should produce the same forward output
    as it did before saving."""
    m = MLP([4, 4, 1], seed=3)
    x = np.random.default_rng(0).normal(0, 1, (5, 4)).astype(np.float32)
    a = m.forward(x)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as fh:
        path = fh.name
    try:
        m.save(path)
        m2 = MLP([4, 4, 1])
        m2.load(path)
        b = m2.forward(x)
        assert np.allclose(a, b, atol=1e-6)
    finally:
        os.unlink(path)


# ============================================================
# ValueFunction protocol
# ============================================================

def test_heuristic_value_fn_implements_protocol():
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    v = HeuristicValueFn()
    assert isinstance(v, ValueFunction)
    s = v(state, 0)
    assert isinstance(s, float)
    assert s == v.score(state, 0)
    assert v.kind == "heuristic"


def test_learned_value_fn_implements_protocol():
    v = LearnedValueFn()
    assert isinstance(v, ValueFunction)
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    s = v(state, 0)
    assert isinstance(s, float)
    # Output is bounded by tanh
    assert -1.0 <= s <= 1.0
    assert v.kind == "mlp"


def test_search_dispatches_to_value_fn():
    """Greedy / alpha-beta / IDS must use whichever value_fn is passed."""
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    lv = LearnedValueFn()
    a1, info1 = greedy_1ply(state, 0, value_fn=lv)
    assert info1["value_kind"] == "mlp"
    a2, info2 = alpha_beta_2ply(state, 0, value_fn=lv, beam=4)
    assert info2["value_kind"] == "mlp"
    a3, info3 = iterative_deepening_search(state, 0, value_fn=lv,
                                            time_budget_ms=200)
    assert info3["value_kind"] == "mlp"


def test_search_value_fn_kind_matches_cfg_path():
    """If only ``cfg`` is given, the search uses ``HeuristicValueFn``."""
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    state = new_game(deck, deck, seed=42)
    from pokemon_tcg.evaluator import EvaluatorConfig
    cfg = EvaluatorConfig()
    a, info = greedy_1ply(state, 0, cfg=cfg)
    assert info["value_kind"] == "heuristic"
    a, info = iterative_deepening_search(state, 0, cfg=cfg,
                                            time_budget_ms=200)
    assert info["value_kind"] == "heuristic"


# ============================================================
# Self-play + training pipeline
# ============================================================

def test_collect_examples_smoke():
    """A few Greedy-vs-Greedy games should be fast and produce balanced data."""
    res = collect_examples(num_games=3, spawn_a=GreedyAgent, spawn_b=GreedyAgent)
    assert res.games_played == 3
    assert len(res.examples) > 0
    # All targets should be in {-1, 0, +1}
    targets = {e.target for e in res.examples}
    assert targets.issubset({-1.0, 0.0, 1.0})


def test_train_decreases_loss():
    """Running the trainer should bring loss down on a non-trivial dataset."""
    sp = collect_examples(num_games=8, spawn_a=GreedyAgent, spawn_b=GreedyAgent)
    m = train(sp, epochs=20, lr=0.01, verbose=False)
    x, y = sp.to_arrays()
    final_loss = float(np.mean((m.forward(x) - y) ** 2))
    # Initial loss (random init) on a binary task approx 0.5; trained loss
    # should drop noticeably.
    assert final_loss < 0.20


def test_champion_with_untrained_value_fn_runs_match():
    """Champion must accept a LearnedValueFn even if untrained."""
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    lv = LearnedValueFn()
    champ = ChampionAgent(config={"value_fn": lv})
    agents = [champ, GreedyAgent()]
    r = simulate_match(deck, deck, agents, seed=42, log=False, max_turns=40)
    assert r["winner"] in (0, 1)


def test_champion_with_trained_value_fn_does_not_explode():
    """After training on self-play, Champion should still produce finite
    scalar values during search."""
    sp = collect_examples(num_games=4, spawn_a=GreedyAgent, spawn_b=GreedyAgent)
    m = train(sp, epochs=10, lr=0.01, verbose=False)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as fh:
        out_path = fh.name
    try:
        save_model(m, out_path)
        lv = LearnedValueFn(path=out_path)
        cards = load_cards()
        deck = build_random_deck(cards, seed=42)
        state = new_game(deck, deck, seed=42)
        a = lv(state, 0)
        assert np.isfinite(a)
        assert -1.0 <= a <= 1.0
    finally:
        os.unlink(out_path)


def test_champion_search_with_mlp_does_not_crash():
    """ChampionAgent with MLP value_fn over a full match returns valid actions."""
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    lv = LearnedValueFn()
    champ = ChampionAgent(config={"value_fn": lv,
                                   "time_budget_ms": 30,    # keep it fast
                                   "max_depth": 3})
    # Run 1 turn
    state = new_game(deck, deck, seed=42)
    act = champ(state, 0)
    assert act is not None
    from pokemon_tcg.actions import Action
    assert isinstance(act, Action)
    # Drive a full match
    r = simulate_match(deck, deck, [champ, GreedyAgent()], seed=42,
                       log=False, max_turns=40)
    assert r["winner"] in (0, 1)
