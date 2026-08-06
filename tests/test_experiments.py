"""Experiment framework + logging + metrics tests."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from pokemon_tcg.cards import load_cards
from pokemon_tcg.deck import build_random_deck
from pokemon_tcg.simulator import simulate_match
from pokemon_tcg.experiments.runner import ExperimentRunner
from pokemon_tcg.experiments.deck_pool import build_deck_pool, DeckPool
from pokemon_tcg.experiments.tournament import run_tournament
from pokemon_tcg.evaluation.metrics import Metrics, summarize
from pokemon_tcg.evaluation.leaderboard import Leaderboard, _exp_score
from pokemon_tcg.logging_utils.game_log import MatchRecord, MatchLogger
from pokemon_tcg.analysis.failure import categorize_failure, summarize_failures, CATEGORIES
from pokemon_tcg.agents.benchmarks import GreedyAgent, DefensiveAgent, AggressiveAgent


def test_build_deck_pool_random():
    pool = build_deck_pool("random", n=4, base_seed=42)
    assert len(pool) == 4
    for d in pool.decks:
        assert len(d) >= 55


def test_build_deck_pool_themed():
    pool = build_deck_pool("themed", n=4, base_seed=42)
    assert len(pool) == 4
    # Should have names with type letters
    assert all("-" in name for name in pool.names)


def test_experiment_runner_smoke():
    decks = build_deck_pool("random", n=4)
    runner = ExperimentRunner(out_dir="results")
    r = runner.run(GreedyAgent(), DefensiveAgent(), decks, seeds=[42, 7], games_per_pair=2)
    assert r.metrics.games == 4
    assert r.metrics.wins_p0 + r.metrics.wins_p1 <= 4


def test_metrics_summarize():
    cards = load_cards()
    deck = build_random_deck(cards, seed=42)
    records = []
    for s in [42, 7, 123]:
        result = simulate_match(deck, deck, [GreedyAgent(), DefensiveAgent()],
                                 seed=s, log=True, max_turns=40)
        rec = MatchRecord(seed=s, winner=result["winner"],
                          turns=result["turns"], events=result["log"],
                          metadata={"x": 1})
        rec.state = result["state"]
        records.append(rec)
    m = summarize(records)
    m.finalize()
    d = m.to_dict()
    assert d["games"] == 3
    assert d["win_rate_p0"] >= 0 and d["win_rate_p0"] <= 1.0


def test_tournament_smoke():
    agents = [GreedyAgent(), DefensiveAgent(), AggressiveAgent()]
    res = run_tournament(agents, seeds=[42, 7], games_per_pair=1, out_dir="results")
    assert len(res.records) >= len(agents) * len(agents) * 2
    assert res.leaderboard.elo
    # Every agent has an Elo
    for a in agents:
        assert a.name in res.leaderboard.elo


def test_match_log_serialization():
    rec = MatchRecord(seed=42, winner=0, turns=10, events=[{"k": 1}], metadata={"x": 1})
    d = rec.to_json()
    assert d["seed"] == 42
    rec2 = MatchRecord.from_json(d)
    assert rec2.winner == 0
    assert rec2.events[0]["k"] == 1


def test_logger_writes_json(tmp_dir="results"):
    rec = MatchRecord(seed=42, winner=0, turns=10, events=[{"k": 1}])
    logger = MatchLogger(out_dir=tmp_dir)
    path = logger.write(rec, name="test.json")
    assert os.path.exists(path)
    # Cleanup
    if os.path.exists(path):
        os.remove(path)


def test_failure_categories_registered():
    for cat in ("NO_KO", "POOR_SETUP", "PRIZE_TRADE", "DECK_OUT",
                 "OVER_EXT", "ENERGY_STARVED", "NOT_CATEGORIZED"):
        assert cat in CATEGORIES


def test_categorize_winner_returns_win():
    rec = MatchRecord(seed=42, winner=0, turns=10, events=[
        {"kind": "PRIZE_TAKEN", "remaining": 5}
    ] * 6)
    assert categorize_failure(rec, perspective=0) == "WIN"


def test_categorize_deck_out():
    rec = MatchRecord(seed=42, winner=1, turns=20, events=[
        {"kind": "GAME_OVER", "winner": 1, "reason": "deck_out"}
    ])
    assert categorize_failure(rec, perspective=0) == "DECK_OUT"


def test_leaderboard_elo_update():
    lb = Leaderboard()
    lb.record(winner=0, a="A", b="B")
    lb.record(winner=1, a="A", b="B")
    assert "A" in lb.elo
    assert "B" in lb.elo
    a_e_before = lb.elo["A"]
    lb.record(winner=0, a="A", b="C")
    assert lb.elo["C"] < 1500  # C lost one


def test_exp_score():
    assert 0 < _exp_score(2000, 1500) < 1
    assert abs(_exp_score(1500, 1500) - 0.5) < 1e-9
