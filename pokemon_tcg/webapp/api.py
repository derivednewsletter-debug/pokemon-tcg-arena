"""Pokemon TCG webapp — API blueprint and data loaders.

Endpoint catalogue
==================
GET /api/leaderboard          Elo + win rate + head-to-head summary
GET /api/matrix               matchup matrix {a: {b: wins/games}}
GET /api/replay/<seed>        structured event log for one match
GET /api/runs                 list of cached tournament runs
GET /api/runs/<id>            one cached tournament run
GET /api/failure/<seed>       failure categorization for one match
GET /api/runs/refresh         trigger a fresh small tournament run
"""
from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from typing import Any

from flask import Blueprint, abort, current_app, jsonify, request

# Make the project importable when running `python3 webapp/app.py`
_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pokemon_tcg.cards import load_cards
from pokemon_tcg.deck import build_random_deck
from pokemon_tcg.simulator import simulate_match
from pokemon_tcg.agents.benchmarks import (
    GreedyAgent, GreedyNoRetreatAgent, SearchAgent, AggressiveAgent,
    DefensiveAgent, EnergyRampAgent, BenchBufferAgent,
)
from pokemon_tcg.agents.champion import ChampionAgent
from pokemon_tcg.experiments.tournament import run_tournament
from pokemon_tcg.experiments.deck_pool import build_deck_pool
from pokemon_tcg.analysis.failure import categorize_failure, summarize_failures
from pokemon_tcg.logging_utils.game_log import MatchRecord


api = Blueprint("api", __name__)

CACHE_LOCK = threading.Lock()
CACHE: dict[str, Any] = {
    "leaderboard": None,
    "matrix": None,
    "last_run": 0.0,
}
RUNS: list[dict[str, Any]] = []  # most recent run first


# ---------------------------------------------------------------------------
# Background tournament runner
# ---------------------------------------------------------------------------

def _all_agent_classes():
    return [ChampionAgent, SearchAgent, GreedyAgent, AggressiveAgent,
            DefensiveAgent, EnergyRampAgent, BenchBufferAgent]


def _default_seeds(n: int = 5) -> list[int]:
    return [42, 7, 123, 999, 2024][:n]


def run_quick_tournament(games_per_pair: int = 1) -> dict[str, Any]:
    """Run a small tournament and cache results. Idempotent under lock."""
    with CACHE_LOCK:
        cards = load_cards()
        decks = build_deck_pool("themed", n=8)
        agents = [cls() for cls in _all_agent_classes()]
        t0 = time.time()
        # Run only the 6 strongest to keep startup snappy.
        agents = [ChampionAgent(), SearchAgent(), GreedyAgent(),
                  GreedyNoRetreatAgent(), AggressiveAgent(), DefensiveAgent()]
        # 3 seeds × 1 game per pair keeps the first paint under ~5s.
        res = run_tournament(agents, _default_seeds(3),
                              pool_mode="themed",
                              games_per_pair=games_per_pair,
                              out_dir=_results_dir())
        CACHE["leaderboard"] = res.leaderboard
        CACHE["matrix"] = dict(res.leaderboard.matchup_wins)
        CACHE["last_run"] = time.time()
        RUNS.insert(0, {
            "id": int(CACHE["last_run"]),
            "agents": [a.name for a in agents],
            "seeds": _default_seeds(3),
            "games_per_pair": games_per_pair,
            "elapsed": round(time.time() - t0, 2),
            "labels": list(res.leaderboard.names),
            "elo": dict(res.leaderboard.elo),
            "matchup": {
                f"{a}_vs_{b}": list(v)
                for (a, b), v in res.leaderboard.matchup_wins.items()
            },
            "records": [_record_with_agents(r) for r in res.records],
        })
        RUNS[:] = RUNS[:8]  # cap history
        return RUNS[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _results_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _record_with_agents(record: MatchRecord) -> dict:
    """Annotate a MatchRecord JSON payload with the names of the two
    players (extracted from `metadata.p0`/`metadata.p1` so the dashboard
    can filter by agent name).
    """
    payload = record.to_json()
    meta = payload.get("metadata") or {}
    payload["p0"] = meta.get("p0", "?")
    payload["p1"] = meta.get("p1", "?")
    return payload


def _ensure_data(force: bool = False) -> dict[str, Any]:
    """Lazy tournament generator — keeps the dashboard snappy on startup.

    Always returns ``{"leaderboard": <Leaderboard>, "matrix": <dict>,
    "records": <list>}`` so downstream consumers can use a stable shape.
    """
    if force or not CACHE["leaderboard"] or (time.time() - CACHE["last_run"]) > 600:
        run_quick_tournament()
    return {
        "leaderboard": CACHE["leaderboard"],
        "matrix": CACHE["matrix"],
        "records": RUNS[0]["records"] if RUNS else [],
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _leaderboard_payload(force: bool = False) -> dict[str, Any]:
    data = _ensure_data(force=force)
    lb = data["leaderboard"]
    rows = []
    names = list(lb.names or [])
    if not names:
        return {"rows": [], "matrix": {}, "updated_at": CACHE["last_run"]}
    # Winrates per agent come straight from matchup table
    matchup = data["matrix"]
    for name in sorted(names, key=lambda n: -lb.elo.get(n, 1500)):
        wins = sum(int(v[0]) for k, v in matchup.items() if k[0] == name)
        games = sum(int(v[1]) for k, v in matchup.items() if k[0] == name)
        wr = round(wins / games * 100, 1) if games else 0.0
        rows.append({"name": name, "elo": round(lb.elo.get(name, 1500), 1),
                     "wins": wins, "games": games, "winrate": wr})
    return {
        "rows": rows,
        "matrix": {f"{a}_vs_{b}": list(v) for (a, b), v in matchup.items()},
        "updated_at": CACHE["last_run"],
        "agents": names,
    }


@api.route("/api/leaderboard")
def api_leaderboard():
    return jsonify(_leaderboard_payload())


@api.route("/api/leaderboard/refresh")
def api_leaderboard_refresh():
    return jsonify(_leaderboard_payload(force=True))


@api.route("/api/matrix")
def api_matrix():
    data = _ensure_data()
    payload = _leaderboard_payload()
    # Restructure { A_vs_B: [w, g] } to { a: { b: w/g } }
    matrix = {}
    for k, v in payload["matrix"].items():
        a, b = k.split("_vs_")
        matrix.setdefault(a, {})[b] = {"wins": v[0], "games": v[1]}
    return jsonify({
        "agents": payload["agents"],
        "matrix": matrix,
    })


@api.route("/api/replay", defaults={"seed": None})
@api.route("/api/replay/<seed>")
def api_replay(seed: str | None = None):
    """Return one structured match record from the most recent run.

    Accepts the seed in the URL path *or* as ``?seed=`` /
    ``?p0=...&p1=...`` query parameters. Falls back to the first record
    if nothing matches.
    """
    data = _ensure_data()
    qs = request.args
    p0 = qs.get("p0")
    p1 = qs.get("p1")
    seed_q = qs.get("seed")
    chosen = seed or seed_q
    candidates = data.get("records", [])
    selected = None
    if chosen and chosen.isdigit():
        seed_int = int(chosen)
        for r in candidates:
            try:
                if int(r.get("seed", -1)) == seed_int:
                    if (p0 is None or r.get("p0") == p0) and \
                       (p1 is None or r.get("p1") == p1):
                        selected = r
                        break
            except (ValueError, TypeError):
                continue
    if selected is None and candidates:
        if p0 is not None or p1 is not None:
            for r in candidates:
                if (p0 is None or r.get("p0") == p0) and \
                   (p1 is None or r.get("p1") == p1):
                    selected = r
                    break
            if selected is None:
                selected = candidates[0]
        else:
            selected = candidates[0]
    if selected is None:
        return jsonify({"error": "no records available"})
    return jsonify(_annotate_record(selected))


def _pick_record(seed: str, data: dict[str, Any]) -> dict[str, Any]:
    if not RUNS:
        return {"error": "no data"}
    latest = RUNS[0]
    seed_int = int(seed) if seed.isdigit() else None
    candidates = latest.get("records", [])
    if seed_int is not None:
        for r in candidates:
            try:
                if int(r.get("seed", -1)) == seed_int:
                    return _annotate_record(r)
            except (ValueError, TypeError):
                continue
    if candidates:
        return _annotate_record(candidates[0])
    return {"error": "no records"}


def _annotate_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Run the failure analyzer over an in-memory MatchRecord.

    We rebuild a MatchRecord from the JSON payload and run
    ``categorize_failure`` for each player's perspective so the UI
    can label each loss with a dominant failure mode.
    """
    events = rec.get("events", [])
    winner = rec.get("winner")
    seed = rec.get("seed", 0)
    try:
        match_rec = MatchRecord(seed=seed, winner=winner,
                                turns=rec.get("turns", 0),
                                events=events,
                                metadata=rec.get("metadata", {}))
    except Exception:
        return rec
    out = dict(rec)
    out["failure"] = {
        "p0": categorize_failure(match_rec, perspective=0),
        "p1": categorize_failure(match_rec, perspective=1),
    }
    return out


@api.route("/api/failure/<seed>")
def api_failure(seed: str):
    data = _ensure_data()
    rec = _pick_record(seed, data)
    if "failure" not in rec:
        rec["failure"] = {"p0": "UNKNOWN", "p1": "UNKNOWN"}
    return jsonify({
        "seed": int(seed) if seed.isdigit() else None,
        "events": rec.get("events", []),
        "failure": rec.get("failure", {}),
        "winner": rec.get("winner"),
        "turns": rec.get("turns"),
    })


@api.route("/api/runs")
def api_runs():
    return jsonify([{"id": r["id"],
                     "agents": r["agents"],
                     "seeds": r["seeds"],
                     "games_per_pair": r["games_per_pair"],
                     "elapsed": r["elapsed"]} for r in RUNS])


@api.route("/api/runs/<int:run_id>")
def api_run_detail(run_id: int):
    for r in RUNS:
        if r["id"] == run_id:
            return jsonify({k: v for k, v in r.items() if k != "records"})
    abort(404)


@api.route("/api/health")
def api_health():
    return jsonify({
        "ok": True,
        "runs": len(RUNS),
        "last_run": CACHE["last_run"],
        "agents": list(_all_agent_classes().__class__.__name__),
    })
