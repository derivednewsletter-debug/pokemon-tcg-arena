"""Game lifecycle API — Vercel Python function at ``/api/game``.

Routes
------
* ``GET  /api/game/meta``  — available decks, difficulties, engine info.
* ``POST /api/game/new``   — start a game: ``{human_deck, ai_deck,
  difficulty}`` -> first player view.
* ``POST /api/game/act``   — submit picks: ``{game_id, picks:[int]}``
  -> next player view (the AI's full reply turn has already run).

Sessions live in server memory keyed by ``game_id``; a game quietly
expires after 30 minutes or a cold start (the frontend handles that).
"""
from __future__ import annotations

import threading
import time

from flask import Flask, jsonify, request

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _paths  # noqa: F401  (sys.path bootstrap)
from _decks import all_decks, get_deck, validate_deck
from _orchestrator import DIFFICULTIES, GameSession
from _telemetry import store

app = Flask(__name__)

_sessions: dict = {}
_session_lock = threading.Lock()
SESSION_TTL = 30 * 60


def _cleanup() -> None:
    """Drop expired sessions and free their engine battles."""
    now = time.time()
    dead = []
    with _session_lock:
        for gid, s in list(_sessions.items()):
            if now - s["ts"] > SESSION_TTL:
                dead.append((gid, s["session"]))
                del _sessions[gid]
    for _, session in dead:
        try:
            session.game.finish()
        except Exception:
            pass


def _get_session(gid: str) -> GameSession | None:
    with _session_lock:
        s = _sessions.get(gid)
        if s is None:
            return None
        s["ts"] = time.time()
        return s["session"]


@app.route("/api/game/meta", methods=["GET"])
def meta():
    decks = all_decks()
    return jsonify({
        "decks": [{"id": d["id"], "name": d["name"], "blurb": d["blurb"],
                   "type": d.get("type", ""), "cards": d["cards"]} for d in decks],
        "difficulties": [
            {"id": "easy", "name": "Easy", "blurb": "No lookahead — pure heuristics."},
            {"id": "medium", "name": "Medium", "blurb": "Light engine-search lookahead."},
            {"id": "hard", "name": "Hard", "blurb": "Deep multi-world lookahead + learned opponent model."},
        ],
        "engine": "Official Pokémon TCG AI Battle engine (cg)",
        "agent": "Engine-search lookahead agent + learned opponent model",
        "custom_deck": {"max": 60, "max_copies": 4, "min_basic": 1},
    })


@app.route("/api/game/new", methods=["POST"])
def new_game():
    body = request.get_json(silent=True) or {}
    difficulty = body.get("difficulty") or "medium"

    # Player's deck: either a curated id or a full custom list.
    custom = body.get("custom_deck")
    if custom is not None:
        ok, reason = validate_deck(custom)
        if not ok:
            return jsonify({"error": reason}), 400
        human_deck = list(custom)
        human_deck_id = "custom"
    else:
        human_deck_id = body.get("human_deck") or "mega_aboma"
        try:
            human_deck = get_deck(human_deck_id)
        except KeyError:
            return jsonify({"error": "unknown deck id"}), 400

    ai_deck_id = body.get("ai_deck") or "mega_aboma"
    try:
        ai_deck = get_deck(ai_deck_id)
    except KeyError:
        return jsonify({"error": "unknown deck id"}), 400

    session = GameSession(human_deck, ai_deck, human_deck_id, ai_deck_id,
                          difficulty, store)
    with _session_lock:
        _sessions[session.game_id] = {"session": session, "ts": time.time()}
    try:
        view = session.advance()
    except Exception as exc:
        session.game.finish()
        with _session_lock:
            _sessions.pop(session.game_id, None)
        return jsonify({"error": f"engine failed: {exc}"}), 500
    if view.get("over"):
        session.game.finish()
        with _session_lock:
            _sessions.pop(session.game_id, None)
    return jsonify(view)


@app.route("/api/game/act", methods=["POST"])
def act():
    body = request.get_json(silent=True) or {}
    gid = body.get("game_id")
    picks = body.get("picks")
    if not gid or not isinstance(picks, list):
        return jsonify({"error": "game_id and picks required"}), 400
    session = _get_session(gid)
    if session is None:
        return jsonify(
            {"error": "game_expired",
             "message": "This game was on a server that went to sleep — start a new one."}
        ), 404
    try:
        picks = [int(p) for p in picks]
        view = session.act(picks)
    except Exception as exc:
        return jsonify({"error": f"engine failed: {exc}"}), 500
    if view.get("over"):
        session.game.finish()
        with _session_lock:
            _sessions.pop(gid, None)
    return jsonify(view)


if __name__ == "__main__":
    # Local dev: python api/game.py  (serves /api/game/* only)
    app.run(host="127.0.0.1", port=5001, debug=False)
