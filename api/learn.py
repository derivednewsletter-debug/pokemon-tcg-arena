"""Learning dashboard API — Vercel Python function at ``/api/learn``.

``GET /api/learn`` returns the aggregated learning profile (games
played, AI win rate vs humans, human action tendencies, openers) plus
recent games. ``POST /api/learn`` accepts externally-recorded games.

Games finished through the arena are recorded automatically server-side;
this endpoint just makes the accumulated knowledge visible (the
frontend's Stats page renders it).
"""
from __future__ import annotations

from flask import Flask, jsonify, request

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _paths  # noqa: F401  (sys.path bootstrap)
from _telemetry import store
from learning.profiles import ACTION_LABELS, build_profile

app = Flask(__name__)


@app.route("/api/learn", methods=["GET"])
def stats():
    records = store.records()
    profile = build_profile(records)
    recent = [{
        "winner": r.get("winner"),
        "human_deck": r.get("human_deck"),
        "ai_deck": r.get("ai_deck"),
        "difficulty": r.get("difficulty"),
        "turns": r.get("turns"),
        "human_prizes_left": r.get("human_prizes_left"),
        "ai_prizes_left": r.get("ai_prizes_left"),
        "duration_s": r.get("duration_s"),
        "ts": r.get("ts"),
    } for r in records[-12:]][::-1]
    return jsonify({
        "profile": profile,
        "recent": recent,
        "labels": ACTION_LABELS,
    })


@app.route("/api/learn", methods=["POST"])
def record():
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "record must be a JSON object"}), 400
    store.add(body)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=False)
