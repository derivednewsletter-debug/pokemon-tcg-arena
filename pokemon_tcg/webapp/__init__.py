"""Web dashboard for the Pokemon TCG agent.

Flask app exposing three pages:

* ``/``         — overview dashboard (Elo leaderboard + win rate bars)
* ``/matrix``   — matchup matrix win-rate heatmap
* ``/replay``   — turn-by-turn replay of a single match with failure
                  annotations

Plus a small JSON API under ``/api/*`` for any external tooling.

Run with::

    python3 -m pokemon_tcg web --port 5055

If no tournament data has been generated yet, the app runs a small
tournament on startup so the dashboard always has something to show.
"""
