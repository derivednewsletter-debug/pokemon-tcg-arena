"""Vercel serverless API for the Pokémon TCG Arena.

Endpoints (each file in this folder is a Vercel Python function):

* ``/api/game/meta`` — decks + cards metadata for the frontend.
* ``/api/game/new``  — start a game; returns the first player view.
* ``/api/game/act``  — submit the human's picks; runs the AI's reply
  turn through the real engine; returns the next player view.
* ``/api/learn``     — telemetry: finished games are recorded here and
  aggregated into the opponent profile the AI plays with (GET returns
  the learning dashboard data).
"""
