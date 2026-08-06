"""Pokemon TCG dashboard — Flask entry point.

Run with::

    python3 -m pokemon_tcg web --port 5055

Or directly::

    python3 -m pokemon_tcg.webapp.app
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from flask import Flask, render_template

from pokemon_tcg.webapp.api import api, _ensure_data, RUNS


def create_app() -> Flask:
    """Build the Flask app.

    The first request triggers a small lazy tournament run via
    ``_ensure_data()`` so the dashboard always has something to show.
    """
    app = Flask(__name__)
    app.register_blueprint(api)
    pkg = os.path.dirname(os.path.abspath(__file__))
    app.template_folder = os.path.join(pkg, "templates")
    app.static_folder = os.path.join(pkg, "static")

    pages = {
        "/": ("dashboard.html", "Pokemon TCG — Director Dashboard"),
        "/matrix": ("matrix.html", "Matchup Matrix"),
        "/replay": ("replay.html", "Match Replay"),
    }

    for i, (route, (tmpl, title)) in enumerate(pages.items()):
        def _make(t=tmpl, ttl=title):
            def view():
                # Trigger lazy data load so first paint has something to show.
                if not RUNS:
                    _ensure_data()
                return render_template(t, title=ttl)
            return view
        app.add_url_rule(route, endpoint=f"page_{i}", view_func=_make())

    @app.context_processor
    def inject_nav():
        return {
            "nav": [
                ("/", "Dashboard"),
                ("/matrix", "Matrix"),
                ("/replay", "Replay"),
            ],
        }

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    create_app().run(host=args.host, port=args.port, debug=args.debug)
