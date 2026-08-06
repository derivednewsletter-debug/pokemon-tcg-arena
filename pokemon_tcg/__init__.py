"""Pokemon TCG AI Battle Challenge — competitive training agent.

A modular Python implementation of a Pokemon TCG-style simulator, agent, and
evaluation framework. Built on a simplified ruleset that captures the core
mechanics of the real game (turn structure, energy, evolution, attacks,
prizes) while remaining deterministic, fast, and easy to instrument.

Sub-packages
============
- cards         card models + CSV loader
- deck          deck construction
- game_state    immutable per-player state snapshots
- actions       legal-action enumeration
- simulator     rules engine (state -> state -> state)
- evaluator     position-evaluation heuristics
- search        alpha-beta / rollout lookahead
- agents        agent interface and benchmark strategies
- experiments   batch runner + tournament
- evaluation    metrics + leaderboard
- logging_utils structured game logs
- analysis      failure attribution
"""

__version__ = "0.5.0"
