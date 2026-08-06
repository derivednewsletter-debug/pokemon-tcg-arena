"""Agent interface — matches the kaggriculture pattern.

An agent is a pure function of (state, who) -> deterministic Action. To
keep search/play symmetric, agents are callables rather than classes;
wrapping classes (Benchmark agents, trainable agents) implement the same
``(state, who) -> Action`` signature."""
from __future__ import annotations

import abc
from typing import Optional

from ..actions import Action
from ..game_state import GameState, make_observation


def observe(state: GameState, who: int) -> dict:
    """Trimmed observation passed to agents that prefer JSON-style reads."""
    return make_observation(state, who)


def coerce_action(value) -> Action:
    """Helper: convert a dict/Action into an Action, raising if neither."""
    if isinstance(value, Action):
        return value
    if isinstance(value, dict):
        return Action.from_json(value)
    raise TypeError(f"Expected Action or dict, got {type(value).__name__}: {value!r}")


class Agent(abc.ABC):
    """Abstract agent — implements a callable ``(state, who) -> Action``."""

    name: str = "Agent"

    def __init__(self, config: Optional[dict] = None):
        self.config = dict(config or {})

    @abc.abstractmethod
    def __call__(self, state: GameState, who: int) -> Action:
        ...

    def reset(self) -> None:
        """Optional hook — invoked at the start of every match."""
        return None
