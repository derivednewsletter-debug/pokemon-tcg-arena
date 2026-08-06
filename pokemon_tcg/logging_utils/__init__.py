"""Structured logging utilities for Pokemon TCG matches.

A match log is a list of dicts (`state.log`). We expose writers that
serialize these to JSONL or pretty-printed text for analysis.
"""
from .game_log import MatchLogger, write_match_json, read_match_json

__all__ = ["MatchLogger", "write_match_json", "read_match_json"]
