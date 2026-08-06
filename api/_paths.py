"""Path bootstrap shared by every API function.

Vercel executes each function with the repo root as cwd, but not
necessarily with the right paths on ``sys.path``. This helper makes the
imports work identically locally and on Vercel:

* repo root  -> ``from learning.profiles import ...``
* submission -> ``from agent import Agent`` / ``from cg.api import ...``
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SUBMISSION = os.path.join(_REPO_ROOT, "submission")

for _p in (_REPO_ROOT, _SUBMISSION):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def repo_root() -> str:
    return _REPO_ROOT
