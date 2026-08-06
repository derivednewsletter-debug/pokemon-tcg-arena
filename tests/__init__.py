"""Run all tests from the kaggle_pokemon package.

Lightweight test runner — keeps test deps minimal (no pytest). Each
test module exposes a list of (name, callable) pairs and the top-level
`run` executes them sequentially.
"""
import importlib
import os
import sys
import time
import traceback
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))


def _discover() -> list:
    out = []
    for fn in sorted(os.listdir(HERE)):
        if fn.startswith("test_") and fn.endswith(".py"):
            mod = importlib.import_module(f"tests.{fn[:-3]}")
            for name in dir(mod):
                if name.startswith("test_") and callable(getattr(mod, name)):
                    out.append((name, getattr(mod, name)))
    return out


def run(filter: Optional[str] = None) -> int:
    """Run all tests (filtered by name substring if requested)."""
    failures = 0
    passed = 0
    for name, fn in _discover():
        if filter and filter not in name:
            continue
        t0 = time.time()
        try:
            fn()
            dt = time.time() - t0
            print(f"PASS  {name}  ({dt*1000:.1f}ms)")
            passed += 1
        except Exception:
            dt = time.time() - t0
            print(f"FAIL  {name}  ({dt*1000:.1f}ms)")
            traceback.print_exc()
            failures += 1
    print(f"\n=== {passed}/{passed + failures} tests passed ===")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(run())
