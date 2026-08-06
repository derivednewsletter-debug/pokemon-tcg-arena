"""Detach a command from the invoking shell (double-fork daemon).

Usage: python3 scripts/daemonize.py <cwd> <cmd> [args...]
The parent exits immediately; the child survives shell exit.
"""
from __future__ import annotations

import os
import sys

if len(sys.argv) < 3:
    print("usage: daemonize.py <cwd> <cmd> [args...]")
    sys.exit(1)

cwd = sys.argv[1]
cmd = sys.argv[2:]

pid = os.fork()
if pid > 0:
    os._exit(0)          # first parent exits
os.setsid()              # new session, detached from controlling tty
pid2 = os.fork()
if pid2 > 0:
    os._exit(0)          # intermediate exits so child can never reacquire tty
os.chdir(cwd)
# re-open stdio to /dev/null so nothing holds our pipes
devnull = os.open(os.devnull, os.O_RDWR)
for fd in (0, 1, 2):
    try:
        os.dup2(devnull, fd)
    except OSError:
        pass
os.execvp(cmd[0], cmd)
