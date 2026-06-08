"""
pipeline_lock.py
================
Simple file-based lock preventing two DES pipeline scripts from running
simultaneously. Uses a PID file so stale locks from crashes are detected.

Usage:
    from pipeline_lock import acquire_lock, release_lock
    acquire_lock()   # raises RuntimeError if another instance is running
    ...
    release_lock()   # call in finally block
"""

import os
from pathlib import Path

LOCK_FILE = Path(__file__).parent / ".pipeline.lock"


def acquire_lock():
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().strip())
        except (ValueError, OSError):
            pid = None
        # Check if the PID is actually still running
        if pid is not None:
            try:
                os.kill(pid, 0)   # signal 0 = check existence only
                raise RuntimeError(
                    f"Another pipeline process is already running (PID {pid}).\n"
                    f"If this is stale, delete {LOCK_FILE} and retry."
                )
            except ProcessLookupError:
                pass   # process is gone — stale lock, safe to overwrite
    LOCK_FILE.write_text(str(os.getpid()))


def release_lock():
    try:
        LOCK_FILE.unlink()
    except FileNotFoundError:
        pass
