"""
case_manager.py
===============
Rigorous per-case state tracking, single-instance enforcement via pidfile,
and runner sentinel files for all OpenFOAM campaign runners.

Per-case state machine (RANS):
  (missing) / pending → meshing → mesh_done → solving → done
  any state → failed

Extended for URANS hot-start (future):
  mesh_done → rans_solving → rans_done → urans_solving → done

State file : <case_dir>/case.status     (written atomically via os.rename)
Pidfile    : <cases_dir>/.runner.pid
Sentinels  : <cases_dir>/.runner.done | .runner.failed
"""

import os, sys, atexit, signal, time
from enum import Enum
from pathlib import Path


# ─── State enum ───────────────────────────────────────────────────────────────

class Status(str, Enum):
    PENDING       = "pending"
    MESHING       = "meshing"
    MESH_DONE     = "mesh_done"
    SOLVING       = "solving"
    RANS_SOLVING  = "rans_solving"    # URANS hot-start: RANS phase
    RANS_DONE     = "rans_done"       # URANS hot-start: RANS fields saved
    URANS_SOLVING = "urans_solving"   # URANS hot-start: pimpleFoam phase
    DONE          = "done"
    FAILED        = "failed"


# ─── Per-case state ───────────────────────────────────────────────────────────

class CaseState:
    """Atomic read/write of <case_dir>/case.status."""

    def __init__(self, case_dir: Path):
        self._dir  = Path(case_dir)
        self._path = self._dir / "case.status"
        self._tmp  = self._dir / "case.status.tmp"

    @property
    def status(self) -> Status:
        if not self._path.exists():
            return Status.PENDING
        try:
            return Status(self._path.read_text().strip().split("\n")[0])
        except ValueError:
            return Status.PENDING

    def set(self, s: Status, detail: str = ""):
        content = s.value + (f"\n{detail}" if detail else "")
        self._tmp.write_text(content)
        os.rename(self._tmp, self._path)   # atomic on POSIX (same fs)

    def detail(self) -> str:
        if not self._path.exists():
            return ""
        lines = self._path.read_text().strip().split("\n")
        return "\n".join(lines[1:]) if len(lines) > 1 else ""


# ─── Resume decision ──────────────────────────────────────────────────────────

def resume_action(case_dir: Path) -> str:
    """
    Inspect <case_dir> and return the action the runner should take:

      'skip'          — done, skip entirely
      'clean_restart' — clear directory, build mesh + solve from scratch
      'solve_only'    — mesh is good (mesh_done); re-write ICs and solve fresh
      'resume_solve'  — solving was interrupted; check for checkpoint, continue
      'skip_rans'     — rans_done; skip RANS, go straight to URANS (hot-start)
      'resume_rans'   — rans_solving interrupted; restart RANS (keep mesh)
      'resume_urans'  — urans_solving interrupted; restart pimpleFoam (keep mesh+RANS)
    """
    case_dir = Path(case_dir)
    state = CaseState(case_dir)
    s = state.status

    # ── Legacy migration: no status file but old pipeline logs exist ──────────
    if s == Status.PENDING and not (case_dir / "case.status").exists():
        log = case_dir / "log.simpleFoam"
        if log.exists():
            text = log.read_text(errors="replace")
            if any(ln.strip() == "End" for ln in text.splitlines()):
                state.set(Status.DONE, "migrated from legacy run")
                return "skip"
            # Log present but no End — solve was interrupted
            state.set(Status.SOLVING, "migrated from legacy run (interrupted)")
            return "resume_solve"
        mesh_log = case_dir / "log.cartesianMesh"
        if mesh_log.exists():
            bl_log = case_dir / "log.generateBoundaryLayers"
            if bl_log.exists() and (case_dir / "constant" / "polyMesh").exists():
                state.set(Status.MESH_DONE, "migrated from legacy run")
                return "solve_only"
            state.set(Status.MESHING, "migrated from legacy run (interrupted)")
            return "clean_restart"

    # ── Normal state table ────────────────────────────────────────────────────
    if s == Status.DONE:
        return "skip"
    elif s == Status.MESH_DONE:
        return "solve_only"
    elif s == Status.SOLVING:
        return "resume_solve"
    elif s == Status.RANS_DONE:
        return "skip_rans"
    elif s == Status.RANS_SOLVING:
        return "resume_rans"
    elif s == Status.URANS_SOLVING:
        return "resume_urans"
    else:
        # PENDING, MESHING, FAILED — clean restart
        return "clean_restart"


# ─── Single-instance pidfile ──────────────────────────────────────────────────

class PidFile:
    """
    Prevents two runner instances from operating on the same cases_dir.

    Usage:
        with PidFile(cases_dir):
            ... run all cases ...

    On entry: checks for stale/live pidfile.
    On exit (clean or signal): removes pidfile.
    """

    def __init__(self, cases_dir: Path):
        self._path = Path(cases_dir) / ".runner.pid"

    def __enter__(self):
        if self._path.exists():
            try:
                pid = int(self._path.read_text().strip())
                os.kill(pid, 0)   # OSError if process doesn't exist
                print(
                    f"ERROR: another runner is already active (PID {pid}).\n"
                    f"  Pidfile: {self._path}\n"
                    f"  Kill it first or delete the pidfile if it is stale.",
                    file=sys.stderr,
                )
                sys.exit(1)
            except (ValueError, ProcessLookupError):
                print(f"WARNING: removing stale pidfile (process gone): {self._path}",
                      file=sys.stderr)
                self._path.unlink(missing_ok=True)
            except PermissionError:
                # Process exists but owned by another user — treat as live
                print(
                    f"ERROR: pidfile {self._path} belongs to another process "
                    f"we cannot signal. Remove it manually if stale.",
                    file=sys.stderr,
                )
                sys.exit(1)

        self._path.write_text(str(os.getpid()))
        atexit.register(self._remove)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT,  self._signal_handler)
        return self

    def __exit__(self, *_):
        self._remove()

    def _remove(self):
        self._path.unlink(missing_ok=True)

    def _signal_handler(self, signum, frame):
        self._remove()
        # Re-raise as default so the process actually exits
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)


# ─── Runner sentinel ──────────────────────────────────────────────────────────

class RunnerSentinel:
    """
    Writes .runner.done or .runner.failed on runner exit so monitors can
    trigger reliably on a file, not on log content.

    Usage:
        with RunnerSentinel(cases_dir, "slant_sweep") as sentinel:
            for case in cases:
                result = run_case(...)
                if result ok:
                    sentinel.n_done += 1
                else:
                    sentinel.n_failed += 1

    The monitor should watch:
        until [ -f <cases_dir>/.runner.done ] || [ -f <cases_dir>/.runner.failed ]
        do sleep 15; done
    """

    def __init__(self, cases_dir: Path, campaign_name: str = ""):
        self._dir     = Path(cases_dir)
        self._campaign = campaign_name
        self.n_done   = 0
        self.n_failed = 0
        self.n_skipped = 0
        self._done_path   = self._dir / ".runner.done"
        self._failed_path = self._dir / ".runner.failed"

    def __enter__(self):
        self._done_path.unlink(missing_ok=True)
        self._failed_path.unlink(missing_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, _tb):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        summary = (
            f"campaign={self._campaign}\n"
            f"done={self.n_done}\n"
            f"skipped={self.n_skipped}\n"
            f"failed={self.n_failed}\n"
            f"finished={ts}\n"
        )
        if exc_type is None and self.n_failed == 0:
            self._done_path.write_text(summary)
        else:
            if exc_type is not None:
                summary += f"exception={exc_val}\n"
            self._failed_path.write_text(summary)
        return False   # never suppress exceptions
