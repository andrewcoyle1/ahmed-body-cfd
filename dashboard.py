"""
dashboard.py  —  Read-only CFD pipeline monitoring dashboard
Shows DoE progress and Bayesian optimisation loop status.
Opens a local web server at http://localhost:8765

Server-Sent Events: the server watches simulation files and pushes updates
to the browser the moment data changes. Zero polling timer in the browser.

Usage:
    python3 dashboard.py          # default port 8765
    python3 dashboard.py 9000     # custom port
"""

import re
import sys
import csv
import json
import time
import queue
import threading
import socketserver
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer


class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """HTTP server that handles each request in a separate thread.
    Required so long-lived SSE /stream connections don't block the / route."""
    daemon_threads = True

CASES_DIR     = Path("openfoam_cases")
DESIGN_MATRIX = Path("design_matrix.csv")
BO_HISTORY    = Path("results/bo_history.csv")
RESULTS_CSV   = Path("results/results_summary.csv")
MESH_CONV_DIR = Path("mesh_convergence")
POLL_INTERVAL = 2   # seconds between file-mtime checks in the monitor thread
PORT          = int(sys.argv[1]) if len(sys.argv) > 1 else 8765

DES_CASES_DIR_PATH = Path("des_cases")
DES_RESULTS_CSV    = Path("des_output/des_results.csv")
MF_OPTIMUM_JSON    = Path("results/mf_optimum_design.json")
MF_PARETO_CSV      = Path("results/mf_pareto_designs.csv")
BO_LOOP_LOG        = Path("bo_loop.log")
LOCAL_SAMPLE_LOG   = Path("local_sampling.log")

F1_LAMBDA = 1.0 / 3.0


# ── data collection (read-only) ───────────────────────────────────────────────

def load_design_matrix():
    if not DESIGN_MATRIX.exists():
        return {}
    params = {}
    with open(DESIGN_MATRIX) as f:
        header = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if header is None:
                header = line.split(",")
                continue
            vals = line.split(",")
            row  = dict(zip(header, vals))
            cid  = row.get("case_id", "")
            params[cid] = {
                "slant":    float(row.get("slant_angle", 0)),
                "diffuser": float(row.get("diffuser_angle", 0)),
                "ride_h":   float(row.get("ride_height", 0)),
                "r_nose":   float(row.get("front_radius", 0)),
            }
    return params


def load_bo_history():
    if not BO_HISTORY.exists():
        return []
    rows = []
    with open(BO_HISTORY) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_all_results():
    points = []
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV) as f:
            for row in csv.DictReader(f):
                try:
                    points.append((float(row["Cd"]), float(row["Cl"]), "doe"))
                except (KeyError, ValueError):
                    pass
    if BO_HISTORY.exists():
        with open(BO_HISTORY) as f:
            for row in csv.DictReader(f):
                try:
                    points.append((float(row["Cd_cfd"]), float(row["Cl_cfd"]), "bo"))
                except (KeyError, ValueError):
                    pass
    if DES_RESULTS_CSV.exists():
        with open(DES_RESULTS_CSV) as f:
            for row in csv.DictReader(f):
                try:
                    if row.get("Cd_DES") and row.get("Cl_DES"):
                        points.append((float(row["Cd_DES"]), float(row["Cl_DES"]), "des"))
                except (KeyError, ValueError):
                    pass
    return points


def load_doe_best():
    if not RESULTS_CSV.exists():
        return None
    best = None
    with open(RESULTS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                f1 = float(row["Cd"]) + F1_LAMBDA * float(row["Cl"])
                if best is None or f1 < best:
                    best = f1
            except (KeyError, ValueError):
                pass
    return best


def load_bo_params():
    import json as _json
    mapping = {}
    if CASES_DIR.exists():
        for d in CASES_DIR.glob("case_bo_*"):
            p = d / "params.json"
            if p.exists():
                try:
                    raw = _json.loads(p.read_text())
                    mapping[d.name] = {
                        "slant":    float(raw["slant_angle"]),
                        "diffuser": float(raw["diffuser_angle"]),
                        "ride_h":   float(raw["ride_height"]),
                        "r_nose":   float(raw["front_radius"]),
                    }
                except (KeyError, ValueError):
                    pass
    if BO_HISTORY.exists():
        with open(BO_HISTORY) as f:
            for row in csv.DictReader(f):
                cid = row.get("case_id", "")
                if not cid or cid in mapping:
                    continue
                try:
                    mapping[cid] = {
                        "slant":    float(row["slant_angle"]),
                        "diffuser": float(row["diffuser_angle"]),
                        "ride_h":   float(row["ride_height"]),
                        "r_nose":   float(row["front_radius"]),
                    }
                except (KeyError, ValueError):
                    pass
    return mapping


def parse_case(case_dir: Path) -> dict:
    cid    = case_dir.name
    result = {"id": cid, "status": "waiting", "iter": 0,
              "cd": None, "cl": None, "elapsed_min": None}

    log = case_dir / "log.simpleFoam"
    if not log.exists():
        if (case_dir / "log.blockMesh").exists():
            result["status"] = "meshing"
        return result

    try:
        text = log.read_text(errors="replace")
    except OSError:
        return result

    iters  = re.findall(r'^Time = (\d+)',             text, re.MULTILINE)
    cds    = re.findall(r'^\s+Cd:\s+([\d.eE+\-]+)',   text, re.MULTILINE)
    cls_   = re.findall(r'^\s+Cl:\s+([\d.eE+\-]+)',   text, re.MULTILINE)
    times  = re.findall(r'ClockTime = (\d+)',          text)
    ended  = 'End\n' in text or text.rstrip().endswith('End')

    result["iter"]        = int(iters[-1])   if iters  else 0
    result["cd"]          = float(cds[-1])   if cds    else None
    result["cl"]          = float(cls_[-1])  if cls_   else None
    result["elapsed_min"] = round(int(times[-1]) / 60, 1) if times else None
    result["status"]      = "done" if ended else "running"
    return result


def _des_phase(case_id: str) -> str:
    """Classify a DES case into pipeline phase by case number."""
    try:
        n = int(case_id.split("_")[-1])
    except (ValueError, IndexError):
        return "unknown"
    if n <= 9:   return "initial"
    if n <= 12:  return "manual_bo"
    if n <= 43:  return "auto_bo"
    return "local"


def load_des_results():
    if not DES_RESULTS_CSV.exists():
        return []
    rows = []
    with open(DES_RESULTS_CSV) as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "case_id":  row["case_id"],
                    "phase":    _des_phase(row["case_id"]),
                    "slant":    float(row["slant_angle"]),
                    "diffuser": float(row["diffuser_angle"]),
                    "ride_h":   float(row["ride_height"]),
                    "fr":       float(row["front_radius"]),
                    "cd_des":   float(row["Cd_DES"])  if row.get("Cd_DES")  else None,
                    "cl_des":   float(row["Cl_DES"])  if row.get("Cl_DES")  else None,
                    "dcd":      float(row["dCd"])      if row.get("dCd")      else None,
                    "dcl":      float(row["dCl"])      if row.get("dCl")      else None,
                })
            except (KeyError, ValueError):
                pass
    return rows


def load_mf_pareto():
    """Load MF surrogate Pareto front for drawing as a curve."""
    if not MF_PARETO_CSV.exists():
        return []
    pts = []
    with open(MF_PARETO_CSV) as f:
        for row in csv.DictReader(f):
            try:
                pts.append({
                    "cd": float(row["Cd_predicted"]),
                    "cl": float(row["Cl_predicted"]),
                })
            except (KeyError, ValueError):
                pass
    return sorted(pts, key=lambda p: p["cd"])


def compute_correction_stats():
    """Return δCd and δCl statistics across all DES cases that have corrections."""
    if not DES_RESULTS_CSV.exists():
        return None
    dcds, dcls = [], []
    with open(DES_RESULTS_CSV) as f:
        for row in csv.DictReader(f):
            try:
                if row.get("dCd") and row.get("dCl"):
                    dcds.append(float(row["dCd"]))
                    dcls.append(float(row["dCl"]))
            except (KeyError, ValueError):
                pass
    if not dcds:
        return None
    import statistics
    return {
        "n":        len(dcds),
        "dcd_mean": round(sum(dcds) / len(dcds), 4),
        "dcd_std":  round(statistics.stdev(dcds), 4),
        "dcd_min":  round(min(dcds), 4),
        "dcd_max":  round(max(dcds), 4),
        "dcl_mean": round(sum(dcls) / len(dcls), 4),
        "dcl_std":  round(statistics.stdev(dcls), 4),
        "dcl_min":  round(min(dcls), 4),
        "dcl_max":  round(max(dcls), 4),
    }


def load_mf_optimum():
    if not MF_OPTIMUM_JSON.exists():
        return None
    try:
        return json.loads(MF_OPTIMUM_JSON.read_text())
    except Exception:
        return None


def load_ei_history():
    ei_vals = []
    for log_path in (BO_LOOP_LOG, LOCAL_SAMPLE_LOG):
        if not log_path.exists():
            continue
        for line in log_path.read_text(errors="replace").splitlines():
            m = re.search(r'New EI = ([\d.eE+\-]+)', line)
            if m:
                try:
                    ei_vals.append(float(m.group(1)))
                except ValueError:
                    pass
    # Remove consecutive duplicates from the double-logging artefact
    deduped = []
    for v in ei_vals:
        if not deduped or abs(v - deduped[-1]) > 1e-10:
            deduped.append(v)
    return deduped


def _active_des_dirs() -> set[str]:
    """Ask Docker which des_case directories are currently mounted."""
    try:
        import subprocess as _sp, json as _json
        out = _sp.check_output(
            ["docker", "ps", "-q"], stderr=_sp.DEVNULL, text=True
        ).split()
        active = set()
        for cid in out:
            info = _sp.check_output(
                ["docker", "inspect", cid], stderr=_sp.DEVNULL, text=True
            )
            for mount in _json.loads(info)[0].get("Mounts", []):
                src = mount.get("Source", "")
                # Match paths ending in des_case_NNN (not _field_init)
                if "/des_case_" in src and "_field_init" not in src:
                    active.add(Path(src).name)
        return active
    except Exception:
        return set()


def parse_active_des_case():
    if not DES_CASES_DIR_PATH.exists():
        return None
    active_dirs = _active_des_dirs()
    for d in sorted(DES_CASES_DIR_PATH.glob("des_case_*"), reverse=True):
        if d.name not in active_dirs:
            continue
        log_path = d / "log.pimpleFoam"
        try:
            text  = log_path.read_text(errors="replace") if log_path.exists() else ""
            times = re.findall(r'^Time = ([\d.eE+\-]+)', text, re.MULTILINE)
            t     = float(times[-1]) if times else 0.0
            ended = 'End\n' in text or text.rstrip().endswith('End')
            if not ended:
                clock   = re.findall(r'ClockTime = (\d+)', text)
                elapsed = round(int(clock[-1]) / 60, 1) if clock else None
                return {
                    "case_id":    d.name,
                    "sim_time":   t,
                    "pct":        round(t / 0.45 * 100, 1),
                    "elapsed_min": elapsed,
                }
        except OSError:
            pass
    return None


MESH_LEVELS = [
    {"name": "L1_coarse", "label": "L1 Coarse (surf 3)"},
    {"name": "L2_medium", "label": "L2 Medium (surf 4)"},
    {"name": "L3_fine",   "label": "L3 Fine   (surf 5)"},
]


def parse_mesh_level(level_name: str) -> dict:
    """Read live state for one mesh convergence level."""
    d = MESH_CONV_DIR / level_name
    result = {
        "name": level_name, "status": "waiting",
        "iter": 0, "cd": None, "cl": None,
        "cells": None, "non_ortho": None, "elapsed_min": None,
    }

    # Final result JSON (written when level completes)
    rj = MESH_CONV_DIR / f"{level_name}_result.json"
    if rj.exists():
        try:
            data = json.loads(rj.read_text())
            result.update({
                "status":    "done",
                "cd":        data.get("Cd"),
                "cl":        data.get("Cl"),
                "cells":     data.get("cells"),
                "non_ortho": data.get("max_non_ortho"),
            })
            # Read final iter and elapsed from log even though case is done
            log = d / "log.simpleFoam"
            if log.exists():
                try:
                    text = log.read_text(errors="replace")
                    iters = re.findall(r'^Time = (\d+)', text, re.MULTILINE)
                    times = re.findall(r'ClockTime = (\d+)', text)
                    if iters:
                        result["iter"] = int(iters[-1])
                    if times:
                        result["elapsed_min"] = round(int(times[-1]) / 60, 1)
                except OSError:
                    pass
            return result
        except Exception:
            pass

    if not d.exists():
        return result

    log = d / "log.simpleFoam"
    if not log.exists():
        if (d / "log.blockMesh").exists() or (d / "log.snappyHexMesh").exists():
            result["status"] = "meshing"
            # Try to get cell count from checkMesh if already done
            cm = d / "log.checkMesh"
            if cm.exists():
                for line in reversed(cm.read_text().splitlines()):
                    if "cells:" in line:
                        try:
                            result["cells"] = int(line.split(":")[1].strip().split()[0])
                            break
                        except (IndexError, ValueError):
                            pass
                for line in cm.read_text().splitlines():
                    if "Mesh non-orthogonality Max:" in line:
                        try:
                            result["non_ortho"] = float(line.split("Max:")[1].split()[0])
                        except (IndexError, ValueError):
                            pass
        return result

    try:
        text = log.read_text(errors="replace")
    except OSError:
        return result

    iters  = re.findall(r'^Time = (\d+)',           text, re.MULTILINE)
    cds    = re.findall(r'^\s+Cd:\s+([\d.eE+\-]+)', text, re.MULTILINE)
    cls_   = re.findall(r'^\s+Cl:\s+([\d.eE+\-]+)', text, re.MULTILINE)
    times  = re.findall(r'ClockTime = (\d+)',        text)
    ended  = 'End\n' in text or text.rstrip().endswith('End')

    result["iter"]        = int(iters[-1])            if iters  else 0
    result["cd"]          = float(cds[-1])            if cds    else None
    result["cl"]          = float(cls_[-1])           if cls_   else None
    result["elapsed_min"] = round(int(times[-1])/60, 1) if times else None
    result["status"]      = "done" if ended else "running"

    # Cell count and non-ortho from checkMesh
    cm = d / "log.checkMesh"
    if cm.exists():
        for line in reversed(cm.read_text().splitlines()):
            if "cells:" in line:
                try:
                    result["cells"] = int(line.split(":")[1].strip().split()[0])
                    break
                except (IndexError, ValueError):
                    pass
        for line in cm.read_text().splitlines():
            if "Mesh non-orthogonality Max:" in line:
                try:
                    result["non_ortho"] = float(line.split("Max:")[1].split()[0])
                except (IndexError, ValueError):
                    pass
    return result


def compute_richardson(levels: list[dict]) -> dict | None:
    """Richardson extrapolation from 3 levels. Returns None if insufficient data."""
    import math
    valid = [(lv["cells"], lv["cd"]) for lv in levels
             if lv.get("cells") and lv.get("cd") is not None]
    if len(valid) < 3:
        return None
    (n3, f3), (n2, f2), (n1, f1) = sorted(valid)
    r21 = (n1 / n2) ** (1/3)
    e21, e32 = f1 - f2, f2 - f3
    if abs(e32) < 1e-10 or abs(e21) < 1e-10:
        return {"cd_extrap": round(f1, 5), "p_order": None, "gci": None}
    try:
        p = abs(math.log(abs(e32 / e21)) / math.log(r21))
    except (ValueError, ZeroDivisionError):
        return None
    cd_extrap = f1 + (f1 - f2) / (r21 ** p - 1)
    gci = 1.25 * abs(e21 / f1) / (r21 ** p - 1) * 100
    return {
        "cd_extrap": round(cd_extrap, 5),
        "p_order":   round(p, 2),
        "gci":       round(gci, 2),
    }


def collect_all():
    dm        = load_design_matrix()
    bo_params = load_bo_params()
    doe_cases, bo_cases = [], []

    for d in sorted(CASES_DIR.glob("case_*")):
        info = parse_case(d)
        if d.name.startswith("case_bo_"):
            info["params"] = bo_params.get(d.name, {})
            bo_cases.append(info)
        else:
            info["params"] = dm.get(d.name, {})
            doe_cases.append(info)

    bo_history = load_bo_history()
    doe_best   = load_doe_best()
    return doe_cases, bo_cases, bo_history, doe_best


# ── data snapshot for SSE ─────────────────────────────────────────────────────

def build_snapshot():
    """Return a dict of all dynamic state. Compared against previous snapshot
    to detect changes; serialised as JSON and pushed to SSE clients."""
    doe_cases, bo_cases, bo_history, doe_best = collect_all()

    n_done    = sum(1 for c in doe_cases if c["status"] == "done")
    n_running = sum(1 for c in doe_cases if c["status"] in ("running", "meshing"))
    n_wait    = sum(1 for c in doe_cases if c["status"] == "waiting")
    n_total   = len(doe_cases)
    doe_pct   = round(n_done / n_total * 100, 1) if n_total else 0.0

    cd_vals   = [c["cd"] for c in doe_cases if c["cd"] is not None and c["status"] == "done"]
    cd_range  = f"{min(cd_vals):.4f} – {max(cd_vals):.4f}" if cd_vals else "—"

    # BO stats
    n_bo       = len([r for r in bo_history if r.get("Cd_cfd", "")])
    bo_phase   = bool(bo_history) or bool(bo_cases)
    bo_best_f1 = None
    bo_best_cd = None
    for row in bo_history:
        try:
            f1 = float(row["f1_cfd"])
            cd = float(row["Cd_cfd"])
            if bo_best_f1 is None or f1 < bo_best_f1:
                bo_best_f1 = f1
                bo_best_cd = cd
        except (KeyError, ValueError):
            pass
    for c in bo_cases:
        if c["cd"] is not None and (bo_best_cd is None or c["cd"] < bo_best_cd):
            bo_best_cd = c["cd"]

    improvement = None
    if bo_best_f1 is not None and doe_best is not None:
        improvement = round(doe_best - bo_best_f1, 4)

    latest_ei = None
    for row in reversed(bo_history):
        try:
            latest_ei = float(row["ei_max"])
            break
        except (KeyError, ValueError):
            pass

    # Mesh convergence levels
    mc_levels = [parse_mesh_level(lv["name"]) for lv in MESH_LEVELS]
    mc_richardson = compute_richardson(mc_levels)

    # Pareto points (raw data — SVG drawn client-side)
    pareto_pts = [{"cd": p[0], "cl": p[1], "src": p[2]} for p in load_all_results()]

    # BO history rows
    bo_hist = []
    for row in bo_history:
        try:
            bo_hist.append({
                "iter":      row.get("iteration", "?"),
                "case_id":   row.get("case_id", "?"),
                "cd_cfd":    float(row["Cd_cfd"]) if row.get("Cd_cfd") else None,
                "f1_cfd":    float(row["f1_cfd"]) if row.get("f1_cfd") else None,
                "f1_pred":   float(row["f1_predicted"]) if row.get("f1_predicted") else None,
                "f1_best":   float(row["f1_best"]) if row.get("f1_best") else None,
                "ei_max":    float(row["ei_max"]) if row.get("ei_max") else None,
                "converged": str(row.get("converged", "")).lower() == "true",
            })
        except (KeyError, ValueError):
            pass

    return {
        "ts":          time.strftime("%H:%M:%S"),
        "doe_pct":     doe_pct,
        "doe_label":   f"{n_done} of {n_total} cases complete",
        "n_done":      n_done,
        "n_running":   n_running,
        "n_wait":      n_wait,
        "n_total":     n_total,
        "cd_range":    cd_range,
        "doe_cases":   doe_cases,
        "bo_phase":    bo_phase,
        "n_bo":        n_bo,
        "bo_best_f1":  bo_best_f1,
        "bo_best_cd":  bo_best_cd,
        "doe_best_f1": doe_best,
        "improvement": improvement,
        "latest_ei":   latest_ei,
        "bo_hist":     bo_hist,
        "bo_cases":    bo_cases,
        "pareto_pts":    pareto_pts,
        "mc_levels":     mc_levels,
        "mc_richardson": mc_richardson,
        "des_results":       load_des_results(),
        "mf_optimum":        load_mf_optimum(),
        "ei_history":        load_ei_history(),
        "active_des":        parse_active_des_case(),
        "mf_pareto":         load_mf_pareto(),
        "correction_stats":  compute_correction_stats(),
    }


# ── SSE broadcast machinery ───────────────────────────────────────────────────

_subscribers: list[queue.Queue] = []
_subscribers_lock = threading.Lock()


def subscribe() -> queue.Queue:
    q = queue.Queue(maxsize=4)
    with _subscribers_lock:
        _subscribers.append(q)
    return q


def unsubscribe(q: queue.Queue):
    with _subscribers_lock:
        try:
            _subscribers.remove(q)
        except ValueError:
            pass


def _broadcast(payload: str):
    with _subscribers_lock:
        live = list(_subscribers)
    for q in live:
        try:
            q.put_nowait(payload)
        except queue.Full:
            pass  # slow client — skip this event, it'll get the next one


def monitor_thread():
    """Watch simulation files for changes and push SSE events on change."""
    prev_snapshot = None
    prev_mtimes: dict[Path, float] = {}

    def watched_files():
        files = [DESIGN_MATRIX, BO_HISTORY, RESULTS_CSV,
                 DES_RESULTS_CSV, MF_OPTIMUM_JSON, BO_LOOP_LOG, LOCAL_SAMPLE_LOG]
        if CASES_DIR.exists():
            files += list(CASES_DIR.glob("*/log.simpleFoam"))
            files += list(CASES_DIR.glob("*/log.blockMesh"))
        if MESH_CONV_DIR.exists():
            files += list(MESH_CONV_DIR.glob("*/log.simpleFoam"))
            files += list(MESH_CONV_DIR.glob("*/log.blockMesh"))
            files += list(MESH_CONV_DIR.glob("*/log.snappyHexMesh"))
            files += list(MESH_CONV_DIR.glob("*_result.json"))
        if DES_CASES_DIR_PATH.exists():
            files += list(DES_CASES_DIR_PATH.glob("*/log.pimpleFoam"))
        return files

    while True:
        time.sleep(POLL_INTERVAL)
        try:
            # Check if any watched file changed
            changed = False
            for f in watched_files():
                try:
                    mtime = f.stat().st_mtime
                except OSError:
                    continue
                if prev_mtimes.get(f) != mtime:
                    prev_mtimes[f] = mtime
                    changed = True

            if not changed:
                continue

            snap = build_snapshot()
            # Only push if something the client cares about actually changed
            comparable = {k: v for k, v in snap.items() if k != "ts"}
            if comparable == prev_snapshot:
                continue
            prev_snapshot = comparable

            payload = f"data: {json.dumps(snap)}\n\n"
            _broadcast(payload)

        except Exception:
            pass  # never crash the monitor thread


# ── initial page HTML ─────────────────────────────────────────────────────────

def initial_html(snap: dict) -> str:
    doe_cases  = snap["doe_cases"]
    doe_pct    = snap["doe_pct"]

    # Build initial table rows (static IDs, JS will patch values)
    doe_rows = ""
    for c in doe_cases:
        pct    = min(c["iter"] / 1800, 1.0) * 100
        status = c["status"]
        color  = {"done": "#22c55e", "running": "#3b82f6",
                  "meshing": "#f59e0b", "waiting": "#6b7280"}[status]
        label  = status.upper()
        p      = c.get("params", {})
        params_html = ""
        if p:
            params_html = (
                f'<span class="param">α={p["slant"]:.0f}°</span>'
                f'<span class="param">δ={p["diffuser"]:.0f}°</span>'
                f'<span class="param">h={p["ride_h"]:.0f}mm</span>'
                f'<span class="param">R={p["r_nose"]:.0f}mm</span>'
            )
        cid = c["id"]
        bar_class = "bar-fill"
        if status == "meshing":
            bar_class += " bar-mesh"
        elif status == "done":
            bar_class += " bar-done"
        doe_rows += f"""<tr id="row-{cid}" class="{status}">
  <td class="mono">{cid}</td>
  <td><span id="{cid}-badge" class="badge" style="background:{color}">{label}</span></td>
  <td><div class="bar-wrap">
    <div id="{cid}-bar" class="{bar_class}" style="width:{pct:.1f}%"></div>
    <span id="{cid}-iter" class="bar-label">{c["iter"]}/3000</span>
  </div></td>
  <td class="mono cd"><span id="{cid}-cd">{"%.4f" % c["cd"] if c["cd"] is not None else "—"}</span></td>
  <td class="mono"><span id="{cid}-cl">{"%.4f" % c["cl"] if c["cl"] is not None else "—"}</span></td>
  <td class="params">{params_html}</td>
  <td class="mono dim"><span id="{cid}-elapsed">{"%.1f min" % c["elapsed_min"] if c["elapsed_min"] else "—"}</span></td>
</tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title id="page-title">Ahmed Body CFD — DoE {snap["n_done"]}/{snap["n_total"]}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #0f172a; color: #e2e8f0; font-size: 13px; padding: 24px; }}
  h1   {{ font-size: 20px; font-weight: 600; color: #f8fafc; margin-bottom: 4px; }}
  .sub {{ color: #94a3b8; font-size: 12px; margin-bottom: 24px; }}

  .section-header {{
    display: flex; align-items: baseline; gap: 12px;
    font-size: 13px; font-weight: 700; color: #f8fafc;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin: 28px 0 12px;
    padding-bottom: 6px; border-bottom: 1px solid #1e293b;
  }}
  .section-sub {{ font-size: 11px; font-weight: 400;
                  text-transform: none; color: #64748b; letter-spacing: 0; }}
  .bo-header {{ color: #a78bfa; border-color: #312e81; }}

  .cards {{ display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }}
  .card {{ background: #1e293b; border-radius: 10px; padding: 16px 20px; min-width: 120px; }}
  .card .num {{ font-size: 26px; font-weight: 700; }}
  .card .lbl {{ font-size: 11px; color: #94a3b8; margin-top: 2px; }}
  .card.done .num {{ color: #22c55e; }}
  .card.run  .num {{ color: #3b82f6; }}
  .card.wait .num {{ color: #6b7280; }}
  .card.cd   .num {{ color: #f8fafc; font-size: 18px; }}
  .card.bo   .num {{ color: #a78bfa; }}

  .overall {{ background: #1e293b; border-radius: 10px;
              padding: 14px 20px; margin-bottom: 20px; }}
  .overall-label {{ font-size: 12px; color: #94a3b8; margin-bottom: 8px; }}
  .overall-bar  {{ background: #0f172a; border-radius: 4px; height: 8px; }}
  .overall-fill {{ background: #22c55e; height: 8px; border-radius: 4px; transition: width 0.6s ease; }}
  .overall-pct  {{ font-size: 11px; color: #94a3b8; margin-top: 5px; }}

  table {{ width: 100%; border-collapse: collapse; background: #1e293b;
           border-radius: 10px; overflow: hidden; margin-bottom: 8px; }}
  th  {{ background: #0f172a; padding: 10px 12px; text-align: left;
         font-size: 11px; color: #64748b; font-weight: 600;
         text-transform: uppercase; letter-spacing: 0.05em; }}
  td  {{ padding: 9px 12px; border-bottom: 1px solid #0f172a; vertical-align: middle; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #263245; }}
  tr.bo-best td {{ background: #1e1b4b !important; }}

  .mono  {{ font-family: "SF Mono", "Fira Code", monospace; }}
  .dim   {{ color: #64748b; }}
  .cd    {{ color: #f8fafc; font-weight: 600; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-size: 10px; font-weight: 700; color: #fff;
            text-transform: uppercase; letter-spacing: 0.05em; }}

  .bar-wrap {{ position: relative; background: #0f172a; border-radius: 4px;
               height: 18px; min-width: 160px; }}
  .bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.6s ease; background: #3b82f6; }}
  .bar-done  {{ background: #22c55e; }}
  .bar-mesh  {{
    background: repeating-linear-gradient(
      45deg, #f59e0b, #f59e0b 4px, #78350f 4px, #78350f 8px);
    animation: slide 1s linear infinite;
  }}
  @keyframes slide {{ from {{ background-position:0 0 }} to {{ background-position:16px 0 }} }}

  .bar-label {{ position: absolute; inset: 0; display: flex; align-items: center;
                justify-content: center; font-size: 10px; font-family: monospace;
                color: #cbd5e1; mix-blend-mode: screen; }}

  .params {{ display: flex; gap: 4px; flex-wrap: wrap; }}
  .param  {{ background: #0f172a; border-radius: 3px; padding: 1px 5px;
             font-size: 10px; color: #94a3b8; font-family: monospace; }}

  @keyframes flash {{ 0%,100% {{ background: transparent }} 50% {{ background: rgba(59,130,246,.25) }} }}
  .flash {{ animation: flash 0.5s ease; border-radius: 3px; }}

  .dot-live {{ display:inline-block; width:7px; height:7px; border-radius:50%;
               background:#22c55e; margin-right:6px;
               animation: pulse 2s ease-in-out infinite; }}
  @keyframes pulse {{ 0%,100% {{ opacity:1; transform:scale(1) }}
                      50%  {{ opacity:.4; transform:scale(.7) }} }}

  .status-bar {{ font-size: 11px; color: #475569; margin-top: 20px; text-align: right; }}
</style>
</head>
<body>
<h1>Ahmed Body CFD Pipeline</h1>
<div class="sub">
  <span class="dot-live"></span>
  Live — updates when simulation data changes &nbsp;·&nbsp;
  Last event: <span id="last-ts">{snap["ts"]}</span>
  &nbsp;·&nbsp; Read-only
</div>

<!-- Pipeline overview -->
<div id="pipeline-overview"></div>

<!-- Mesh convergence section -->
<div id="mc-section"></div>

<!-- DoE section -->
<div class="section-header">
  <span>Design of Experiments</span>
  <span class="section-sub">30-case Latin Hypercube · k-ω SST low-Re · 2 parallel workers · 6 cores each</span>
</div>

<div class="cards" id="doe-cards">
  <div class="card done"><div class="num" id="card-done">{snap["n_done"]}</div><div class="lbl">Completed</div></div>
  <div class="card run" ><div class="num" id="card-running">{snap["n_running"]}</div><div class="lbl">Running</div></div>
  <div class="card wait"><div class="num" id="card-wait">{snap["n_wait"]}</div><div class="lbl">Waiting</div></div>
  <div class="card cd"  ><div class="num" id="card-cd-range">{snap["cd_range"]}</div><div class="lbl">Cd range (done)</div></div>
</div>

<div class="overall">
  <div class="overall-label" id="overall-label">DoE progress — {snap["doe_label"]}</div>
  <div class="overall-bar"><div class="overall-fill" id="overall-fill" style="width:{doe_pct:.1f}%"></div></div>
  <div class="overall-pct" id="overall-pct">{doe_pct:.0f}%</div>
</div>

<table>
  <thead>
    <tr>
      <th>Case</th><th>Status</th><th style="min-width:200px">Progress</th>
      <th>Cd</th><th>Cl</th><th>Parameters</th><th>Elapsed</th>
    </tr>
  </thead>
  <tbody id="doe-tbody">{doe_rows}</tbody>
</table>

<!-- DES / co-Kriging + BO section -->
<div id="des-section"></div>

<!-- MF Pareto front (after DES so it reflects co-Kriging output) -->
<div id="pareto-container"></div>

<div class="status-bar" id="status-bar">Connected — waiting for data</div>

<script>
const STATUS_COLOR = {{
  done: "#22c55e", running: "#3b82f6", meshing: "#f59e0b", waiting: "#6b7280"
}};
const STATUS_LABEL = {{
  done: "DONE", running: "RUNNING", meshing: "MESHING", waiting: "WAITING"
}};

function setText(id, val) {{
  const el = document.getElementById(id);
  if (!el || el.textContent === String(val)) return;
  el.textContent = val;
  el.classList.remove("flash");
  void el.offsetWidth;
  el.classList.add("flash");
}}

function setHtml(id, val) {{
  const el = document.getElementById(id);
  if (el) el.innerHTML = val;
}}

function setStyle(id, prop, val) {{
  const el = document.getElementById(id);
  if (el) el.style[prop] = val;
}}

// ── Pareto SVG (drawn client-side) ──────────────────────────────────────────
function paretoFront(pts) {{
  const dominated = new Array(pts.length).fill(false);
  for (let i = 0; i < pts.length; i++) {{
    for (let j = 0; j < pts.length; j++) {{
      if (i === j) continue;
      if (pts[j][0] <= pts[i][0] && pts[j][1] <= pts[i][1] &&
          (pts[j][0] < pts[i][0] || pts[j][1] < pts[i][1])) {{
        dominated[i] = true; break;
      }}
    }}
  }}
  return dominated.map((d, i) => !d ? i : -1).filter(i => i >= 0);
}}

function renderPareto(pts, mfPareto) {{
  if (!pts || pts.length === 0) {{ setHtml("pareto-container", ""); return; }}

  const W = 520, H = 320;
  const PAD = {{ l: 54, r: 20, t: 20, b: 44 }};
  const pw = W - PAD.l - PAD.r, ph = H - PAD.t - PAD.b;

  const cds = pts.map(p => p.cd), cls = pts.map(p => p.cl);
  const cdMin = Math.min(...cds), cdMax = Math.max(...cds);
  const clMin = Math.min(...cls), clMax = Math.max(...cls);
  const cdPad = Math.max((cdMax - cdMin) * 0.08, 0.005);
  const clPad = Math.max((clMax - clMin) * 0.08, 0.05);
  const xlo = cdMin - cdPad, xhi = cdMax + cdPad;
  const ylo = clMin - clPad, yhi = clMax + clPad;

  const tx = cd => PAD.l + (cd - xlo) / (xhi - xlo) * pw;
  const ty = cl => PAD.t + ph - (cl - ylo) / (yhi - ylo) * ph;

  const ptsArr = pts.map(p => [p.cd, p.cl, p.src]);
  const pfIdx  = new Set(paretoFront(ptsArr));

  let grid = "";
  for (let i = 0; i <= 5; i++) {{
    const xv = xlo + i / 5 * (xhi - xlo);
    const yv = ylo + i / 5 * (yhi - ylo);
    const gx = tx(xv), gy = ty(yv);
    grid += `<line x1="${{gx.toFixed(1)}}" y1="${{PAD.t}}" x2="${{gx.toFixed(1)}}" y2="${{PAD.t+ph}}" stroke="#1e293b" stroke-width="1"/>`;
    grid += `<line x1="${{PAD.l}}" y1="${{gy.toFixed(1)}}" x2="${{PAD.l+pw}}" y2="${{gy.toFixed(1)}}" stroke="#1e293b" stroke-width="1"/>`;
    grid += `<text x="${{gx.toFixed(1)}}" y="${{PAD.t+ph+14}}" text-anchor="middle" font-size="9" fill="#475569">${{xv.toFixed(3)}}</text>`;
    grid += `<text x="${{PAD.l-6}}" y="${{(gy+3).toFixed(1)}}" text-anchor="end" font-size="9" fill="#475569">${{yv.toFixed(2)}}</text>`;
  }}

  const pfPts = [...pfIdx].map(i => ptsArr[i]).sort((a, b) => a[0] - b[0]);
  let stepPath = "";
  if (pfPts.length > 1) {{
    const coords = pfPts.map(p => [tx(p[0]), ty(p[1])]);
    let d = `M ${{coords[0][0].toFixed(1)}} ${{coords[0][1].toFixed(1)}}`;
    for (let k = 1; k < coords.length; k++)
      d += ` L ${{coords[k-1][0].toFixed(1)}} ${{coords[k][1].toFixed(1)}} L ${{coords[k][0].toFixed(1)}} ${{coords[k][1].toFixed(1)}}`;
    stepPath = `<path d="${{d}}" fill="none" stroke="#22c55e" stroke-width="1.5" stroke-dasharray="4 2" opacity="0.8"/>`;
  }}

  let dots = "";
  ptsArr.forEach(([cd, cl, src], i) => {{
    const x = tx(cd), y = ty(cl), onPf = pfIdx.has(i);
    const colour = src === "des" ? (onPf ? "#f97316" : "#fb923c")
                 : src === "bo"  ? (onPf ? "#7c3aed" : "#a78bfa")
                 :                 (onPf ? "#22c55e" : "#475569");
    const r = src === "des" ? 5 : src === "bo" ? 5 : 4;
    const stroke = onPf ? "stroke='#fff' stroke-width='1'" : "";
    const f1 = cd + {F1_LAMBDA} * cl;
    const tip = `Cd=${{cd.toFixed(4)}} Cl=${{cl.toFixed(4)}} f₁=${{f1.toFixed(4)}} (${{src.toUpperCase()}})${{onPf ? " ★" : ""}}`;
    dots += `<circle cx="${{x.toFixed(1)}}" cy="${{y.toFixed(1)}}" r="${{r}}" fill="${{colour}}" ${{stroke}} opacity="0.9"><title>${{tip}}</title></circle>`;
  }});

  // MF surrogate Pareto curve
  let mfCurve = "";
  if (mfPareto && mfPareto.length > 1) {{
    const mfSorted = [...mfPareto].sort((a, b) => a.cd - b.cd);
    const coords   = mfSorted.map(p => [tx(p.cd), ty(p.cl)]);
    let d = `M ${{coords[0][0].toFixed(1)}} ${{coords[0][1].toFixed(1)}}`;
    for (let k = 1; k < coords.length; k++)
      d += ` L ${{coords[k][0].toFixed(1)}} ${{coords[k][1].toFixed(1)}}`;
    mfCurve = `<path d="${{d}}" fill="none" stroke="#a78bfa" stroke-width="2" opacity="0.9"/>`;
    // End-cap dots
    mfSorted.forEach(p => {{
      mfCurve += `<circle cx="${{tx(p.cd).toFixed(1)}}" cy="${{ty(p.cl).toFixed(1)}}" r="3" fill="#a78bfa" opacity="0.7"><title>MF Pareto: Cd=${{p.cd.toFixed(4)}} Cl=${{p.cl.toFixed(4)}}</title></circle>`;
    }});
  }}

  const nDoe = pts.filter(p => p.src === "doe").length;
  const nDes = pts.filter(p => p.src === "des").length;
  const nMf  = mfPareto ? mfPareto.length : 0;

  setHtml("pareto-container", `
<div style="margin:16px 0 8px">
  <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px">
    Multi-fidelity Pareto Front &nbsp;·&nbsp; ${{nDoe}} RANS + ${{nDes}} DES + ${{nMf}}-pt MF curve &nbsp;·&nbsp; hover for values
  </div>
  <svg width="${{W}}" height="${{H}}" style="background:#1e293b;border-radius:10px;display:block">
    ${{grid}}${{stepPath}}${{mfCurve}}${{dots}}
    <text x="${{PAD.l+pw/2}}" y="${{H-4}}" text-anchor="middle" font-size="10" fill="#64748b">Drag coefficient Cd →</text>
    <text transform="rotate(-90)" x="${{-(PAD.t+ph/2)}}" y="12" text-anchor="middle" font-size="10" fill="#64748b">Lift coefficient Cl (↓ = downforce)</text>
    <circle cx="${{PAD.l+8}}"   cy="${{PAD.t+8}}" r="4" fill="#475569"/>
    <text x="${{PAD.l+16}}"  y="${{PAD.t+12}}" font-size="9" fill="#94a3b8">RANS DoE</text>
    <circle cx="${{PAD.l+62}}"  cy="${{PAD.t+8}}" r="5" fill="#fb923c"/>
    <text x="${{PAD.l+70}}"  y="${{PAD.t+12}}" font-size="9" fill="#94a3b8">DES</text>
    <line x1="${{PAD.l+100}}" y1="${{PAD.t+8}}" x2="${{PAD.l+116}}" y2="${{PAD.t+8}}" stroke="#a78bfa" stroke-width="2"/>
    <text x="${{PAD.l+120}}" y="${{PAD.t+12}}" font-size="9" fill="#94a3b8">MF Pareto</text>
    <circle cx="${{PAD.l+166}}" cy="${{PAD.t+8}}" r="4" fill="#22c55e" stroke="#fff" stroke-width="1"/>
    <text x="${{PAD.l+174}}" y="${{PAD.t+12}}" font-size="9" fill="#94a3b8">RANS Pareto</text>
  </svg>
</div>`);
}}

// ── Pipeline overview ────────────────────────────────────────────────────────
function renderPipelineOverview(d) {{
  const nDes  = (d.des_results || []).length;
  const nEI   = (d.ei_history  || []).length;
  const curEI = nEI > 0 ? d.ei_history[nEI - 1] : null;
  const eiCol = curEI === null ? "#6b7280" : curEI < 0.005 ? "#22c55e" : curEI < 0.015 ? "#f59e0b" : "#f87171";

  const phases = [
    {{ label: "Mesh Verification",  detail: "GCI = 0.89%",                    color: "#22c55e",  done: true }},
    {{ label: "RANS DoE",           detail: d.n_done + " / " + d.n_total + " cases", color: "#3b82f6", done: d.n_done === d.n_total }},
    {{ label: "DES Campaign",       detail: nDes + " HF evaluations",          color: "#fb923c",  done: nDes > 0 }},
    {{ label: "Co-Kriging Surrogate", detail: d.mf_optimum ? "fitted" : "pending", color: "#a78bfa", done: !!d.mf_optimum }},
    {{ label: "Bayesian Optimisation", detail: nEI + " iterations · EI " + (curEI !== null ? curEI.toFixed(4) : "—"), color: eiCol, done: curEI !== null && curEI < 0.005 }},
  ];

  const steps = phases.map((p, i) => `
    <div style="display:flex;align-items:center;gap:8px;flex:1;min-width:140px">
      ${{i > 0 ? `<div style="height:2px;flex:0 0 16px;background:${{phases[i-1].done ? "#22c55e" : "#1e293b"}}"></div>` : ""}}
      <div style="background:#1e293b;border-radius:8px;padding:10px 14px;flex:1;border-left:3px solid ${{p.color}}">
        <div style="font-size:10px;font-weight:700;color:${{p.color}};text-transform:uppercase;letter-spacing:.05em">${{p.label}}</div>
        <div style="font-size:11px;color:#94a3b8;margin-top:2px">${{p.detail}}</div>
      </div>
    </div>`).join("");

  setHtml("pipeline-overview", `
<div style="margin-bottom:20px">
  <div style="font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px">Pipeline</div>
  <div style="display:flex;gap:0;align-items:stretch;flex-wrap:wrap">${{steps}}</div>
</div>`);
}}

// ── Mesh convergence section ─────────────────────────────────────────────────
function fmtCells(n) {{
  if (n === null || n === undefined) return "—";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return Math.round(n / 1e3) + "k";
  return String(n);
}}

function renderMeshConv(d) {{
  const levels = d.mc_levels;
  const rich   = d.mc_richardson;

  if (!levels || levels.length === 0) {{ setHtml("mc-section", ""); return; }}

  const LEVEL_LABELS = {{
    L1_coarse: "L1 Coarse (surf 3)",
    L2_medium: "L2 Medium (surf 4)",
    L3_fine:   "L3 Fine   (surf 5)",
  }};

  let lvlRows = "";
  levels.forEach(lv => {{
    const label  = LEVEL_LABELS[lv.name] || lv.name;
    const color  = STATUS_COLOR[lv.status] || "#6b7280";
    const badge  = STATUS_LABEL[lv.status] || lv.status.toUpperCase();
    const pct    = lv.status === "meshing" ? 100 : Math.min(lv.iter / 3000, 1.0) * 100;
    const barCls = lv.status === "done"    ? "bar-fill bar-done"
                 : lv.status === "meshing" ? "bar-fill bar-mesh"
                 : "bar-fill";
    const iterLbl = lv.status === "meshing" ? "meshing…"
                  : lv.status === "waiting"  ? "waiting"
                  : lv.iter + "/3000";
    lvlRows += `<tr>
      <td class="mono" style="color:#67e8f9">${{label}}</td>
      <td><span class="badge" style="background:${{color}}">${{badge}}</span></td>
      <td><div class="bar-wrap" style="min-width:180px">
        <div class="${{barCls}}" style="width:${{pct.toFixed(1)}}%"></div>
        <span class="bar-label">${{iterLbl}}</span>
      </div></td>
      <td class="mono cd">${{lv.cd   !== null ? lv.cd.toFixed(4)   : "—"}}</td>
      <td class="mono"   >${{lv.cl   !== null ? lv.cl.toFixed(4)   : "—"}}</td>
      <td class="mono dim">${{fmtCells(lv.cells)}}</td>
      <td class="mono dim">${{lv.non_ortho !== null ? lv.non_ortho.toFixed(1) + "°" : "—"}}</td>
      <td class="mono dim">${{lv.elapsed_min !== null ? lv.elapsed_min.toFixed(1) + " min" : "—"}}</td>
    </tr>`;
  }});

  let richHtml = "";
  if (rich) {{
    const gciOk     = rich.gci !== null && rich.gci < 5.0;
    const gciStr    = rich.gci     !== null ? rich.gci.toFixed(2) + "%" : "—";
    const pStr      = rich.p_order !== null ? rich.p_order.toFixed(2)   : "—";
    const vColor    = gciOk ? "#22c55e" : "#f59e0b";
    const vText     = gciOk
      ? "L2 VALID — GCI &lt; 5%"
      : `GCI = ${{gciStr}} — REVIEW BEFORE PROCEEDING`;
    richHtml = `
<div style="background:#0c2d3d;border:1px solid #164e63;border-radius:8px;padding:14px 18px;margin-top:10px;display:flex;gap:32px;align-items:center;flex-wrap:wrap">
  <div style="font-size:11px;color:#67e8f9;text-transform:uppercase;letter-spacing:.06em;width:100%;margin-bottom:4px">Richardson Extrapolation</div>
  <div class="card" style="min-width:0;padding:10px 16px">
    <div class="num" style="font-size:18px;color:#f8fafc">${{rich.cd_extrap.toFixed(5)}}</div>
    <div class="lbl">Cd extrapolated</div>
  </div>
  <div class="card" style="min-width:0;padding:10px 16px">
    <div class="num" style="font-size:18px;color:#94a3b8">${{pStr}}</div>
    <div class="lbl">Order of accuracy p</div>
  </div>
  <div class="card" style="min-width:0;padding:10px 16px">
    <div class="num" style="font-size:18px;color:${{gciOk ? "#22c55e" : "#f59e0b"}}">${{gciStr}}</div>
    <div class="lbl">GCI (L2 → L3)</div>
  </div>
  <div style="margin-left:auto">
    <span class="badge" style="background:${{vColor}};font-size:11px;padding:5px 14px">${{vText}}</span>
  </div>
</div>`;
  }}

  setHtml("mc-section", `
<div class="section-header" style="color:#67e8f9;border-color:#164e63">
  <span>Mesh Convergence Study</span>
  <span class="section-sub">k-ω SST low-Re &nbsp;·&nbsp; y&#x207a; &lt; 1 &nbsp;·&nbsp; 15 BL layers &nbsp;·&nbsp; Richardson extrapolation</span>
</div>
<table style="margin-bottom:0">
  <thead><tr>
    <th>Level</th><th>Status</th><th style="min-width:200px">Progress</th>
    <th>Cd</th><th>Cl</th><th>Cells</th><th>Max non-ortho</th><th>Elapsed</th>
  </tr></thead>
  <tbody>${{lvlRows}}</tbody>
</table>
${{richHtml}}`);
}}

// ── BO section ───────────────────────────────────────────────────────────────
function renderBO(d) {{
  if (!d.bo_phase) {{
    setHtml("bo-section", `
<div class="section-header bo-header">
  <span>Bayesian Optimisation Loop</span>
  <span class="section-sub">Not yet started — run: python3 surrogate_optimiser.py --bayesian-loop 30</span>
</div>`);
    return;
  }}

  const imprStr = d.improvement > 0
    ? `<span style="color:#22c55e;font-weight:700">▼ ${{d.improvement.toFixed(4)}} vs DoE best</span>`
    : `<span style="color:#94a3b8">no improvement yet</span>`;
  const eiStr = d.latest_ei !== null ? d.latest_ei.toExponential(2) : "—";

  let histRows = "";
  const bestF1 = d.bo_best_f1;
  d.bo_hist.forEach(r => {{
    const isBest = bestF1 !== null && r.f1_cfd !== null && Math.abs(r.f1_cfd - bestF1) < 1e-6;
    const conv = r.converged
      ? '<span class="badge" style="background:#22c55e">YES</span>'
      : '<span class="badge" style="background:#ef4444">NO</span>';
    histRows += `<tr${{isBest ? ' class="bo-best"' : ''}}>
      <td class="mono dim">${{r.iter}}</td>
      <td class="mono">${{r.case_id}}</td>
      <td class="mono cd">${{r.cd_cfd !== null ? r.cd_cfd.toFixed(4) : "—"}}</td>
      <td class="mono cd">${{r.f1_cfd !== null ? r.f1_cfd.toFixed(4) : "—"}}</td>
      <td class="mono dim">${{r.f1_pred !== null ? r.f1_pred.toFixed(4) : "—"}}</td>
      <td class="mono">${{r.f1_best !== null ? r.f1_best.toFixed(4) : "—"}}</td>
      <td class="mono dim">${{r.ei_max !== null ? r.ei_max.toExponential(2) : "—"}}</td>
      <td>${{conv}}</td>
    </tr>`;
  }});

  let liveRows = "";
  d.bo_cases.forEach(c => {{
    const pct = Math.min(c.iter / 3000, 1.0) * 100;
    const color = STATUS_COLOR[c.status] || "#6b7280";
    const label = STATUS_LABEL[c.status] || c.status.toUpperCase();
    liveRows += `<tr class="${{c.status}}">
      <td class="mono">${{c.id}} <span class="badge" style="background:#7c3aed;margin-left:4px">10c</span></td>
      <td><span class="badge" style="background:${{color}}">${{label}}</span></td>
      <td><div class="bar-wrap">
        <div class="bar-fill${{c.status==="done"?" bar-done":""}}" style="width:${{pct.toFixed(1)}}%"></div>
        <span class="bar-label">${{c.iter}}/3000</span>
      </div></td>
      <td class="mono cd">${{c.cd !== null ? c.cd.toFixed(4) : "—"}}</td>
      <td class="mono">${{c.cl !== null ? c.cl.toFixed(4) : "—"}}</td>
      <td></td>
      <td class="mono dim">${{c.elapsed_min !== null ? c.elapsed_min.toFixed(1)+" min" : "—"}}</td>
    </tr>`;
  }});

  setHtml("bo-section", `
<div class="section-header bo-header">
  <span>Bayesian Optimisation Loop</span>
  <span class="section-sub">${{d.n_bo}} iteration${{d.n_bo !== 1 ? "s" : ""}} complete · objective: f = C&#x2093; + ⅓·C&#x2097; · EI acquisition</span>
</div>
<div class="cards" style="margin-top:0">
  <div class="card bo"><div class="num">${{d.n_bo}}</div><div class="lbl">BO iterations</div></div>
  <div class="card bo"><div class="num">${{d.bo_best_f1 !== null ? d.bo_best_f1.toFixed(4) : "—"}}</div><div class="lbl">Best f₁ (BO)</div></div>
  <div class="card bo"><div class="num" style="font-size:14px">${{imprStr}}</div><div class="lbl">vs DoE f₁ (${{d.doe_best_f1 !== null ? d.doe_best_f1.toFixed(4) : "—"}})</div></div>
  <div class="card bo"><div class="num" style="font-size:14px">${{eiStr}}</div><div class="lbl">Latest EI</div></div>
</div>
${{histRows ? `<table style="margin-bottom:16px"><thead><tr>
  <th>#</th><th>Case</th><th>Cd (CFD)</th><th>f₁ = Cd+⅓Cl</th><th>f₁ predicted</th>
  <th>Best f₁</th><th>EI</th><th>Conv</th>
</tr></thead><tbody>${{histRows}}</tbody></table>` : ""}}
${{liveRows ? `<table style="margin-bottom:16px"><thead><tr>
  <th>Case</th><th>Status</th><th style="min-width:200px">Progress</th>
  <th>Cd</th><th>Cl</th><th>Parameters</th><th>Elapsed</th>
</tr></thead><tbody>${{liveRows}}</tbody></table>` : ""}}`);
}}

// ── DES / co-Kriging section ─────────────────────────────────────────────────
const PHASE_BADGE = {{
  "initial":   {{ label: "Initial",    color: "#0369a1" }},
  "manual_bo": {{ label: "Manual BO",  color: "#7c3aed" }},
  "auto_bo":   {{ label: "Auto BO",    color: "#6d28d9" }},
  "local":     {{ label: "Local",      color: "#065f46" }},
  "unknown":   {{ label: "?",          color: "#475569" }},
}};

function renderDES(d) {{
  const des  = d.des_results || [];
  const opt  = d.mf_optimum;
  const ei   = d.ei_history  || [];
  const act  = d.active_des;
  const corr = d.correction_stats;

  // Active case progress
  let activeHtml = "";
  if (act) {{
    activeHtml = `
    <div style="background:#1e293b;border-radius:8px;padding:12px 16px;margin-bottom:12px">
      <div style="font-size:11px;color:#a78bfa;margin-bottom:6px">▶ Running: ${{act.case_id}}</div>
      <div class="bar-wrap" style="min-width:340px">
        <div class="bar-fill" style="width:${{act.pct}}%;background:#a78bfa"></div>
        <span class="bar-label">t=${{act.sim_time.toFixed(3)}} / 0.450 s &nbsp;(${{act.pct}}%)</span>
      </div>
      ${{act.elapsed_min ? `<div style="font-size:10px;color:#64748b;margin-top:4px">${{act.elapsed_min}} min elapsed</div>` : ""}}
    </div>`;
  }}

  // Phase counts
  const phaseCounts = {{}};
  des.forEach(r => {{ phaseCounts[r.phase] = (phaseCounts[r.phase] || 0) + 1; }});
  const phaseCards = Object.entries(phaseCounts).map(([ph, n]) => {{
    const b = PHASE_BADGE[ph] || PHASE_BADGE["unknown"];
    return `<div class="card" style="min-width:0;padding:10px 14px;border-left:3px solid ${{b.color}}">
      <div class="num" style="font-size:18px;color:#f8fafc">${{n}}</div>
      <div class="lbl">${{b.label}}</div>
    </div>`;
  }}).join("");

  // Correction statistics
  let corrHtml = "";
  if (corr) {{
    const dcdCol = Math.abs(corr.dcd_mean) > 0.02 ? "#f87171" : "#94a3b8";
    corrHtml = `
    <div style="background:#1e293b;border-radius:8px;padding:12px 16px;margin-bottom:12px">
      <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px">
        DES correction δ = f<sub>HF</sub> − f<sub>LF</sub> &nbsp;·&nbsp; ${{corr.n}} points
      </div>
      <div style="display:flex;gap:24px;flex-wrap:wrap">
        <div>
          <div style="font-size:11px;color:#64748b;margin-bottom:4px">δCd</div>
          <div style="font-size:14px;font-weight:700;color:${{dcdCol}}">
            ${{corr.dcd_mean >= 0 ? "+" : ""}}${{corr.dcd_mean.toFixed(4)}} <span style="font-size:10px;color:#64748b">± ${{corr.dcd_std.toFixed(4)}}</span>
          </div>
          <div style="font-size:10px;color:#475569">range [${{corr.dcd_min.toFixed(4)}}, ${{corr.dcd_max >= 0 ? "+" : ""}}${{corr.dcd_max.toFixed(4)}}]</div>
        </div>
        <div>
          <div style="font-size:11px;color:#64748b;margin-bottom:4px">δCl</div>
          <div style="font-size:14px;font-weight:700;color:#94a3b8">
            ${{corr.dcl_mean >= 0 ? "+" : ""}}${{corr.dcl_mean.toFixed(4)}} <span style="font-size:10px;color:#64748b">± ${{corr.dcl_std.toFixed(4)}}</span>
          </div>
          <div style="font-size:10px;color:#475569">range [${{corr.dcl_min.toFixed(4)}}, ${{corr.dcl_max >= 0 ? "+" : ""}}${{corr.dcl_max.toFixed(4)}}]</div>
        </div>
        <div style="margin-left:auto;font-size:10px;color:#475569;align-self:flex-end">
          RANS systematically ${{corr.dcd_mean > 0 ? "under-predicts" : "over-predicts"}} Cd<br>
          by mean |δCd| = ${{Math.abs(corr.dcd_mean).toFixed(4)}}
        </div>
      </div>
    </div>`;
  }}

  // EI convergence bar chart
  let eiHtml = "";
  if (ei.length > 0) {{
    const eiMax = Math.max(...ei);
    const eiMin = Math.min(...ei);
    const latest = ei[ei.length - 1];
    const col = latest < 0.005 ? "#22c55e" : latest < 0.015 ? "#f59e0b" : "#f87171";
    const bars = ei.map((v, i) => {{
      const h = eiMax > 0 ? Math.max(4, v / eiMax * 56) : 4;
      const c = v < 0.005 ? "#22c55e" : v < 0.015 ? "#f59e0b" : "#f87171";
      return `<div title="Iter ${{i+1}}: EI=${{v.toFixed(5)}}" style="display:inline-flex;flex-direction:column;align-items:center;margin:0 1px;cursor:default">
        <div style="width:7px;height:${{h.toFixed(0)}}px;background:${{c}};border-radius:2px 2px 0 0"></div>
      </div>`;
    }}).join("");
    eiHtml = `
    <div style="background:#1e293b;border-radius:8px;padding:12px 16px;margin-bottom:12px;display:flex;gap:24px;align-items:center">
      <div>
        <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px">EI convergence — ${{ei.length}} iterations</div>
        <div style="display:flex;align-items:flex-end;height:60px">${{bars}}</div>
      </div>
      <div style="margin-left:auto;text-align:right">
        <div style="font-size:22px;font-weight:700;color:${{col}}">${{latest.toFixed(4)}}</div>
        <div style="font-size:11px;color:#94a3b8">current EI</div>
        <div style="font-size:10px;color:#475569;margin-top:4px">threshold 0.005</div>
      </div>
    </div>`;
  }}

  // MF optimum card
  let optHtml = "";
  if (opt) {{
    optHtml = `
    <div style="background:#1e1b4b;border:1px solid #312e81;border-radius:8px;padding:14px 18px;margin-bottom:12px;display:flex;gap:20px;flex-wrap:wrap;align-items:center">
      <div style="font-size:11px;color:#a78bfa;text-transform:uppercase;letter-spacing:.06em;width:100%;margin-bottom:2px">MF Co-Kriging Optimum</div>
      <div><div style="font-size:15px;font-weight:700;color:#e0e7ff">${{opt.slant_angle.toFixed(2)}}°</div><div style="font-size:10px;color:#64748b">slant</div></div>
      <div><div style="font-size:15px;font-weight:700;color:#e0e7ff">${{opt.diffuser_angle.toFixed(2)}}°</div><div style="font-size:10px;color:#64748b">diffuser</div></div>
      <div><div style="font-size:15px;font-weight:700;color:#e0e7ff">${{opt.ride_height.toFixed(1)}} mm</div><div style="font-size:10px;color:#64748b">ride height</div></div>
      <div><div style="font-size:15px;font-weight:700;color:#e0e7ff">${{opt.front_radius.toFixed(1)}} mm</div><div style="font-size:10px;color:#64748b">front radius</div></div>
      <div style="margin-left:auto;display:flex;gap:20px">
        <div style="text-align:right">
          <div style="font-size:15px;font-weight:700;color:#f8fafc">${{opt.Cd_predicted.toFixed(4)}} <span style="font-size:10px;color:#64748b">±${{opt.Cd_uncertainty.toFixed(4)}}</span></div>
          <div style="font-size:10px;color:#64748b">Cd predicted</div>
        </div>
        <div style="text-align:right">
          <div style="font-size:15px;font-weight:700;color:#a78bfa">${{opt.Cl_predicted.toFixed(4)}} <span style="font-size:10px;color:#64748b">±${{opt.Cl_uncertainty.toFixed(4)}}</span></div>
          <div style="font-size:10px;color:#64748b">Cl predicted</div>
        </div>
        <div style="text-align:right">
          <div style="font-size:15px;font-weight:700;color:#22c55e">${{opt.f1_objective.toFixed(4)}}</div>
          <div style="font-size:10px;color:#64748b">f₁ = Cd+⅓Cl</div>
        </div>
      </div>
    </div>`;
  }}

  // DES results table sorted by f1
  const sorted = [...des].sort((a, b) => {{
    const fa = a.cd_des !== null && a.cl_des !== null ? a.cd_des + a.cl_des / 3 : 999;
    const fb = b.cd_des !== null && b.cl_des !== null ? b.cd_des + b.cl_des / 3 : 999;
    return fa - fb;
  }});
  const bestF1 = sorted.length && sorted[0].cd_des !== null
    ? sorted[0].cd_des + sorted[0].cl_des / 3 : null;

  const desRows = sorted.map(r => {{
    const f1 = r.cd_des !== null && r.cl_des !== null ? r.cd_des + r.cl_des / 3 : null;
    const isBest = f1 !== null && bestF1 !== null && Math.abs(f1 - bestF1) < 1e-6;
    const dcdCol = r.dcd === null ? "#94a3b8" : r.dcd > 0.01 ? "#f87171" : r.dcd < -0.01 ? "#4ade80" : "#94a3b8";
    const pb = PHASE_BADGE[r.phase] || PHASE_BADGE["unknown"];
    return `<tr${{isBest ? ' class="bo-best"' : ''}}>
      <td class="mono" style="color:#fb923c">${{r.case_id}}</td>
      <td><span class="badge" style="background:${{pb.color}};font-size:9px">${{pb.label}}</span></td>
      <td class="mono dim">${{r.slant.toFixed(2)}}°</td>
      <td class="mono dim">${{r.diffuser.toFixed(2)}}°</td>
      <td class="mono dim">${{r.ride_h.toFixed(1)}}</td>
      <td class="mono cd">${{r.cd_des !== null ? r.cd_des.toFixed(4) : "—"}}</td>
      <td class="mono">${{r.cl_des !== null ? r.cl_des.toFixed(4) : "—"}}</td>
      <td class="mono" style="color:${{dcdCol}}">${{r.dcd !== null ? (r.dcd >= 0 ? "+" : "") + r.dcd.toFixed(4) : "—"}}</td>
      <td class="mono dim">${{r.dcl !== null ? (r.dcl >= 0 ? "+" : "") + r.dcl.toFixed(4) : "—"}}</td>
      <td class="mono cd">${{f1 !== null ? f1.toFixed(4) : "—"}}${{isBest ? " ★" : ""}}</td>
    </tr>`;
  }}).join("");

  if (!des.length && !act) {{ setHtml("des-section", ""); return; }}

  setHtml("des-section", `
<div class="section-header" style="color:#fb923c;border-color:#431407">
  <span>DES Campaign &amp; Bayesian Optimisation</span>
  <span class="section-sub">${{des.length}} HF evaluations · kOmegaSSTDES/IDDES · Kennedy &amp; O'Hagan co-Kriging</span>
</div>
<div class="cards" style="margin-top:0">${{phaseCards}}</div>
${{activeHtml}}${{corrHtml}}${{eiHtml}}${{optHtml}}
${{desRows ? `<table>
  <thead><tr>
    <th>Case</th><th>Phase</th><th>Slant</th><th>Diffuser</th><th>Rh mm</th>
    <th>Cd DES</th><th>Cl DES</th><th>δCd</th><th>δCl</th><th>f₁ ★best</th>
  </tr></thead>
  <tbody>${{desRows}}</tbody>
</table>` : ""}}`);
}}

// ── apply a full snapshot to the page ────────────────────────────────────────
function applySnapshot(d) {{
  setText("last-ts", d.ts);
  document.title = `Ahmed Body CFD — DoE ${{d.n_done}}/${{d.n_total}}`;

  setText("card-done",     d.n_done);
  setText("card-running",  d.n_running);
  setText("card-wait",     d.n_wait);
  setText("card-cd-range", d.cd_range);

  setText("overall-label", "DoE progress — " + d.doe_label);
  setStyle("overall-fill", "width", d.doe_pct.toFixed(1) + "%");
  setText("overall-pct",   d.doe_pct.toFixed(0) + "%");

  d.doe_cases.forEach(c => {{
    const cid = c.id;
    const pct = Math.min(c.iter / 1800, 1.0) * 100;
    const color = STATUS_COLOR[c.status] || "#6b7280";
    const label = STATUS_LABEL[c.status] || c.status.toUpperCase();

    const row = document.getElementById("row-" + cid);
    if (row) {{
      row.className = c.status;
    }}
    const badge = document.getElementById(cid + "-badge");
    if (badge) {{
      badge.style.background = color;
      badge.textContent = label;
    }}
    const bar = document.getElementById(cid + "-bar");
    if (bar) {{
      bar.style.width = pct.toFixed(1) + "%";
      if (c.status === "done") bar.classList.add("bar-done");
    }}
    setText(cid + "-iter",    c.iter + "/3000");
    setText(cid + "-cd",      c.cd    !== null ? c.cd.toFixed(4)    : "—");
    setText(cid + "-cl",      c.cl    !== null ? c.cl.toFixed(4)    : "—");
    setText(cid + "-elapsed", c.elapsed_min !== null ? c.elapsed_min.toFixed(1) + " min" : "—");
  }});

  renderPipelineOverview(d);
  renderMeshConv(d);
  renderPareto(d.pareto_pts, d.mf_pareto);
  renderDES(d);
}}

// ── SSE connection ────────────────────────────────────────────────────────────
function connect() {{
  const es = new EventSource("/stream");

  es.onopen = () => {{
    setText("status-bar", "Connected — waiting for data");
  }};

  es.onmessage = (event) => {{
    try {{
      const d = JSON.parse(event.data);
      applySnapshot(d);
      setText("status-bar", "Last update: " + d.ts);
    }} catch(e) {{
      console.warn("bad SSE payload", e);
    }}
  }};

  es.onerror = () => {{
    setText("status-bar", "Connection lost — reconnecting…");
    es.close();
    setTimeout(connect, 3000);
  }};
}}

// Apply initial state synchronously so the MC section is never blank
applySnapshot({json.dumps(snap, default=str)});
connect();
</script>
</body>
</html>"""


# ── HTTP server ───────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            snap = build_snapshot()
            html = initial_html(snap).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",  "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection",    "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            q = subscribe()
            # Send current state immediately so new tab isn't blank
            try:
                snap    = build_snapshot()
                payload = f"data: {json.dumps(snap)}\n\n"
                self.wfile.write(payload.encode("utf-8"))
                self.wfile.flush()
            except Exception:
                unsubscribe(q)
                return

            try:
                while True:
                    try:
                        payload = q.get(timeout=30)
                    except queue.Empty:
                        # Keepalive comment to prevent proxy timeouts
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    self.wfile.write(payload.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                unsubscribe(q)

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *_):
        pass


def main():
    t = threading.Thread(target=monitor_thread, daemon=True)
    t.start()

    server   = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    import socket
    local_ip = socket.gethostbyname(socket.gethostname())
    print("Dashboard running →")
    print(f"  Local:   http://localhost:{PORT}")
    print(f"  Network: http://{local_ip}:{PORT}")
    print(f"Reactive — pushes updates when simulation data changes.")
    print("(Read-only — zero impact on running simulations)")

    try:
        import webbrowser
        threading.Timer(0.3, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


if __name__ == "__main__":
    main()
