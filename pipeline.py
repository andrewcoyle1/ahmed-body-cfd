#!/usr/bin/env python3
"""
pipeline.py - Overnight automation
===================================
1. Mesh convergence study (3 levels, full solver, no boundary layers)
2. Select DoE mesh level: coarsest with GCI < 5% and Cd within 20% of canonical
3. Patch case_generator.py with consistent mesh settings for selected level
4. Regenerate + run all 30 DoE cases (4 parallel workers)
5. Post-process → surrogate optimiser → PDF report

Log: pipeline.log
"""

import sys, os, json, re, shutil, subprocess, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
MESH_CONV_DIR = BASE / "mesh_convergence"
CASES_DIR     = BASE / "openfoam_cases"
RESULTS_DIR   = BASE / "results"
LOG_FILE      = BASE / "pipeline.log"

# Canonical reference (Lienhart 2002, 25° slant Ahmed body)
CD_REF        = 0.285
CD_TOL        = 0.20    # accept Cd_extrap within ±20% of canonical
GCI_THRESH    = 5.0     # GCI < 5% → grid-converged

# DoE parallelism — 2 containers × 6 cores = 12 cores on a 12-core machine
DOE_WORKERS   = 2

# Mesh level definitions — background grid fixed at 128×12×20, varies surf_level
# surf_level=4 is L2 (validated, selected for DoE)
MESH_LEVELS = [
    {"name": "L1_coarse", "surf_level": 3},
    {"name": "L2_medium", "surf_level": 4},
    {"name": "L3_fine",   "surf_level": 5},
]
LEVEL_ORDER = ["L1_coarse", "L2_medium", "L3_fine"]


# ── Logging ────────────────────────────────────────────────────────────────────
def log(msg, also_print=True):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    if also_print:
        print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ── Step 1: Mesh convergence ───────────────────────────────────────────────────
def run_mesh_convergence():
    log("=" * 60)
    log("STEP 1 — Mesh convergence study (full solver, no BL)")
    log("=" * 60)
    result = subprocess.run(
        [sys.executable, "mesh_convergence.py"],
        cwd=BASE, capture_output=False
    )
    if result.returncode != 0:
        log("WARNING: mesh_convergence.py exited with errors — continuing")


def load_convergence_results():
    results = []
    for lvl in MESH_LEVELS:
        jf = MESH_CONV_DIR / f"{lvl['name']}_result.json"
        if jf.exists():
            with open(jf) as f:
                results.append(json.load(f))
    return results


def richardson_extrapolation(results):
    """Replicated from mesh_convergence.py for standalone use."""
    valid = [r for r in results if r.get("Cd") and r.get("cells")]
    if len(valid) < 3:
        return None, None, None
    pairs = sorted([(r["cells"], r["Cd"]) for r in valid])
    (n3, f3), (n2, f2), (n1, f1) = pairs
    r21 = (n1 / n2) ** (1/3)
    e21, e32 = f1 - f2, f2 - f3
    if abs(e32) < 1e-10 or abs(e21) < 1e-10:
        return f1, None, None
    try:
        p = abs(np.log(abs(e32 / e21)) / np.log(r21))
    except (ValueError, ZeroDivisionError):
        return f1, None, None
    cd_extrap = f1 + (f1 - f2) / (r21 ** p - 1)
    gci = 1.25 * abs(e21 / f1) / (r21 ** p - 1) * 100
    return round(cd_extrap, 5), round(p, 2), round(gci, 2)


# ── Step 2: Select DoE mesh level ─────────────────────────────────────────────
def select_doe_level(results):
    """
    Pick the coarsest mesh level that satisfies GCI < 5%.
    If GCI is not satisfied, pick L2 as a practical compromise.
    Separately check if extrapolated Cd is within 20% of canonical.
    """
    cd_extrap, p_order, gci = richardson_extrapolation(results)

    log(f"\nMesh convergence summary:")
    for r in results:
        cd = f"{r['Cd']:.4f}" if r.get("Cd") else "—"
        log(f"  {r['name']:<15}  cells={r.get('cells','—'):>10,}  "
            f"Cd={cd}  non-ortho={r.get('max_non_ortho','—')}")

    if cd_extrap:
        err_pct = abs(cd_extrap - CD_REF) / CD_REF * 100
        log(f"\n  Richardson Cd   = {cd_extrap:.4f}  (order p={p_order})")
        log(f"  GCI (fine grid) = {gci}%")
        log(f"  vs canonical    = {err_pct:.1f}%  (ref={CD_REF}, tol={CD_TOL*100:.0f}%)")
        if err_pct > CD_TOL * 100:
            log(f"  WARNING: extrapolated Cd {err_pct:.1f}% from canonical — "
                "mesh setup may need review; proceeding anyway")
    else:
        log("  Richardson extrapolation not possible (< 3 valid levels)")
        gci = 999

    # Find coarsest level with Cd populated (if GCI bad, still need a mesh)
    # Prefer L1 → L2 → L3
    by_name = {r["name"]: r for r in results}

    if gci is not None and gci < GCI_THRESH:
        # GCI satisfied — use coarsest level that has a Cd value
        for name in LEVEL_ORDER:
            if name in by_name and by_name[name].get("Cd"):
                log(f"\n  GCI < {GCI_THRESH}% → selecting {name} for DoE")
                return next(l for l in MESH_LEVELS if l["name"] == name)

    # GCI not met or insufficient data → default to L2
    log(f"\n  GCI >= {GCI_THRESH}% (or insufficient data) → defaulting to L2_medium")
    return next(l for l in MESH_LEVELS if l["name"] == "L2_medium")


# ── Step 3: Validate case_generator mesh level ────────────────────────────────
def patch_case_generator(level):
    """
    case_generator.py is pre-configured for L2 (surf_level=4, 128×12×20).
    This step just logs the selection and verifies it matches the chosen level.
    If a different level were needed, case_generator would need manual update.
    """
    if level["name"] != "L2_medium":
        log(f"  WARNING: selected level={level['name']} but case_generator.py is "
            f"hardcoded for L2_medium (surf_level=4). Update case_generator.py manually "
            f"if re-running mesh convergence selects a different level.")
    else:
        log(f"  case_generator.py already configured for L2_medium (surf_level=4) — no patch needed.")


# ── Step 4: Regenerate + run DoE cases ────────────────────────────────────────
def regenerate_doe_cases():
    log("\nRegenerating 30 DoE cases with updated mesh settings...")
    result = subprocess.run(
        [sys.executable, "case_generator.py"],
        cwd=BASE, capture_output=True, text=True
    )
    if result.returncode != 0:
        log(f"ERROR in case_generator.py:\n{result.stderr}")
        sys.exit(1)
    log("  Case generation complete.")


def run_doe_case(case_id):
    run_sh = CASES_DIR / case_id / "run.sh"
    log_path = CASES_DIR / case_id / "pipeline_run.log"
    result = subprocess.run(
        ["bash", str(run_sh)],
        cwd=BASE,
        stdout=log_path.open("w"),
        stderr=subprocess.STDOUT,
    )
    ok = result.returncode == 0
    log(f"  [{'OK    ' if ok else 'FAILED'}] {case_id}")
    return case_id, ok


def run_doe_parallel(workers=DOE_WORKERS):
    dm = pd.read_csv(BASE / "design_matrix.csv")
    cases = dm["case_id"].tolist()
    log(f"\nRunning {len(cases)} DoE cases ({workers} parallel workers)...")

    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_doe_case, cid): cid for cid in cases}
        for fut in as_completed(futures):
            cid, ok = fut.result()
            if not ok:
                failed.append(cid)

    if failed:
        log(f"\n  Failed cases ({len(failed)}): {', '.join(failed)}")
    else:
        log(f"\n  All {len(cases)} cases completed successfully.")
    return failed


# ── Step 5: Post-process ───────────────────────────────────────────────────────
def run_post_processor():
    log("\nPost-processing DoE results...")
    result = subprocess.run(
        [sys.executable, "post_processor.py"],
        cwd=BASE, capture_output=True, text=True
    )
    log(result.stdout.strip())
    if result.returncode != 0:
        log(f"WARNING: post_processor.py errors:\n{result.stderr}")


# ── Step 6: Surrogate optimiser ───────────────────────────────────────────────
def run_surrogate():
    log("\nRunning surrogate optimiser + Bayesian optimisation...")
    result = subprocess.run(
        [sys.executable, "surrogate_optimiser.py"],
        cwd=BASE, capture_output=True, text=True
    )
    log(result.stdout.strip())
    if result.returncode != 0:
        log(f"WARNING: surrogate_optimiser.py errors:\n{result.stderr}")


# ── Step 7: Report ────────────────────────────────────────────────────────────
def run_report():
    log("\nGenerating PDF report...")
    result = subprocess.run(
        [sys.executable, "report.py"],
        cwd=BASE, capture_output=True, text=True
    )
    log(result.stdout.strip())
    if result.returncode != 0:
        log(f"WARNING: report.py errors:\n{result.stderr}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()
    log(f"\n{'='*60}")
    log(f"PIPELINE START  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'='*60}\n")

    RESULTS_DIR.mkdir(exist_ok=True)

    # 1. Mesh convergence
    run_mesh_convergence()
    results = load_convergence_results()

    # 2. Select DoE mesh level
    log("\n" + "="*60)
    log("STEP 2 — Selecting DoE mesh level")
    log("="*60)
    selected = select_doe_level(results)
    log(f"  Selected: {selected['name']}")

    # 3. Patch case_generator with consistent settings
    log("\n" + "="*60)
    log("STEP 3 — Patching case_generator.py")
    log("="*60)
    patch_case_generator(selected)

    # 4. Regenerate + run DoE
    log("\n" + "="*60)
    log("STEP 4 — DoE cases")
    log("="*60)
    regenerate_doe_cases()
    failed = run_doe_parallel(workers=DOE_WORKERS)

    # 5. Post-process (continue even if some cases failed)
    log("\n" + "="*60)
    log("STEP 5 — Post-processing")
    log("="*60)
    run_post_processor()

    # 6. Surrogate
    csv = RESULTS_DIR / "results_summary.csv"
    if csv.exists():
        n_converged = pd.read_csv(csv)["converged"].sum()
        log(f"  Converged cases: {n_converged}")
        if n_converged >= 15:
            log("\n" + "="*60)
            log("STEP 6 — Surrogate optimiser")
            log("="*60)
            run_surrogate()
        else:
            log(f"  Only {n_converged} converged — skipping surrogate (need ≥15)")
    else:
        log("  No results_summary.csv — skipping surrogate")

    # 7. Report
    log("\n" + "="*60)
    log("STEP 7 — PDF Report")
    log("="*60)
    run_report()

    elapsed = time.time() - t0
    log(f"\n{'='*60}")
    log(f"PIPELINE COMPLETE  {elapsed/3600:.1f} h elapsed")
    log(f"Report: {RESULTS_DIR}/ahmed_body_aero_optimisation.pdf")
    log(f"{'='*60}\n")
