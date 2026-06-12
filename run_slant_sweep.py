"""
run_slant_sweep.py
==================
5 L2 steady-RANS solves at fixed diffuser_angle=0° across slant angles that
span the Ahmed body critical-angle regime. Used to validate the computed
Cd-vs-slant trend against Ahmed (1984) experimental data.

Points (diffuser_angle=0° for all, matching Ahmed original geometry):
  slant_sw_000  12.5° — deep sub-critical, attached flow
  slant_sw_001  20.0° — sub-critical with growing vortex system
  slant_sw_002  25.0° — Lienhart baseline (experimental reference, Cd≈0.299)
  slant_sw_003  30.0° — near critical, maximum C-pillar vortex strength
  slant_sw_004  35.0° — super-critical, fully separated

Results → results/slant_sweep.csv
Cases   → openfoam_cases_slant_sweep/

Usage:
  python3 run_slant_sweep.py           # run all (clean restart for any non-done case)
  python3 run_slant_sweep.py --resume  # skip done cases, resume interrupted ones
"""

import importlib.util, sys, time
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent

# ── Load runner and case manager ──────────────────────────────────────────────
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

runner  = _load("cfmesh_doe_runner", BASE / "cfmesh_doe_runner.py")
cm      = _load("case_manager",      BASE / "case_manager.py")

# ── Campaign config ───────────────────────────────────────────────────────────
CASES_DIR   = BASE / "openfoam_cases_slant_sweep"
RESULTS_CSV = BASE / "results" / "slant_sweep.csv"

RIDE_HEIGHT  = 50.8
FRONT_RADIUS = 100.0

SWEEP = [
    ("slant_sw_000", 12.5, 0.0),
    ("slant_sw_001", 20.0, 0.0),
    ("slant_sw_002", 25.0, 0.0),
    ("slant_sw_003", 30.0, 0.0),
    ("slant_sw_004", 35.0, 0.0),
]


# ── CSV helpers ───────────────────────────────────────────────────────────────
def append_to_csv(row: dict, csv_path: Path):
    """Append one result row atomically (read → dedup → write)."""
    df_new = pd.DataFrame([row])
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        df_existing = df_existing[df_existing["case_id"] != row["case_id"]]
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    tmp = csv_path.with_suffix(".tmp")
    df.to_csv(tmp, index=False, float_format="%.6f")
    tmp.rename(csv_path)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    resume = "--resume" in sys.argv
    CASES_DIR.mkdir(exist_ok=True)
    RESULTS_CSV.parent.mkdir(exist_ok=True)

    print(f"\nSlant-angle sweep — {len(SWEEP)} cases  (resume={resume})")
    print(f"  Cases : {CASES_DIR}")
    print(f"  CSV   : {RESULTS_CSV}\n")

    with cm.PidFile(CASES_DIR), cm.RunnerSentinel(CASES_DIR, "slant_sweep") as sentinel:
        for case_id, slant, diffuser in SWEEP:
            params = {
                "slant_angle":    slant,
                "diffuser_angle": diffuser,
                "ride_height":    RIDE_HEIGHT,
                "front_radius":   FRONT_RADIUS,
            }

            print(f"\n── {case_id}  slant={slant}°  diff={diffuser}° ──")
            res = runner.run_case(case_id, params, resume=resume, cases_dir=CASES_DIR)

            if res.get("skipped"):
                sentinel.n_skipped += 1
                continue

            if res["Cd"] is not None:
                row = {
                    "case_id":       case_id,
                    "slant_angle":   slant,
                    "diffuser_angle": diffuser,
                    "Cd":            res["Cd"],
                    "Cl":            res["Cl"],
                }
                append_to_csv(row, RESULTS_CSV)
                f1 = res["Cd"] + res["Cl"] / 3.0
                print(f"  → Cd={res['Cd']:.4f}  Cl={res['Cl']:.4f}  f1={f1:.4f}")
                sentinel.n_done += 1
            else:
                print(f"  → FAILED: {res.get('error', 'unknown')}")
                sentinel.n_failed += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    if RESULTS_CSV.exists():
        df = pd.read_csv(RESULTS_CSV)
        n_done = len(df)
        print(f"\n{'='*55}")
        print(f"Slant sweep: {n_done}/{len(SWEEP)} complete")
        print(f"{'Case':<15} {'Slant':>6} {'Cd':>8} {'Cl':>8} {'f1':>8}")
        print("-" * 50)
        for _, r in df.iterrows():
            if pd.notna(r["Cd"]):
                f1 = r["Cd"] + r["Cl"] / 3.0
                print(f"{r['case_id']:<15} {r['slant_angle']:>6.1f}"
                      f" {r['Cd']:>8.4f} {r['Cl']:>8.4f} {f1:>8.4f}")
        print(f"{'='*55}")


if __name__ == "__main__":
    main()
