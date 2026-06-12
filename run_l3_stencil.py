"""
run_l3_stencil.py
=================
Runs 6 L3 solves around the MF optimum (38.9°/10.5°) and the critical-angle
cliff (35.7°/9.3°) to map whether the optimum sits in a basin.

Points:
  l3_stn_000  35.7   9.3   — BO iter 1 cliff point (L3 re-solve)
  l3_stn_001  37.3  10.0   — midpoint between cliff and optimum
  l3_stn_002  38.0  10.5   — near-optimum approach
  l3_stn_003  38.9   9.5   — optimum −1° diffuser
  l3_stn_004  38.9  11.5   — optimum +1° diffuser
  l3_stn_005  39.5  10.5   — optimum +0.6° slant

Results → results/l3_stencil.csv
Cases   → openfoam_cases_l3_stencil/

NOTE: keep these OUT of results/l3_anchors.csv — validation, not training data.

Usage:
  python3 run_l3_stencil.py           # run all
  python3 run_l3_stencil.py --resume  # skip done, resume interrupted
"""

import importlib.util, sys
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = _load("cfmesh_doe_runner", BASE / "cfmesh_doe_runner.py")
cm     = _load("case_manager",      BASE / "case_manager.py")

runner.L2_TEMPLATE = BASE / "mesh_convergence" / "L3_fine"
runner.L2_SYM      = BASE / "mesh_convergence" / "L3_symmetry"

CASES_DIR   = BASE / "openfoam_cases_l3_stencil"
RESULTS_CSV = BASE / "results" / "l3_stencil.csv"

RIDE_HEIGHT  = 50.8
FRONT_RADIUS = 100.0

STENCIL = [
    ("l3_stn_000", 35.7,  9.3),
    ("l3_stn_001", 37.3, 10.0),
    ("l3_stn_002", 38.0, 10.5),
    ("l3_stn_003", 38.9,  9.5),
    ("l3_stn_004", 38.9, 11.5),
    ("l3_stn_005", 39.5, 10.5),
]


def append_to_csv(row: dict, csv_path: Path):
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


def main():
    resume = "--resume" in sys.argv
    CASES_DIR.mkdir(exist_ok=True)
    RESULTS_CSV.parent.mkdir(exist_ok=True)

    print(f"\nL3 Stencil Campaign — {len(STENCIL)} points  (resume={resume})")
    print(f"  Template : {runner.L2_TEMPLATE}")
    print(f"  Cases    : {CASES_DIR}")
    print(f"  CSV      : {RESULTS_CSV}\n")

    with cm.PidFile(CASES_DIR), cm.RunnerSentinel(CASES_DIR, "l3_stencil") as sentinel:
        for case_id, slant, diffuser in STENCIL:
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

    if RESULTS_CSV.exists():
        df = pd.read_csv(RESULTS_CSV)
        print(f"\n{'='*60}")
        print(f"Stencil: {len(df)}/{len(STENCIL)} complete")
        print(f"{'Case':<15} {'Slant':>6} {'Diff':>6} {'Cd':>8} {'Cl':>8} {'f1':>8}")
        print("-" * 60)
        for _, r in df.iterrows():
            if pd.notna(r["Cd"]):
                f1 = r["Cd"] + r["Cl"] / 3.0
                print(f"{r['case_id']:<15} {r['slant_angle']:>6.1f} "
                      f"{r['diffuser_angle']:>6.1f} {r['Cd']:>8.4f} "
                      f"{r['Cl']:>8.4f} {f1:>8.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
