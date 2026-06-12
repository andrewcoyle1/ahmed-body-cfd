"""
run_l3_anchors.py
=================
Runs the L3 anchor campaign for multi-fidelity co-Kriging correction.

8 anchor points in the active 2D design space (slant 15–40°, diffuser 0–20°;
ride_height=50.8mm, front_radius=100mm fixed per Sobol analysis):

  #0  32.5 °  8.9 °   — existing case_l3_opt (reused, not re-run)
  #1  38.9 °  5.6 °   — recorded L2 BO optimum (no L3 data there yet)
  #2–7: 6 LHS space-filling points biased toward the high-slant basin

Results appended to results/l3_anchors.csv.
Cases written to openfoam_cases_l3_anchors/.

Usage:
  python3 run_l3_anchors.py              # run all missing anchors
  python3 run_l3_anchors.py --resume     # skip cases that already finished
"""

import importlib.util, sys, json
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent

# ── Load cfmesh_doe_runner without executing its main() ───────────────────────
spec = importlib.util.spec_from_file_location(
    "cfmesh_doe_runner", BASE / "cfmesh_doe_runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

# Switch template to L3_fine for every case this script runs
runner.L2_TEMPLATE = BASE / "mesh_convergence" / "L3_fine"
runner.L2_SYM      = BASE / "mesh_convergence" / "L3_symmetry"

CASES_DIR   = BASE / "openfoam_cases_l3_anchors"
RESULTS_CSV = BASE / "results" / "l3_anchors.csv"
CASES_DIR.mkdir(exist_ok=True)

# Fixed parameters
RIDE_HEIGHT  = 50.8
FRONT_RADIUS = 100.0

# ── Anchor matrix ─────────────────────────────────────────────────────────────
# Columns: case_id, slant_angle, diffuser_angle
# #0: existing L3 point at old optimum — we copy results rather than re-run.
# #1: current recorded optimum from results/optimum_design.json.
# #2–7: LHS space-filling over (slant 15–40°, diffuser 0–20°), seed=7,
#        biased by constraining 2 points in the high-slant basin (slant>34°).
ANCHORS = [
    ("l3_anc_000", 32.492,  8.948),   # existing — will be copied from case_l3_opt
    ("l3_anc_001", 38.932,  5.579),   # recorded L2 BO optimum
    # LHS block (seed=7, 6 points)
    ("l3_anc_002", 15.8,    2.1),
    ("l3_anc_003", 21.3,   15.4),
    ("l3_anc_004", 27.6,    9.2),
    ("l3_anc_005", 33.4,   17.8),
    ("l3_anc_006", 36.7,    3.8),
    ("l3_anc_007", 40.0,   12.0),
]


def copy_existing_result(src_case_dir: Path, case_id: str) -> dict:
    """Re-extract results from an already-solved case directory."""
    res = runner.extract_results(src_case_dir)
    print(f"  [copy] {case_id}  Cd={res['Cd']}  Cl={res['Cl']}")
    return res


def already_in_csv(case_id: str, csv_path: Path) -> bool:
    if not csv_path.exists():
        return False
    df = pd.read_csv(csv_path)
    return case_id in df["case_id"].values


def append_to_csv(row: dict, csv_path: Path):
    df_new = pd.DataFrame([row])
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        # Avoid duplicates
        df_existing = df_existing[df_existing["case_id"] != row["case_id"]]
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(csv_path, index=False, float_format="%.6f")


def main():
    resume = "--resume" in sys.argv
    RESULTS_CSV.parent.mkdir(exist_ok=True)

    print(f"\nL3 Anchor Campaign — {len(ANCHORS)} points")
    print(f"  Template : {runner.L2_TEMPLATE}")
    print(f"  Cases dir: {CASES_DIR}")
    print(f"  Results  : {RESULTS_CSV}\n")

    for case_id, slant, diffuser in ANCHORS:
        params = {
            "slant_angle":    slant,
            "diffuser_angle": diffuser,
            "ride_height":    RIDE_HEIGHT,
            "front_radius":   FRONT_RADIUS,
        }

        if already_in_csv(case_id, RESULTS_CSV):
            print(f"  SKIP {case_id} — already in {RESULTS_CSV.name}")
            continue

        # l3_anc_000 already solved in openfoam_cases_l3_validation/case_l3_opt
        if case_id == "l3_anc_000":
            src = BASE / "openfoam_cases_l3_validation" / "case_l3_opt"
            if src.exists():
                res = copy_existing_result(src, case_id)
                if res["Cd"] is not None:
                    row = {
                        "case_id": case_id, "slant_angle": slant,
                        "diffuser_angle": diffuser,
                        "Cd": res["Cd"], "Cl": res["Cl"],
                    }
                    append_to_csv(row, RESULTS_CSV)
                    continue
            print(f"  WARNING: {src} not found — will run l3_anc_000 fresh")

        print(f"\n── {case_id}  slant={slant}°  diff={diffuser}° ──")
        res = runner.run_case(case_id, params, resume=resume, cases_dir=CASES_DIR)

        if res["Cd"] is not None:
            row = {
                "case_id": case_id, "slant_angle": slant,
                "diffuser_angle": diffuser,
                "Cd": res["Cd"], "Cl": res["Cl"],
            }
            append_to_csv(row, RESULTS_CSV)
            f1 = res["Cd"] + (1.0 / 3.0) * res["Cl"]
            print(f"  → Cd={res['Cd']:.4f}  Cl={res['Cl']:.4f}  f1={f1:.4f}")
        else:
            print(f"  → FAILED: {res.get('error','unknown error')}")

    # Print summary
    if RESULTS_CSV.exists():
        df = pd.read_csv(RESULTS_CSV)
        print(f"\n{'='*55}")
        print(f"Anchor summary ({len(df)} / {len(ANCHORS)} complete):")
        print(f"{'Case':<15} {'Slant':>6} {'Diff':>6} {'Cd':>8} {'Cl':>8} {'f1':>8}")
        print("-" * 55)
        for _, r in df.iterrows():
            if pd.notna(r["Cd"]):
                f1 = r["Cd"] + (1.0 / 3.0) * r["Cl"]
                print(f"{r['case_id']:<15} {r['slant_angle']:>6.1f} "
                      f"{r['diffuser_angle']:>6.1f} {r['Cd']:>8.4f} "
                      f"{r['Cl']:>8.4f} {f1:>8.4f}")
        print(f"{'='*55}")


if __name__ == "__main__":
    main()
