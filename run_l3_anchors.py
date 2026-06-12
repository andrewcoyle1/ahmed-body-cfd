"""
run_l3_anchors.py
=================
Runs the L3 anchor campaign for multi-fidelity co-Kriging correction.

8 anchor points in the active 2D design space (slant 15–40°, diffuser 0–20°;
ride_height=50.8mm, front_radius=100mm fixed per Sobol analysis):

  #0  32.5 °  8.9 °   — existing case_l3_opt (copied, not re-run)
  #1  38.9 °  5.6 °   — recorded L2 BO optimum
  #2–7: 6 LHS space-filling points biased toward the high-slant basin

Results → results/l3_anchors.csv
Cases   → openfoam_cases_l3_anchors/

Usage:
  python3 run_l3_anchors.py           # run all
  python3 run_l3_anchors.py --resume  # skip done, resume interrupted
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

CASES_DIR   = BASE / "openfoam_cases_l3_anchors"
RESULTS_CSV = BASE / "results" / "l3_anchors.csv"

RIDE_HEIGHT  = 50.8
FRONT_RADIUS = 100.0

ANCHORS = [
    ("l3_anc_000", 32.492,  8.948),
    ("l3_anc_001", 38.932,  5.579),
    ("l3_anc_002", 15.8,    2.1),
    ("l3_anc_003", 21.3,   15.4),
    ("l3_anc_004", 27.6,    9.2),
    ("l3_anc_005", 33.4,   17.8),
    ("l3_anc_006", 36.7,    3.8),
    ("l3_anc_007", 40.0,   12.0),
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

    print(f"\nL3 Anchor Campaign — {len(ANCHORS)} points  (resume={resume})")
    print(f"  Template : {runner.L2_TEMPLATE}")
    print(f"  Cases    : {CASES_DIR}")
    print(f"  CSV      : {RESULTS_CSV}\n")

    with cm.PidFile(CASES_DIR), cm.RunnerSentinel(CASES_DIR, "l3_anchors") as sentinel:
        for case_id, slant, diffuser in ANCHORS:
            params = {
                "slant_angle":    slant,
                "diffuser_angle": diffuser,
                "ride_height":    RIDE_HEIGHT,
                "front_radius":   FRONT_RADIUS,
            }

            # l3_anc_000: copy from existing validation case rather than re-solving
            if case_id == "l3_anc_000":
                src = BASE / "openfoam_cases_l3_validation" / "case_l3_opt"
                state = cm.CaseState(CASES_DIR / case_id)
                if state.status == cm.Status.DONE:
                    sentinel.n_skipped += 1
                    print(f"  SKIP {case_id} (done)")
                    continue
                if src.exists():
                    res = runner.extract_results(src)
                    if res["Cd"] is not None:
                        (CASES_DIR / case_id).mkdir(exist_ok=True)
                        state.set(cm.Status.DONE, "copied from case_l3_opt")
                        append_to_csv({
                            "case_id": case_id, "slant_angle": slant,
                            "diffuser_angle": diffuser,
                            "Cd": res["Cd"], "Cl": res["Cl"],
                        }, RESULTS_CSV)
                        print(f"  [copy] {case_id}  Cd={res['Cd']}  Cl={res['Cl']}")
                        sentinel.n_done += 1
                        continue
                print(f"  WARNING: {src} not found — running l3_anc_000 fresh")

            print(f"\n── {case_id}  slant={slant}°  diff={diffuser}° ──")
            res = runner.run_case(case_id, params, resume=resume, cases_dir=CASES_DIR)

            if res.get("skipped"):
                sentinel.n_skipped += 1
                continue

            if res["Cd"] is not None:
                append_to_csv({
                    "case_id":       case_id,
                    "slant_angle":   slant,
                    "diffuser_angle": diffuser,
                    "Cd":            res["Cd"],
                    "Cl":            res["Cl"],
                }, RESULTS_CSV)
                f1 = res["Cd"] + res["Cl"] / 3.0
                print(f"  → Cd={res['Cd']:.4f}  Cl={res['Cl']:.4f}  f1={f1:.4f}")
                sentinel.n_done += 1
            else:
                print(f"  → FAILED: {res.get('error', 'unknown')}")
                sentinel.n_failed += 1

    if RESULTS_CSV.exists():
        df = pd.read_csv(RESULTS_CSV)
        print(f"\n{'='*55}")
        print(f"Anchors: {len(df)}/{len(ANCHORS)} complete")
        print(f"{'Case':<15} {'Slant':>6} {'Diff':>6} {'Cd':>8} {'Cl':>8} {'f1':>8}")
        print("-" * 55)
        for _, r in df.iterrows():
            if pd.notna(r["Cd"]):
                f1 = r["Cd"] + r["Cl"] / 3.0
                print(f"{r['case_id']:<15} {r['slant_angle']:>6.1f} "
                      f"{r['diffuser_angle']:>6.1f} {r['Cd']:>8.4f} "
                      f"{r['Cl']:>8.4f} {f1:>8.4f}")
        print(f"{'='*55}")


if __name__ == "__main__":
    main()
