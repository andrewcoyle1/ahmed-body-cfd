"""
compute_force_stats.py
======================
Walks all completed OpenFOAM cases and computes per-case mean/std of Cd and Cl
over the averaging window (iters 600-800). Identifies oscillating cases.
Output: results/force_stats.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE      = Path(__file__).parent
AVG_START = 600
N_ITERS   = 800

CASE_DIRS = [
    BASE / "openfoam_cases_phase2",
    BASE / "openfoam_cases_l3_anchors",
    BASE / "openfoam_cases_l3_validation",
]


def load_force_history(case_dir: Path):
    """Return (time, Cd, Cl) arrays from forceCoeffs, or None."""
    base = case_dir / "postProcessing" / "forceCoeffs"
    if not base.exists():
        return None
    dat = None
    for td in sorted(base.iterdir()):
        for name in ("coefficient.dat", "forceCoeffs.dat"):
            cand = td / name
            if cand.exists():
                dat = cand
    if dat is None:
        return None
    rows = []
    with open(dat) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    rows.append([float(x) for x in parts[:5]])
                except ValueError:
                    pass
    if not rows:
        return None
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1], arr[:, 4]


def stats_for_case(case_dir: Path) -> dict | None:
    result = load_force_history(case_dir)
    if result is None:
        return None
    t, cd, cl = result
    mask = (t >= AVG_START) & (t <= N_ITERS)
    if not mask.any():
        mask = np.ones(len(t), dtype=bool)
    cd_w, cl_w = cd[mask], cl[mask]
    return {
        "case_id":  case_dir.name,
        "case_dir": str(case_dir.parent.name),
        "n_samples": int(mask.sum()),
        "cd_mean": float(np.mean(cd_w)),
        "cd_std":  float(np.std(cd_w)),
        "cl_mean": float(np.mean(cl_w)),
        "cl_std":  float(np.std(cl_w)),
        "f1_mean": float(np.mean(cd_w) + (1.0/3.0) * np.mean(cl_w)),
    }


def main():
    rows = []
    for parent in CASE_DIRS:
        if not parent.exists():
            continue
        for case_dir in sorted(parent.iterdir()):
            if not case_dir.is_dir():
                continue
            s = stats_for_case(case_dir)
            if s is not None:
                rows.append(s)

    df = pd.DataFrame(rows)
    df = df.sort_values(["case_dir", "case_id"]).reset_index(drop=True)

    out = BASE / "results" / "force_stats.csv"
    df.to_csv(out, index=False, float_format="%.6f")
    print(f"Written {len(df)} cases → {out}")

    print(f"\n{'='*75}")
    print(f"{'Case':<20} {'Parent':<28} {'Cd mean':>8} {'Cd std':>8} {'Cl mean':>8} {'Cl std':>8}")
    print("-" * 75)
    for _, r in df.iterrows():
        flag = " *** OSCILLATING" if r["cd_std"] > 0.01 else ""
        print(f"{r['case_id']:<20} {r['case_dir']:<28} "
              f"{r['cd_mean']:>8.4f} {r['cd_std']:>8.4f} "
              f"{r['cl_mean']:>8.4f} {r['cl_std']:>8.4f}{flag}")

    print(f"\nCd std: mean={df['cd_std'].mean():.4f}, "
          f"max={df['cd_std'].max():.4f} ({df.loc[df['cd_std'].idxmax(), 'case_id']})")
    print(f"Cl std: mean={df['cl_std'].mean():.4f}, "
          f"max={df['cl_std'].max():.4f} ({df.loc[df['cl_std'].idxmax(), 'case_id']})")

    oscillating = df[df["cd_std"] > 0.005]
    print(f"\nOscillating cases (Cd std > 0.005): {len(oscillating)}")
    if not oscillating.empty:
        for _, r in oscillating.iterrows():
            print(f"  {r['case_id']} ({r['case_dir']}): Cd std={r['cd_std']:.4f}, Cl std={r['cl_std']:.4f}")


if __name__ == "__main__":
    main()
