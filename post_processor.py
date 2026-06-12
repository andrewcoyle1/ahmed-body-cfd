"""
post_processor.py
=================
Module 3 of the Ahmed body aerodynamic optimisation pipeline.

For each completed OpenFOAM case:
  1. Reads forceCoeffs.dat output from simpleFoam
  2. Extracts converged Cd and Cl (mean of final 20% of time steps)
  3. Checks convergence via final residuals in log.simpleFoam
  4. Updates the case JSON config with results
  5. Writes a consolidated results_summary.csv

Run after one or more cases have completed:
  python3 post_processor.py                  # process all completed cases
  python3 post_processor.py case_000         # process one specific case

Dependencies: numpy, pandas (already used in earlier modules)
"""

import sys
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

CASES_DIR      = Path("openfoam_cases")
JSON_DIR       = Path("doe_output/cases")
DESIGN_MATRIX  = Path("design_matrix.csv")
RESULTS_DIR    = Path("results")
RESULTS_CSV    = RESULTS_DIR / "results_summary.csv"

CONVERGENCE_THRESHOLD = 1e-3   # final residual above this → flagged unconverged
TAIL_FRACTION         = 0.20   # use last 20% of time steps for averaging


# ─── 1. FORCE COEFFICIENT EXTRACTION ─────────────────────────────────────────

def find_force_coeffs_file(case_dir: Path) -> Path | None:
    """
    OpenFOAM v2512 writes forceCoeffs output to:
      postProcessing/forceCoeffs/<startTime>/coefficient.dat
    Earlier versions used forceCoeffs.dat. Check both.
    """
    base = case_dir / "postProcessing" / "forceCoeffs"
    if not base.exists():
        return None
    for td in sorted(base.iterdir()):
        for name in ("coefficient.dat", "forceCoeffs.dat"):
            candidate = td / name
            if candidate.exists():
                return candidate
    return None


def _parse_dat_rows(dat_file: Path) -> list:
    rows = []
    with open(dat_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue
    return rows


def extract_cd_from_log(case_dir: Path) -> dict:
    """
    Fallback: parse forceCoeffs summary blocks from log.simpleFoam.
    Each block written at writeInterval looks like:
      forceCoeffs forceCoeffs write:
          Cd:  <total>  <pressure>  <viscous>  0
          Cl:  ...
    Returns tail-averaged Cd/Cl or None on failure.
    """
    log_file = case_dir / "log.simpleFoam"
    if not log_file.exists():
        return {"Cd": None, "Cl": None, "source": "log_missing"}

    cd_vals, cl_vals = [], []
    lines = log_file.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("Cd:") and i > 0:
            parts = line.split()
            try:
                cd_vals.append(float(parts[1]))
            except (IndexError, ValueError):
                pass
        elif line.strip().startswith("Cl:") and i > 0:
            parts = line.split()
            try:
                cl_vals.append(float(parts[1]))
            except (IndexError, ValueError):
                pass

    if not cd_vals:
        return {"Cd": None, "Cl": None, "source": "log_no_cd"}

    n_tail = max(1, int(len(cd_vals) * TAIL_FRACTION))
    cd = float(np.mean(cd_vals[-n_tail:]))
    cl = float(np.mean(cl_vals[-n_tail:])) if cl_vals else None

    return {
        "Cd":    round(cd, 6),
        "Cl":    round(cl, 6) if cl is not None else None,
        "source": "log",
    }


# Minimum data rows for a valid dat file — catches stale partial files (e.g. 5 rows
# from a killed 50-iter run) without penalising early-converged cases (~45+ rows).
_MIN_DAT_ROWS = 10


def extract_force_coeffs(case_dir: Path) -> dict:
    """
    Parse forceCoeffs output and return mean Cd and Cl from the converged tail.

    OpenFOAM v2512 column layout (coefficient.dat):
      Time  Cd  Cd(f)  Cd(r)  Cl  Cl(f)  Cl(r)  CmPitch  CmRoll  CmYaw  Cs  Cs(f)  Cs(r)
    Cd = col 1, Cl = col 4  (0-indexed)

    Falls back to log.simpleFoam if coefficient.dat is absent or has fewer than
    _MIN_DAT_ROWS rows (stale partial data from an earlier killed run).
    """
    dat_file = find_force_coeffs_file(case_dir)
    rows = _parse_dat_rows(dat_file) if dat_file is not None else []

    if len(rows) < _MIN_DAT_ROWS:
        reason = "coefficient.dat not found" if dat_file is None else f"coefficient.dat sparse ({len(rows)} rows < {_MIN_DAT_ROWS})"
        log_result = extract_cd_from_log(case_dir)
        if log_result["Cd"] is not None:
            cd, cl = log_result["Cd"], log_result["Cl"]
            return {
                "Cd":    cd,
                "Cl":    cl,
                "Cd_Cl": round(cd / cl, 6) if cl is not None and abs(cl) > 1e-6 else None,
                "error": f"fallback_log ({reason})",
            }
        return {"Cd": None, "Cl": None, "Cd_Cl": None, "error": reason}

    arr  = np.array(rows)
    tail = arr[max(0, int(len(arr) * (1 - TAIL_FRACTION))):]

    cd = float(np.mean(tail[:, 1]))   # Cd column
    cl = float(np.mean(tail[:, 4]))   # Cl column (v2512 layout)

    return {
        "Cd":    round(cd, 6),
        "Cl":    round(cl, 6),
        "Cd_Cl": round(cd / cl, 6) if abs(cl) > 1e-6 else None,
        "error": None,
    }


# ─── 2. CONVERGENCE CHECK ─────────────────────────────────────────────────────

def check_convergence(case_dir: Path) -> bool:
    """
    Two-part convergence check:

    1. Cd stability (primary) — relative std dev of Cd over the last 20% of
       force-coefficient samples must be below 0.5%.  This is the physically
       meaningful criterion: the drag reading is stable regardless of whether
       residuals are still ticking down.

    2. Residual drop (secondary) — the final initial residual for p, k, and
       omega must each be at least 2 orders of magnitude below their respective
       first-iteration values.  This guards against cases that happened to
       have a flat (but unconverged) Cd trace.

    Both conditions must pass.  A run that was killed (no 'End' marker) fails
    automatically so that incomplete cases are not fed to the surrogate.
    """
    log_file = case_dir / "log.simpleFoam"
    if not log_file.exists():
        return False

    log_text = log_file.read_text(errors="replace")

    # Must have completed normally
    if not ('End\n' in log_text or log_text.rstrip().endswith('End')):
        return False

    # ── 1. Cd stability ───────────────────────────────────────────────────────
    dat_file = find_force_coeffs_file(case_dir)
    rows = _parse_dat_rows(dat_file) if dat_file is not None else []
    if len(rows) >= 20:
        arr    = np.array(rows)
        n_tail = max(10, int(len(arr) * TAIL_FRACTION))
        tail_cd = arr[-n_tail:, 1]
        cd_mean = float(np.mean(tail_cd))
        cd_std  = float(np.std(tail_cd))
        if abs(cd_mean) > 1e-6 and (cd_std / abs(cd_mean)) > 0.005:
            return False   # Cd still oscillating > 0.5 % of mean

    # ── 2. Residual drop ─────────────────────────────────────────────────────
    field_pattern = re.compile(
        r"Solving for (\w+),\s*Initial residual\s*=\s*([0-9eE+\-.]+)",
        re.IGNORECASE,
    )
    all_matches = field_pattern.findall(log_text)
    # Collect first and last initial residual per field
    first_res: dict[str, float] = {}
    last_res:  dict[str, float] = {}
    for field, val in all_matches:
        try:
            v = float(val)
        except ValueError:
            continue
        if field not in first_res:
            first_res[field] = v
        last_res[field] = v

    for field in ("p", "k", "omega"):
        r0 = first_res.get(field)
        rf = last_res.get(field)
        if r0 is None or rf is None:
            continue
        if r0 > 1e-10 and rf / r0 > 0.01:   # less than 2 orders drop → flag
            return False

    return True


# ─── 3. JSON UPDATE ───────────────────────────────────────────────────────────

def update_case_json(case_id: str, force_results: dict, converged: bool):
    """Fill in the results fields of the case JSON config."""
    json_path = JSON_DIR / f"{case_id}.json"
    if not json_path.exists():
        return
    with open(json_path) as f:
        config = json.load(f)
    config["results"]["Cd"]         = force_results.get("Cd")
    config["results"]["Cl"]         = force_results.get("Cl")
    config["results"]["Cd_Cl"]      = force_results.get("Cd_Cl")
    config["results"]["converged"]  = converged
    config["results"]["error"]      = force_results.get("error")
    with open(json_path, "w") as f:
        json.dump(config, f, indent=2)


# ─── 4. RESULTS CSV ───────────────────────────────────────────────────────────

def write_results_csv(all_rows: list[dict]):
    """Write or overwrite results_summary.csv with all processed cases."""
    RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(all_rows, columns=[
        "case_id", "slant_angle", "diffuser_angle",
        "ride_height", "front_radius",
        "Cd", "Cl", "Cd_Cl", "converged", "error"
    ])
    df.to_csv(RESULTS_CSV, index=False, float_format="%.6f")
    print(f"\nResults saved → {RESULTS_CSV}  ({len(df)} cases)")


# ─── 5. MAIN ─────────────────────────────────────────────────────────────────

def process_case(case_id: str, design_row: pd.Series) -> dict:
    case_dir = CASES_DIR / case_id

    if not case_dir.exists():
        print(f"  {case_id}  SKIP — directory not found")
        return None

    if not (case_dir / "log.simpleFoam").exists():
        print(f"  {case_id}  SKIP — not yet run (no log.simpleFoam)")
        return None

    force  = extract_force_coeffs(case_dir)
    conv   = check_convergence(case_dir)
    update_case_json(case_id, force, conv)

    status = "OK" if (force["Cd"] is not None and conv) else ("NO_CONV" if force["Cd"] is not None else "ERROR")
    print(f"  {case_id}  Cd={force['Cd']}  Cl={force['Cl']}  converged={conv}  [{status}]")

    return {
        "case_id":        case_id,
        "slant_angle":    design_row["slant_angle"],
        "diffuser_angle": design_row["diffuser_angle"],
        "ride_height":    design_row["ride_height"],
        "front_radius":   design_row["front_radius"],
        "Cd":             force["Cd"],
        "Cl":             force["Cl"],
        "Cd_Cl":          force["Cd_Cl"],
        "converged":      conv,
        "error":          force.get("error"),
    }


def main():
    global CASES_DIR, DESIGN_MATRIX, RESULTS_CSV, RESULTS_DIR

    # Allow --cases-dir, --design-matrix, --results-csv overrides
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--cases-dir" and i + 1 < len(args):
            CASES_DIR = Path(args[i + 1]).resolve()
        if a == "--design-matrix" and i + 1 < len(args):
            DESIGN_MATRIX = Path(args[i + 1]).resolve()
        if a == "--results-csv" and i + 1 < len(args):
            RESULTS_CSV = Path(args[i + 1]).resolve()
            RESULTS_DIR = RESULTS_CSV.parent
            RESULTS_DIR.mkdir(exist_ok=True)

    dm = pd.read_csv(DESIGN_MATRIX).set_index("case_id")

    # If a specific case_id is passed on the command line, process only that one
    target_cases_arg = [a for a in args if a.startswith("case_")]
    if target_cases_arg:
        target_cases = target_cases_arg
    else:
        target_cases = list(dm.index)

    print(f"Post-processing {len(target_cases)} case(s)...\n")
    rows = []
    for case_id in target_cases:
        if case_id not in dm.index:
            print(f"  {case_id}  SKIP — not in design_matrix.csv")
            continue
        row = process_case(case_id, dm.loc[case_id])
        if row is not None:
            rows.append(row)

    if rows:
        write_results_csv(rows)
    else:
        print("\nNo completed cases found. Run at least one case first:")
        print("  bash docker_run_case.sh case_000")


if __name__ == "__main__":
    main()
