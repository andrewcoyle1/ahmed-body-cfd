"""
run_bo_iteration.py
===================
Single BO iteration: generates and runs des_case_010 at the EI-optimal point
identified by co_kriging.py, then appends the result to des_results.csv and
re-runs co_kriging.py so we can confirm EI has dropped (convergence).
"""

import json, subprocess, sys, time
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from des_case_generator import write_des_case, DES_CASES_DIR

# ─── EI candidate ─────────────────────────────────────────────────────────────
candidate = json.loads(Path("results/mf_next_candidate.json").read_text())
CASE_ID = "des_case_012"
params = {
    "slant_angle":    candidate["slant_angle"],
    "diffuser_angle": candidate["diffuser_angle"],
    "ride_height":    candidate["ride_height"],
    "front_radius":   candidate["front_radius"],
}

print("── BO iteration: des_case_010 ───────────────────────────────────────────")
print(f"  slant:     {params['slant_angle']:.4f}°")
print(f"  diffuser:  {params['diffuser_angle']:.4f}°")
print(f"  ride_ht:   {params['ride_height']:.4f} mm")
print(f"  front_rad: {params['front_radius']:.4f} mm")

# ─── Generate case files ───────────────────────────────────────────────────────
DES_CASES_DIR.mkdir(exist_ok=True)
case_dir = write_des_case(CASE_ID, params, DES_CASES_DIR)
print(f"\n  Case written → {case_dir}")

# ─── Run case ──────────────────────────────────────────────────────────────────
run_sh = case_dir / "run.sh"
print(f"\n  Launching {run_sh} ...\n")
t0 = time.time()
result = subprocess.run(["bash", str(run_sh)], check=False)
elapsed = (time.time() - t0) / 60
print(f"\n  run.sh finished in {elapsed:.1f} min  (return code {result.returncode})")

# ─── Extract forces ────────────────────────────────────────────────────────────
import re

def extract_time_averaged_forces(case_dir: Path):
    """Read forceCoeffs postProcessing and return time-averaged Cd, Cl."""
    pp = case_dir / "postProcessing" / "forceCoeffs" / "0"
    # OpenFOAM version differences: coefficient.dat or forceCoeffs.dat
    for fname in ("coefficient.dat", "forceCoeffs.dat"):
        coeff_path = pp / fname
        if coeff_path.exists():
            break
    else:
        raise FileNotFoundError(f"No coefficient file found in {pp}")
    df = pd.read_csv(coeff_path, sep=r"\s+", comment="#",
                     names=["Time","Cd","CdF","CdR","Cl","ClF","ClR",
                            "CmPitch","CmRoll","CmYaw","Cs","CsF","CsR"])
    # average over t > AVG_START
    AVG_START = 0.13
    avg = df[df["Time"] > AVG_START]
    if len(avg) < 5:
        avg = df.tail(20)
    return float(avg["Cd"].mean()), float(avg["Cl"].mean())

print("\n── Extracting time-averaged forces ──────────────────────────────────────")
try:
    cd_des, cl_des = extract_time_averaged_forces(case_dir)
    print(f"  Cd_DES = {cd_des:.4f}")
    print(f"  Cl_DES = {cl_des:.4f}")
except FileNotFoundError as e:
    print(f"  ERROR: {e}")
    print("  Check that pimpleFoam ran to completion and fieldAverage was active.")
    sys.exit(1)

# ─── Also get RANS prediction at this point (from RANS GP) ────────────────────
rans_csv = Path("results/results_summary.csv")
df_rans   = pd.read_csv(rans_csv)
FEATURES  = ["slant_angle", "diffuser_angle", "ride_height", "front_radius"]
X_rans    = df_rans[FEATURES].values
cd_rans_vals = df_rans["Cd"].values
cl_rans_vals = df_rans["Cl"].values

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

def fit_rans_gp(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(X.shape[1]),
                                           length_scale_bounds=(1e-2, 1e2), nu=2.5) \
             + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                   normalize_y=True, random_state=0)
    gp.fit(Xs, y)
    return gp, scaler

gp_cd, sc_cd = fit_rans_gp(X_rans, cd_rans_vals)
gp_cl, sc_cl = fit_rans_gp(X_rans, cl_rans_vals)

x_new = np.array([[params["slant_angle"], params["diffuser_angle"],
                   params["ride_height"],  params["front_radius"]]])
cd_rans_pred = float(gp_cd.predict(sc_cd.transform(x_new))[0])
cl_rans_pred = float(gp_cl.predict(sc_cl.transform(x_new))[0])
print(f"  Cd_RANS (GP pred) = {cd_rans_pred:.4f}")
print(f"  Cl_RANS (GP pred) = {cl_rans_pred:.4f}")

dcd = cd_des - cd_rans_pred
dcl = cl_des - cl_rans_pred
print(f"  δCd = {dcd:+.4f},  δCl = {dcl:+.4f}")

# ─── Append to des_results.csv ────────────────────────────────────────────────
des_csv = Path("des_output/des_results.csv")
df_des  = pd.read_csv(des_csv)

new_row = {
    "case_id":        CASE_ID,
    "slant_angle":    params["slant_angle"],
    "diffuser_angle": params["diffuser_angle"],
    "ride_height":    params["ride_height"],
    "front_radius":   params["front_radius"],
    "Cd_DES":         cd_des,
    "Cl_DES":         cl_des,
    "Cd_RANS":        cd_rans_pred,
    "Cl_RANS":        cl_rans_pred,
    "dCd":            dcd,
    "dCl":            dcl,
}
df_des = pd.concat([df_des, pd.DataFrame([new_row])], ignore_index=True)
df_des.to_csv(des_csv, index=False)
print(f"\n  Appended des_case_010 → {des_csv}")

# ─── Re-run co_kriging.py ──────────────────────────────────────────────────────
print("\n── Re-running co_kriging.py with 11 HF points ───────────────────────────")
subprocess.run([sys.executable, "co_kriging.py"], check=True)

print("\n── BO iteration complete ────────────────────────────────────────────────")
print("  Check results/mf_next_candidate.json for updated EI value.")
print("  If EI << 0.023 the loop has converged.")
