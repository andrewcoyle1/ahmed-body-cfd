"""
run_bo_loop.py
==============
Automated multi-fidelity BO loop. Runs DES cases at EI-optimal points,
updates the co-Kriging surrogate after each, and stops when EI drops
below the convergence threshold or max iterations are reached.

Usage:
    python3 run_bo_loop.py [--max-iter N] [--ei-threshold T]

Defaults: max 10 iterations, EI threshold 0.005.
Progress is logged to bo_loop.log in real time.
"""

import argparse, json, subprocess, sys, time, logging
from pathlib import Path
from pipeline_lock import acquire_lock, release_lock

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from des_case_generator import write_des_case, DES_CASES_DIR

# ─── Configuration ────────────────────────────────────────────────────────────
FEATURES      = ["slant_angle", "diffuser_angle", "ride_height", "front_radius"]
AVG_START     = 0.13
DES_RESULTS   = Path("des_output/des_results.csv")
RANS_RESULTS  = Path("results/results_summary.csv")
CANDIDATE_JSON = Path("results/mf_next_candidate.json")
LOG_FILE      = Path("bo_loop.log")

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


def next_case_id() -> str:
    existing = sorted(DES_CASES_DIR.glob("des_case_*"))
    ids = [int(p.name.split("_")[-1]) for p in existing if p.name.split("_")[-1].isdigit()]
    return f"des_case_{(max(ids) + 1):03d}" if ids else "des_case_013"


def fit_rans_gp(X, y):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    kernel = ConstantKernel(1.0) * Matern(
        length_scale=np.ones(X.shape[1]),
        length_scale_bounds=(1e-2, 1e2), nu=2.5
    ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                  normalize_y=True, random_state=0)
    gp.fit(Xs, y)
    return gp, sc


def extract_forces(case_dir: Path) -> tuple[float, float]:
    pp = case_dir / "postProcessing" / "forceCoeffs" / "0"
    for fname in ("coefficient.dat", "forceCoeffs.dat"):
        p = pp / fname
        if p.exists():
            df = pd.read_csv(p, sep=r"\s+", comment="#",
                             names=["Time","Cd","CdF","CdR","Cl","ClF","ClR",
                                    "CmPitch","CmRoll","CmYaw","Cs","CsF","CsR"])
            avg = df[df["Time"] > AVG_START]
            if len(avg) < 5:
                avg = df.tail(20)
            return float(avg["Cd"].mean()), float(avg["Cl"].mean())
    raise FileNotFoundError(f"No force coefficient file in {pp}")


def get_rans_prediction(params: dict) -> tuple[float, float]:
    df = pd.read_csv(RANS_RESULTS)
    X = df[FEATURES].values
    x_new = np.array([[params[f] for f in FEATURES]])
    gp_cd, sc_cd = fit_rans_gp(X, df["Cd"].values)
    gp_cl, sc_cl = fit_rans_gp(X, df["Cl"].values)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cd = float(gp_cd.predict(sc_cd.transform(x_new))[0])
        cl = float(gp_cl.predict(sc_cl.transform(x_new))[0])
    return cd, cl


def append_result(case_id: str, params: dict,
                  cd_des: float, cl_des: float,
                  cd_rans: float, cl_rans: float):
    df = pd.read_csv(DES_RESULTS)
    row = {
        "case_id":        case_id,
        "slant_angle":    params["slant_angle"],
        "diffuser_angle": params["diffuser_angle"],
        "ride_height":    params["ride_height"],
        "front_radius":   params["front_radius"],
        "Cd_DES":  cd_des,  "Cl_DES":  cl_des,
        "Cd_RANS": cd_rans, "Cl_RANS": cl_rans,
        "dCd": cd_des - cd_rans,
        "dCl": cl_des - cl_rans,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(DES_RESULTS, index=False)
    log.info(f"  Appended {case_id} → {DES_RESULTS}")


def run_iteration(case_id: str, params: dict) -> bool:
    """Generate, run, extract, append one DES case. Returns True on success."""
    log.info(f"\n{'─'*60}")
    log.info(f"  Case: {case_id}")
    log.info(f"  slant={params['slant_angle']:.3f}°  diffuser={params['diffuser_angle']:.3f}°"
             f"  rh={params['ride_height']:.3f}mm  fr={params['front_radius']:.3f}mm")

    DES_CASES_DIR.mkdir(exist_ok=True)
    case_dir = write_des_case(case_id, params, DES_CASES_DIR)
    log.info(f"  Case written → {case_dir}")

    t0 = time.time()
    log.info(f"  Launching pimpleFoam ...")
    result = subprocess.run(["bash", str(case_dir / "run.sh")], check=False)
    elapsed = (time.time() - t0) / 60
    log.info(f"  Finished in {elapsed:.1f} min  (rc={result.returncode})")

    if result.returncode != 0:
        log.warning(f"  run.sh non-zero return code — attempting force extraction anyway")

    try:
        cd_des, cl_des = extract_forces(case_dir)
        log.info(f"  Cd_DES={cd_des:.4f}  Cl_DES={cl_des:.4f}")
    except FileNotFoundError as e:
        log.error(f"  Force extraction failed: {e}")
        return False

    cd_rans, cl_rans = get_rans_prediction(params)
    log.info(f"  Cd_RANS={cd_rans:.4f}  Cl_RANS={cl_rans:.4f}"
             f"  δCd={cd_des-cd_rans:+.4f}  δCl={cl_des-cl_rans:+.4f}")

    append_result(case_id, params, cd_des, cl_des, cd_rans, cl_rans)
    return True


def update_surrogate() -> float:
    """Re-run co_kriging.py and return the new EI value."""
    log.info("\n  Re-fitting co-Kriging surrogate ...")
    subprocess.run([sys.executable, "co_kriging.py"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ei = json.loads(CANDIDATE_JSON.read_text())["EI"]
    candidate = json.loads(CANDIDATE_JSON.read_text())
    log.info(f"  New EI = {ei:.6f}")
    log.info(f"  Next candidate: slant={candidate['slant_angle']:.3f}°"
             f"  diffuser={candidate['diffuser_angle']:.3f}°"
             f"  rh={candidate['ride_height']:.3f}mm"
             f"  fr={candidate['front_radius']:.3f}mm")
    return ei, candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter",     type=int,   default=10)
    parser.add_argument("--ei-threshold", type=float, default=0.005)
    args = parser.parse_args()
    acquire_lock()
    try:
        _main(args)
    finally:
        release_lock()


def _main(args):
    log.info("=" * 60)
    log.info("  Multi-fidelity BO loop")
    log.info(f"  Max iterations:  {args.max_iter}")
    log.info(f"  EI threshold:    {args.ei_threshold}")
    log.info("=" * 60)

    # Load current EI from last co_kriging run
    candidate_data = json.loads(CANDIDATE_JSON.read_text())
    ei = candidate_data["EI"]
    log.info(f"\n  Starting EI = {ei:.6f}")

    for iteration in range(1, args.max_iter + 1):
        log.info(f"\n{'='*60}")
        log.info(f"  BO ITERATION {iteration}/{args.max_iter}  (current EI={ei:.6f})")
        log.info(f"{'='*60}")

        if ei < args.ei_threshold:
            log.info(f"\n  EI={ei:.6f} < threshold={args.ei_threshold} — converged. Stopping.")
            break

        # Read current best candidate
        candidate_data = json.loads(CANDIDATE_JSON.read_text())
        params = {f: candidate_data[f] for f in FEATURES}
        case_id = next_case_id()

        success = run_iteration(case_id, params)
        if not success:
            log.error(f"  Iteration {iteration} failed — stopping loop.")
            break

        ei, candidate_data = update_surrogate()

    log.info(f"\n{'='*60}")
    log.info(f"  Loop complete. Final EI = {ei:.6f}")
    if ei < args.ei_threshold:
        log.info("  Status: CONVERGED")
    else:
        log.info(f"  Status: max iterations reached (EI={ei:.6f} > threshold={args.ei_threshold})")
    log.info(f"  Results: results/mf_optimum_design.json")
    log.info(f"  Pareto:  results/mf_pareto_front.png")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
