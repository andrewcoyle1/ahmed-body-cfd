"""
run_local_sampling.py
=====================
Focused local sampling around the identified MF optimum region.
Places a Latin Hypercube in a tight box around slant~39°, diffuser~18°,
ride_height~46mm — the region consistently identified by BO.
Front radius is fixed at 100mm (shown to be low-sensitivity).

Runs each case, appends to des_results.csv, refits co_kriging after each.
EI should decay monotonically once the correction GP is dense locally.
"""

import json, subprocess, sys, logging, warnings
from pathlib import Path
from pipeline_lock import acquire_lock, release_lock

import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from des_case_generator import write_des_case, DES_CASES_DIR

# ─── Local sampling box ───────────────────────────────────────────────────────
LOCAL_BOUNDS = {
    "slant_angle":    (37.5, 40.5),
    "diffuser_angle": (16.5, 19.5),
    "ride_height":    (43.0, 49.0),
    "front_radius":   (100.0, 100.0),   # fixed
}
N_LOCAL     = 12
FEATURES    = ["slant_angle", "diffuser_angle", "ride_height", "front_radius"]
AVG_START   = 0.13
DES_RESULTS = Path("des_output/des_results.csv")
RANS_CSV    = Path("results/results_summary.csv")
LOG_FILE    = Path("local_sampling.log")

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
    ids = [int(p.name.split("_")[-1]) for p in existing
           if p.name.split("_")[-1].isdigit()]
    return f"des_case_{(max(ids) + 1):03d}" if ids else "des_case_044"


def lhs_local(n: int, bounds: dict, seed: int = 42) -> list[dict]:
    """Generate n LHS points in the local box, excluding fixed variables."""
    free_vars = [k for k, (lo, hi) in bounds.items() if lo != hi]
    fixed     = {k: lo for k, (lo, hi) in bounds.items() if lo == hi}

    sampler = qmc.LatinHypercube(d=len(free_vars), seed=seed)
    sample  = sampler.random(n)
    lows  = np.array([bounds[k][0] for k in free_vars])
    highs = np.array([bounds[k][1] for k in free_vars])
    scaled = qmc.scale(sample, lows, highs)

    points = []
    for row in scaled:
        p = {k: float(v) for k, v in zip(free_vars, row)}
        p.update(fixed)
        points.append(p)
    return points


def fit_rans_gp(X, y):
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    kernel = ConstantKernel(1.0) * Matern(
        length_scale=np.ones(X.shape[1]),
        length_scale_bounds=(1e-2, 1e2), nu=2.5
    ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                  normalize_y=True, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
    df = pd.read_csv(RANS_CSV)
    X  = df[FEATURES].values
    xn = np.array([[params[f] for f in FEATURES]])
    gp_cd, sc_cd = fit_rans_gp(X, df["Cd"].values)
    gp_cl, sc_cl = fit_rans_gp(X, df["Cl"].values)
    return (float(gp_cd.predict(sc_cd.transform(xn))[0]),
            float(gp_cl.predict(sc_cl.transform(xn))[0]))


def append_result(case_id, params, cd_des, cl_des, cd_rans, cl_rans):
    df  = pd.read_csv(DES_RESULTS)
    row = {"case_id": case_id, **{f: params[f] for f in FEATURES},
           "Cd_DES": cd_des, "Cl_DES": cl_des,
           "Cd_RANS": cd_rans, "Cl_RANS": cl_rans,
           "dCd": cd_des - cd_rans, "dCl": cl_des - cl_rans}
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(DES_RESULTS, index=False)


def update_surrogate() -> float:
    subprocess.run([sys.executable, "co_kriging.py"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.loads(Path("results/mf_next_candidate.json").read_text())["EI"]


def main():
    acquire_lock()
    try:
        _main()
    finally:
        release_lock()


def _main():
    log.info("=" * 60)
    log.info("  Local sampling — tight box around MF optimum")
    log.info(f"  Slant:     {LOCAL_BOUNDS['slant_angle']}")
    log.info(f"  Diffuser:  {LOCAL_BOUNDS['diffuser_angle']}")
    log.info(f"  Rh:        {LOCAL_BOUNDS['ride_height']}")
    log.info(f"  Fr:        fixed at {LOCAL_BOUNDS['front_radius'][0]} mm")
    log.info(f"  N points:  {N_LOCAL}")
    log.info("=" * 60)

    points = lhs_local(N_LOCAL, LOCAL_BOUNDS)

    log.info("\n  Sampling plan:")
    for i, p in enumerate(points):
        log.info(f"  {i+1:2d}. slant={p['slant_angle']:.3f}°  "
                 f"diff={p['diffuser_angle']:.3f}°  rh={p['ride_height']:.3f}mm")

    for i, params in enumerate(points, 1):
        case_id = next_case_id()
        log.info(f"\n{'='*60}")
        log.info(f"  LOCAL POINT {i}/{N_LOCAL} — {case_id}")
        log.info(f"  slant={params['slant_angle']:.3f}°  diff={params['diffuser_angle']:.3f}°  "
                 f"rh={params['ride_height']:.3f}mm  fr={params['front_radius']:.1f}mm")

        DES_CASES_DIR.mkdir(exist_ok=True)
        case_dir = write_des_case(case_id, params, DES_CASES_DIR)

        result = subprocess.run(["bash", str(case_dir / "run.sh")], check=False)
        log.info(f"  pimpleFoam rc={result.returncode}")

        try:
            cd_des, cl_des = extract_forces(case_dir)
            log.info(f"  Cd_DES={cd_des:.4f}  Cl_DES={cl_des:.4f}")
        except FileNotFoundError as e:
            log.error(f"  {e} — skipping")
            continue

        cd_rans, cl_rans = get_rans_prediction(params)
        log.info(f"  δCd={cd_des-cd_rans:+.4f}  δCl={cl_des-cl_rans:+.4f}")
        append_result(case_id, params, cd_des, cl_des, cd_rans, cl_rans)

        ei = update_surrogate()
        log.info(f"  EI after = {ei:.6f}")

    log.info(f"\n{'='*60}")
    log.info("  Local sampling complete.")
    log.info(f"  Final EI = {json.loads(Path('results/mf_next_candidate.json').read_text())['EI']:.6f}")
    log.info(f"  Optimum  → results/mf_optimum_design.json")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
