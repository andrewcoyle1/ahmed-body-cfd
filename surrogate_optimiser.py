"""
surrogate_optimiser.py
======================
Module 4 of the Ahmed body aerodynamic optimisation pipeline.

Fits Gaussian Process surrogate models to the 30-case DoE results, then:
  1. Validates GP accuracy (leave-one-out cross-validation)
  2. Runs single-objective Bayesian optimisation (minimise Cd)
  3. Builds a multi-objective Pareto front (Cd vs downforce Cl)
  4. Produces response surface slices across the 4D design space
  5. (--validate)        Runs CFD at the surrogate optimum and compares
  6. (--bayesian-loop N) Runs N iterations of the full Bayesian optimisation
                         loop: EI acquisition → CFD → GP refit → repeat

Bayesian Optimisation Loop
--------------------------
The loop extends the DoE by intelligently selecting the next simulation
point using Expected Improvement (EI), the standard acquisition function
for Bayesian optimisation:

    EI(x) = E[max(f_best - f(x), 0)]
           = (f_best - μ(x) - ξ) · Φ(Z) + σ(x) · φ(Z)

    where  Z = (f_best - μ(x) - ξ) / σ(x)
           μ(x), σ(x) = GP posterior mean and std at x
           Φ, φ       = standard normal CDF and PDF
           ξ          = exploration parameter (default 0.01)

EI automatically balances:
  - Exploitation: regions where the GP mean is low (likely good)
  - Exploration:  regions where GP uncertainty σ is high (might be better)

Each iteration:
  1. Maximise EI over the 4D design space (differential evolution)
  2. Generate an OpenFOAM case at the selected point
  3. Run simpleFoam via Docker (~53 min)
  4. Append the new (x, Cd) observation to the training set
  5. Refit the GP on all data collected so far
  6. Record EI magnitude — convergence when max(EI) < 1e-4

Outputs
-------
  results/gp_validation.png         — predicted vs actual Cd/Cl (LOO-CV)
  results/response_surfaces.png     — 2D slices through the 4D space
  results/pareto_front.png          — Pareto front Cd vs |Cl|
  results/optimum_design.json       — optimal single-objective design
  results/pareto_designs.csv        — Pareto-optimal design table
  results/optimum_validation.json   — (--validate) surrogate vs CFD at optimum
  results/bo_history.csv            — per-iteration BO record
  results/bo_convergence.png        — incumbent Cd trace + EI decay plot

Usage
-----
  python3 surrogate_optimiser.py                   # requires results_summary.csv
  python3 surrogate_optimiser.py --plot-only        # skip model fit, reload saved
  python3 surrogate_optimiser.py --validate         # run CFD at optimum + compare
  python3 surrogate_optimiser.py --bayesian-loop 15 # run 15 BO iterations

Dependencies: numpy, pandas, scipy, scikit-learn, matplotlib
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from pathlib import Path
from itertools import combinations

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.optimize import differential_evolution
import pickle

warnings.filterwarnings("ignore")

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

RESULTS_CSV  = Path("results/results_summary.csv")
RESULTS_DIR  = Path("results")

FEATURES     = ["slant_angle", "diffuser_angle", "ride_height", "front_radius"]
FEATURE_LABELS = ["Slant angle (°)", "Diffuser angle (°)", "Ride height (mm)", "Front radius (mm)"]

# Design space bounds — match doe_setup.py ranges exactly.
# front_radius upper bound clamped to 139 mm (H/2 - 5 mm geometric limit).
BOUNDS = {
    "slant_angle":    (15.0,  40.0),
    "diffuser_angle": ( 0.0,  20.0),
    "ride_height":    (30.0,  80.0),
    "front_radius":   (50.0, 139.0),
}

# Exploration–exploitation trade-off for EI.  ξ=0.01 is the standard default;
# increase toward 0.1 if the loop gets stuck exploiting a narrow basin.
EI_XI       = 0.01
EI_XI_LOCAL = 0.001   # tighter exploitation for local refinement

# ── Local refinement: tighter search box around the known best ────────────────
# Best found: slant=40°, diffuser=11.9°, ride_height=68.6mm, front_radius=50mm
LOCAL_BOUNDS = {
    "slant_angle":    (35.0, 40.0),
    "diffuser_angle": ( 7.0, 17.0),
    "ride_height":    (58.0, 78.0),
    "front_radius":   (50.0, 75.0),
}

# Cases excluded from GP training — physically inconsistent outliers that
# saturate the noise kernel and corrupt EI calculations.
OUTLIER_CASES = set()   # populated manually if a case proves physically inconsistent

BO_CASES_DIR = Path("openfoam_cases")   # same directory as DoE cases

# ── F1 aerodynamic objective ──────────────────────────────────────────────────
# In Formula 1 aerodynamics, downforce is worth approximately 3× more than
# equivalent drag in lap time terms across most circuits.  This is derived
# from the relationship between cornering speed (∝ √Cl) and straight-line
# speed (∝ 1/√Cd), and is consistent with published sensitivity studies
# (e.g. Milliken & Milliken, Race Car Vehicle Dynamics, SAE 1994).
#
# The scalarised F1 objective is therefore:
#
#     f(x) = Cd + λ · Cl      (minimise)
#
# where λ = 1/3.  A reduction of ΔCl = −0.3 (more downforce) is worth as
# much as ΔCd = +0.1 extra drag — the BO will naturally trade drag for
# downforce at this ratio.
#
# Note: our sign convention has Cl > 0 = lift (bad), Cl < 0 = downforce (good).
# Minimising Cd + λ·Cl simultaneously pushes for low drag AND negative Cl.
F1_LAMBDA = 1.0 / 3.0


def f1_objective(y_cd: np.ndarray, y_cl: np.ndarray) -> np.ndarray:
    """
    Scalarised F1 aerodynamic objective.

        f = Cd + (1/3) · Cl      [minimise]

    Lower is better: rewards low drag and negative Cl (downforce).
    The 1/3 weighting reflects that downforce is ~3× more valuable than
    drag reduction in lap time across a representative F1 circuit mix.
    """
    return y_cd + F1_LAMBDA * y_cl


# ─── 1. DATA LOADING ──────────────────────────────────────────────────────────

def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"{RESULTS_CSV} not found — run post_processor.py first"
        )
    df = pd.read_csv(RESULTS_CSV)
    valid = df.dropna(subset=["Cd", "Cl"])
    if len(valid) < 5:
        raise ValueError(f"Only {len(valid)} converged cases — need at least 5 to fit GP")

    print(f"Loaded {len(valid)} converged cases from {RESULTS_CSV}")
    X = valid[FEATURES].values.astype(float)
    y_cd = valid["Cd"].values.astype(float)
    y_cl = valid["Cl"].values.astype(float)
    return X, y_cd, y_cl, valid


# ─── 2. GP SURROGATE ──────────────────────────────────────────────────────────

def build_kernel():
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(len(FEATURES)), length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
    )


def fit_gp(X: np.ndarray, y: np.ndarray, scaler: StandardScaler) -> GaussianProcessRegressor:
    Xs = scaler.transform(X)
    gp = GaussianProcessRegressor(
        kernel=build_kernel(),
        n_restarts_optimizer=10,
        normalize_y=True,
        random_state=42,
    )
    gp.fit(Xs, y)
    return gp


def loo_cv(X: np.ndarray, y: np.ndarray, scaler: StandardScaler) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-out cross-validation predictions."""
    preds, stds = np.zeros_like(y), np.zeros_like(y)
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X):
        sc = StandardScaler().fit(X[train_idx])
        gp = GaussianProcessRegressor(
            kernel=build_kernel(),
            n_restarts_optimizer=5,
            normalize_y=True,
            random_state=42,
        )
        gp.fit(sc.transform(X[train_idx]), y[train_idx])
        p, s = gp.predict(sc.transform(X[test_idx]), return_std=True)
        preds[test_idx] = p
        stds[test_idx] = s
    return preds, stds


# ─── 3. VALIDATION PLOTS ──────────────────────────────────────────────────────

def plot_validation(y_cd, y_cl, cd_loo, cl_loo, cd_std, cl_std):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("GP Surrogate Validation — Leave-One-Out Cross-Validation", fontsize=13)

    for ax, y_true, y_pred, std, label in zip(
        axes,
        [y_cd, y_cl],
        [cd_loo, cl_loo],
        [cd_std, cl_std],
        ["$C_d$", "$C_l$"],
    ):
        lo = min(y_true.min(), y_pred.min()) * 0.97
        hi = max(y_true.max(), y_pred.max()) * 1.03
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="Perfect prediction")
        ax.errorbar(y_true, y_pred, yerr=2 * std, fmt="o", color="#1f77b4",
                    ecolor="#aec7e8", elinewidth=1.5, capsize=3, ms=6, label="LOO prediction ± 2σ")
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        ax.set_xlabel(f"CFD {label}", fontsize=11)
        ax.set_ylabel(f"GP {label}", fontsize=11)
        ax.set_title(f"{label}  |  R² = {r2:.3f}  |  MAE = {mae:.4f}", fontsize=10)
        ax.legend(fontsize=9)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")

    plt.tight_layout()
    out = RESULTS_DIR / "gp_validation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ─── 4. RESPONSE SURFACES ─────────────────────────────────────────────────────

def plot_response_surfaces(gp_cd: GaussianProcessRegressor, scaler: StandardScaler, X: np.ndarray):
    """Six 2D slices through the 4D space; remaining dims held at their median."""
    medians = np.median(X, axis=0)  # shape (4,)
    pairs = list(combinations(range(4), 2))   # 6 pairs

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("GP Surrogate — $C_d$ Response Surfaces (other dims at median)", fontsize=13)

    for ax, (i, j) in zip(axes.flat, pairs):
        b_i = BOUNDS[FEATURES[i]]
        b_j = BOUNDS[FEATURES[j]]
        xi = np.linspace(*b_i, 50)
        xj = np.linspace(*b_j, 50)
        II, JJ = np.meshgrid(xi, xj)

        grid = np.tile(medians, (II.size, 1))
        grid[:, i] = II.ravel()
        grid[:, j] = JJ.ravel()

        Cd_pred = gp_cd.predict(scaler.transform(grid)).reshape(II.shape)

        cf = ax.contourf(II, JJ, Cd_pred, levels=20, cmap="RdYlGn_r")
        plt.colorbar(cf, ax=ax, label="$C_d$")
        ax.scatter(X[:, i], X[:, j], c="white", edgecolors="black", s=25, zorder=5,
                   label="DoE samples")
        ax.set_xlabel(FEATURE_LABELS[i], fontsize=9)
        ax.set_ylabel(FEATURE_LABELS[j], fontsize=9)
        ax.set_title(f"{FEATURES[i]} vs {FEATURES[j]}", fontsize=9)

    plt.tight_layout()
    out = RESULTS_DIR / "response_surfaces.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ─── 5. SINGLE-OBJECTIVE OPTIMISATION ─────────────────────────────────────────

def optimise_f1(
    gp_cd: GaussianProcessRegressor,
    gp_cl: GaussianProcessRegressor,
    scaler: StandardScaler,
) -> dict:
    """
    Minimise the F1 scalarised objective  f = Cd + (1/3)·Cl  over the full
    design space using differential evolution.

    This finds the design that best balances drag reduction and downforce
    generation at an F1-relevant trade-off ratio, rather than minimising
    drag alone (which would ignore downforce entirely).
    """
    bounds_list = [BOUNDS[f] for f in FEATURES]

    def objective(x):
        xs     = scaler.transform(x.reshape(1, -1))
        cd_hat = float(gp_cd.predict(xs)[0])
        cl_hat = float(gp_cl.predict(xs)[0])
        return float(f1_objective(np.array([cd_hat]), np.array([cl_hat]))[0])

    result = differential_evolution(
        objective,
        bounds=bounds_list,
        seed=42,
        maxiter=500,
        tol=1e-6,
    )

    xs      = scaler.transform(result.x.reshape(1, -1))
    cd_pred = float(gp_cd.predict(xs)[0])
    cl_pred = float(gp_cl.predict(xs)[0])

    opt = {feat: round(float(val), 4) for feat, val in zip(FEATURES, result.x)}
    opt["Cd_predicted"]  = round(cd_pred, 6)
    opt["Cl_predicted"]  = round(cl_pred, 6)
    opt["f1_objective"]  = round(float(result.fun), 6)
    opt["F1_lambda"]     = F1_LAMBDA
    return opt


# ─── 6. PARETO FRONT ──────────────────────────────────────────────────────────

def pareto_front(costs: np.ndarray) -> np.ndarray:
    """Return boolean mask of non-dominated points (minimise both objectives)."""
    n = len(costs)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
                dominated[i] = True
                break
    return ~dominated


def compute_and_plot_pareto(
    gp_cd: GaussianProcessRegressor,
    gp_cl: GaussianProcessRegressor,
    scaler: StandardScaler,
    y_cd: np.ndarray,
    y_cl: np.ndarray,
) -> pd.DataFrame:
    """Sample 20k random designs, predict Cd and |Cl|, extract Pareto front."""
    rng = np.random.default_rng(42)
    N = 20_000
    X_rand = np.column_stack([
        rng.uniform(*BOUNDS[f], N) for f in FEATURES
    ])
    Xs = scaler.transform(X_rand)
    cd_pred = gp_cd.predict(Xs)
    cl_pred = gp_cl.predict(Xs)
    downforce = -cl_pred          # positive = more downforce

    # Objectives: minimise Cd, maximise downforce (= minimise -downforce)
    costs = np.column_stack([cd_pred, -downforce])
    mask = pareto_front(costs)

    pareto_cd  = cd_pred[mask]
    pareto_cl  = cl_pred[mask]
    pareto_X   = X_rand[mask]
    sort_idx   = np.argsort(pareto_cd)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(cd_pred, -cl_pred, c="#cccccc", s=5, alpha=0.3, label="Sampled designs")
    ax.scatter(y_cd, -y_cl, c="#1f77b4", s=50, zorder=5, label="CFD cases (DoE)")
    ax.plot(pareto_cd[sort_idx], -pareto_cl[sort_idx],
            "ro-", ms=6, lw=2, zorder=6, label="Pareto front")

    ax.set_xlabel("Drag coefficient $C_d$", fontsize=12)
    ax.set_ylabel("Downforce $-C_l$", fontsize=12)
    ax.set_title("Multi-objective Pareto Front: Drag vs Downforce", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out = RESULTS_DIR / "pareto_front.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")

    df_pareto = pd.DataFrame(pareto_X[sort_idx], columns=FEATURES)
    df_pareto["Cd_predicted"] = pareto_cd[sort_idx]
    df_pareto["Cl_predicted"] = pareto_cl[sort_idx]
    df_pareto = df_pareto.round(4)
    return df_pareto


# ─── 7. BAYESIAN OPTIMISATION LOOP ───────────────────────────────────────────

def expected_improvement(
    X_cand: np.ndarray,
    gp: GaussianProcessRegressor,
    y_best: float,
    scaler: StandardScaler,
    xi: float = EI_XI,
) -> np.ndarray:
    """
    Expected Improvement acquisition function.

    For minimisation (lower Cd is better):

        EI(x) = (f_best - μ(x) - ξ) · Φ(Z) + σ(x) · φ(Z)
        Z      = (f_best - μ(x) - ξ) / σ(x)

    Parameters
    ----------
    X_cand : (N, 4) array of candidate points in original (unscaled) space
    gp     : fitted GaussianProcessRegressor for Cd
    y_best : current best observed Cd (incumbent)
    scaler : StandardScaler fitted on training X
    xi     : exploration parameter — larger → more exploration

    Returns
    -------
    ei : (N,) array — EI value at each candidate point
    """
    from scipy.stats import norm
    Xs = scaler.transform(X_cand)
    mu, sigma = gp.predict(Xs, return_std=True)
    sigma = np.maximum(sigma, 1e-9)          # avoid division by zero
    Z  = (y_best - mu - xi) / sigma
    ei = (y_best - mu - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei = np.maximum(ei, 0.0)                 # EI is non-negative by definition
    return ei


def maximise_ei(
    gp_cd: GaussianProcessRegressor,
    scaler: StandardScaler,
    y_best: float,
    bounds_override: dict | None = None,
    xi: float = EI_XI,
) -> tuple[np.ndarray, float]:
    """
    Find the point in the design space that maximises Expected Improvement.

    Uses differential evolution (global) to avoid local optima in the
    acquisition surface, which is often multi-modal.

    Parameters
    ----------
    bounds_override : optional dict mapping feature name → (lo, hi).
                      Defaults to global BOUNDS (full design space).
                      Pass LOCAL_BOUNDS for the local refinement phase.

    Returns
    -------
    x_next   : (4,) array — next design point to evaluate
    ei_max   : scalar — EI value at x_next (convergence indicator)
    """
    b = bounds_override if bounds_override is not None else BOUNDS
    bounds_list = [b[f] for f in FEATURES]

    def neg_ei(x):
        ei = expected_improvement(x.reshape(1, -1), gp_cd, y_best, scaler, xi=xi)
        return -float(np.ravel(ei)[0])

    result = differential_evolution(
        neg_ei,
        bounds=bounds_list,
        seed=None,   # vary seed so failed-case repeats don't lock onto the same point
        maxiter=300,
        tol=1e-8,
        popsize=15,
    )
    return result.x, -result.fun


BO_N_CORES = 10  # cores per Bayesian loop case (one case at a time → use all cores)


def run_bo_case(
    case_id: str,
    params: dict,
) -> tuple[float | None, float | None, bool]:
    """
    Write, run, and post-process a single Bayesian optimisation CFD case.

    The case is written into BO_CASES_DIR using the same case_generator
    machinery as the DoE, so all mesh settings, BCs, and solver config are
    identical to the training data.

    Unlike the DoE (4 serial cases in parallel), BO runs one case at a time
    so all BO_N_CORES cores are given to that single case via MPI parallelism
    (decomposePar + mpirun -np {BO_N_CORES} + reconstructPar).

    Returns
    -------
    cd        : converged Cd or None on failure
    cl        : converged Cl or None on failure
    converged : True if residuals met threshold
    """
    import subprocess
    import importlib.util

    # Lazy-import case_generator and post_processor to avoid circular imports
    def _load(name, fpath):
        spec = importlib.util.spec_from_file_location(name, fpath)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    cg = _load("case_generator", Path(__file__).parent / "case_generator.py")
    pp = _load("post_processor",  Path(__file__).parent / "post_processor.py")

    case_dir = cg.write_case(case_id, params, BO_CASES_DIR, n_cores=BO_N_CORES, n_iters=3000)
    print(f"    Case written → {case_dir}  ({BO_N_CORES} cores)")

    # Write params immediately so dashboard can show them while CFD runs
    import json as _json
    (case_dir / "params.json").write_text(_json.dumps(params, indent=2))

    run_sh = case_dir / "run.sh"
    log_path = case_dir / "bo_run.log"

    # Kill any orphaned containers mounting this case directory before launching.
    # docker --filter volume= only matches named volumes, not bind-mounts, so we
    # inspect all running containers and stop any whose Mounts include this path.
    case_path = str(case_dir.resolve())
    subprocess.run(
        ["bash", "-c",
         f'docker ps -q | xargs -r docker inspect --format '
         f'"{{{{.Id}}}} {{{{range .Mounts}}}}{{{{.Source}}}} {{{{end}}}}"'
         f' | grep "{case_path}" | awk \'{{print $1}}\' | xargs -r docker stop'],
        capture_output=True,
    )

    print(f"    Running CFD (this takes ~53 min)…")
    result = subprocess.run(
        ["bash", str(run_sh)],
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
    )

    if result.returncode != 0:
        print(f"    ERROR: CFD failed for {case_id} — check {log_path}")
        return None, None, False

    force     = pp.extract_force_coeffs(case_dir)
    converged = pp.check_convergence(case_dir)
    return force.get("Cd"), force.get("Cl"), converged


def bayesian_loop(
    n_iterations: int,
    X_init: np.ndarray,
    y_cd_init: np.ndarray,
    y_cl_init: np.ndarray,
    scaler_init: StandardScaler,
    bounds_override: dict | None = None,
    exclude_cases: set | None = None,
    ei_xi: float = EI_XI,
) -> pd.DataFrame:
    """
    Run the Bayesian optimisation loop for n_iterations steps.

    Objective
    ---------
    The loop targets the F1 scalarised objective:

        f(x) = Cd + (1/3) · Cl      [minimise]

    A dedicated GP is fitted to f each iteration.  EI is computed against f,
    so the acquisition function simultaneously accounts for both drag reduction
    and downforce gain — rather than treating them as separate objectives.

    The Cd and Cl GPs (for the Pareto front) are maintained separately and
    updated each iteration so that the Pareto analysis remains current.

    Algorithm
    ---------
    For i = 1 … n_iterations:
      1. Compute y_f1 = Cd + λ·Cl for all training data collected so far.
      2. Fit GP_f1 on (X, y_f1).  Also refit GP_cd, GP_cl for Pareto tracking.
      3. Compute EI(x) w.r.t. GP_f1; find x* = argmax EI.
      4. If max(EI) < 1e-4 — terminate early (surrogate converged).
      5. Generate and run a CFD case at x* using 8 MPI cores.
      6. Append (x*, Cd, Cl) to training set; log to bo_history.csv.

    Returns
    -------
    history : DataFrame with one row per BO iteration
    """
    EI_CONVERGENCE = 1e-4

    X    = X_init.copy()
    y_cd = y_cd_init.copy()
    y_cl = y_cl_init.copy()

    history_rows = []
    history_path = RESULTS_DIR / "bo_history.csv"
    start_iter   = 1

    if history_path.exists():
        prev = pd.read_csv(history_path)
        if len(prev) > 0:
            excluded = set(exclude_cases or [])
            n_excl = prev["case_id"].isin(excluded).sum()
            print(f"  Resuming from existing bo_history.csv ({len(prev)} prior iterations"
                  + (f", {n_excl} excluded from GP training: {excluded}" if n_excl else "") + ")")
            start_iter   = int(prev["iteration"].max()) + 1
            history_rows = prev.to_dict("records")
            for _, row in prev.iterrows():
                if row.get("case_id") in excluded:
                    continue   # outlier — omit from GP training
                if pd.notna(row.get("Cd_cfd")):
                    x_row = np.array([[row[f] for f in FEATURES]])
                    X    = np.vstack([X,    x_row])
                    y_cd = np.append(y_cd, row["Cd_cfd"])
                    y_cl = np.append(y_cl, row["Cl_cfd"] if pd.notna(row.get("Cl_cfd")) else 0.0)

    y_f1_init = f1_objective(y_cd, y_cl)
    print(f"\n  Objective: f = Cd + (1/{1/F1_LAMBDA:.0f})·Cl  [F1 aero efficiency]")
    print(f"  Starting BO loop — {n_iterations} iterations")
    print(f"  DoE best f1 = {y_f1_init.min():.4f}  "
          f"(Cd={y_cd[y_f1_init.argmin()]:.4f}, "
          f"Cl={y_cl[y_f1_init.argmin()]:.4f})")

    for i in range(start_iter, n_iterations + 1):
        print(f"\n── BO Iteration {i} ──────────────────────────────────────")

        y_f1   = f1_objective(y_cd, y_cl)
        scaler = StandardScaler().fit(X)
        Xs     = scaler.transform(X)

        gp_f1 = GaussianProcessRegressor(
            kernel=build_kernel(), n_restarts_optimizer=5,
            normalize_y=True, random_state=42,
        )
        gp_f1.fit(Xs, y_f1)
        print(f"  GP_f1 refitted on {len(X)} points  |  kernel: {gp_f1.kernel_}")

        y_best         = float(y_f1.min())
        x_next, ei_max = maximise_ei(gp_f1, scaler, y_best,
                                     bounds_override=bounds_override, xi=ei_xi)

        x_next_s              = scaler.transform(x_next.reshape(1, -1))
        f1_pred, f1_std       = gp_f1.predict(x_next_s, return_std=True)

        # Also predict Cd and Cl individually for logging
        gp_cd_iter = GaussianProcessRegressor(
            kernel=build_kernel(), n_restarts_optimizer=3,
            normalize_y=True, random_state=42,
        )
        gp_cd_iter.fit(Xs, y_cd)
        gp_cl_iter = GaussianProcessRegressor(
            kernel=build_kernel(), n_restarts_optimizer=3,
            normalize_y=True, random_state=42,
        )
        gp_cl_iter.fit(Xs, y_cl)
        cd_pred = float(gp_cd_iter.predict(x_next_s)[0])
        cl_pred = float(gp_cl_iter.predict(x_next_s)[0])

        params_next = {f: float(v) for f, v in zip(FEATURES, x_next)}
        print(f"  f1_best = {y_best:.4f}  |  max EI = {ei_max:.6f}")
        print(f"  Next point: { {k: round(v,2) for k,v in params_next.items()} }")
        print(f"  GP prediction: f1={float(f1_pred[0]):.4f} ± {float(f1_std[0]):.4f}  "
              f"(Cd≈{cd_pred:.4f}, Cl≈{cl_pred:.4f})")

        if ei_max < EI_CONVERGENCE:
            print(f"  EI < {EI_CONVERGENCE} — F1 objective has converged. Stopping.")
            break

        case_id = f"case_bo_{i:03d}"
        cd_cfd, cl_cfd, converged = run_bo_case(case_id, params_next)

        if cd_cfd is None:
            print(f"  CFD failed for {case_id} — skipping iteration")
            # f1_best = incumbent unchanged (no new data appended)
            row = {
                "iteration": i, "case_id": case_id,
                **{f: round(v, 4) for f, v in params_next.items()},
                "f1_predicted": round(float(f1_pred[0]), 6),
                "f1_std":       round(float(f1_std[0]), 6),
                "Cd_predicted": round(cd_pred, 6),
                "Cl_predicted": round(cl_pred, 6),
                "ei_max":       round(ei_max, 8),
                "Cd_cfd": None, "Cl_cfd": None, "f1_cfd": None,
                "converged": False,
                "f1_best": round(y_best, 6),           # incumbent unchanged
                "Cd_best": round(float(y_cd[y_f1.argmin()]), 6),
            }
            history_rows.append(row)
            pd.DataFrame(history_rows).to_csv(history_path, index=False)
            # Vary the next candidate by perturbing the failed point slightly
            # so the GP doesn't recommend the identical point next iteration.
            # (GP landscape unchanged since no new data was added.)
            continue

        f1_cfd = float(f1_objective(np.array([cd_cfd]),
                                    np.array([cl_cfd if cl_cfd else 0.0]))[0])
        X    = np.vstack([X,    x_next.reshape(1, -1)])
        y_cd = np.append(y_cd, cd_cfd)
        y_cl = np.append(y_cl, cl_cfd if cl_cfd is not None else 0.0)

        new_f1_best  = float(f1_objective(y_cd, y_cl).min())
        improvement  = y_best - new_f1_best
        print(f"  CFD: Cd={cd_cfd:.4f}  Cl={cl_cfd:.4f}  f1={f1_cfd:.4f}  conv={converged}")
        print(f"  f1 incumbent: {y_best:.4f} → {new_f1_best:.4f}  "
              f"({'↓ ' + f'{improvement:.4f}' if improvement > 0 else 'no improvement'})")

        row = {
            "iteration": i, "case_id": case_id,
            **{f: round(v, 4) for f, v in params_next.items()},
            "f1_predicted": round(float(f1_pred[0]), 6),
            "f1_std":       round(float(f1_std[0]), 6),
            "Cd_predicted": round(cd_pred, 6),
            "Cl_predicted": round(cl_pred, 6),
            "ei_max":       round(ei_max, 8),
            "Cd_cfd":       round(cd_cfd, 6),
            "Cl_cfd":       round(cl_cfd, 6) if cl_cfd is not None else None,
            "f1_cfd":       round(f1_cfd, 6),
            "converged":    converged,
            "f1_best":      round(new_f1_best, 6),
            "Cd_best":      round(float(y_cd[f1_objective(y_cd, y_cl).argmin()]), 6),
        }
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(history_path, index=False)
        print(f"  History saved → {history_path}")

    return pd.DataFrame(history_rows)


def plot_bo_convergence(history: pd.DataFrame):
    """
    Generate two-panel report figure for the Bayesian optimisation loop.

    Top panel — Incumbent trace:
        Best Cd found so far plotted against BO iteration number.
        A flat line means the current iteration did not improve on the
        best known design; a step down marks a genuine improvement.

    Bottom panel — EI decay:
        Maximum Expected Improvement at each iteration on a log scale.
        EI naturally decreases as the surrogate fills in the design space
        and uncertainty is resolved. Convergence is declared when
        max(EI) < 1e-4 (dashed threshold line).

    The figure is saved to results/bo_convergence.png for inclusion in
    the report.
    """
    if history.empty or "Cd_cfd" not in history.columns:
        print("  No BO history to plot.")
        return

    valid = history.dropna(subset=["Cd_cfd"])
    if valid.empty:
        print("  No successful BO iterations to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.suptitle("Bayesian Optimisation Loop — Convergence", fontsize=13, fontweight="bold")

    iters    = valid["iteration"].values
    f1_best  = valid["f1_best"].values
    ei_vals  = valid["ei_max"].values
    f1_cfd   = valid["f1_cfd"].values

    # Top: incumbent trace (f1 = Cd + λ·Cl — the actual optimisation objective)
    ax1.step(iters, f1_best, where="post", color="#1f77b4", lw=2.5, label="Best f1 incumbent")
    ax1.scatter(iters, f1_cfd, color="#ff7f0e", zorder=5, s=60, label="CFD f1 (this iter)")
    ax1.set_ylabel("F1 objective  $f_1 = C_d + \\frac{1}{3}C_l$", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Incumbent trace (F1 objective)", fontsize=10)

    # Bottom: EI decay
    ax2.semilogy(iters, ei_vals, "s-", color="#2ca02c", lw=2, ms=7, label="max EI")
    ax2.axhline(1e-4, color="red", lw=1.5, ls="--", label="Convergence threshold (1e-4)")
    ax2.set_xlabel("BO Iteration", fontsize=11)
    ax2.set_ylabel("max Expected Improvement", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_title("EI decay — convergence indicator", fontsize=10)

    plt.tight_layout()
    out = RESULTS_DIR / "bo_convergence.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ─── 9. OPTIMUM VALIDATION ────────────────────────────────────────────────────

def validate_optimum(gp_cd: GaussianProcessRegressor, gp_cl: GaussianProcessRegressor,
                     scaler: StandardScaler):
    """
    Generate and run a CFD case at the surrogate optimum, then compare
    predicted vs simulated Cd/Cl to quantify surrogate error at the optimum.
    """
    import subprocess
    import importlib.util

    opt_path = RESULTS_DIR / "optimum_design.json"
    if not opt_path.exists():
        print("  ERROR: results/optimum_design.json not found — run optimisation first")
        return

    with open(opt_path) as f:
        opt = json.load(f)

    print(f"  Optimal design: {opt}")

    # Build the params dict expected by case_generator.write_case()
    params = {
        "slant_angle":    opt["slant_angle"],
        "diffuser_angle": opt["diffuser_angle"],
        "ride_height":    opt["ride_height"],
        "front_radius":   opt["front_radius"],
    }

    # Import write_case from case_generator without executing its main()
    spec = importlib.util.spec_from_file_location(
        "case_generator", Path(__file__).parent / "case_generator.py"
    )
    cg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cg)

    case_id  = "case_optimum"
    case_dir = Path("openfoam_cases")
    case_dir.mkdir(exist_ok=True)

    print(f"  Generating OpenFOAM case → openfoam_cases/{case_id}/")
    cg.write_case(case_id, params, case_dir)

    print(f"  Running CFD via Docker (this takes ~20 min)...")
    result = subprocess.run(
        ["bash", "docker_run_case.sh", case_id],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"  ERROR: Docker run failed for {case_id}")
        return

    # Extract Cd/Cl using the same logic as post_processor.py
    spec2 = importlib.util.spec_from_file_location(
        "post_processor", Path(__file__).parent / "post_processor.py"
    )
    pp = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(pp)

    cfd_result = pp.extract_force_coeffs(case_dir / case_id)
    converged  = pp.check_convergence(case_dir / case_id)

    if cfd_result["Cd"] is None:
        print(f"  ERROR: Could not extract Cd from {case_id} — check log.simpleFoam")
        return

    # GP prediction with uncertainty at this point
    x_opt = np.array([[params[f] for f in FEATURES]])
    cd_pred, cd_std = gp_cd.predict(scaler.transform(x_opt), return_std=True)
    cl_pred, cl_std = gp_cl.predict(scaler.transform(x_opt), return_std=True)

    cd_cfd = cfd_result["Cd"]
    cl_cfd = cfd_result["Cl"]
    cd_err = abs(cd_cfd - float(cd_pred[0])) / abs(cd_cfd) * 100
    cl_err = abs(cl_cfd - float(cl_pred[0])) / (abs(cl_cfd) + 1e-9) * 100

    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │           SURROGATE VALIDATION AT OPTIMUM       │")
    print(f"  ├──────────────┬──────────────┬───────────────────┤")
    print(f"  │              │   Surrogate  │        CFD        │")
    print(f"  ├──────────────┼──────────────┼───────────────────┤")
    print(f"  │  Cd          │  {float(cd_pred[0]):.4f} ± {float(cd_std[0]):.4f}  │  {cd_cfd:.4f}  ({cd_err:.1f}% err)  │")
    print(f"  │  Cl          │  {float(cl_pred[0]):.4f} ± {float(cl_std[0]):.4f}  │  {cl_cfd:.4f}  ({cl_err:.1f}% err)  │")
    print(f"  │  Converged   │      —       │  {converged}             │")
    print(f"  └──────────────┴──────────────┴───────────────────┘")

    validation = {
        "case_id":          case_id,
        "design_params":    params,
        "Cd_surrogate":     round(float(cd_pred[0]), 6),
        "Cd_surrogate_std": round(float(cd_std[0]), 6),
        "Cd_cfd":           round(cd_cfd, 6),
        "Cd_error_pct":     round(cd_err, 2),
        "Cl_surrogate":     round(float(cl_pred[0]), 6),
        "Cl_surrogate_std": round(float(cl_std[0]), 6),
        "Cl_cfd":           round(cl_cfd, 6),
        "Cl_error_pct":     round(cl_err, 2),
        "converged":        converged,
    }
    out = RESULTS_DIR / "optimum_validation.json"
    with open(out, "w") as f:
        json.dump(validation, f, indent=2)
    print(f"\n  Saved → {out}")


# ─── 10. MAIN ─────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    model_cache = RESULTS_DIR / "gp_models.pkl"
    plot_only    = "--plot-only"     in sys.argv
    validate     = "--validate"      in sys.argv
    bo_flag      = "--bayesian-loop" in sys.argv
    local_flag   = "--local-refine"  in sys.argv

    n_bo_iters = 0
    if bo_flag:
        idx = sys.argv.index("--bayesian-loop")
        try:
            n_bo_iters = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Usage: python3 surrogate_optimiser.py --bayesian-loop <N>")
            sys.exit(1)

    n_local_iters = 0
    if local_flag:
        idx = sys.argv.index("--local-refine")
        try:
            n_local_iters = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Usage: python3 surrogate_optimiser.py --local-refine <N>")
            sys.exit(1)

    X, y_cd, y_cl, df = load_data()
    scaler = StandardScaler().fit(X)

    if plot_only and model_cache.exists():
        print("Loading cached GP models...")
        with open(model_cache, "rb") as f:
            gp_cd, gp_cl, scaler = pickle.load(f)
    else:
        print("\n── Fitting GP surrogates ──────────────────────────────────────")
        gp_cd = fit_gp(X, y_cd, scaler)
        gp_cl = fit_gp(X, y_cl, scaler)
        with open(model_cache, "wb") as f:
            pickle.dump((gp_cd, gp_cl, scaler), f)
        print(f"  Cd GP kernel: {gp_cd.kernel_}")
        print(f"  Cl GP kernel: {gp_cl.kernel_}")

        print("\n── Leave-one-out cross-validation ────────────────────────────")
        cd_loo, cd_std = loo_cv(X, y_cd, scaler)
        cl_loo, cl_std = loo_cv(X, y_cl, scaler)
        print(f"  Cd  R²={r2_score(y_cd, cd_loo):.3f}  MAE={mean_absolute_error(y_cd, cd_loo):.4f}")
        print(f"  Cl  R²={r2_score(y_cl, cl_loo):.3f}  MAE={mean_absolute_error(y_cl, cl_loo):.4f}")
        plot_validation(y_cd, y_cl, cd_loo, cl_loo, cd_std, cl_std)

    print("\n── Response surfaces ─────────────────────────────────────────")
    plot_response_surfaces(gp_cd, scaler, X)

    print("\n── Single-objective optimisation (min f1 = Cd + λ·Cl) ────────")
    opt = optimise_f1(gp_cd, gp_cl, scaler)
    print(f"  Optimal design:")
    for k, v in opt.items():
        print(f"    {k}: {v}")

    # Only write if no CFD-validated result already exists that is better.
    opt_path = RESULTS_DIR / "optimum_design.json"
    existing_f1_cfd = None
    if opt_path.exists():
        try:
            existing_f1_cfd = json.load(open(opt_path)).get("f1_cfd")
        except Exception:
            pass
    if existing_f1_cfd is not None and existing_f1_cfd <= opt["f1_objective"]:
        print(f"  Keeping existing optimum_design.json "
              f"(CFD-validated f1={existing_f1_cfd:.6f} ≤ surrogate f1={opt['f1_objective']:.6f})")
    else:
        with open(opt_path, "w") as f:
            json.dump(opt, f, indent=2)
        print(f"  Saved → results/optimum_design.json")

    print("\n── Pareto front (Cd vs downforce) ────────────────────────────")
    df_pareto = compute_and_plot_pareto(gp_cd, gp_cl, scaler, y_cd, y_cl)
    df_pareto.to_csv(RESULTS_DIR / "pareto_designs.csv", index=False)
    print(f"  {len(df_pareto)} Pareto-optimal designs")
    print(f"  Saved → results/pareto_designs.csv")
    print(f"\n  Pareto front extremes:")
    print(f"    Min Cd : {df_pareto.iloc[0].to_dict()}")
    print(f"    Max -Cl: {df_pareto.iloc[-1].to_dict()}")

    if validate:
        print("\n── Optimum validation (CFD run) ──────────────────────────────")
        validate_optimum(gp_cd, gp_cl, scaler)

    if bo_flag:
        current_count = 0
        hist_path = RESULTS_DIR / "bo_history.csv"
        if hist_path.exists():
            current_count = len(pd.read_csv(hist_path))
        total_iters = current_count + n_bo_iters
        print(f"\n── Bayesian optimisation loop ({n_bo_iters} additional iterations) ────────")
        print(f"  Outliers excluded from GP: {OUTLIER_CASES}")
        history = bayesian_loop(total_iters, X, y_cd, y_cl, scaler,
                                exclude_cases=OUTLIER_CASES)
        if not history.empty:
            plot_bo_convergence(history)
            valid_rows = history.dropna(subset=["f1_cfd"])
            if not valid_rows.empty:
                best_row = valid_rows.loc[valid_rows["f1_cfd"].idxmin()]
                print(f"\n  BO complete. Best design found:")
                for f in FEATURES:
                    print(f"    {f}: {best_row[f]:.4f}")
                print(f"    Cd_cfd:      {best_row['Cd_cfd']:.6f}")
                print(f"    Cd_best_DoE: {y_cd_init_best:.6f}" if 'y_cd_init_best' in dir() else "")
                best_params = {f: round(float(best_row[f]), 4) for f in FEATURES}
                best_params["Cd_predicted"] = round(float(best_row["Cd_predicted"]), 6)
                best_params["Cl_predicted"] = round(float(best_row["Cl_predicted"]), 6)
                best_params["Cd_cfd"]       = round(float(best_row["Cd_cfd"]), 6)
                best_params["Cl_cfd"]       = round(float(best_row["Cl_cfd"]), 6) if pd.notna(best_row.get("Cl_cfd")) else None
                best_params["f1_cfd"]       = round(float(best_row["f1_cfd"]), 6)
                best_params["source"]       = "bayesian_loop"
                with open(RESULTS_DIR / "optimum_design.json", "w") as f:
                    json.dump(best_params, f, indent=2)
                print(f"  Updated → results/optimum_design.json")

    if local_flag:
        # Compute total iterations needed so resume logic works correctly
        current_count = 0
        hist_path = RESULTS_DIR / "bo_history.csv"
        if hist_path.exists():
            current_count = len(pd.read_csv(hist_path))
        total_iters = current_count + n_local_iters

        print(f"\n── Local refinement loop ({n_local_iters} additional iterations) ──")
        print(f"  Outliers excluded from GP: {OUTLIER_CASES}")
        print(f"  Search bounds: {LOCAL_BOUNDS}")
        print(f"  ξ (EI exploration): {EI_XI_LOCAL}  (exploitation focus)")
        history = bayesian_loop(
            total_iters, X, y_cd, y_cl, scaler,
            bounds_override=LOCAL_BOUNDS,
            exclude_cases=OUTLIER_CASES,
            ei_xi=EI_XI_LOCAL,
        )
        if not history.empty:
            plot_bo_convergence(history)
            valid_rows = history.dropna(subset=["f1_cfd"])
            if not valid_rows.empty:
                best_row = valid_rows.loc[valid_rows["f1_cfd"].idxmin()]
                print(f"\n  Local refinement complete. Best design:")
                for feat in FEATURES:
                    print(f"    {feat}: {best_row[feat]:.4f}")
                print(f"    f1_cfd = {best_row['f1_cfd']:.6f}  "
                      f"(Cd={best_row['Cd_cfd']:.4f}, Cl={best_row['Cl_cfd']:.4f})")
                best_params = {feat: round(float(best_row[feat]), 4) for feat in FEATURES}
                best_params["Cd_predicted"] = round(float(best_row["Cd_predicted"]), 6)
                best_params["Cd_cfd"]       = round(float(best_row["Cd_cfd"]), 6)
                best_params["Cl_cfd"]       = round(float(best_row["Cl_cfd"]), 6)
                best_params["f1_cfd"]       = round(float(best_row["f1_cfd"]), 6)
                best_params["source"]       = "local_refinement"
                with open(RESULTS_DIR / "optimum_design.json", "w") as f:
                    json.dump(best_params, f, indent=2)
                print(f"  Updated → results/optimum_design.json")

    print("\nDone. All outputs in results/")


if __name__ == "__main__":
    main()
