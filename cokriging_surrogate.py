"""
cokriging_surrogate.py
======================
Kennedy & O'Hagan (2000) additive co-Kriging surrogate for the Ahmed body
optimisation campaign.

Fidelity assignment (per CLAUDE.md):
  LF = L2 RANS (simpleFoam, ~435K cells, ~10 min/solve)
  HF = L3 RANS (simpleFoam, ~912K cells, ~25 min/solve)

Model
-----
The additive (ρ=1) co-Kriging model decomposes the HF response as:

    f_HF(x) = f_LF(x) + δ(x)         [Eq. 1 in report]

where  f_LF(x) is the L2-trained GP (from surrogate_optimiser.py)
       δ(x)    is an independent GP trained on the fidelity correction
                δ_i = y_HF(x_i) − GP_LF.predict(x_i)   at the 8 L3 anchors.

Separate correction GPs are fitted for Cd and Cl, so:
    Cd_HF(x) = Cd_LF(x) + δ_Cd(x)
    Cl_HF(x) = Cl_LF(x) + δ_Cl(x)

with combined uncertainty:
    σ²_HF(x) ≈ σ²_LF(x) + σ²_δ(x)

The F1 objective is then:
    f₁(x) = Cd_HF(x) + (1/3)·Cl_HF(x)     [minimise]

Outputs
-------
  results/cokriging_correction.png    — δ_Cd and δ_Cl response surfaces
  results/cokriging_validation.png    — HF prediction vs L3 CFD at anchors
  results/optimum_design_mf.json      — MF optimum with surrogate uncertainty
  results/mf_verification.json        — (if --verify) L3 CFD at MF optimum

Usage
-----
  python3 cokriging_surrogate.py                # fit + re-optimise
  python3 cokriging_surrogate.py --verify       # also run L3 at MF optimum
  python3 cokriging_surrogate.py --plot-only    # reload cached models, replot

Dependencies: numpy, pandas, scipy, scikit-learn, matplotlib
"""

import sys, json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import differential_evolution
from scipy.stats import norm as sp_norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE        = Path(__file__).parent
RESULTS_DIR = BASE / "results"

# ── Re-use constants from surrogate_optimiser ────────────────────────────────
FEATURES       = ["slant_angle", "diffuser_angle"]
FEATURE_LABELS = ["Slant angle (°)", "Diffuser angle (°)"]
BOUNDS = {
    "slant_angle":    (15.0, 40.0),
    "diffuser_angle": ( 0.0, 20.0),
}
F1_LAMBDA = 1.0 / 3.0

def f1(cd, cl):
    return cd + F1_LAMBDA * cl


# ── 1. Load LF GPs (from surrogate_optimiser cache) ──────────────────────────

def load_lf_gps() -> tuple:
    """Load the cached L2 GP models from surrogate_optimiser.py."""
    cache = RESULTS_DIR / "gp_models.pkl"
    if not cache.exists():
        raise FileNotFoundError(
            f"{cache} not found — run `python3 surrogate_optimiser.py` first")
    with open(cache, "rb") as fh:
        gp_cd_lf, gp_cl_lf, scaler_lf = pickle.load(fh)
    print(f"Loaded LF GPs from {cache}")
    print(f"  Cd kernel: {gp_cd_lf.kernel_}")
    print(f"  Cl kernel: {gp_cl_lf.kernel_}")
    return gp_cd_lf, gp_cl_lf, scaler_lf


# ── 2. Load L3 anchor data ────────────────────────────────────────────────────

def load_anchors() -> pd.DataFrame:
    csv = RESULTS_DIR / "l3_anchors.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found — run `python3 run_l3_anchors.py` first")
    df = pd.read_csv(csv).dropna(subset=["Cd", "Cl"])
    print(f"Loaded {len(df)} L3 anchor points from {csv}")
    if len(df) < 3:
        raise ValueError(
            f"Only {len(df)} anchor points — need at least 3 to fit correction GP. "
            "Wait for more L3 solves to finish.")
    return df


# ── 3. Fit correction GPs ─────────────────────────────────────────────────────

def build_correction_kernel():
    """Matérn-5/2 + white noise for the δ GP."""
    return (
        ConstantKernel(0.1, (1e-4, 1e2))
        * Matern(length_scale=np.ones(len(FEATURES)),
                 length_scale_bounds=(0.5, 30.0), nu=2.5)
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-2))
    )


def fit_correction_gps(
    df_anchors: pd.DataFrame,
    gp_cd_lf: GaussianProcessRegressor,
    gp_cl_lf: GaussianProcessRegressor,
    scaler_lf: StandardScaler,
) -> tuple:
    """
    Compute δ = y_HF − ŷ_LF at the anchor locations and fit correction GPs.

    Returns
    -------
    gp_dcd, gp_dcl  : correction GPs for Cd and Cl
    scaler_anc      : StandardScaler fitted on the anchor X matrix
    delta_cd, delta_cl : raw correction values at anchors (for reporting)
    """
    X_anc = df_anchors[FEATURES].values.astype(float)
    y_cd_hf = df_anchors["Cd"].values
    y_cl_hf = df_anchors["Cl"].values

    # LF prediction at anchor locations
    cd_lf_pred = gp_cd_lf.predict(scaler_lf.transform(X_anc))
    cl_lf_pred = gp_cl_lf.predict(scaler_lf.transform(X_anc))

    delta_cd = y_cd_hf - cd_lf_pred
    delta_cl = y_cl_hf - cl_lf_pred

    print(f"\n── Fidelity correction at L3 anchors ────────────────────────")
    print(f"  {'Case':<15} {'Slant':>6} {'Diff':>6}  "
          f"{'δCd':>8} {'δCl':>8}  {'Cd_L2':>8} {'Cd_L3':>8}")
    print("  " + "-" * 70)
    for i, (_, row) in enumerate(df_anchors.iterrows()):
        print(f"  {row['case_id']:<15} {row['slant_angle']:>6.1f} "
              f"{row['diffuser_angle']:>6.1f}  "
              f"{delta_cd[i]:>+8.4f} {delta_cl[i]:>+8.4f}  "
              f"{cd_lf_pred[i]:>8.4f} {row['Cd']:>8.4f}")

    scaler_anc = StandardScaler().fit(X_anc)
    X_anc_s = scaler_anc.transform(X_anc)

    gp_dcd = GaussianProcessRegressor(
        kernel=build_correction_kernel(),
        n_restarts_optimizer=8, normalize_y=True, random_state=42)
    gp_dcd.fit(X_anc_s, delta_cd)

    gp_dcl = GaussianProcessRegressor(
        kernel=build_correction_kernel(),
        n_restarts_optimizer=8, normalize_y=True, random_state=42)
    gp_dcl.fit(X_anc_s, delta_cl)

    print(f"\n  δCd kernel: {gp_dcd.kernel_}")
    print(f"  δCl kernel: {gp_dcl.kernel_}")
    print(f"\n  Mean δCd = {delta_cd.mean():+.4f}  (std {delta_cd.std():.4f})")
    print(f"  Mean δCl = {delta_cl.mean():+.4f}  (std {delta_cl.std():.4f})")

    return gp_dcd, gp_dcl, scaler_anc, delta_cd, delta_cl


# ── 4. HF predictor ───────────────────────────────────────────────────────────

def predict_hf(
    X: np.ndarray,
    gp_cd_lf, gp_cl_lf, scaler_lf,
    gp_dcd, gp_dcl, scaler_anc,
) -> tuple:
    """
    Return (cd_hf, cd_hf_std, cl_hf, cl_hf_std) arrays at query points X.

    Uses additive co-Kriging:
        Cd_HF = Cd_LF + δ_Cd
        σ²_HF ≈ σ²_LF + σ²_δ
    """
    cd_lf, cd_lf_std = gp_cd_lf.predict(scaler_lf.transform(X), return_std=True)
    cl_lf, cl_lf_std = gp_cl_lf.predict(scaler_lf.transform(X), return_std=True)

    d_cd, d_cd_std = gp_dcd.predict(scaler_anc.transform(X), return_std=True)
    d_cl, d_cl_std = gp_dcl.predict(scaler_anc.transform(X), return_std=True)

    cd_hf     = cd_lf + d_cd
    cl_hf     = cl_lf + d_cl
    cd_hf_std = np.sqrt(cd_lf_std**2 + d_cd_std**2)
    cl_hf_std = np.sqrt(cl_lf_std**2 + d_cl_std**2)

    return cd_hf, cd_hf_std, cl_hf, cl_hf_std


# ── 5. Leave-one-out validation ───────────────────────────────────────────────

def loo_validation(
    df_anchors: pd.DataFrame,
    gp_cd_lf, gp_cl_lf, scaler_lf,
) -> tuple:
    """
    Leave-one-out cross-validation of the HF prediction at each anchor.
    Trains correction GP on n-1 anchors, predicts at the held-out point.
    Returns arrays (cd_loo, cl_loo, cd_loo_std, cl_loo_std).
    """
    X_anc = df_anchors[FEATURES].values.astype(float)
    y_cd  = df_anchors["Cd"].values
    y_cl  = df_anchors["Cl"].values
    n     = len(df_anchors)

    cd_loo, cl_loo = np.zeros(n), np.zeros(n)
    cd_std, cl_std = np.zeros(n), np.zeros(n)

    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        X_tr = X_anc[mask]; y_cd_tr = y_cd[mask]; y_cl_tr = y_cl[mask]

        gp_cd_lf_i, gp_cl_lf_i, scaler_lf_i = gp_cd_lf, gp_cl_lf, scaler_lf

        cd_lf_tr = gp_cd_lf_i.predict(scaler_lf_i.transform(X_tr))
        cl_lf_tr = gp_cl_lf_i.predict(scaler_lf_i.transform(X_tr))
        dcd_tr   = y_cd_tr - cd_lf_tr
        dcl_tr   = y_cl_tr - cl_lf_tr

        sc_i = StandardScaler().fit(X_tr)
        g_cd = GaussianProcessRegressor(kernel=build_correction_kernel(),
                                        n_restarts_optimizer=3,
                                        normalize_y=True, random_state=42)
        g_cl = GaussianProcessRegressor(kernel=build_correction_kernel(),
                                        n_restarts_optimizer=3,
                                        normalize_y=True, random_state=42)
        g_cd.fit(sc_i.transform(X_tr), dcd_tr)
        g_cl.fit(sc_i.transform(X_tr), dcl_tr)

        x_test = X_anc[[i]]
        # Use scaler_lf (global) for LF prediction at test point
        cd_lf_t = float(gp_cd_lf_i.predict(scaler_lf_i.transform(x_test))[0])
        cl_lf_t = float(gp_cl_lf_i.predict(scaler_lf_i.transform(x_test))[0])
        d_cd_t, d_cd_s = g_cd.predict(sc_i.transform(x_test), return_std=True)
        d_cl_t, d_cl_s = g_cl.predict(sc_i.transform(x_test), return_std=True)

        cd_loo[i] = cd_lf_t + float(np.ravel(d_cd_t)[0])
        cl_loo[i] = cl_lf_t + float(np.ravel(d_cl_t)[0])
        cd_std[i] = float(np.ravel(d_cd_s)[0])
        cl_std[i] = float(np.ravel(d_cl_s)[0])

    return cd_loo, cl_loo, cd_std, cl_std


# ── 6. MF optimisation ────────────────────────────────────────────────────────

def optimise_mf(
    gp_cd_lf, gp_cl_lf, scaler_lf,
    gp_dcd, gp_dcl, scaler_anc,
) -> dict:
    """Minimise f₁_HF = Cd_HF + (1/3)·Cl_HF over the design space."""
    bounds_list = [BOUNDS[f] for f in FEATURES]

    def obj(x):
        X = x.reshape(1, -1)
        cd_hf, _, cl_hf, _ = predict_hf(
            X, gp_cd_lf, gp_cl_lf, scaler_lf, gp_dcd, gp_dcl, scaler_anc)
        return float(f1(cd_hf[0], cl_hf[0]))

    result = differential_evolution(obj, bounds=bounds_list, seed=42,
                                    maxiter=500, tol=1e-7, popsize=20)
    x_opt = result.x.reshape(1, -1)
    cd_hf, cd_std, cl_hf, cl_std = predict_hf(
        x_opt, gp_cd_lf, gp_cl_lf, scaler_lf, gp_dcd, gp_dcl, scaler_anc)

    opt = {feat: round(float(v), 4) for feat, v in zip(FEATURES, result.x)}
    opt["Cd_HF_predicted"]  = round(float(cd_hf[0]), 6)
    opt["Cd_HF_std"]        = round(float(cd_std[0]), 6)
    opt["Cl_HF_predicted"]  = round(float(cl_hf[0]), 6)
    opt["Cl_HF_std"]        = round(float(cl_std[0]), 6)
    opt["f1_HF_predicted"]  = round(float(result.fun), 6)
    opt["ride_height"]      = 50.8
    opt["front_radius"]     = 100.0
    return opt


# ── 6b. λ-sensitivity sweep ───────────────────────────────────────────────────

def lambda_sweep(
    gp_cd_lf, gp_cl_lf, scaler_lf,
    gp_dcd, gp_dcl, scaler_anc,
    lambdas=(0.0, 1/6, 1/3, 1/2, 2/3, 1.0),
) -> pd.DataFrame:
    """Re-optimise f = Cd_HF + λ·Cl_HF for each λ value."""
    bounds_list = [BOUNDS[f] for f in FEATURES]
    rows = []
    for lam in lambdas:
        def obj(x, lam=lam):
            X = x.reshape(1, -1)
            cd_hf, _, cl_hf, _ = predict_hf(
                X, gp_cd_lf, gp_cl_lf, scaler_lf, gp_dcd, gp_dcl, scaler_anc)
            return float(cd_hf[0] + lam * cl_hf[0])

        result = differential_evolution(obj, bounds=bounds_list, seed=42,
                                        maxiter=500, tol=1e-7, popsize=20)
        x_opt = result.x.reshape(1, -1)
        cd_hf, _, cl_hf, _ = predict_hf(
            x_opt, gp_cd_lf, gp_cl_lf, scaler_lf, gp_dcd, gp_dcl, scaler_anc)
        rows.append({
            "lambda":          round(float(lam), 6),
            "slant_angle":     round(float(result.x[0]), 2),
            "diffuser_angle":  round(float(result.x[1]), 2),
            "Cd_HF":           round(float(cd_hf[0]), 4),
            "Cl_HF":           round(float(cl_hf[0]), 4),
            "f_HF":            round(float(result.fun), 4),
        })
        print(f"  λ={lam:.4f}  slant={result.x[0]:.1f}°  diff={result.x[1]:.1f}°  "
              f"Cd={cd_hf[0]:.4f}  Cl={cl_hf[0]:.4f}  f={result.fun:.4f}")

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "lambda_sweep.csv"
    df.to_csv(out, index=False, float_format="%.6f")
    print(f"  Saved → {out}")
    return df


def plot_lambda_sweep(df: pd.DataFrame):
    """Plot optimum location and objective vs λ."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("MF Optimum Migration with Objective Weighting $\\lambda$", fontsize=12)

    lam = df["lambda"].values
    markers = "o"

    axes[0].plot(lam, df["slant_angle"].values, f"{markers}-", color="#1f77b4", ms=7)
    axes[0].set_xlabel("$\\lambda$", fontsize=11)
    axes[0].set_ylabel("Optimal $\\alpha_s^*$ (°)", fontsize=11)
    axes[0].set_title("Slant angle", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(lam, df["diffuser_angle"].values, f"{markers}-", color="#ff7f0e", ms=7)
    axes[1].set_xlabel("$\\lambda$", fontsize=11)
    axes[1].set_ylabel("Optimal $\\alpha_d^*$ (°)", fontsize=11)
    axes[1].set_title("Diffuser angle", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(lam, df["Cd_HF"].values, f"{markers}-", color="#2ca02c", ms=7, label="$C_d^*$")
    axes[2].plot(lam, df["Cl_HF"].values, f"s-", color="#d62728", ms=7, label="$C_l^*$")
    axes[2].set_xlabel("$\\lambda$", fontsize=11)
    axes[2].set_ylabel("Coefficient", fontsize=11)
    axes[2].set_title("$C_d^*$ and $C_l^*$ at optimum", fontsize=10)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    # Mark λ = 1/3
    for ax in axes:
        ax.axvline(1/3, color="gray", lw=1, ls="--", alpha=0.7)

    plt.tight_layout()
    out = RESULTS_DIR / "lambda_sweep.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ── 7. Plots ──────────────────────────────────────────────────────────────────

def plot_validation(df_anchors, cd_loo, cl_loo, cd_std, cl_std):
    y_cd = df_anchors["Cd"].values
    y_cl = df_anchors["Cl"].values

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Co-Kriging HF Surrogate — LOO Validation at L3 Anchors",
                 fontsize=13)

    for ax, y_true, y_pred, std, label in zip(
        axes, [y_cd, y_cl], [cd_loo, cl_loo], [cd_std, cl_std],
        ["$C_d$ (L3)", "$C_l$ (L3)"]
    ):
        lo = min(y_true.min(), y_pred.min()) * 0.97
        hi = max(y_true.max(), y_pred.max()) * 1.03
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
        ax.errorbar(y_true, y_pred, yerr=2 * std, fmt="o", color="#d62728",
                    ecolor="#f7a7a8", elinewidth=1.5, capsize=3, ms=7,
                    label="LOO prediction ± 2σ")
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
        ax.set_xlabel(f"CFD {label}", fontsize=11)
        ax.set_ylabel(f"MF GP {label}", fontsize=11)
        ax.set_title(f"{label}  |  MAE = {mae:.4f}  RMSE = {rmse:.4f}", fontsize=10)
        ax.legend(fontsize=9)
        ax.set_aspect("equal")

    plt.tight_layout()
    out = RESULTS_DIR / "cokriging_validation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


def plot_correction_surface(
    gp_dcd, gp_dcl, scaler_anc, df_anchors,
):
    """2D correction (δ) response surfaces over the design space."""
    sa = np.linspace(*BOUNDS["slant_angle"], 80)
    da = np.linspace(*BOUNDS["diffuser_angle"], 80)
    SA, DA = np.meshgrid(sa, da)
    X_grid = np.column_stack([SA.ravel(), DA.ravel()])
    Xs = scaler_anc.transform(X_grid)

    dcd = gp_dcd.predict(Xs).reshape(SA.shape)
    dcl = gp_dcl.predict(Xs).reshape(SA.shape)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Co-Kriging Fidelity Correction $\\delta(x) = f_\\mathrm{HF} - f_\\mathrm{LF}$",
                 fontsize=13)

    for ax, Z, title, clabel in zip(
        axes, [dcd, dcl],
        ["$\\delta_{C_d}$ (L3 − L2)", "$\\delta_{C_l}$ (L3 − L2)"],
        ["$\\delta C_d$", "$\\delta C_l$"]
    ):
        vmax = max(abs(Z.min()), abs(Z.max()))
        cf = ax.contourf(SA, DA, Z, levels=20, cmap="RdBu_r",
                         vmin=-vmax, vmax=vmax)
        plt.colorbar(cf, ax=ax, label=clabel)
        ax.scatter(df_anchors["slant_angle"], df_anchors["diffuser_angle"],
                   c="k", s=40, zorder=5, label="L3 anchors")
        ax.set_xlabel("Slant angle (°)", fontsize=10)
        ax.set_ylabel("Diffuser angle (°)", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)

    plt.tight_layout()
    out = RESULTS_DIR / "cokriging_correction.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


def plot_hf_response_surface(
    gp_cd_lf, gp_cl_lf, scaler_lf,
    gp_dcd, gp_dcl, scaler_anc,
    opt_l2: dict, opt_mf: dict,
):
    """HF f₁ response surface with L2 and MF optima marked."""
    sa = np.linspace(*BOUNDS["slant_angle"], 80)
    da = np.linspace(*BOUNDS["diffuser_angle"], 80)
    SA, DA = np.meshgrid(sa, da)
    X_grid = np.column_stack([SA.ravel(), DA.ravel()])

    cd_hf, _, cl_hf, _ = predict_hf(
        X_grid, gp_cd_lf, gp_cl_lf, scaler_lf, gp_dcd, gp_dcl, scaler_anc)
    F1 = (cd_hf + F1_LAMBDA * cl_hf).reshape(SA.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(SA, DA, F1, levels=25, cmap="RdYlGn_r")
    plt.colorbar(cf, ax=ax, label="$f_1 = C_d + \\frac{1}{3}C_l$ (HF)")
    ax.contour(SA, DA, F1, levels=10, colors="k", linewidths=0.4, alpha=0.4)

    ax.scatter(opt_l2["slant_angle"], opt_l2["diffuser_angle"],
               c="#1f77b4", s=120, zorder=7, marker="*",
               label=f"L2 optimum ({opt_l2['slant_angle']}°, {opt_l2['diffuser_angle']}°)")
    ax.scatter(opt_mf["slant_angle"], opt_mf["diffuser_angle"],
               c="#d62728", s=120, zorder=7, marker="*",
               label=f"MF optimum ({opt_mf['slant_angle']}°, {opt_mf['diffuser_angle']}°)")

    ax.set_xlabel("Slant angle (°)", fontsize=11)
    ax.set_ylabel("Diffuser angle (°)", fontsize=11)
    ax.set_title("Co-Kriging HF Surrogate — $f_1$ Response Surface", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = RESULTS_DIR / "hf_response_surface.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ── 8. MF optimum verification (L3 solve) ────────────────────────────────────

def verify_mf_optimum(opt_mf: dict) -> dict | None:
    """
    Run L3 at the MF optimum and compare with surrogate prediction.
    Returns the verification result dict, or None if CFD fails.
    """
    import importlib.util as ilu

    spec = ilu.spec_from_file_location("runner", BASE / "cfmesh_doe_runner.py")
    runner = ilu.module_from_spec(spec); spec.loader.exec_module(runner)
    runner.L2_TEMPLATE = BASE / "mesh_convergence" / "L3_fine"
    runner.L2_SYM      = BASE / "mesh_convergence" / "L3_symmetry"

    params = {
        "slant_angle":    opt_mf["slant_angle"],
        "diffuser_angle": opt_mf["diffuser_angle"],
        "ride_height":    opt_mf["ride_height"],
        "front_radius":   opt_mf["front_radius"],
    }
    cases_dir = BASE / "openfoam_cases_l3_anchors"
    print(f"\n── L3 verification at MF optimum ────────────────────────────")
    print(f"  slant={params['slant_angle']}°  diff={params['diffuser_angle']}°")

    res = runner.run_case("l3_mf_verify", params, resume=True, cases_dir=cases_dir)
    if res["Cd"] is None:
        print(f"  FAILED: {res.get('error')}")
        return None

    f1_cfd = f1(res["Cd"], res["Cl"])
    f1_pred = opt_mf["f1_HF_predicted"]
    err = abs(f1_cfd - f1_pred) / abs(f1_cfd) * 100

    print(f"\n  Surrogate  : f1 = {f1_pred:.4f}  "
          f"(Cd={opt_mf['Cd_HF_predicted']:.4f}, Cl={opt_mf['Cl_HF_predicted']:.4f})")
    print(f"  CFD (L3)   : f1 = {f1_cfd:.4f}  "
          f"(Cd={res['Cd']:.4f}, Cl={res['Cl']:.4f})")
    print(f"  |error|    : {err:.1f}%")

    verif = {
        "slant_angle":       opt_mf["slant_angle"],
        "diffuser_angle":    opt_mf["diffuser_angle"],
        "f1_HF_predicted":   round(f1_pred, 6),
        "f1_HF_std":         round(opt_mf["Cd_HF_std"] + F1_LAMBDA * opt_mf["Cl_HF_std"], 6),
        "Cd_HF_predicted":   opt_mf["Cd_HF_predicted"],
        "Cl_HF_predicted":   opt_mf["Cl_HF_predicted"],
        "Cd_cfd":            round(res["Cd"], 6),
        "Cl_cfd":            round(res["Cl"], 6),
        "f1_cfd":            round(f1_cfd, 6),
        "f1_error_pct":      round(err, 2),
    }
    out = RESULTS_DIR / "mf_verification.json"
    with open(out, "w") as fh:
        json.dump(verif, fh, indent=2)
    print(f"  Saved → {out}")
    return verif


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    plot_only = "--plot-only" in sys.argv
    verify    = "--verify"    in sys.argv

    # Load cached L2 GPs
    gp_cd_lf, gp_cl_lf, scaler_lf = load_lf_gps()

    # Load L3 anchors
    df_anchors = load_anchors()

    # Fit correction GPs
    if not plot_only:
        print("\n── Fitting correction GPs ────────────────────────────────────")
        gp_dcd, gp_dcl, scaler_anc, delta_cd, delta_cl = fit_correction_gps(
            df_anchors, gp_cd_lf, gp_cl_lf, scaler_lf)

        # Cache
        cache = RESULTS_DIR / "cokriging_models.pkl"
        with open(cache, "wb") as fh:
            pickle.dump((gp_dcd, gp_dcl, scaler_anc), fh)
        print(f"\n  Cached → {cache}")
    else:
        cache = RESULTS_DIR / "cokriging_models.pkl"
        if not cache.exists():
            raise FileNotFoundError(f"{cache} not found — run without --plot-only first")
        with open(cache, "rb") as fh:
            gp_dcd, gp_dcl, scaler_anc = pickle.load(fh)
        print("Loaded cached correction GPs")

    # LOO validation
    if not plot_only:
        print("\n── LOO validation ────────────────────────────────────────────")
        cd_loo, cl_loo, cd_std, cl_std = loo_validation(
            df_anchors, gp_cd_lf, gp_cl_lf, scaler_lf)
        y_cd = df_anchors["Cd"].values
        y_cl = df_anchors["Cl"].values
        mae_cd  = float(np.mean(np.abs(y_cd - cd_loo)))
        mae_cl  = float(np.mean(np.abs(y_cl - cl_loo)))
        mae_f1  = float(np.mean(np.abs(f1(y_cd, y_cl) - f1(cd_loo, cl_loo))))
        print(f"  LOO MAE Cd  = {mae_cd:.4f}")
        print(f"  LOO MAE Cl  = {mae_cl:.4f}")
        print(f"  LOO MAE f1  = {mae_f1:.4f}")
        plot_validation(df_anchors, cd_loo, cl_loo, cd_std, cl_std)

    # Correction surface
    plot_correction_surface(gp_dcd, gp_dcl, scaler_anc, df_anchors)

    # MF optimisation
    if not plot_only:
        print("\n── MF optimisation (minimise f₁_HF) ─────────────────────────")
        opt_mf = optimise_mf(gp_cd_lf, gp_cl_lf, scaler_lf, gp_dcd, gp_dcl, scaler_anc)
        print(f"  MF optimum:")
        for k, v in opt_mf.items():
            print(f"    {k}: {v}")

        # Compare with L2 optimum
        l2_path = RESULTS_DIR / "optimum_design.json"
        opt_l2 = json.loads(l2_path.read_text()) if l2_path.exists() else {}
        if opt_l2:
            print(f"\n  L2 optimum: slant={opt_l2['slant_angle']}°  "
                  f"diff={opt_l2.get('diffuser_angle', '?')}°  "
                  f"f1_cfd={opt_l2.get('f1_cfd', '?')}")
            dslant = abs(opt_mf["slant_angle"] - opt_l2.get("slant_angle", opt_mf["slant_angle"]))
            ddiff  = abs(opt_mf["diffuser_angle"] - opt_l2.get("diffuser_angle", opt_mf["diffuser_angle"]))
            print(f"  Shift: Δslant={dslant:.2f}°  Δdiffuser={ddiff:.2f}°")

        with open(RESULTS_DIR / "optimum_design_mf.json", "w") as fh:
            json.dump(opt_mf, fh, indent=2)
        print(f"\n  Saved → results/optimum_design_mf.json")

        plot_hf_response_surface(
            gp_cd_lf, gp_cl_lf, scaler_lf,
            gp_dcd, gp_dcl, scaler_anc,
            opt_l2 or opt_mf, opt_mf)
    else:
        opt_mf = json.loads((RESULTS_DIR / "optimum_design_mf.json").read_text())
        opt_l2 = json.loads((RESULTS_DIR / "optimum_design.json").read_text())
        plot_hf_response_surface(
            gp_cd_lf, gp_cl_lf, scaler_lf,
            gp_dcd, gp_dcl, scaler_anc,
            opt_l2, opt_mf)

    # λ-sweep
    print("\n── λ-sensitivity sweep ───────────────────────────────────────")
    df_lam = lambda_sweep(gp_cd_lf, gp_cl_lf, scaler_lf, gp_dcd, gp_dcl, scaler_anc)
    plot_lambda_sweep(df_lam)

    # Verification
    if verify:
        verify_mf_optimum(opt_mf)

    print("\nDone. All outputs in results/")


if __name__ == "__main__":
    main()
