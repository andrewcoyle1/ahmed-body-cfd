"""
co_kriging.py
=============
Multi-fidelity co-Kriging surrogate for the Ahmed body optimisation.

Model (Kennedy & O'Hagan 2000, additive form with ρ=1):
    f_HF(x) = f_LF(x) + δ(x)

    f_LF  — RANS GP trained on 30 DoE points
    δ     — correction GP trained on 10 DES points
             δ_i = f_HF(x_i) - f_LF_pred(x_i)

Combined prediction:
    μ_MF(x)   = μ_LF(x) + μ_δ(x)
    σ²_MF(x)  = σ²_LF(x) + σ²_δ(x)

Bayesian optimisation then uses EI on the MF surrogate.

Outputs
-------
  results/mf_pareto_front.png        — updated Pareto front (MF vs RANS)
  results/mf_optimum_design.json     — MF optimal design
  results/mf_correction_surface.png  — δ(x) correction landscape
  des_output/des_results.csv         — DES results (already written)
"""

import json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import differential_evolution
from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

FEATURES = ["slant_angle", "diffuser_angle", "ride_height", "front_radius"]
BOUNDS   = {
    "slant_angle":    (15.0,  40.0),
    "diffuser_angle": ( 0.0,  20.0),
    "ride_height":    (30.0,  80.0),
    "front_radius":   (50.0, 139.0),
}
F1_LAMBDA = 1.0 / 3.0
EI_XI     = 0.01

RESULTS_DIR = Path("results")
DES_DIR     = Path("des_output")


# ─── KERNELS ──────────────────────────────────────────────────────────────────

def rans_kernel():
    """Kernel for the 30-point RANS surrogate."""
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(len(FEATURES)),
                 length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
    )


def delta_kernel():
    """Kernel for the 10-point correction GP.
    Smoother and less noisy than the RANS kernel — the correction function
    δ(x) = f_HF - f_LF is expected to be a smoother function than either
    fidelity level alone.
    """
    return (
        ConstantKernel(0.1, (1e-4, 1e1))
        * Matern(length_scale=np.ones(len(FEATURES)),
                 length_scale_bounds=(1e-1, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1e-2))
    )


# ─── GP FITTING ───────────────────────────────────────────────────────────────

def fit_gp(X, y, kernel_fn, n_restarts=10):
    scaler = StandardScaler().fit(X)
    gp = GaussianProcessRegressor(
        kernel=kernel_fn(),
        n_restarts_optimizer=n_restarts,
        normalize_y=True,
        random_state=42,
    )
    gp.fit(scaler.transform(X), y)
    return gp, scaler


def predict(gp, scaler, X):
    mu, sigma = gp.predict(scaler.transform(X), return_std=True)
    return mu, sigma


# ─── MULTI-FIDELITY SURROGATE ─────────────────────────────────────────────────

class MultiFidelitySurrogate:
    """
    Wraps four GPs: RANS Cd, RANS Cl, correction δCd, correction δCl.
    Prediction combines both levels with additive correction.
    """

    def __init__(self, gp_cd_lf, sc_cd_lf, gp_cl_lf, sc_cl_lf,
                 gp_dcd, sc_dcd, gp_dcl, sc_dcl):
        self.gp_cd_lf  = gp_cd_lf;  self.sc_cd_lf  = sc_cd_lf
        self.gp_cl_lf  = gp_cl_lf;  self.sc_cl_lf  = sc_cl_lf
        self.gp_dcd    = gp_dcd;    self.sc_dcd    = sc_dcd
        self.gp_dcl    = gp_dcl;    self.sc_dcl    = sc_dcl

    def predict_cd(self, X):
        mu_lf, s_lf   = predict(self.gp_cd_lf, self.sc_cd_lf, X)
        mu_d,  s_d    = predict(self.gp_dcd,   self.sc_dcd,   X)
        return mu_lf + mu_d, np.sqrt(s_lf**2 + s_d**2)

    def predict_cl(self, X):
        mu_lf, s_lf   = predict(self.gp_cl_lf, self.sc_cl_lf, X)
        mu_d,  s_d    = predict(self.gp_dcl,   self.sc_dcl,   X)
        return mu_lf + mu_d, np.sqrt(s_lf**2 + s_d**2)

    def predict_f1(self, X):
        mu_cd, s_cd = self.predict_cd(X)
        mu_cl, s_cl = self.predict_cl(X)
        mu_f1 = mu_cd + F1_LAMBDA * mu_cl
        s_f1  = np.sqrt(s_cd**2 + (F1_LAMBDA * s_cl)**2)
        return mu_f1, s_f1


# ─── EXPECTED IMPROVEMENT ─────────────────────────────────────────────────────

def expected_improvement(X, mf, f_best, xi=EI_XI):
    X = np.atleast_2d(X)
    mu, sigma = mf.predict_f1(X)
    sigma = np.maximum(sigma, 1e-9)
    Z  = (f_best - mu - xi) / sigma
    ei = (f_best - mu - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    return np.maximum(ei, 0.0)


def maximise_ei(mf, f_best, bounds_list, n_restarts=20):
    best_ei, best_x = -np.inf, None
    def neg_ei(x):
        return -expected_improvement(x.reshape(1, -1), mf, f_best)[0]
    for _ in range(n_restarts):
        res = differential_evolution(neg_ei, bounds_list, seed=None,
                                     maxiter=500, tol=1e-6,
                                     mutation=(0.5, 1.5), recombination=0.9)
        if -res.fun > best_ei:
            best_ei = -res.fun
            best_x  = res.x
    return best_x, best_ei


# ─── PARETO FRONT ─────────────────────────────────────────────────────────────

def pareto_front(costs):
    n = costs.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
                dominated[i] = True
                break
    return ~dominated


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── 1. Load RANS DoE data ─────────────────────────────────────────────────
    rans_results = pd.read_csv("results/results_summary.csv")
    doe_dm       = pd.read_csv("doe_output/design_matrix.csv")
    rans_df      = doe_dm.merge(rans_results[["case_id", "Cd", "Cl"]], on="case_id")
    rans_df      = rans_df.dropna(subset=["Cd", "Cl"])

    X_lf  = rans_df[FEATURES].values.astype(float)
    y_cd_lf = rans_df["Cd"].values.astype(float)
    y_cl_lf = rans_df["Cl"].values.astype(float)
    print(f"RANS training points: {len(X_lf)}")

    # ── 2. Fit RANS GPs ───────────────────────────────────────────────────────
    print("\n── Fitting RANS (low-fidelity) GPs ──────────────────────────────────")
    gp_cd_lf, sc_cd_lf = fit_gp(X_lf, y_cd_lf, rans_kernel)
    gp_cl_lf, sc_cl_lf = fit_gp(X_lf, y_cl_lf, rans_kernel)
    print(f"  Cd kernel: {gp_cd_lf.kernel_}")
    print(f"  Cl kernel: {gp_cl_lf.kernel_}")

    # ── 3. Load DES data ──────────────────────────────────────────────────────
    des_df = pd.read_csv("des_output/des_results.csv")
    print(f"\nDES (high-fidelity) points: {len(des_df)}")

    # Evaluate RANS GP at DES locations to get f_LF predictions
    X_hf = des_df[FEATURES].values.astype(float)
    mu_cd_lf_at_hf, _ = predict(gp_cd_lf, sc_cd_lf, X_hf)
    mu_cl_lf_at_hf, _ = predict(gp_cl_lf, sc_cl_lf, X_hf)

    # Compute additive corrections: δ = f_HF - f_LF_pred
    dcd = des_df["Cd_DES"].values - mu_cd_lf_at_hf
    dcl = des_df["Cl_DES"].values - mu_cl_lf_at_hf

    print("\n── Correction statistics (DES − RANS GP prediction) ────────────────")
    print(f"  δCd: mean={dcd.mean():+.4f}  std={dcd.std():.4f}  "
          f"range=[{dcd.min():+.4f}, {dcd.max():+.4f}]")
    print(f"  δCl: mean={dcl.mean():+.4f}  std={dcl.std():.4f}  "
          f"range=[{dcl.min():+.4f}, {dcl.max():+.4f}]")

    # ── 4. Fit correction GPs ─────────────────────────────────────────────────
    print("\n── Fitting correction (δ) GPs ───────────────────────────────────────")
    gp_dcd, sc_dcd = fit_gp(X_hf, dcd, delta_kernel, n_restarts=15)
    gp_dcl, sc_dcl = fit_gp(X_hf, dcl, delta_kernel, n_restarts=15)
    print(f"  δCd kernel: {gp_dcd.kernel_}")
    print(f"  δCl kernel: {gp_dcl.kernel_}")

    # ── 5. Build MF surrogate ─────────────────────────────────────────────────
    mf = MultiFidelitySurrogate(gp_cd_lf, sc_cd_lf, gp_cl_lf, sc_cl_lf,
                                gp_dcd, sc_dcd, gp_dcl, sc_dcl)

    # ── 6. Validate at DES training points ───────────────────────────────────
    print("\n── MF surrogate accuracy at DES points ──────────────────────────────")
    mu_cd_mf, _ = mf.predict_cd(X_hf)
    mu_cl_mf, _ = mf.predict_cl(X_hf)
    cd_err = np.abs(mu_cd_mf - des_df["Cd_DES"].values)
    cl_err = np.abs(mu_cl_mf - des_df["Cl_DES"].values)
    print(f"  Cd MAE at HF points: {cd_err.mean():.4f}  (max {cd_err.max():.4f})")
    print(f"  Cl MAE at HF points: {cl_err.mean():.4f}  (max {cl_err.max():.4f})")

    # ── 7. Single-objective MF optimisation ───────────────────────────────────
    print("\n── MF single-objective optimisation (min Cd + λ·Cl) ─────────────────")
    bounds_list = [BOUNDS[f] for f in FEATURES]

    def neg_f1_mf(x):
        X = np.array(x).reshape(1, -1)
        mu_cd, _ = mf.predict_cd(X)
        mu_cl, _ = mf.predict_cl(X)
        return (mu_cd + F1_LAMBDA * mu_cl)[0]

    best_result = differential_evolution(neg_f1_mf, bounds_list,
                                          seed=42, maxiter=1000,
                                          mutation=(0.5, 1.5), recombination=0.9,
                                          tol=1e-8, popsize=20)
    x_opt = best_result.x
    mu_cd_opt, s_cd_opt = mf.predict_cd(x_opt.reshape(1, -1))
    mu_cl_opt, s_cl_opt = mf.predict_cl(x_opt.reshape(1, -1))
    mu_cd_opt = float(np.atleast_1d(mu_cd_opt).ravel()[0])
    mu_cl_opt = float(np.atleast_1d(mu_cl_opt).ravel()[0])
    s_cd_opt  = float(np.atleast_1d(s_cd_opt).ravel()[0])
    s_cl_opt  = float(np.atleast_1d(s_cl_opt).ravel()[0])
    f1_opt = mu_cd_opt + F1_LAMBDA * mu_cl_opt

    opt_design = {f: float(x_opt[i]) for i, f in enumerate(FEATURES)}
    opt_design.update({
        "Cd_predicted": mu_cd_opt,
        "Cd_uncertainty": s_cd_opt,
        "Cl_predicted": mu_cl_opt,
        "Cl_uncertainty": s_cl_opt,
        "f1_objective": f1_opt,
        "F1_lambda": F1_LAMBDA,
        "fidelity": "multi-fidelity (RANS+DES co-Kriging)",
    })

    print(f"  MF optimal design:")
    for k, v in opt_design.items():
        print(f"    {k}: {v}")

    (RESULTS_DIR / "mf_optimum_design.json").write_text(
        json.dumps(opt_design, indent=2))
    print(f"  Saved → results/mf_optimum_design.json")

    # ── 8. MF Pareto front ────────────────────────────────────────────────────
    print("\n── MF Pareto front ───────────────────────────────────────────────────")
    rng = np.random.default_rng(42)
    n_samples = 50000
    X_sample = np.column_stack([
        rng.uniform(lo, hi, n_samples)
        for lo, hi in bounds_list
    ])
    cd_pred, _ = mf.predict_cd(X_sample)
    cl_pred, _ = mf.predict_cl(X_sample)

    downforce = -cl_pred
    costs     = np.column_stack([cd_pred, -downforce])
    mask      = pareto_front(costs)

    pf_cd = cd_pred[mask]
    pf_cl = cl_pred[mask]
    pf_X  = X_sample[mask]

    # Sort by Cd
    sort_idx = np.argsort(pf_cd)
    pf_cd, pf_cl, pf_X = pf_cd[sort_idx], pf_cl[sort_idx], pf_X[sort_idx]

    pf_df = pd.DataFrame(pf_X, columns=FEATURES)
    pf_df["Cd_predicted"] = pf_cd
    pf_df["Cl_predicted"] = pf_cl
    pf_df.to_csv(RESULTS_DIR / "mf_pareto_designs.csv", index=False)
    print(f"  {mask.sum()} MF Pareto-optimal designs")
    print(f"  Min Cd:     Cd={pf_cd.min():.4f}  Cl={pf_cl[pf_cd.argmin()]:.4f}")
    print(f"  Max downforce: Cd={pf_cd[(-pf_cl).argmax()]:.4f}  Cl={pf_cl.min():.4f}")

    # ── 9. Comparison plot: RANS vs MF Pareto ─────────────────────────────────
    # Also load the RANS-only Pareto for comparison
    rans_pf = pd.read_csv(RESULTS_DIR / "pareto_designs.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("RANS vs Multi-Fidelity (RANS + DES Co-Kriging) Pareto Front",
                 fontsize=13)

    # Left: Pareto comparison
    ax = axes[0]
    ax.plot(rans_pf["Cd_predicted"], -rans_pf["Cl_predicted"],
            "o--", color="#aaaaaa", ms=5, lw=1.5, label="RANS only (30 pts)")
    ax.plot(pf_cd, -pf_cl, "o-", color="#c0392b", ms=6, lw=2,
            label="Multi-fidelity (RANS+DES)")
    ax.scatter([float(mu_cd_opt)], [float(-mu_cl_opt)], s=120, zorder=5,
               color="#e74c3c", marker="*", label=f"MF optimum")
    # DES training points
    ax.scatter(des_df["Cd_DES"], -des_df["Cl_DES"], s=40, zorder=4,
               color="#2980b9", alpha=0.7, label="DES training pts")
    ax.set_xlabel("Drag coefficient Cd")
    ax.set_ylabel("Downforce coefficient −Cl")
    ax.set_title("Pareto Front Comparison")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: correction surface (slant vs diffuser, at median rh and fr)
    ax2 = axes[1]
    sa_grid = np.linspace(15, 40, 60)
    da_grid = np.linspace(0, 20, 60)
    SA, DA  = np.meshgrid(sa_grid, da_grid)
    rh_med  = np.median(X_lf[:, 2])
    fr_med  = np.median(X_lf[:, 3])
    X_grid  = np.column_stack([
        SA.ravel(), DA.ravel(),
        np.full(SA.size, rh_med),
        np.full(SA.size, fr_med),
    ])
    dcd_grid, _ = predict(gp_dcd, sc_dcd, X_grid)
    c = ax2.contourf(SA, DA, dcd_grid.reshape(SA.shape), levels=20, cmap="RdBu_r")
    plt.colorbar(c, ax=ax2, label="δCd (DES − RANS GP)")
    ax2.scatter(des_df["slant_angle"], des_df["diffuser_angle"],
                s=60, c="black", zorder=5, label="DES pts")
    ax2.set_xlabel("Slant angle (°)")
    ax2.set_ylabel("Diffuser angle (°)")
    ax2.set_title(f"Cd Correction δ(x)  [rh={rh_med:.0f}mm, fr={fr_med:.0f}mm]")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "mf_pareto_front.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → results/mf_pareto_front.png")

    # ── 10. EI on MF surrogate ────────────────────────────────────────────────
    print("\n── EI acquisition on MF surrogate ───────────────────────────────────")
    f_best = f1_opt
    x_ei, ei_val = maximise_ei(mf, f_best, bounds_list)
    x_ei_design = {f: float(x_ei[i]) for i, f in enumerate(FEATURES)}
    mu_cd_ei, _ = mf.predict_cd(x_ei.reshape(1, -1))
    mu_cl_ei, _ = mf.predict_cl(x_ei.reshape(1, -1))
    mu_cd_ei = float(np.atleast_1d(mu_cd_ei).ravel()[0])
    mu_cl_ei = float(np.atleast_1d(mu_cl_ei).ravel()[0])
    print(f"  Best EI point:  EI={ei_val:.6f}")
    for k, v in x_ei_design.items():
        print(f"    {k}: {v:.4f}")
    print(f"    Cd_predicted: {mu_cd_ei:.4f}")
    print(f"    Cl_predicted: {mu_cl_ei:.4f}")
    print(f"    f1_predicted: {mu_cd_ei + F1_LAMBDA*mu_cl_ei:.4f}")

    ei_design = x_ei_design.copy()
    ei_design.update({
        "Cd_predicted": mu_cd_ei,
        "Cl_predicted": mu_cl_ei,
        "f1_predicted": mu_cd_ei + F1_LAMBDA * mu_cl_ei,
        "EI": float(ei_val),
    })
    (RESULTS_DIR / "mf_next_candidate.json").write_text(
        json.dumps(ei_design, indent=2))
    print(f"  Saved → results/mf_next_candidate.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
