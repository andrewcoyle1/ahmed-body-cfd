"""
Generate PGF figures for the LaTeX report.
Outputs → cfd_report/figures/
"""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("pgf")  # switch to "pgf" for final report output
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from pathlib import Path

BASE   = Path(__file__).parent
FIGS   = BASE / "cfd_report" / "figures"
FIGS.mkdir(exist_ok=True)

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "pgf.texsystem":    "pdflatex",
    "pgf.preamble":     r"\usepackage[T1]{fontenc}\usepackage{mathpazo}\usepackage{amsmath}",
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   10,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "text.usetex":      False,  # set True when using pgf backend
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

PNG_MODE = False  # set False to output PGF for the report

def save(fig, name):
    if PNG_MODE:
        path = FIGS / f"{name}.png"
        fig.savefig(path, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        path = FIGS / f"{name}.pgf"
        fig.savefig(path, format="pgf", bbox_inches="tight")
        plt.close(fig)
        # Prefix companion image paths so LaTeX finds them from cfd_report/
        txt = path.read_text()
        txt = txt.replace(f"{{{name}-img", f"{{figures/{name}-img")
        path.write_text(txt)
    print(f"  Saved → {path}")


# ── 1. DoE design-space scatter ───────────────────────────────────────────────
def fig_doe_scatter():
    df = pd.read_csv(BASE / "results" / "results_phase2.csv")
    df = df[df["Cd"].notna()].copy()
    df["f"] = df["Cd"] + (1/3) * df["Cl"]

    # best point
    best = df.loc[df["f"].idxmin()]

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    sc = ax.scatter(df["slant_angle"], df["diffuser_angle"],
                    c=df["f"], cmap="viridis_r", s=40, zorder=3,
                    edgecolors="none", vmin=0.20, vmax=0.45)
    cb = fig.colorbar(sc, ax=ax, label=r"$f = C_d + \frac{1}{3}C_l$")
    cb.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    ax.scatter(best["slant_angle"], best["diffuser_angle"],
               marker="*", s=200, c="red", zorder=5, label=f"Best CFD  $f$={best['f']:.4f}")

    ax.set_xlabel(r"Slant angle $\alpha_\mathrm{slant}$ (°)")
    ax.set_ylabel(r"Diffuser angle $\alpha_\mathrm{diff}$ (°)")
    ax.legend(loc="center", bbox_to_anchor=(0.5, -0.2), borderaxespad=0, frameon=False)
    ax.set_xlim(14, 41)
    ax.set_ylim(-0.5, 20.5)
    save(fig, "doe_scatter")


# ── 2. Full campaign objective convergence ────────────────────────────────────
def fig_campaign_convergence():
    # Phase 1
    p1 = pd.read_csv(BASE / "results" / "results_summary.csv")
    p1 = p1[p1["Cd"].notna()].copy()
    p1["f"] = p1["Cd"] + (1/3) * p1["Cl"]
    p1["phase"] = "Phase 1"

    # Phase 2
    p2 = pd.read_csv(BASE / "results" / "results_phase2.csv")
    p2 = p2[p2["Cd"].notna()].copy()
    p2["f"] = p2["Cd"] + (1/3) * p2["Cl"]
    p2["phase"] = "Phase 2"

    # BO
    bo = pd.read_csv(BASE / "results" / "bo_history.csv")
    bo = bo[bo["converged"] == True].copy()
    bo["f"] = bo["f1_cfd"]
    bo["phase"] = "BO"

    n1 = len(p1)
    n2 = len(p2)
    p1["seq"] = range(1, n1 + 1)
    p2["seq"] = range(n1 + 1, n1 + n2 + 1)
    bo["seq"] = range(n1 + n2 + 1, n1 + n2 + len(bo) + 1)

    all_f = (list(p1["f"]) + list(p2["f"]) + list(bo["f"]))
    all_seq = (list(p1["seq"]) + list(p2["seq"]) + list(bo["seq"]))
    running_best = np.minimum.accumulate(all_f)

    colors = {"Phase 1": "#4878d0", "Phase 2": "#ee854a", "BO": "#6acc65"}

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    for phase, sub in [("Phase 1", p1), ("Phase 2", p2), ("BO", bo)]:
        ax.scatter(sub["seq"], sub["f"], s=18, alpha=0.7,
                   color=colors[phase], label=phase, zorder=3)

    ax.plot(all_seq, running_best, "k-", lw=1.5, label="Running best", zorder=4)

    # Phase boundary lines and labels
    ax.axvline(n1 + 0.5, color="grey", lw=0.8, ls="--")
    ax.axvline(n1 + n2 + 0.5, color="grey", lw=0.8, ls="--")
    ax.text(n1 / 2,            0.78, "Phase 1", ha="center", fontsize=8, color="grey")
    ax.text(n1 + n2 / 2,      0.78, "Phase 2", ha="center", fontsize=8, color="grey")
    ax.text(n1 + n2 + 10 / 2, 0.78, "BO",      ha="center", fontsize=8, color="grey")

    ax.set_xlabel("Cumulative CFD evaluation")
    ax.set_ylabel(r"$f = C_d + \frac{1}{3}C_l$")
    ax.set_ylim(bottom=0.15)
    ax.legend(loc="upper left", bbox_to_anchor=(0.075, -0.15), borderaxespad=0, frameon=False, ncol=4)
    save(fig, "campaign_convergence")


# ── 3. Residual convergence at optimum (case_021) ────────────────────────────
def fig_residuals():
    log = BASE / "openfoam_cases_phase2" / "case_021" / "log.simpleFoam"
    pattern = re.compile(
        r"^Time = (\d+)\s*$|"
        r"smoothSolver:.*?Solving for (Ux|Uy|Uz|p|k|omega).*?Initial residual = ([0-9.eE+\-]+)"
    )
    times, fields = [], {}
    current_time = None
    with open(log) as fh:
        for line in fh:
            line = line.strip()
            m_time = re.match(r"^Time = (\d+)$", line)
            if m_time:
                current_time = int(m_time.group(1))
                times.append(current_time)
                continue
            m_res = re.match(
                r"smoothSolver:.*?Solving for (\w+),.*?Initial residual = ([0-9.eE+\-]+)",
                line)
            if m_res and current_time is not None:
                field = m_res.group(1)
                val = float(m_res.group(2))
                fields.setdefault(field, []).append((current_time, val))

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    style = {"Ux": ("-",  "#4878d0"), "p": ("-",  "#ee854a"),
             "k":  ("--", "#6acc65"), "omega": ("--", "#d65f5f")}
    labels = {"Ux": r"$U_x$", "p": r"$p$", "k": r"$k$", "omega": r"$\omega$"}

    for field, pairs in fields.items():
        if field not in style:
            continue
        t, v = zip(*pairs)
        ls, col = style[field]
        ax.semilogy(t, v, ls=ls, color=col, lw=1.2, label=labels[field])

    ax.axvspan(600, 800, alpha=0.10, color="grey", label="Averaging window")
    ax.axhline(1e-4, color="grey", lw=0.8, ls=":", label=r"$10^{-4}$ target")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Initial residual")
    ax.set_xlim(0, 800)
    ax.legend(loc="upper left", frameon=False, ncol=2)
    save(fig, "residuals_optimum")


# ── 4. Force coefficient time-history at optimum ──────────────────────────────
def fig_forcecoeffs():
    dat = BASE / "openfoam_cases_phase2" / "case_021" / \
          "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat"

    rows = []
    with open(dat) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    rows.append([float(x) for x in parts])
                except ValueError:
                    pass

    arr = np.array(rows)
    t  = arr[:, 0]
    Cd = arr[:, 1]
    Cl = arr[:, 4]

    mask = (t >= 600) & (t <= 800)
    cd_mean = np.mean(Cd[mask])
    cl_mean = np.mean(Cl[mask])
    f_mean  = cd_mean + (1/3) * cl_mean

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.0, 4.5), sharex=True)

    ax1.plot(t, Cd, color="#4878d0", lw=0.9, label=r"$C_d$")
    ax1.axvspan(600, 800, alpha=0.12, color="grey")
    ax1.axhline(cd_mean, color="#4878d0", lw=1.0, ls="--",
                label=rf"Mean $C_d$ = {cd_mean:.4f}")
    ax1.set_ylabel(r"$C_d$")
    ax1.set_ylim(0.25, 0.55)

    ax2.plot(t, Cl, color="#ee854a", lw=0.9, label=r"$C_l$")
    ax2.axvspan(600, 800, alpha=0.12, color="grey", label="Averaging window")
    ax2.axhline(cl_mean, color="#ee854a", lw=1.0, ls="--",
                label=rf"Mean $C_l$ = {cl_mean:.4f}")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel(r"$C_l$")
    ax2.set_xlim(0, 800)
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.075), frameon=False, ncol=3)  # global legend
    fig.tight_layout()
    save(fig, "forcecoeffs_optimum")


# ── 5. Ahmed body geometry schematic ─────────────────────────────────────────
def fig_geometry_schematic():
    """Side-view dimensioned sketch of the Ahmed body (25 deg baseline geometry)."""
    # Key dimensions (mm, Lienhart geometry)
    L   = 1044.0   # total length
    H   = 288.0    # height
    h   = 50.8     # ride height (ground clearance)
    R   = 100.0    # nose radius (simplified as quarter-circle bounding box)
    sl  = 222.0    # slant length (projected horizontal, approx for 25 deg)
    dh  = 50.0     # diffuser height (approx)
    dl  = 200.0    # diffuser length (approx)

    # Normalise to unit length
    s = 1.0 / L
    def sc(v): return v * s

    # Ground line at y=0; body sits at y=sc(h)
    floor = sc(h)
    roof  = floor + sc(H)
    nose_x = sc(R)

    # Body outline (simplified profile, right-facing)
    # Rear slant starts at x_slant_start, angled to rear base
    x_slant_start = sc(820.0)   # approx where slant begins
    y_slant_start = roof
    x_rear_top    = sc(1044.0 - 0.0)
    y_rear_bot    = floor + sc(dh)

    body_x = [nose_x, sc(L - dl - 0.0), x_slant_start, sc(L), sc(L), sc(R), nose_x]
    body_y = [roof,   roof,              roof,           y_rear_bot + sc(dh*0.5),
              floor + sc(dh), floor, floor]

    # Simplified smooth nose: just use the rectangular outline
    body_x = [
        sc(R),          # nose top
        x_slant_start,  # roof flat
        sc(L),          # rear top (after slant)
        sc(L),          # rear base
        sc(L - dl),     # diffuser start
        sc(0),          # underbody front
        sc(0),          # front face bottom
        sc(R),          # front face top
    ]
    body_y = [
        roof,           # nose top
        roof,           # roof flat
        roof - sc(H*0.38),  # rear top (slant drops ~25 deg over 222 mm)
        floor + sc(dh),    # rear base
        floor,          # diffuser start
        floor,          # underbody
        roof,           # front face bottom → top
        roof,
    ]

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.set_aspect("equal")

    # Ground
    ax.axhline(0, color="black", lw=1.5)
    ax.fill_between([0, 1.05], [-0.01, -0.01], [0, 0], color="#d0d0d0")

    # Body silhouette
    ax.fill(body_x, body_y, color="#c8d8e8", ec="black", lw=1.2, zorder=3)

    # Nose radius arc (quarter circle approximation)
    theta = np.linspace(np.pi/2, np.pi, 40)
    ax.plot(sc(R) + sc(R)*np.cos(theta),
            roof  + sc(R)*np.sin(theta) - sc(R),
            color="black", lw=1.2, zorder=4)

    # Dimension annotations
    ann_kw = dict(arrowprops=dict(arrowstyle="<->", color="black", lw=0.8),
                  fontsize=7.5, color="black", ha="center")

    # Total length
    ax.annotate("", xy=(sc(L), -0.06), xytext=(0, -0.06),
                arrowprops=dict(arrowstyle="<->", color="black", lw=0.8))
    ax.text(0.5, -0.085, r"$L = 1044$ mm", ha="center", va="top", fontsize=7.5)

    # Ride height
    ax.annotate("", xy=(-0.04, floor), xytext=(-0.04, 0),
                arrowprops=dict(arrowstyle="<->", color="black", lw=0.8))
    ax.text(-0.07, floor/2, r"$h$", ha="center", va="center", fontsize=7.5)

    # Body height
    ax.annotate("", xy=(-0.04, roof), xytext=(-0.04, floor),
                arrowprops=dict(arrowstyle="<->", color="black", lw=0.8))
    ax.text(-0.07, (roof+floor)/2, r"$H$", ha="center", va="center", fontsize=7.5)

    # Slant angle arc
    ax.annotate(r"$\alpha_\mathrm{slant}$",
                xy=(x_slant_start + 0.04, roof - 0.03),
                fontsize=8, color="#c0392b")

    # Diffuser angle
    ax.annotate(r"$\alpha_\mathrm{diff}$",
                xy=(sc(L - dl) + 0.02, floor + 0.015),
                fontsize=8, color="#2980b9")

    ax.set_xlim(-0.12, 1.10)
    ax.set_ylim(-0.13, 0.55)
    ax.axis("off")
    ax.set_title("Ahmed body --- side-view geometry (not to scale)", fontsize=9)

    save(fig, "geometry_schematic")


# ── 6. GP surrogate validation (LOO cross-validation) ────────────────────────
def fig_gp_validation():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.model_selection import LeaveOneOut

    df = pd.read_csv(BASE / "results" / "results_phase2.csv")
    df = df[df["Cd"].notna() & df["Cl"].notna()].copy()
    X = df[["slant_angle", "diffuser_angle"]].values
    y_cd = df["Cd"].values
    y_cl = df["Cl"].values

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    kernel = ConstantKernel() * Matern(nu=2.5) + WhiteKernel()
    loo = LeaveOneOut()

    for y, name in [(y_cd, "Cd"), (y_cl, "Cl")]:
        preds, stds = [], []
        for train, test in loo.split(Xs):
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                          normalize_y=True)
            gp.fit(Xs[train], y[train])
            mu, sigma = gp.predict(Xs[test], return_std=True)
            preds.append(float(mu[0]))
            stds.append(float(sigma[0]))
        if name == "Cd":
            cd_loo, cd_std = np.array(preds), np.array(stds)
        else:
            cl_loo, cl_std = np.array(preds), np.array(stds)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, y_true, y_pred, std, label in zip(
        axes,
        [y_cd, y_cl],
        [cd_loo, cl_loo],
        [cd_std, cl_std],
        [r"$C_d$", r"$C_l$"],
    ):
        lo = min(y_true.min(), y_pred.min()) * 0.97
        hi = max(y_true.max(), y_pred.max()) * 1.03
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="Perfect prediction")
        ax.errorbar(y_true, y_pred, yerr=2 * std, fmt="o", color="#1f77b4",
                    ecolor="#aec7e8", elinewidth=1.5, capsize=3, ms=5,
                    label=r"LOO prediction $\pm 2\sigma$")
        r2  = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        ax.set_xlabel(f"CFD {label}")
        ax.set_ylabel(f"GP {label}")
        ax.set_title(f"{label}  |  $R^2 = {r2:.3f}$  |  MAE $= {mae:.4f}$")
        ax.legend(frameon=True)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")

    fig.tight_layout()
    save(fig, "gp_validation")


# ── 7. GP response surface ────────────────────────────────────────────────────
def fig_response_surfaces():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(BASE / "results" / "results_phase2.csv")
    df = df[df["Cd"].notna() & df["Cl"].notna()].copy()
    X = df[["slant_angle", "diffuser_angle"]].values
    y_cd = df["Cd"].values
    y_cl = df["Cl"].values

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    kernel = ConstantKernel() * Matern(nu=2.5) + WhiteKernel()

    gp_cd = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                     normalize_y=True).fit(Xs, y_cd)
    gp_cl = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                     normalize_y=True).fit(Xs, y_cl)

    bounds = {"slant_angle": (15.0, 40.0), "diffuser_angle": (0.0, 20.0)}
    xi = np.linspace(*bounds["slant_angle"], 80)
    xj = np.linspace(*bounds["diffuser_angle"], 80)
    II, JJ = np.meshgrid(xi, xj)
    grid = np.column_stack([II.ravel(), JJ.ravel()])
    Xs_grid = scaler.transform(grid)

    cd_pred = gp_cd.predict(Xs_grid).reshape(II.shape)
    cl_pred = gp_cl.predict(Xs_grid).reshape(II.shape)
    f_pred  = cd_pred + (1/3) * cl_pred

    # Surrogate optimum
    idx = np.unravel_index(np.argmin(f_pred), f_pred.shape)
    opt_slant = II[idx]
    opt_diff  = JJ[idx]

    # DoE best
    df["f"] = df["Cd"] + (1/3) * df["Cl"]
    best = df.loc[df["f"].idxmin()]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    titles = [r"$C_d$", r"$C_l$", r"$f = C_d + \frac{1}{3}C_l$"]
    surfaces = [cd_pred, cl_pred, f_pred]
    cmaps = ["RdYlGn_r", "RdYlGn", "RdYlGn_r"]

    for ax, surf, title, cmap in zip(axes, surfaces, titles, cmaps):
        cf = ax.contourf(II, JJ, surf, levels=20, cmap=cmap)
        fig.colorbar(cf, ax=ax)
        ax.scatter(X[:, 0], X[:, 1], c="white", edgecolors="black",
                   s=18, zorder=5, label="DoE samples")
        ax.plot(opt_slant, opt_diff, "k+", ms=10, mew=2, zorder=6,
                label="Surrogate optimum")
        ax.plot(best["slant_angle"], best["diffuser_angle"], "r*",
                ms=12, zorder=7, label="CFD best")
        ax.set_xlabel(r"Slant angle (°)")
        ax.set_ylabel(r"Diffuser angle (°)")
        ax.set_title(title)

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="lower center", bbox_to_anchor=(0.5, -0.08),
               fontsize=8, frameon=False, ncol=3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    save(fig, "response_surfaces")


# ── 8. Sobol sensitivity bar chart ────────────────────────────────────────────
def fig_sobol():
    df = pd.read_csv(BASE / "results" / "sobol_indices.csv")
    objectives  = ["Cd", "Cl", "f"]
    titles      = [r"$C_d$", r"$C_l$", r"$f = C_d + \frac{1}{3}C_l$"]
    features    = df["variable"].unique().tolist()
    labels      = [f.replace("slant_angle", "Slant\nangle")
                    .replace("diffuser_angle", "Diffuser\nangle") for f in features]
    x = np.arange(len(features))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=False)

    for ax, obj, title in zip(axes, objectives, titles):
        sub = df[df["objective"] == obj].set_index("variable").reindex(features)
        S1  = np.maximum(sub["S1"].values, 0)
        S1c = sub["S1_conf"].values
        ST  = np.maximum(sub["ST"].values, 0)
        STc = sub["ST_conf"].values

        ax.bar(x - width/2, S1, width, label=r"$S_1$ (first-order)",
               color="#1A6FAF", alpha=0.85, yerr=S1c, capsize=4,
               error_kw={"elinewidth": 1})
        ax.bar(x + width/2, ST, width, label=r"$S_T$ (total-order)",
               color="#E87722", alpha=0.85, yerr=STc, capsize=4,
               error_kw={"elinewidth": 1})
        ax.axhline(0.05, color="red", ls="--", lw=0.8, alpha=0.7, label="5\\% threshold")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Sensitivity index" if ax == axes[0] else "")
        for i, (st, stc) in enumerate(zip(ST, STc)):
            ax.text(i + width/2, min(st + stc + 0.02, 0.92),
                    f"{st:.2f}", ha="center", va="bottom", fontsize=7)

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="lower center", bbox_to_anchor=(0.5, -0.0),
               fontsize=8, frameon=False, ncol=3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    save(fig, "sobol_sensitivity")


# ── 9. Bayesian optimisation convergence ──────────────────────────────────────
def fig_bo_convergence():
    history = pd.read_csv(BASE / "results" / "bo_history.csv")
    valid   = history.dropna(subset=["Cd_cfd"])

    iters   = valid["iteration"].values
    f1_best = valid["f1_best"].values
    ei_vals = valid["ei_max"].values
    f1_cfd  = valid["f1_cfd"].values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)

    ax1.step(iters, f1_best, where="post", color="#1f77b4", lw=2,
             label="Best $f$ incumbent")
    ax1.scatter(iters, f1_cfd, color="#ff7f0e", zorder=5, s=40,
                label="CFD $f$ (this iteration)")
    ax1.axhline(f1_best[0], color="grey", lw=0.8, ls=":",
                label=f"DoE best $f={f1_best[0]:.4f}$")
    ax1.set_ylabel(r"$f = C_d + \frac{1}{3}C_l$")
    ax1.legend(loc="upper center", frameon=False, fontsize=8)

    ax2.semilogy(iters, ei_vals, "s-", color="#2ca02c", lw=1.5, ms=5,
                 label="max EI")
    ax2.axhline(1e-4, color="red", lw=1, ls="--",
                label=r"Convergence threshold ($10^{-4}$)")
    ax2.set_xlabel("BO iteration")
    ax2.set_ylabel("max Expected Improvement")
    ax2.legend(loc="center left", frameon=False, fontsize=8)

    fig.suptitle("Bayesian Optimisation --- Convergence History")
    fig.tight_layout()
    save(fig, "bo_convergence")


# ── 10. Pareto front ──────────────────────────────────────────────────────────
def fig_pareto():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(BASE / "results" / "results_phase2.csv")
    df = df[df["Cd"].notna() & df["Cl"].notna()].copy()
    X = df[["slant_angle", "diffuser_angle"]].values
    y_cd = df["Cd"].values
    y_cl = df["Cl"].values

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    kernel = ConstantKernel() * Matern(nu=2.5) + WhiteKernel()
    gp_cd = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                     normalize_y=True).fit(Xs, y_cd)
    gp_cl = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                     normalize_y=True).fit(Xs, y_cl)

    rng = np.random.default_rng(42)
    X_rand = np.column_stack([rng.uniform(15, 40, 20000),
                               rng.uniform(0,  20, 20000)])
    Xs_r   = scaler.transform(X_rand)
    cd_p   = gp_cd.predict(Xs_r)
    cl_p   = gp_cl.predict(Xs_r)

    # Restrict to downforce-generating designs (Cl <= 0)
    df_mask = cl_p <= 0
    cd_df   = cd_p[df_mask]
    cl_df   = cl_p[df_mask]

    # 2-objective Pareto front via sorted sweep (exact for 2D):
    # Sort by Cd asc; a point is non-dominated if it has the best downforce seen so far.
    order      = np.argsort(cd_df)
    cd_sorted  = cd_df[order]
    cl_sorted  = cl_df[order]
    running_min_cl = np.inf
    pareto_idx = []
    for i, cl_val in enumerate(cl_sorted):
        if cl_val < running_min_cl:   # improves downforce (more negative Cl)
            pareto_idx.append(i)
            running_min_cl = cl_val

    p_cd = cd_sorted[pareto_idx]
    p_cl = cl_sorted[pareto_idx]

    # DoE CFD points with downforce
    cfd_df = y_cl <= 0

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(cd_df, -cl_df, c="#cccccc", s=4, alpha=0.25, label="Sampled designs")
    ax.scatter(y_cd[cfd_df], -y_cl[cfd_df], c="#1f77b4", s=40,
               zorder=5, label="CFD cases (DoE)")
    ax.plot(p_cd, -p_cl, "ro-", ms=4, lw=1.5, zorder=6, label="Pareto front")
    ax.set_xlabel(r"Drag coefficient $C_d$")
    ax.set_ylabel(r"Downforce $-C_l$")
    ax.set_title("Pareto Front: Drag vs Downforce")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "pareto_front")


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating report figures → cfd_report/figures/\n")
    fig_doe_scatter()
    fig_campaign_convergence()
    fig_residuals()
    fig_forcecoeffs()
    fig_geometry_schematic()
    fig_gp_validation()
    fig_response_surfaces()
    fig_sobol()
    fig_bo_convergence()
    fig_pareto()
    print("\nDone.")
