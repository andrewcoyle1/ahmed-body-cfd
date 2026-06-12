"""
doe_setup.py
============
Design of Experiments (DoE) framework for Ahmed body aerodynamic optimisation.
Covers:
  1. Parameter space definition
  2. Latin Hypercube Sampling
  3. Space-filling quality diagnostics
  4. Design point export (CSV + per-case config files ready for CFD submission)

Dependencies: numpy, scipy, pandas, matplotlib
  pip install numpy scipy pandas matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats.qmc import LatinHypercube, discrepancy
from scipy.spatial.distance import pdist
from pathlib import Path
import json

# ─── 1. PARAMETER SPACE DEFINITION ───────────────────────────────────────────
#
# Each parameter is defined by:
#   name        : identifier used in mesh morphing / solver scripts later
#   lower       : lower bound (physical units)
#   upper       : upper bound (physical units)
#   unit        : for documentation
#   description : human-readable label
#
# Ahmed body parameters chosen for aerodynamic significance:
#   - slant_angle     : rear slant angle (°) — most influential; drives Cd step at ~30°
#   - diffuser_angle  : underbody diffuser angle (°) — affects underbody suction
#   - ride_height     : ground clearance (mm) — affects underbody flow and drag
#   - front_radius    : front edge radius (mm) — affects separation at nose
#
# Add or remove parameters here; everything downstream adapts automatically.

PARAMETERS = [
    {
        "name":        "slant_angle",
        "lower":       15.0,
        "upper":       40.0,
        "unit":        "deg",
        "description": "Rear slant angle"
    },
    {
        "name":        "diffuser_angle",
        "lower":       0.0,
        "upper":       20.0,
        "unit":        "deg",
        "description": "Underbody diffuser angle"
    },
]

# Parameters fixed at nominal values after Phase 1 Sobol sensitivity analysis.
# front_radius: ST=0.001 (negligible); ride_height: ST=0.096 (moderate but weak
# relative to slant/diffuser). Both fixed at Lienhart reference geometry defaults.
FIXED_PARAMETERS = {
    "ride_height":  50.8,   # mm — Lienhart nominal ground clearance
    "front_radius": 100.0,  # mm — default nose fillet radius
}

# ─── 2. SAMPLING CONFIGURATION ───────────────────────────────────────────────

N_SAMPLES    = 50      # denser phase-2 sweep: 25× per parameter in 2D
RANDOM_SEED  = 42      # fix for reproducibility — document this in your report
N_CANDIDATES = 10      # number of LHS replicates to pick the best space-filling design

OUTPUT_DIR   = Path("doe_phase2")
OUTPUT_DIR.mkdir(exist_ok=True)


# ─── 3. LATIN HYPERCUBE SAMPLING ─────────────────────────────────────────────

def generate_lhs(n_samples: int, n_params: int, seed: int, n_candidates: int) -> np.ndarray:
    """
    Generate a Latin Hypercube Sample in [0, 1]^n_params.

    Uses 'strength=2' (scipy default) which ensures each stratum is hit
    exactly once in every 1D projection — better space-filling than basic LHS.

    To maximise space-filling quality, we generate n_candidates independent
    LHS designs and select the one with the lowest centered L2-discrepancy
    (a standard measure of how uniformly the points fill the space).
    """
    best_design   = None
    best_disc     = np.inf

    for i in range(n_candidates):
        sampler = LatinHypercube(d=n_params, seed=seed + i)
        sample  = sampler.random(n=n_samples)
        disc    = discrepancy(sample, method="CD")   # centered L2-discrepancy
        if disc < best_disc:
            best_disc   = disc
            best_design = sample

    print(f"Selected LHS design: centered L2-discrepancy = {best_disc:.6f}")
    print(f"  (lower = better space-filling; random uniform ≈ 0.13 for reference)")
    return best_design


def scale_to_physical(unit_samples: np.ndarray, params: list) -> np.ndarray:
    """
    Scale unit-hypercube samples [0,1] to physical parameter ranges.
    Simple linear scaling: x_physical = lower + x_unit * (upper - lower)
    """
    lowers = np.array([p["lower"] for p in params])
    uppers = np.array([p["upper"] for p in params])
    return lowers + unit_samples * (uppers - lowers)


# ─── 4. SPACE-FILLING DIAGNOSTICS ────────────────────────────────────────────

def min_interpoint_distance(samples: np.ndarray) -> float:
    """
    Minimum pairwise Euclidean distance between design points (unit space).
    Higher = better spread. Useful sanity check alongside discrepancy.
    """
    dists = pdist(samples)
    return float(np.min(dists))


def plot_scatter_matrix(unit_samples: np.ndarray, physical_samples: np.ndarray,
                        params: list, output_dir: Path):
    """
    Scatter matrix of all parameter pairs (unit space).
    Shows 1D marginal histograms on the diagonal.
    A well-designed LHS should show even coverage in every 2D projection.
    """
    n = len(params)
    names = [p["name"] for p in params]
    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    fig.suptitle("DoE Scatter Matrix — Unit Parameter Space", fontsize=13, y=1.01)

    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            if i == j:
                ax.hist(unit_samples[:, i], bins=8, color="#1A6FAF", edgecolor="white", linewidth=0.5)
                ax.set_ylabel("Count", fontsize=7)
            else:
                ax.scatter(unit_samples[:, j], unit_samples[:, i],
                           s=18, color="#1A6FAF", alpha=0.75, edgecolors="white", linewidth=0.3)
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)

            if i == n - 1:
                ax.set_xlabel(names[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(names[i], fontsize=8)

            ax.tick_params(labelsize=6)

    plt.tight_layout()
    path = output_dir / "doe_scatter_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Scatter matrix saved → {path}")


def plot_1d_projections(unit_samples: np.ndarray, physical_samples: np.ndarray,
                        params: list, output_dir: Path):
    """
    1D projection plots showing how each parameter is sampled.
    Each point is shown as a rug plot; ideal LHS has one point per stratum.
    """
    n = len(params)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 2.5))
    fig.suptitle("1D Parameter Projections (Physical Space)", fontsize=11)

    for k, (ax, param) in enumerate(zip(axes, params)):
        vals = physical_samples[:, k]
        ax.scatter(vals, np.zeros_like(vals), s=40, color="#1A6FAF",
                   alpha=0.8, zorder=3, edgecolors="white", linewidth=0.5)
        # draw stratum boundaries
        strata = np.linspace(param["lower"], param["upper"], N_SAMPLES + 1)
        for s in strata:
            ax.axvline(s, color="#CCCCCC", linewidth=0.5, zorder=1)
        ax.set_xlim(param["lower"], param["upper"])
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel(f"{param['name']} ({param['unit']})", fontsize=9)
        ax.set_yticks([])
        ax.set_title(param["description"], fontsize=8)

    plt.tight_layout()
    path = output_dir / "doe_1d_projections.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"1D projections saved  → {path}")


# ─── 5. EXPORT DESIGN POINTS ─────────────────────────────────────────────────

def export_csv(physical_samples: np.ndarray, params: list, output_dir: Path) -> pd.DataFrame:
    """
    Export all design points to CSV.
    Columns: case_id + one column per active parameter + fixed parameters.
    Fixed parameters are appended as constant columns so downstream runners
    (cfmesh_doe_runner.py) require no changes.
    """
    df = pd.DataFrame(physical_samples, columns=[p["name"] for p in params])
    df.insert(0, "case_id", [f"case_{i:03d}" for i in range(len(df))])
    for name, val in FIXED_PARAMETERS.items():
        df[name] = val
    path = output_dir / "design_matrix.csv"
    df.to_csv(path, index=False, float_format="%.4f")
    print(f"Design matrix saved   → {path}  ({len(df)} cases)")
    return df


def export_case_configs(df: pd.DataFrame, params: list, output_dir: Path):
    """
    Write one JSON config file per CFD case.
    Your CFD submission script (next module) will read these to set boundary
    conditions, mesh morph targets, and solver parameters automatically.

    JSON format chosen for readability and easy parsing in Python / shell.
    """
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(exist_ok=True)

    for _, row in df.iterrows():
        config = {
            "case_id":    row["case_id"],
            "parameters": {p["name"]: {
                "value": round(row[p["name"]], 4),
                "unit":  p["unit"]
            } for p in params},
            # Placeholders — populated by CFD post-processing later
            "results": {
                "Cd":    None,
                "Cl":    None,
                "Cd_Cl": None
            }
        }
        path = cases_dir / f"{row['case_id']}.json"
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    print(f"Case configs saved    → {cases_dir}/  ({len(df)} files)")


def export_summary(unit_samples, physical_samples, params, output_dir):
    """Print and save a summary of the DoE to a text file."""
    lines = [
        "=" * 60,
        "DoE SUMMARY",
        "=" * 60,
        f"  N samples     : {N_SAMPLES}",
        f"  N parameters  : {len(params)}",
        f"  Random seed   : {RANDOM_SEED}",
        f"  N candidates  : {N_CANDIDATES}",
        "",
        "Parameters:",
    ]
    for p in params:
        lines.append(f"  {p['name']:20s}  [{p['lower']:>7.2f} – {p['upper']:>7.2f}] {p['unit']}")

    lines += [
        "",
        "Space-filling metrics (unit space):",
        f"  Centered L2 discrepancy : see console output above",
        f"  Min interpoint distance : {min_interpoint_distance(unit_samples):.4f}",
        "",
        "Physical parameter ranges actually sampled:",
    ]
    df_phys = pd.DataFrame(physical_samples, columns=[p["name"] for p in params])
    for p in params:
        col = df_phys[p["name"]]
        lines.append(f"  {p['name']:20s}  min={col.min():.3f}  max={col.max():.3f}  mean={col.mean():.3f}  ({p['unit']})")

    lines += ["", "=" * 60]
    summary = "\n".join(lines)
    print("\n" + summary)

    path = output_dir / "doe_summary.txt"
    with open(path, "w") as f:
        f.write(summary)
    print(f"\nSummary saved         → {path}")


# ─── 6. MAIN ─────────────────────────────────────────────────────────────────

def main():
    n_params = len(PARAMETERS)

    print(f"\nGenerating {N_SAMPLES}-point LHS for {n_params} parameters...")
    print(f"Evaluating {N_CANDIDATES} candidate designs to maximise space-filling...\n")

    # Sample in unit hypercube, then scale to physical space
    unit_samples     = generate_lhs(N_SAMPLES, n_params, RANDOM_SEED, N_CANDIDATES)
    physical_samples = scale_to_physical(unit_samples, PARAMETERS)

    # Diagnostics
    plot_scatter_matrix(unit_samples, physical_samples, PARAMETERS, OUTPUT_DIR)
    plot_1d_projections(unit_samples, physical_samples, PARAMETERS, OUTPUT_DIR)
    export_summary(unit_samples, physical_samples, PARAMETERS, OUTPUT_DIR)

    # Export
    df = export_csv(physical_samples, PARAMETERS, OUTPUT_DIR)
    export_case_configs(df, PARAMETERS, OUTPUT_DIR)

    print("\nDone. Next step: feed design_matrix.csv into your CFD submission script.")
    print("Each case_XXX.json will be populated with Cd/Cl results after CFD runs.")


if __name__ == "__main__":
    main()
