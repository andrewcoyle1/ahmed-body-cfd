"""
generate_flow_viz.py
====================
Generates CFD flow field visualisation plots from the optimal BO case
(case_bo_020) using PyVista to read OpenFOAM fields and matplotlib to render.

Outputs
-------
  results/flow_viz_surface_cp.png    — Cp contour on ahmed_body surface (side+top views)
  results/flow_viz_symmetry_plane.png — U_mag + pressure in XZ symmetry plane
  results/flow_viz_streamlines.png   — streamlines in XZ symmetry plane

Usage
-----
  python3 generate_flow_viz.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.interpolate import griddata
from pathlib import Path

os.chdir(Path(__file__).parent)

import pyvista as pv
pv.OFF_SCREEN = True

RESULTS_DIR = Path("results")
CASE_DIR    = Path("openfoam_cases/case_bo_020")
U_INF       = 40.0          # m/s freestream
Q_INF       = 0.5 * 1.225 * U_INF**2   # dynamic pressure (Pa equivalent, kinematic)

AM_GREEN = "#1A3C34"
AM_LIGHT = "#F5F5F0"

# ─── Load OpenFOAM case ───────────────────────────────────────────────────────

foam_file = CASE_DIR / "case_bo_020.foam"
foam_file.touch()

reader = pv.OpenFOAMReader(str(foam_file))
reader.set_active_time_value(reader.time_values[-1])
mesh    = reader.read()
internal = mesh["internalMesh"]
body_surf = mesh["boundary"]["ahmed_body"]
ground    = mesh["boundary"].get("ground", None)

# ─── Helper: extract numpy arrays from a PyVista mesh ─────────────────────────

def get_field(pvmesh, name):
    """Return first occurrence of a named array (handles duplicate names)."""
    for arr_name in pvmesh.array_names:
        if arr_name == name:
            return np.array(pvmesh[name])
    return None


def cp(p_kinematic):
    """Pressure coefficient: Cp = 2*p / U_inf^2  (kinematic p in m²/s²)."""
    return 2.0 * p_kinematic / U_INF**2


# ─── 1. SURFACE Cp — side + top projections ───────────────────────────────────

def plot_surface_cp():
    # cell_data → point_data for consistent indexing
    body_pt = body_surf.cell_data_to_point_data()
    pts   = np.array(body_pt.points)
    p_arr = get_field(body_pt, "p")
    cp_arr = cp(p_arr)

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    cp_lim = max(abs(cp_arr.min()), abs(cp_arr.max()))
    cp_lim = min(cp_lim, 2.5)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(AM_LIGHT)
    fig.suptitle("Surface Pressure Coefficient  $C_p = 2p / U_\\infty^2$\n"
                 "Optimal design  —  slant=39.1°  diffuser=8.5°  ride_height=71.8 mm",
                 fontsize=11, fontweight="bold", color=AM_GREEN)

    cmap = plt.cm.RdBu_r

    # Side view (XZ, showing lateral face at y_max)
    ax = axes[0]
    # Use points visible from the side: keep the outermost y face
    y_thresh = y.max() * 0.98
    mask_side = y >= y_thresh
    if mask_side.sum() < 10:
        mask_side = np.ones(len(y), dtype=bool)
    tri_s = mtri.Triangulation(x[mask_side], z[mask_side])
    tcf = ax.tricontourf(tri_s, cp_arr[mask_side], levels=20,
                         cmap=cmap, vmin=-cp_lim, vmax=cp_lim)
    ax.set_xlabel("x  (m)", fontsize=10)
    ax.set_ylabel("z  (m)", fontsize=10)
    ax.set_title("Side view  (XZ projection)", fontsize=10)
    ax.set_aspect("equal")
    ax.set_facecolor("#222222")
    plt.colorbar(tcf, ax=ax, label="$C_p$", shrink=0.85)

    # Top view (XY)
    ax = axes[1]
    tri_t = mtri.Triangulation(x, y)
    tcf2  = ax.tricontourf(tri_t, cp_arr, levels=20,
                           cmap=cmap, vmin=-cp_lim, vmax=cp_lim)
    ax.set_xlabel("x  (m)", fontsize=10)
    ax.set_ylabel("y  (m)", fontsize=10)
    ax.set_title("Top view  (XY projection)", fontsize=10)
    ax.set_aspect("equal")
    ax.set_facecolor("#222222")
    plt.colorbar(tcf2, ax=ax, label="$C_p$", shrink=0.85)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out = RESULTS_DIR / "flow_viz_surface_cp.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ─── 2. SYMMETRY PLANE — velocity magnitude + pressure ────────────────────────

def get_symmetry_slice():
    """Return (x, z, p, U_mag, Ux, Uz) arrays on the XZ symmetry slice."""
    slc = internal.slice(normal="y", origin=(0, 0.01, 0))
    slc = slc.cell_data_to_point_data()   # align arrays to vertices
    pts = np.array(slc.points)
    p_s = get_field(slc, "p")
    U_s = get_field(slc, "U")
    if U_s is None or U_s.ndim == 1:
        return None
    U_mag = np.linalg.norm(U_s[:, :3], axis=1)
    Ux    = U_s[:, 0]
    Uz    = U_s[:, 2]
    return pts[:, 0], pts[:, 2], p_s, U_mag, Ux, Uz


def interpolate_to_grid(x, z, values, nx=400, nz=200):
    x_grid = np.linspace(x.min(), x.max(), nx)
    z_grid = np.linspace(z.min(), z.max(), nz)
    XX, ZZ = np.meshgrid(x_grid, z_grid)
    VV = griddata((x, z), values, (XX, ZZ), method="linear")
    return XX, ZZ, VV


def plot_symmetry_plane():
    data = get_symmetry_slice()
    if data is None:
        print("Could not extract symmetry slice")
        return
    x, z, p_arr, U_mag, Ux, Uz = data

    # Focus region: just around the body and near wake
    mask = (x > -2.0) & (x < 5.0) & (z < 1.2)
    x, z, p_arr, U_mag, Ux, Uz = (a[mask] for a in (x, z, p_arr, U_mag, Ux, Uz))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.patch.set_facecolor(AM_LIGHT)
    fig.suptitle("Symmetry Plane  (y ≈ 0)  —  Optimal design  slant=39.1°",
                 fontsize=11, fontweight="bold", color=AM_GREEN)

    # ── Velocity magnitude ──
    ax = axes[0]
    XX, ZZ, UU = interpolate_to_grid(x, z, U_mag)
    cf = ax.contourf(XX, ZZ, UU, levels=30, cmap="viridis")
    plt.colorbar(cf, ax=ax, label="|U|  (m/s)", shrink=0.85)

    # Body outline (approximate)
    body_x  = [0.0, 1.044, 1.044, 0.0, 0.0]
    body_z  = [0.0718, 0.0718, 0.360, 0.360, 0.0718]
    ax.fill(body_x, body_z, color="white", alpha=0.85, zorder=3)
    ax.plot(body_x, body_z, "k-", lw=1.2, zorder=4, label="Body outline")
    ax.fill_between([-2, 5], 0, 0.0, color="#666", alpha=0.5)   # ground

    ax.set_ylabel("z  (m)", fontsize=10)
    ax.set_title("Velocity magnitude  |U|  (m/s)", fontsize=10)
    ax.set_facecolor("#111111")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 1.2)

    # ── Pressure ──
    ax = axes[1]
    XX2, ZZ2, PP = interpolate_to_grid(x, z, cp(p_arr))
    cp_lim = 1.5
    cf2 = ax.contourf(XX2, ZZ2, PP, levels=30, cmap="RdBu_r",
                      vmin=-cp_lim, vmax=cp_lim)
    plt.colorbar(cf2, ax=ax, label="$C_p$", shrink=0.85)
    ax.fill(body_x, body_z, color="white", alpha=0.85, zorder=3)
    ax.plot(body_x, body_z, "k-", lw=1.2, zorder=4)
    ax.set_xlabel("x  (m)", fontsize=10)
    ax.set_ylabel("z  (m)", fontsize=10)
    ax.set_title("Pressure coefficient  $C_p$", fontsize=10)
    ax.set_facecolor("#111111")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 1.2)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = RESULTS_DIR / "flow_viz_symmetry_plane.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ─── 3. STREAMLINES in symmetry plane ─────────────────────────────────────────

def plot_streamlines():
    data = get_symmetry_slice()
    if data is None:
        return
    x, z, p_arr, U_mag, Ux, Uz = data

    mask = (x > -2.5) & (x < 4.5) & (z < 1.0)
    x, z, p_arr, U_mag, Ux, Uz = (a[mask] for a in (x, z, p_arr, U_mag, Ux, Uz))

    nx, nz = 500, 250
    x_grid = np.linspace(float(x.min()), float(x.max()), nx)
    z_grid = np.linspace(0.001, float(z.max()), nz)
    XX, ZZ = np.meshgrid(x_grid, z_grid)

    UU = griddata((x, z), Ux,    (XX, ZZ), method="linear")
    WW = griddata((x, z), Uz,    (XX, ZZ), method="linear")
    MM = griddata((x, z), U_mag, (XX, ZZ), method="linear")

    # Replace NaNs (inside body / outside domain) with freestream
    UU = np.where(np.isnan(UU), U_INF, UU)
    WW = np.where(np.isnan(WW), 0.0,   WW)
    MM = np.where(np.isnan(MM), U_INF, MM)

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")

    # Background: velocity magnitude contour
    cf = ax.contourf(XX, ZZ, MM, levels=40, cmap="inferno", alpha=0.85)
    plt.colorbar(cf, ax=ax, label="|U|  (m/s)", shrink=0.9)

    # Streamlines
    seed_z   = np.linspace(0.02, 0.90, 28)
    seed_x   = np.full_like(seed_z, x.min() + 0.05)
    ax.streamplot(x_grid, z_grid, UU, WW,
                  start_points=np.column_stack([seed_x, seed_z]),
                  color="white", linewidth=0.7, density=3,
                  arrowsize=0.6, arrowstyle="->")

    # Body
    body_x = [0.0, 1.044, 1.044, 0.0, 0.0]
    body_z = [0.0718, 0.0718, 0.360, 0.360, 0.0718]
    ax.fill(body_x, body_z, color="#555", alpha=1.0, zorder=5)
    ax.plot(body_x, body_z, "w-", lw=1.5, zorder=6)

    # Ground
    ax.fill_between([x.min(), x.max()], 0, 0.001, color="#444", zorder=4)

    ax.set_xlabel("x  (m)", fontsize=11, color="white")
    ax.set_ylabel("z  (m)", fontsize=11, color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("white")
    ax.set_title("Streamlines in symmetry plane  —  optimal design  slant=39.1°  f₁=0.2541",
                 fontsize=11, color="white", pad=10)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 0.95)

    plt.tight_layout()
    out = RESULTS_DIR / "flow_viz_streamlines.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close()
    print(f"Saved → {out}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Reading OpenFOAM case (t=2065)…")
    print("── Surface Cp ────────────────────────────────")
    plot_surface_cp()
    print("── Symmetry plane ────────────────────────────")
    plot_symmetry_plane()
    print("── Streamlines ───────────────────────────────")
    plot_streamlines()
    print("\nDone.")
