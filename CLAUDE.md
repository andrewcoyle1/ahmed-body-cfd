# Ahmed Body CFD — Project Guidelines

## Report Documentation Rule

**Every engineering decision made in this project must be documented in the LaTeX report.**

Report location: `cfd_report/`
- Methodology decisions → `cfd_report/Methodology/Methodology.tex`
- Numerical results, plots, tables → `cfd_report/Results/Results.tex`
- Per-section subdirs: `Abstract/`, `introduction/`, `conclusion/`, `Discussion/`

When you make or validate a decision during a session, update the relevant `.tex` file
immediately — do not defer. Decisions include but are not limited to:

- Turbulence model selection and rationale (kOmegaSST, realizableKE, etc.)
- Mesh quality metrics (non-orthogonality, skewness, cell counts per level)
- Solver settings (non-orthogonal correctors, relaxation factors, convergence criteria)
- Time-stepping strategy for URANS (maxCo, deltaT, pimpleFoam settings)
- Wall treatment choice (nutUSpaldingWallFunction, kLowReWallFunction — why all-y+)
- Co-Kriging fidelity level assignment (which mesh level = LF, which = HF)
- Any fix applied to a simulation error (include what the error was and why the fix works)
- Simulation results: Cd, Cl, f = Cd + (1/3)*Cl per mesh level and turbulence model

## Critical Constraints

- **Never invoke mesh or solve scripts as `./script.sh`** — always use `bash ./script.sh` (macOS SIP strips DYLD_LIBRARY_PATH on shebang exec).

## Project Overview

Ahmed body (25° slant) drag/lift optimisation using OpenFOAM v2512 + cfMesh on macOS (Mac Mini M4 Pro).

**Objective:** minimise f = Cd + (1/3)*Cl

**Reference:** Lienhart 2002 experimental Cd = 0.299 (25° slant, Re ≈ 2.78×10⁶)

**Geometry:** length 1.044 m, Aref = 0.056 m², U∞ = 40 m/s

**Pipeline:**
1. RANS DoE (30 LHS pts, simpleFoam, kOmegaSST) — low-fidelity: L2 mesh (~897K cells)
2. URANS/DES anchors (10 pts, pimpleFoam) — high-fidelity: L3 mesh (~1.82M cells)
3. Co-Kriging surrogate → EI acquisition → 44 BO iterations
4. Local LHS refinement (12 pts) around optimum

**Fidelity levels:**
- L2 (~897K cells, ~10 min/solve) = low-fidelity (LF)
- L3 (~1.82M cells, ~25 min/solve) = high-fidelity (HF)
- L4 excluded (diverged due to high-skewness faces)

## Key File Locations

| File | Purpose |
|------|---------|
| `cfmesh_validation/` | Validated baseline case (kOmegaSST, Cd≈0.324) |
| `mesh_convergence/L{1-4}_*/` | Mesh convergence study cases |
| `mesh_convergence/L2_realizableKE/` | realizableKE model comparison |
| `mesh_convergence/L2_URANS/` | URANS pimpleFoam hot-start case |
| `mesh_convergence/build_all_meshes.sh` | Mesh build script (run as `bash ./build_all_meshes.sh`) |
| `mesh_convergence/run_all_solves.sh` | Solve script with auto NP scaling |
| `cfd_report/` | LaTeX report |

## Force Coefficient Column Order

forceCoeffs output columns: Time, **Cd**, Cd(f), Cd(r), **Cl**, Cl(f), Cl(r), CmPitch, CmRoll, CmYaw, Cs

In awk: `$2=Cd`, `$5=Cl`, `$11=Cs`

## Methodology.tex Known Issue

The current `Methodology/Methodology.tex` references blockMesh/snappyHexMesh — this is outdated.
The actual meshing uses cfMesh `cartesianMesh` + `generateBoundaryLayers`. Update this section
when next editing the methodology.
