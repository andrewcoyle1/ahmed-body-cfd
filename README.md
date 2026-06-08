# F1 Aerodynamic Shape Optimisation — Ahmed Body

Independent research project implementing a full aerodynamic shape optimisation pipeline for a parametric Ahmed body bluff vehicle, using multi-fidelity CFD (RANS + DES) and Bayesian optimisation with Gaussian Process surrogates.

## Methodology

### 1. Geometry & Mesh

- Parametric Ahmed body with 4 design variables: slant angle (20°–45°), diffuser angle (0°–20°), ride height (40–90 mm), front-edge radius (25–150 mm)
- Geometry generated headless via FreeCAD; exported as STL for snappyHexMesh
- snappyHexMesh surface refinement + boundary layer inflation (15 prism layers on body, 8 on ground, y⁺ ≈ 1)
- Three mesh levels: 318 K / 1.46 M / 10.2 M cells

### 2. Mesh Verification

- Richardson extrapolation + Celik (2008) GCI method applied to Cd
- GCI(L2→L3) = 0.89% — medium mesh (1.46 M cells) confirmed grid-independent
- Results in `mesh_convergence/mesh_convergence.csv`

### 3. Turbulence Model Validation

Validated against Lienhart & Becker (2003) wind-tunnel measurements at 25° slant:

| Model | Cd | Error |
|---|---|---|
| Wall-fn k-ω SST (moving ground) | 0.326 | +9.0% |
| k-ω SST low-Re (stationary ground) | 0.284 | −5.0% |
| DES kOmegaSSTDES / IDDES | 0.299 | 0.0% |
| Experiment (Lienhart & Becker) | 0.299 | — |

DES selected as the high-fidelity model; k-ω SST low-Re RANS selected for the DoE sweep (good accuracy, tractable cost).

### 4. Design of Experiments

- 30-point Latin Hypercube sampling across the 4-dimensional design space
- Solver: simpleFoam (steady incompressible RANS), k-ω SST low-Re, 1.46 M cell mesh
- Parallelised: 2 concurrent workers × 6 MPI processes each, Docker containerised
- Design matrix in `design_matrix.csv`; sampled points in `doe_output/`

### 5. Surrogate & Bayesian Optimisation

- Gaussian Process surrogate with Matérn kernel + white noise regularisation
- Expected Improvement acquisition function with analytical gradient
- F1 objective: minimise J = Cd + (1/3)·|Cl| (drag plus weighted downforce penalty)
- Pareto front computed for drag vs. downforce trade-off across the design space
- Results, GP models, and Pareto front in `results/`

### 6. Multi-Fidelity Correction

- 10 strategically placed DES evaluations: RANS optimum + boundary designs + mid-space samples
- Co-Kriging model: f_HF(x) = ρ·f_LF(x) + δ(x), where δ is a GP correction term
- Provides DES-accurate predictions across the full design space at a fraction of full-DES cost

## Results

[PLACEHOLDER — DoE campaign in progress. Cd/Cl response surfaces, Pareto front, and optimal design parameters will be added on completion.]

## Repository Structure

```
.
├── case_generator.py        # Writes complete OpenFOAM case directories from a design vector
│                            # (blockMeshDict, snappyHexMeshDict, ICs, run.sh)
├── pipeline.py              # Orchestrates the full DoE + Bayesian loop:
│                            # runs cases in parallel, extracts forces, feeds the surrogate
├── post_processor.py        # Parses OpenFOAM forceCoeffs logs → Cd/Cl; writes results CSV
├── surrogate_optimiser.py   # Gaussian Process surrogate, Expected Improvement, Pareto front,
│                            # and Co-Kriging multi-fidelity correction
├── doe_setup.py             # Generates the Latin Hypercube design_matrix.csv
├── generate_ahmed_freecad.py # Headless FreeCAD script — builds the parametric Ahmed body STL
├── design_matrix.csv        # 30-point LHS design matrix (tracked)
├── doe_output/              # Per-run DoE case archives and summary statistics
├── mesh_convergence/        # GCI study: three mesh levels + mesh_convergence.csv
├── sandbox/                 # Validation cases (wall-fn, low-Re, DES)
├── results/                 # GP models, Pareto front, response surfaces, flow visualisations
└── openfoam_cases/          # Live case directories (solver output excluded from git)
```

## Requirements

- **OpenFOAM** via Docker: `opencfd/openfoam-default:latest`
- **Python 3.9+**: `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`
- **FreeCAD** (headless): `/Applications/FreeCAD.app` or equivalent — used only for STL generation

Install Python dependencies:

```bash
pip install numpy scipy pandas scikit-learn matplotlib
```

## Running

**1. Generate the design matrix (if not already present):**

```bash
python doe_setup.py
```

**2. Generate all OpenFOAM case directories:**

```bash
python case_generator.py
```

This writes one case per row of `design_matrix.csv` into `openfoam_cases/`, each with a self-contained `run.sh`.

**3. Run the full pipeline (DoE sweep + Bayesian optimisation loop):**

```bash
python pipeline.py
```

The pipeline runs 2 cases concurrently by default. Results are written to `results/results_summary.csv` as cases complete.

**4. Run a single case manually:**

```bash
bash openfoam_cases/case_000/run.sh
```

**5. Post-process and update the surrogate:**

```bash
python post_processor.py
python surrogate_optimiser.py
```
