"""
mesh_convergence.py
===================
VVUQ Module: Mesh Independence Study for Ahmed body.

Half-domain (Y: 0→1.95, symmetry at y=0), k-ω SST low-Re (y+<1), FreeCAD STL,
snappyHexMesh with explicit eMesh feature snapping.

Mesh levels — fixed 128×12×20 background, varying surface refinement:
  L1 (coarse) : surface level 3  (~400k cells)
  L2 (medium) : surface level 4  (~1.4M cells, validated Cd≈0.315)
  L3 (fine)   : surface level 5  (~5M+ cells)

Usage
-----
  python3 mesh_convergence.py                    # generate + run all levels
  python3 mesh_convergence.py --plot             # plot only (cases already run)
  python3 mesh_convergence.py --level L2_medium  # re-run one level
  python3 mesh_convergence.py --mesh-only        # mesh only, no solver
"""

import sys
import os
import json
import shutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from case_generator import (
    TRANSPORT_PROPERTIES, TURBULENCE_PROPERTIES,
    FREESTREAM_U,
)

# Ahmed body reference dimensions (Lienhart 2002)
L_BODY = 1.044   # m
W_BODY = 0.389   # m
H_BODY = 0.288   # m

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

BASE_PARAMS = {
    "slant_angle": 25.0,    # degrees — Lienhart 2002 high-drag configuration
    "r_nose":      100.0,   # mm — nose fillet radius
}

MESH_LEVELS = [
    {
        "name":       "L1_coarse",
        "label":      "Coarse\n128×12×20\nlevel 3",
        "surf_level": 3,
    },
    {
        "name":       "L2_medium",
        "label":      "Medium\n128×12×20\nlevel 4",
        "surf_level": 4,
    },
    {
        "name":       "L3_fine",
        "label":      "Fine\n128×12×20\nlevel 5",
        "surf_level": 5,
    },
]

OUTPUT_DIR  = Path("mesh_convergence")
DOCKER_IMG  = "opencfd/openfoam-default:latest"
NCORES      = 8
TAIL_FRAC   = 0.20   # fraction of iterations to average for Cd/Cl
FREECADCMD  = "/Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd"
SCRIPT_DIR  = Path(__file__).parent

# Turbulence ICs — nut_ratio=1 matches validated snappy_validation setup (omega≈16000)
_NU = 1.5e-5
_K  = 1.5 * (FREESTREAM_U * 0.01) ** 2   # ≈ 0.24
_NUT = 1.0 * _NU
_OMEGA = _K / _NUT                          # ≈ 16000


# ─── 1. FILE TEMPLATES ────────────────────────────────────────────────────────

def block_mesh_dict():
    return """\
FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
scale 1;
// Half domain: X -5.22→15.66, Y 0→1.95 (symmetry at y=0), Z 0→1.70
// Background cell size ~163mm  (128×12×20 cells, fixed across all levels)
vertices
(
    ( -5.22   0.00  0.00 )
    ( 15.66   0.00  0.00 )
    ( 15.66   1.95  0.00 )
    ( -5.22   1.95  0.00 )
    ( -5.22   0.00  1.70 )
    ( 15.66   0.00  1.70 )
    ( 15.66   1.95  1.70 )
    ( -5.22   1.95  1.70 )
);
blocks
(
    hex (0 1 2 3 4 5 6 7)
    (128 12 20)
    simpleGrading (1 1 1)
);
boundary
(
    inlet      { type patch;    faces ((0 3 7 4)); }
    outlet     { type patch;    faces ((1 2 6 5)); }
    ground     { type wall;     faces ((0 1 2 3)); }
    top        { type slip;     faces ((4 5 6 7)); }
    side_y_neg { type symmetry; faces ((0 1 5 4)); }
    side_y_pos { type slip;     faces ((3 2 6 7)); }
);
"""


def surface_feature_extract_dict():
    return """\
FoamFile { version 2.0; format ascii; class dictionary; object surfaceFeatureExtractDict; }
ahmed_body_hq.stl
{
    extractionMethod extractFromSurface;
    extractFromSurfaceCoeffs { includedAngle 150; }
    writeObj yes;
}
"""


def snappy_hex_mesh_dict(surf_level):
    nb = surf_level              # nearBody refinement
    wb = surf_level - 1         # wakeBox refinement
    sw = surf_level + 1         # slantWake refinement (shear layer capture)
    fe = max(1, surf_level - 2) # feature edge refinement
    return f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }}
castellatedMesh true;
snap            true;
addLayers       true;
mergeTolerance  1e-6;

geometry
{{
    ahmed_body_hq.stl
    {{
        type triSurfaceMesh;
        name ahmed_body;
    }}
    wakeBox   {{ type searchableBox; min (0.0   0.00 0.0);  max (3.60  0.40 0.60); }}
    nearBody  {{ type searchableBox; min (-0.40 0.00 0.0);  max (1.44  0.30 0.50); }}
    slantWake {{ type searchableBox; min (0.65  0.00 0.13); max (1.55  0.25 0.43); }}
    frontFoot {{ type searchableBox; min (0.140 0.050 0.000); max (0.260 0.145 0.075); }}
    rearFoot  {{ type searchableBox; min (0.794 0.050 0.000); max (0.914 0.145 0.075); }}
}}

castellatedMeshControls
{{
    maxLocalCells       8000000;
    maxGlobalCells      16000000;
    minRefinementCells  10;
    nCellsBetweenLevels 3;
    features ( {{ file "ahmed_body_hq.eMesh"; level {fe}; }} );
    refinementSurfaces
    {{
        ahmed_body {{ level ({surf_level} {surf_level}); patchInfo {{ type wall; }} }}
    }}
    refinementRegions
    {{
        nearBody  {{ mode inside; levels ((1E15 {nb})); }}
        wakeBox   {{ mode inside; levels ((1E15 {wb})); }}
        slantWake {{ mode inside; levels ((1E15 {sw})); }}
        frontFoot {{ mode inside; levels ((1E15 6)); }}
        rearFoot  {{ mode inside; levels ((1E15 6)); }}
    }}
    resolveFeatureAngle     30;
    locationInMesh          (5.0 0.15 0.5);
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 5; tolerance 2.0; nSolveIter 50; nRelaxIter 8;
    implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes        false;
    firstLayerThickness  9.3e-6;
    expansionRatio       1.2;
    minThickness         1e-7;
    layers
    {{
        ahmed_body {{ nSurfaceLayers 15; }}
        ground     {{ nSurfaceLayers 8; }}
    }}
    nGrow 0; featureAngle 100; nRelaxIter 5;
    nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
    maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.6;
    minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; nRelaxedIter 20;
}}

meshQualityControls
{{
    maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4;
    maxConcave 80; minVol 1e-13; minTetQuality 1e-30; minArea -1;
    minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.05;
    minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
    relaxed {{ maxNonOrtho 75; minFaceWeight 0.05; }}
}}
"""


def control_dict():
    return """\
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1800;
deltaT          1;
writeControl    timeStep;
writeInterval   500;
purgeWrite      2;
writeFormat     binary;
writePrecision  8;
runTimeModifiable true;
functions
{
    forceCoeffs
    {
        type            forceCoeffs;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   10;
        patches         (ahmed_body);
        rho             rhoInf;
        rhoInf          1.225;
        magUInf         40.0;
        lRef            1.044;
        Aref            0.056;
        coordinateSystem
        {
            origin  (0.522 0 0);
            e1      (1 0 0);
            e2      (0 1 0);
        }
    }
}
"""


FV_SCHEMES = """\
FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes      { default steadyState; }
gradSchemes     { default cellLimited leastSquares 1; grad(p) cellLimited leastSquares 0.5; }
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div(div(phi,U)) Gauss linear;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear limited 1; }
interpolationSchemes { default linear; }
snGradSchemes   { default limited 1; }
wallDist        { method meshWave; }
"""


FV_SOLUTION = """\
FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers
{
    p
    {
        solver          PCG;
        preconditioner
        {
            preconditioner  GAMG;
            tolerance       1e-8;
            relTol          0.0;
            nVcycles        2;
            smoother        GaussSeidel;
            nPreSweeps      0;
            nPostSweeps     2;
            cacheAgglomeration false;
            nCellsInCoarsestLevel 10;
            agglomerator    faceAreaPair;
            mergeLevels     1;
        }
        tolerance       1e-6;
        relTol          0.01;
        minIter         3;
    }
    Phi { $p; }
    "(U|k|omega)"
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
    }
}
SIMPLE
{
    nNonOrthogonalCorrectors 2;
    consistent      yes;
    residualControl { p 1e-5; U 1e-5; "(k|omega)" 1e-4; }
}
potentialFlow { nNonOrthogonalCorrectors 4; }
relaxationFactors { equations { ".*" 0.7; } }
cache { grad(U); }
"""


def _field_files(k=_K, omega=_OMEGA):
    """Uniform initial fields matching validated snappy_validation BCs."""
    U_file = f"""\
FoamFile {{ version 2.0; format ascii; class volVectorField; object U; }}
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform ({FREESTREAM_U} 0 0);
boundaryField
{{
    inlet       {{ type fixedValue; value uniform ({FREESTREAM_U} 0 0); }}
    outlet      {{ type inletOutlet; inletValue uniform (0 0 0); value uniform (0 0 0); }}
    ground      {{ type movingWallVelocity; value uniform ({FREESTREAM_U} 0 0); }}
    top         {{ type slip; }}
    side_y_neg  {{ type symmetry; }}
    side_y_pos  {{ type slip; }}
    ahmed_body  {{ type fixedValue; value uniform (0 0 0); }}
}}
"""
    p_file = """\
FoamFile { version 2.0; format ascii; class volScalarField; object p; }
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    inlet       { type zeroGradient; }
    outlet      { type fixedValue; value uniform 0; }
    ground      { type zeroGradient; }
    top         { type slip; }
    side_y_neg  { type symmetry; }
    side_y_pos  { type slip; }
    ahmed_body  { type zeroGradient; }
}
"""
    k_file = f"""\
FoamFile {{ version 2.0; format ascii; class volScalarField; object k; }}
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform {k:.6f};
boundaryField
{{
    inlet       {{ type fixedValue; value uniform {k:.6f}; }}
    outlet      {{ type inletOutlet; inletValue uniform {k:.6f}; value uniform {k:.6f}; }}
    ground      {{ type fixedValue; value uniform 0; }}
    top         {{ type zeroGradient; }}
    side_y_neg  {{ type symmetry; }}
    side_y_pos  {{ type zeroGradient; }}
    ahmed_body  {{ type fixedValue; value uniform 0; }}
}}
"""
    omega_file = f"""\
FoamFile {{ version 2.0; format ascii; class volScalarField; object omega; }}
dimensions      [0 0 -1 0 0 0 0];
internalField   uniform {omega:.1f};
boundaryField
{{
    inlet       {{ type fixedValue; value uniform {omega:.1f}; }}
    outlet      {{ type inletOutlet; inletValue uniform {omega:.1f}; value uniform {omega:.1f}; }}
    ground      {{ type omegaWallFunction; value uniform 100000; }}
    top         {{ type zeroGradient; }}
    side_y_neg  {{ type symmetry; }}
    side_y_pos  {{ type zeroGradient; }}
    ahmed_body  {{ type omegaWallFunction; value uniform 100000; }}
}}
"""
    nut_file = """\
FoamFile { version 2.0; format ascii; class volScalarField; object nut; }
dimensions      [0 2 -1 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    inlet       { type calculated; value uniform 0; }
    outlet      { type calculated; value uniform 0; }
    ground      { type nutLowReWallFunction; value uniform 0; }
    top         { type slip; }
    side_y_neg  { type symmetry; }
    side_y_pos  { type slip; }
    ahmed_body  { type nutLowReWallFunction; value uniform 0; }
}
"""
    return {"U": U_file, "p": p_file, "k": k_file, "omega": omega_file, "nut": nut_file}


# ─── 2. CASE GENERATION ───────────────────────────────────────────────────────

def generate_stl(case_dir):
    """Generate ahmed_body_hq.stl via FreeCAD headless."""
    stl_out = Path(case_dir) / "constant" / "triSurface" / "ahmed_body_hq.stl"
    env = os.environ.copy()
    env["AHMED_SLANT_ANGLE"] = str(BASE_PARAMS["slant_angle"])
    env["AHMED_R_NOSE"]      = str(BASE_PARAMS["r_nose"])
    env["AHMED_OUT"]         = str(stl_out)
    result = subprocess.run(
        [FREECADCMD, str(SCRIPT_DIR / "generate_ahmed_freecad.py")],
        env=env, capture_output=True, text=True
    )
    if not stl_out.exists():
        print(f"    STL generation failed:\n{result.stdout}\n{result.stderr}")
        return False
    return True


def generate_case(level, case_dir):
    case_dir = Path(case_dir)
    if case_dir.exists():
        shutil.rmtree(case_dir)

    (case_dir / "0").mkdir(parents=True)
    (case_dir / "constant" / "triSurface").mkdir(parents=True)
    (case_dir / "system").mkdir(parents=True)

    print(f"    Generating STL via FreeCAD...")
    if not generate_stl(case_dir):
        raise RuntimeError("STL generation failed")

    # Write initial fields (overwritten after snappy, but needed for patch recognition)
    for fname, content in _field_files().items():
        (case_dir / "0" / fname).write_text(content)

    # Write field_init dir (uniform fields, mounted read-only into Docker)
    field_init = case_dir / "_field_init"
    field_init.mkdir(exist_ok=True)
    for fname, content in _field_files().items():
        (field_init / fname).write_text(content)

    (case_dir / "constant" / "transportProperties").write_text(TRANSPORT_PROPERTIES)
    (case_dir / "constant" / "turbulenceProperties").write_text(TURBULENCE_PROPERTIES)
    (case_dir / "system" / "blockMeshDict").write_text(block_mesh_dict())
    (case_dir / "system" / "surfaceFeatureExtractDict").write_text(surface_feature_extract_dict())
    (case_dir / "system" / "snappyHexMeshDict").write_text(snappy_hex_mesh_dict(level["surf_level"]))
    (case_dir / "system" / "controlDict").write_text(control_dict())
    (case_dir / "system" / "fvSchemes").write_text(FV_SCHEMES)
    (case_dir / "system" / "fvSolution").write_text(FV_SOLUTION)
    (case_dir / "system" / "decomposeParDict").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }\n"
        f"numberOfSubdomains {NCORES};\nmethod scotch;\n"
    )


# ─── 3. DOCKER RUNNER ─────────────────────────────────────────────────────────

def _docker_bash(case_dir, script, extra_mounts=None):
    case_dir = Path(case_dir).resolve()
    mounts = ["-v", f"{case_dir}:/case"]
    if extra_mounts:
        for src, dst in extra_mounts:
            mounts += ["-v", f"{src}:{dst}"]
    cmd = ["docker", "run", "--rm"] + mounts + [DOCKER_IMG, "bash", "-c", script]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr


def run_mesh_docker(case_dir):
    """blockMesh + surfaceFeatureExtract + snappyHexMesh only."""
    script = (
        "source /openfoam/bash.rc && cd /case && set -euo pipefail && "
        "blockMesh > log.blockMesh 2>&1 && "
        "surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1 && "
        "snappyHexMesh -overwrite > log.snappyHexMesh 2>&1"
    )
    ok, _, err = _docker_bash(case_dir, script)
    if not ok:
        print(f"    Meshing failed: {err[:200]}")
    return ok


def run_case_docker(case_dir):
    """Full pipeline: mesh → potentialFoam → decomposePar → simpleFoam → reconstructPar."""
    field_init = Path(case_dir).resolve() / "_field_init"
    script = (
        "source /openfoam/bash.rc && cd /case && set -euo pipefail && "
        # cleanup
        "rm -rf processor[0-9]* constant/polyMesh postProcessing 2>/dev/null || true && "
        "for d in [1-9] [1-9][0-9] [1-9][0-9][0-9] [1-9][0-9][0-9][0-9]; do rm -rf \"$d\" 2>/dev/null || true; done && "
        # mesh
        "blockMesh > log.blockMesh 2>&1 && "
        "surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1 && "
        "snappyHexMesh -overwrite > log.snappyHexMesh 2>&1 && "
        "checkMesh > log.checkMesh 2>&1 && "
        # restore uniform fields after snappy (snappy may write non-uniform data to 0/)
        "cp /field_init/U 0/U && cp /field_init/p 0/p && "
        "cp /field_init/k 0/k && cp /field_init/omega 0/omega && cp /field_init/nut 0/nut && "
        # solve
        "potentialFoam -initialiseUBCs > log.potentialFoam 2>&1 && "
        f"decomposePar -force > log.decomposePar 2>&1 && "
        f"mpirun --allow-run-as-root -np {NCORES} simpleFoam -parallel > log.simpleFoam 2>&1 && "
        "reconstructPar -latestTime > log.reconstructPar 2>&1 || true"
    )
    ok, _, err = _docker_bash(
        case_dir, script,
        extra_mounts=[(str(field_init), "/field_init:ro")]
    )
    if not ok:
        print(f"    Run failed: {err[:200]}")
    return ok


# ─── 4. POST-PROCESSING ───────────────────────────────────────────────────────

def snappy_cell_count(case_dir):
    log = Path(case_dir) / "log.checkMesh"
    if not log.exists():
        log = Path(case_dir) / "log.snappyHexMesh"
    if not log.exists():
        return None
    for line in reversed(log.read_text().splitlines()):
        if "cells:" in line:
            try:
                return int(line.split(":")[1].strip().split()[0].replace(",", ""))
            except (ValueError, IndexError):
                pass
    return None


def extract_cd_cl(case_dir):
    base = Path(case_dir) / "postProcessing" / "forceCoeffs"
    if not base.exists():
        return None, None
    dat = None
    for td in sorted(base.iterdir()):
        for name in ("coefficient.dat", "forceCoeffs.dat"):
            p = td / name
            if p.exists():
                dat = p
                break
    if dat is None:
        return None, None

    rows = []
    with open(dat) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            try:
                rows.append([float(x) for x in line.split()])
            except ValueError:
                continue

    if not rows:
        return None, None

    arr  = np.array(rows)
    tail = arr[max(0, int(len(arr) * (1 - TAIL_FRAC))):]
    return float(np.mean(tail[:, 1])), float(np.mean(tail[:, 4]))


def check_mesh_quality(case_dir):
    log = Path(case_dir) / "log.checkMesh"
    metrics = {"max_non_ortho": None, "max_skewness": None, "cells": None}
    if not log.exists():
        return metrics
    for line in log.read_text().splitlines():
        if "Mesh non-orthogonality Max:" in line:
            try:
                metrics["max_non_ortho"] = float(line.split("Max:")[1].split()[0])
            except (IndexError, ValueError):
                pass
        if "Max skewness" in line:
            try:
                metrics["max_skewness"] = float(line.split("=")[1].split(",")[0])
            except (IndexError, ValueError):
                pass
        if "cells:" in line and metrics["cells"] is None:
            try:
                metrics["cells"] = int(line.split(":")[1].strip())
            except (IndexError, ValueError):
                pass
    return metrics


# ─── 5. PLOTTING ──────────────────────────────────────────────────────────────

def richardson_extrapolation(cd_values, cell_counts):
    if len(cd_values) < 3 or any(c is None for c in cd_values):
        return None, None, None

    pairs = sorted(zip(cell_counts, cd_values))
    n1, f1 = pairs[2]
    n2, f2 = pairs[1]
    n3, f3 = pairs[0]

    r21 = (n1 / n2) ** (1/3)
    r32 = (n2 / n3) ** (1/3)
    e21 = f1 - f2
    e32 = f2 - f3

    if abs(e32) < 1e-10 or abs(e21) < 1e-10:
        return f1, None, None

    try:
        p = abs(np.log(abs(e32 / e21)) / np.log(r21))
    except (ValueError, ZeroDivisionError):
        return f1, None, None

    cd_extrap = f1 + (f1 - f2) / (r21 ** p - 1)
    GCI_fine  = 1.25 * abs(e21 / f1) / (r21 ** p - 1)
    return cd_extrap, round(p, 2), round(GCI_fine * 100, 2)


def plot_convergence(results, output_path):
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax_cd   = fig.add_subplot(gs[0, 0])
    ax_cl   = fig.add_subplot(gs[0, 1])
    ax_qual = fig.add_subplot(gs[1, 0])
    ax_conv = fig.add_subplot(gs[1, 1])

    cells  = [r["cells"] for r in results]
    cds    = [r["Cd"]    for r in results]
    cls    = [r["Cl"]    for r in results]
    nonort = [r["max_non_ortho"] for r in results]
    skew   = [r["max_skewness"]  for r in results]
    labels = [r["label"] for r in results]
    x      = range(len(results))

    valid_cd = [(c, cd) for c, cd in zip(cells, cds) if cd is not None and c is not None]
    if valid_cd:
        vc, vcd = zip(*valid_cd)
        ax_cd.plot(vc, vcd, "o-", color="#1A6FAF", linewidth=2, markersize=8)
        ax_cd.axhline(0.285, color="red",   linestyle="--", linewidth=1.5,
                      label="Experiment (Lienhart 2002)")
        ax_cd.axhline(0.310, color="orange", linestyle=":",  linewidth=1.5,
                      label="WolfDynamics validation")
        cd_extrap, p_order, gci = richardson_extrapolation(
            [r["Cd"] for r in results], [r["cells"] for r in results])
        if cd_extrap:
            ax_cd.axhline(cd_extrap, color="green", linestyle=":", linewidth=1.5,
                          label=f"Richardson extrap: {cd_extrap:.3f}  (p={p_order}, GCI={gci}%)")
        ax_cd.set_xlabel("Cell count")
        ax_cd.set_ylabel("Cd")
        ax_cd.set_title("Drag coefficient vs mesh density")
        ax_cd.legend(fontsize=7)
        ax_cd.grid(True, alpha=0.4)
        ax_cd.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

    valid_cl = [(c, cl) for c, cl in zip(cells, cls) if cl is not None and c is not None]
    if valid_cl:
        vc, vcl = zip(*valid_cl)
        ax_cl.plot(vc, vcl, "s-", color="#E8A020", linewidth=2, markersize=8)
        ax_cl.set_xlabel("Cell count")
        ax_cl.set_ylabel("Cl")
        ax_cl.set_title("Lift coefficient vs mesh density")
        ax_cl.grid(True, alpha=0.4)
        ax_cl.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

    xi = list(x)
    ax_qual.bar([i - 0.2 for i in xi],
                [v if v else 0 for v in nonort], 0.35,
                color="#1A6FAF", alpha=0.8, label="Max non-ortho (°)")
    ax_qual_r = ax_qual.twinx()
    ax_qual_r.bar([i + 0.2 for i in xi],
                  [v if v else 0 for v in skew], 0.35,
                  color="#E85020", alpha=0.8, label="Max skewness")
    ax_qual.axhline(70, color="#1A6FAF", linestyle="--", alpha=0.5, linewidth=1)
    ax_qual_r.axhline(4,  color="#E85020", linestyle="--", alpha=0.5, linewidth=1)
    ax_qual.set_xticks(xi); ax_qual.set_xticklabels(labels, fontsize=8)
    ax_qual.set_ylabel("Max non-orthogonality (°)", color="#1A6FAF")
    ax_qual_r.set_ylabel("Max skewness", color="#E85020")
    ax_qual.set_title("Mesh quality metrics")
    lines1, labels1 = ax_qual.get_legend_handles_labels()
    lines2, labels2 = ax_qual_r.get_legend_handles_labels()
    ax_qual.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    colors = ["#AAAAAA", "#1A6FAF", "#003070"]
    for i, r in enumerate(results):
        dat_path = OUTPUT_DIR / r["name"] / "postProcessing" / "forceCoeffs" / "0"
        for fname in ("coefficient.dat", "forceCoeffs.dat"):
            p = dat_path / fname
            if p.exists():
                rows = []
                with open(p) as f:
                    for line in f:
                        if line.startswith("#") or not line.strip():
                            continue
                        try:
                            rows.append([float(v) for v in line.split()])
                        except ValueError:
                            pass
                if rows:
                    arr = np.array(rows)
                    ax_conv.plot(arr[:, 0], arr[:, 1],
                                 color=colors[i % len(colors)], linewidth=1.2,
                                 label=r["name"].replace("_", " "))
                break
    ax_conv.axhline(0.285, color="red", linestyle="--", linewidth=1,
                    label="Experiment ≈ 0.285")
    ax_conv.axhline(0.310, color="orange", linestyle=":", linewidth=1,
                    label="WolfDynamics ≈ 0.310")
    ax_conv.set_xlabel("Iteration")
    ax_conv.set_ylabel("Cd")
    ax_conv.set_title("Cd convergence history by mesh level")
    ax_conv.legend(fontsize=7)
    ax_conv.grid(True, alpha=0.4)

    fig.suptitle(
        "Ahmed Body Mesh Independence Study\n"
        f"slant={BASE_PARAMS['slant_angle']:.0f}°  R_nose={BASE_PARAMS['r_nose']:.0f}mm  "
        f"U∞={FREESTREAM_U:.0f} m/s  half-domain k-ω SST low-Re (y+<1, 15 BL layers)",
        fontsize=12, fontweight="bold"
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved → {output_path}")


# ─── 6. RESULTS TABLE ─────────────────────────────────────────────────────────

def print_results_table(results):
    print("\n" + "=" * 75)
    print(f"{'Level':<15} {'Cells':>10} {'Cd':>10} {'Cl':>10} "
          f"{'Non-ortho':>10} {'Skewness':>10}")
    print("-" * 75)
    for r in results:
        cd = f"{r['Cd']:.4f}"         if r["Cd"]           else "—"
        cl = f"{r['Cl']:.4f}"         if r["Cl"]           else "—"
        no = f"{r['max_non_ortho']:.1f}°" if r["max_non_ortho"] else "—"
        sk = f"{r['max_skewness']:.2f}"   if r["max_skewness"]  else "—"
        nc = f"{r['cells']:,}"         if r["cells"]        else "—"
        print(f"{r['name']:<15} {nc:>10} {cd:>10} {cl:>10} {no:>10} {sk:>10}")
    print("=" * 75)

    cd_values = [r["Cd"]    for r in results]
    cells     = [r["cells"] for r in results]
    cd_extrap, p_order, gci = richardson_extrapolation(cd_values, cells)
    if cd_extrap:
        print(f"\nRichardson extrapolated Cd : {cd_extrap:.4f}")
        print(f"Observed order of accuracy : {p_order}")
        print(f"GCI (fine grid)            : {gci}%")
        print(f"WolfDynamics reference     : 0.310")
        print(f"Experimental reference     : ~0.285 (Lienhart 2002)")


# ─── 7. MAIN ──────────────────────────────────────────────────────────────────

def _run_level(level, plot_only, mesh_only=False):
    case_dir = OUTPUT_DIR / level["name"]

    if not plot_only:
        print(f"[{level['name']}] generating case...")
        generate_case(level, case_dir)

        if mesh_only:
            print(f"[{level['name']}] meshing...")
            success = run_mesh_docker(case_dir)
        else:
            print(f"[{level['name']}] running OpenFOAM pipeline...")
            success = run_case_docker(case_dir)
        if not success:
            print(f"[{level['name']}] WARNING: pipeline failed")

    quality = check_mesh_quality(case_dir)
    cd, cl  = (None, None) if mesh_only else extract_cd_cl(case_dir)
    if quality["cells"] is None:
        quality["cells"] = snappy_cell_count(case_dir)

    r = {
        "name":          level["name"],
        "label":         level["label"],
        "cells":         quality["cells"],
        "Cd":            round(cd, 6) if cd is not None else None,
        "Cl":            round(cl, 6) if cl is not None else None,
        "max_non_ortho": quality["max_non_ortho"],
        "max_skewness":  quality["max_skewness"],
    }
    print(f"[{level['name']}] Cd={r['Cd']}  Cl={r['Cl']}  cells={r['cells']}  "
          f"non-ortho={r['max_non_ortho']}  skewness={r['max_skewness']}")
    with open(OUTPUT_DIR / f"{level['name']}_result.json", "w") as f:
        json.dump(r, f, indent=2)
    return r


def main():
    plot_only = "--plot"      in sys.argv
    mesh_only = "--mesh-only" in sys.argv
    level_arg = next((sys.argv[i+1] for i, a in enumerate(sys.argv)
                      if a == "--level" and i+1 < len(sys.argv)), None)
    OUTPUT_DIR.mkdir(exist_ok=True)

    if level_arg:
        target = next((l for l in MESH_LEVELS if l["name"] == level_arg), None)
        if target is None:
            print(f"Unknown level '{level_arg}'. Choose from: "
                  f"{[l['name'] for l in MESH_LEVELS]}")
            sys.exit(1)
        _run_level(target, plot_only=False, mesh_only=mesh_only)
    else:
        mode = "mesh-only" if mesh_only else "full"
        print(f"Running {len(MESH_LEVELS)} mesh levels sequentially [{mode}]...")
        print(f"Start: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}\n")
        for level in MESH_LEVELS:
            try:
                _run_level(level, plot_only, mesh_only=mesh_only)
            except Exception as exc:
                print(f"[{level['name']}] ERROR: {exc}")

    results = []
    for level in MESH_LEVELS:
        json_path = OUTPUT_DIR / f"{level['name']}_result.json"
        if json_path.exists():
            with open(json_path) as f:
                results.append(json.load(f))

    print(f"\nEnd: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}")
    if results:
        print_results_table(results)
        plot_convergence(results, OUTPUT_DIR / "mesh_convergence.png")
        pd.DataFrame(results).to_csv(OUTPUT_DIR / "mesh_convergence.csv", index=False)
        print(f"\nAll results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
