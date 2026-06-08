"""
case_generator.py
=================
Reads design_matrix.csv and writes one OpenFOAM case per row using the
validated snappy_validation model (FreeCAD STL, half-domain, L2 mesh).

Solver:  simpleFoam (steady incompressible RANS)
Turb:    k-omega Wilcox
Speed:   40 m/s
Meshing: blockMesh + surfaceFeatureExtract + snappyHexMesh (L2: surface level 4)
Domain:  half-domain, Y 0→1.95 m, symmetry at y=0
Aref:    0.056 m² (half frontal area)

Dependencies: numpy, pandas
"""

import subprocess, os
import numpy as np
import pandas as pd
from pathlib import Path

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

DESIGN_MATRIX = Path("design_matrix.csv")
CASES_DIR     = Path("openfoam_cases")
FREECAD_CMD   = "/Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd"
AHMED_SCRIPT  = str(Path(__file__).parent / "generate_ahmed_freecad.py")
DOCKER_IMAGE  = "opencfd/openfoam-default:latest"

FREESTREAM_U = 40.0
NU           = 1.5e-5
RHO          = 1.225

# Validated k-omega Wilcox ICs (nut_ratio=1.0, 1% intensity @ 40 m/s)
_K     = 0.24
_OMEGA = 16000.0
_NUT   = 1.5e-5


# ─── 1. STL GENERATION (FreeCAD headless) ─────────────────────────────────────

def generate_stl(params, case_dir: Path) -> Path:
    stl_path = case_dir / "constant" / "triSurface" / "ahmed_body.stl"
    stl_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "AHMED_SLANT_ANGLE":    str(params["slant_angle"]),
        "AHMED_R_NOSE":         str(params["front_radius"]),
        "AHMED_DIFFUSER_ANGLE": str(params["diffuser_angle"]),
        "AHMED_h":              str(params["ride_height"]),
        "AHMED_OUT":            str(stl_path.resolve()),
        "_AHMED_RUNNING":       "",   # reset guard for fresh run
    })
    result = subprocess.run(
        [FREECAD_CMD, AHMED_SCRIPT],
        env=env, capture_output=True, text=True
    )
    if result.returncode != 0 or not stl_path.exists():
        raise RuntimeError(f"FreeCAD STL generation failed:\n{result.stdout}\n{result.stderr}")
    return stl_path


# ─── 2. OPENFOAM FILE TEMPLATES ───────────────────────────────────────────────

def block_mesh_dict():
    return """\
FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
scale 1;
vertices
(
    ( -5.220  0.00  0.000 )
    ( 15.660  0.00  0.000 )
    ( 15.660  1.95  0.000 )
    ( -5.220  1.95  0.000 )
    ( -5.220  0.00  2.880 )
    ( 15.660  0.00  2.880 )
    ( 15.660  1.95  2.880 )
    ( -5.220  1.95  2.880 )
);
blocks
(
    hex (0 1 2 3 4 5 6 7) (128 12 20) simpleGrading (1 1 1)
);
boundary
(
    inlet      { type patch;     faces ((0 3 7 4)); }
    outlet     { type patch;     faces ((1 2 6 5)); }
    top        { type symmetry;  faces ((4 7 6 5)); }
    ground     { type wall;      faces ((0 1 2 3)); }
    side_y_neg { type symmetry;  faces ((0 1 5 4)); }
    side_y_pos { type symmetry;  faces ((3 7 6 2)); }
);
"""


def surface_feature_extract_dict():
    return """\
FoamFile { version 2.0; format ascii; class dictionary; object surfaceFeatureExtractDict; }
ahmed_body.stl
{
    extractionMethod    extractFromSurface;
    extractFromSurfaceCoeffs { includedAngle 130; }
    writeObj yes;
}
"""


def snappy_hex_mesh_dict():
    return """\
FoamFile { version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true;
snap            true;
addLayers       true;
mergeTolerance  1e-6;

geometry
{
    ahmed_body.stl { type triSurfaceMesh; name ahmed_body; }
    wakeBox   { type searchableBox; min ( 0.00 0.00 0.00); max (3.60 0.40 0.60); }
    nearBody  { type searchableBox; min (-0.40 0.00 0.00); max (1.44 0.30 0.50); }
    slantWake { type searchableBox; min ( 0.65 0.00 0.13); max (1.55 0.25 0.43); }
    frontFoot { type searchableBox; min (0.140 0.050 0.000); max (0.260 0.145 0.075); }
    rearFoot  { type searchableBox; min (0.794 0.050 0.000); max (0.914 0.145 0.075); }
}

castellatedMeshControls
{
    maxLocalCells        2000000;
    maxGlobalCells       6000000;
    minRefinementCells   10;
    nCellsBetweenLevels  3;
    resolveFeatureAngle  130;
    locationInMesh       (5.0 0.15 0.5);
    allowFreeStandingZoneFaces true;

    features
    (
        { file "ahmed_body.eMesh"; level 2; }
    );

    refinementSurfaces
    {
        ahmed_body { level (4 4); patchInfo { type wall; } }
    }

    refinementRegions
    {
        wakeBox   { mode inside; levels ((1E15 3)); }
        nearBody  { mode inside; levels ((1E15 4)); }
        slantWake { mode inside; levels ((1E15 5)); }
        frontFoot { mode inside; levels ((1E15 6)); }
        rearFoot  { mode inside; levels ((1E15 6)); }
    }
}

snapControls
{
    nSmoothPatch        3;
    tolerance           2.0;
    nSolveIter          30;
    nRelaxIter          5;
    nFeatureSnapIter    10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}

addLayersControls
{
    relativeSizes false;
    layers
    {
        ahmed_body { nSurfaceLayers 15; }
        ground     { nSurfaceLayers 8; }
    }
    expansionRatio            1.2;
    firstLayerThickness       9.3e-6;
    minThickness              1e-7;
    nGrow                     0;
    featureAngle              100;
    nRelaxIter                5;
    nSmoothSurfaceNormals     1;
    nSmoothNormals            3;
    nSmoothThickness          10;
    maxFaceThicknessRatio     0.5;
    maxThicknessToMedialRatio 0.6;
    minMedialAxisAngle        90;
    nBufferCellsNoExtrude     0;
    nLayerIter                50;
}

meshQualityControls
{
    maxNonOrtho          75;
    maxBoundarySkewness  20;
    maxInternalSkewness  4;
    maxConcave           80;
    minVol               1e-13;
    minTetQuality        1e-30;
    minArea              -1;
    minTwist             0.02;
    minDeterminant       0.001;
    minFaceWeight        0.05;
    minVolRatio          0.01;
    minTriangleTwist     -1;
    nSmoothScale         4;
    errorReduction       0.75;
    relaxed { maxNonOrtho 75; minFaceWeight 0.02; }
}
"""


def initial_conditions():
    u = f"""\
FoamFile {{ version 2.0; format ascii; class volVectorField; object U; }}
dimensions [0 1 -1 0 0 0 0];
internalField uniform ({FREESTREAM_U} 0 0);
boundaryField
{{
    inlet      {{ type fixedValue;        value uniform ({FREESTREAM_U} 0 0); }}
    outlet     {{ type inletOutlet;       inletValue uniform (0 0 0); value uniform ({FREESTREAM_U} 0 0); }}
    ground     {{ type movingWallVelocity; value uniform ({FREESTREAM_U} 0 0); }}
    top        {{ type symmetry; }}
    side_y_neg {{ type symmetry; }}
    side_y_pos {{ type symmetry; }}
    ahmed_body {{ type noSlip; }}
}}
"""
    p = """\
FoamFile { version 2.0; format ascii; class volScalarField; object p; }
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet      { type zeroGradient; }
    outlet     { type fixedValue; value uniform 0; }
    ground     { type zeroGradient; }
    top        { type symmetry; }
    side_y_neg { type symmetry; }
    side_y_pos { type symmetry; }
    ahmed_body { type zeroGradient; }
}
"""
    k = f"""\
FoamFile {{ version 2.0; format ascii; class volScalarField; object k; }}
dimensions [0 2 -2 0 0 0 0];
internalField uniform {_K};
boundaryField
{{
    inlet      {{ type fixedValue;      value uniform {_K}; }}
    outlet     {{ type zeroGradient; }}
    ground     {{ type fixedValue;      value uniform 0; }}
    top        {{ type symmetry; }}
    side_y_neg {{ type symmetry; }}
    side_y_pos {{ type symmetry; }}
    ahmed_body {{ type fixedValue;      value uniform 0; }}
}}
"""
    omega = f"""\
FoamFile {{ version 2.0; format ascii; class volScalarField; object omega; }}
dimensions [0 0 -1 0 0 0 0];
internalField uniform {_OMEGA};
boundaryField
{{
    inlet      {{ type fixedValue;        value uniform {_OMEGA}; }}
    outlet     {{ type zeroGradient; }}
    ground     {{ type omegaWallFunction; value uniform {_OMEGA}; }}
    top        {{ type symmetry; }}
    side_y_neg {{ type symmetry; }}
    side_y_pos {{ type symmetry; }}
    ahmed_body {{ type omegaWallFunction; value uniform {_OMEGA}; }}
}}
"""
    nut = f"""\
FoamFile {{ version 2.0; format ascii; class volScalarField; object nut; }}
dimensions [0 2 -1 0 0 0 0];
internalField uniform {_NUT:.2e};
boundaryField
{{
    inlet      {{ type calculated;         value uniform {_NUT:.2e}; }}
    outlet     {{ type calculated;         value uniform {_NUT:.2e}; }}
    ground     {{ type nutLowReWallFunction; value uniform 0; }}
    top        {{ type symmetry; }}
    side_y_neg {{ type symmetry; }}
    side_y_pos {{ type symmetry; }}
    ahmed_body {{ type nutLowReWallFunction; value uniform 0; }}
}}
"""
    return {"U": u, "p": p, "k": k, "omega": omega, "nut": nut}


TRANSPORT_PROPERTIES = f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object transportProperties; }}
transportModel Newtonian;
nu {NU};
"""

TURBULENCE_PROPERTIES = """\
FoamFile { version 2.0; format ascii; class dictionary; object turbulenceProperties; }
simulationType RAS;
RAS { RASModel kOmegaSST; turbulence on; printCoeffs on; }
"""

def control_dict(end_time: int = 1500) -> str:
    return f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object controlDict; }}
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time};
deltaT          1;
writeControl    timeStep;
writeInterval   500;
purgeWrite      2;
writeFormat     binary;
writePrecision  8;
runTimeModifiable true;
functions
{{
    forceCoeffs
    {{
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
        {{
            origin  (0.522 0 0);
            e1      (1 0 0);
            e2      (0 1 0);
        }}
    }}
}}
"""

# Keep a default constant for the DoE (unchanged behaviour)
CONTROL_DICT = control_dict(end_time=1500)

FV_SCHEMES = """\
FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; grad(U) cellLimited Gauss linear 1; }
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwindV grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
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


# ─── 3. RUN SCRIPT ────────────────────────────────────────────────────────────

def decompose_par_dict(n_cores: int) -> str:
    """
    decomposeParDict for scotch decomposition.
    Scotch requires no manual subdomain geometry — it balances load automatically.
    """
    return f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; location "system"; object decomposeParDict; }}
numberOfSubdomains  {n_cores};
method              scotch;
"""


def run_script(case_id, case_abs_path: str, n_cores: int = 1) -> str:
    """
    Generate the Docker run script for a case.

    n_cores=1  (default) — serial run, used for the DoE batch (4 parallel cases)
    n_cores>1            — parallel run: decomposePar + mpirun -np N + reconstructPar
                           Used for single Bayesian loop cases to exploit all cores.

    blockMesh, surfaceFeatureExtract, and snappyHexMesh always run serially;
    only potentialFoam and simpleFoam are parallelised.
    """
    if n_cores > 1:
        solver_block = f"""\
        decomposePar -force > log.decomposePar 2>&1
        mpirun -np {n_cores} --allow-run-as-root potentialFoam -parallel -initialiseUBCs > log.potentialFoam 2>&1 || true
        mpirun -np {n_cores} --allow-run-as-root simpleFoam    -parallel                > log.simpleFoam 2>&1
        reconstructPar -latestTime > log.reconstructPar 2>&1
        reconstructPar -withZero   >> log.reconstructPar 2>&1 || true"""
        cores_flag = f"--cpus={n_cores}"
    else:
        solver_block = """\
        potentialFoam -initialiseUBCs > log.potentialFoam 2>&1 || true
        simpleFoam > log.simpleFoam 2>&1"""
        cores_flag = ""

    return f"""#!/bin/bash
# Auto-generated Docker run script for {case_id}  (n_cores={n_cores})
set -e
CASE_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="{DOCKER_IMAGE}"

echo "=== {case_id}: meshing ==="

mkdir -p "$CASE_DIR/_field_init"
for f in U p k omega nut; do
    cp "$CASE_DIR/0/$f" "$CASE_DIR/_field_init/$f"
done

docker run --rm {cores_flag} \\
    -v "$CASE_DIR:/case" \\
    -v "$CASE_DIR/_field_init:/field_init:ro" \\
    "$IMAGE" \\
    bash -c "
        source /openfoam/bash.rc
        cd /case
        set -euo pipefail

        rm -rf processor[0-9]* constant/polyMesh postProcessing 2>/dev/null || true
        for d in [1-9] [1-9][0-9] [1-9][0-9][0-9] [1-9][0-9][0-9][0-9]; do
            rm -rf \\\"\\$d\\\" 2>/dev/null || true
        done

        blockMesh             > log.blockMesh 2>&1
        surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1
        snappyHexMesh -overwrite > log.snappyHexMesh 2>&1
        mkdir -p 0

        cp /field_init/U     0/U
        cp /field_init/p     0/p
        cp /field_init/k     0/k
        cp /field_init/omega 0/omega
        cp /field_init/nut   0/nut

{solver_block}
    "

echo "Done: {case_id}"
"""


# ─── 4. CASE DIRECTORY BUILDER ────────────────────────────────────────────────

def write_case(case_id: str, params: dict, base_dir: Path, n_cores: int = 1, n_iters: int = 3000) -> Path:
    """
    Write a complete OpenFOAM case directory for one design point.

    n_cores=1  (default) — serial; use for batched DoE runs (2 cases in parallel)
    n_cores=10           — parallel; use for single Bayesian loop evaluations
    n_iters=3000         — hard-cap endTime; residualControl stops early when
                           p<1e-5, U<1e-5, k/omega<1e-4 are all satisfied
    """
    case_dir = base_dir / case_id
    (case_dir / "0").mkdir(parents=True, exist_ok=True)
    (case_dir / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)
    (case_dir / "system").mkdir(parents=True, exist_ok=True)

    generate_stl(params, case_dir)

    ics = initial_conditions()
    for fname, content in ics.items():
        (case_dir / "0" / fname).write_text(content)

    (case_dir / "constant" / "transportProperties").write_text(TRANSPORT_PROPERTIES)
    (case_dir / "constant" / "turbulenceProperties").write_text(TURBULENCE_PROPERTIES)

    (case_dir / "system" / "blockMeshDict").write_text(block_mesh_dict())
    (case_dir / "system" / "surfaceFeatureExtractDict").write_text(surface_feature_extract_dict())
    (case_dir / "system" / "snappyHexMeshDict").write_text(snappy_hex_mesh_dict())
    (case_dir / "system" / "controlDict").write_text(control_dict(n_iters))
    (case_dir / "system" / "fvSchemes").write_text(FV_SCHEMES)
    (case_dir / "system" / "fvSolution").write_text(FV_SOLUTION)

    if n_cores > 1:
        (case_dir / "system" / "decomposeParDict").write_text(decompose_par_dict(n_cores))

    run_sh = case_dir / "run.sh"
    run_sh.write_text(run_script(case_id, str(case_dir.resolve()), n_cores=n_cores))
    os.chmod(run_sh, 0o755)

    return case_dir


# ─── 5. MAIN ─────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(DESIGN_MATRIX)
    CASES_DIR.mkdir(exist_ok=True)

    print(f"Generating {len(df)} OpenFOAM cases → {CASES_DIR}/\n")

    for _, row in df.iterrows():
        params = {
            "slant_angle":    float(row["slant_angle"]),
            "diffuser_angle": float(row["diffuser_angle"]),
            "ride_height":    float(row["ride_height"]),
            "front_radius":   float(row["front_radius"]),
        }
        write_case(str(row["case_id"]), params, CASES_DIR, n_cores=6)
        print(f"  {row['case_id']}  slant={params['slant_angle']:.1f}°  "
              f"diff={params['diffuser_angle']:.1f}°  "
              f"rh={params['ride_height']:.0f}mm  "
              f"r_nose={params['front_radius']:.0f}mm")

    master = CASES_DIR / "run_all.sh"
    lines = ["#!/bin/bash", "# Run all cases sequentially", "set -e", ""]
    for _, row in df.iterrows():
        lines.append(f"bash {CASES_DIR}/{row['case_id']}/run.sh")
    master.write_text("\n".join(lines))
    os.chmod(master, 0o755)

    print(f"\nAll cases written to {CASES_DIR}/")
    print(f"  Run all:       bash {CASES_DIR}/run_all.sh")
    print(f"  Run one:       bash {CASES_DIR}/case_000/run.sh")


if __name__ == "__main__":
    main()
