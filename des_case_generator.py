"""
des_case_generator.py
=====================
Generates DES (kOmegaSSTDES / IDDES) OpenFOAM cases for the multi-fidelity
co-Kriging campaign. Uses pimpleFoam with time-averaged force coefficients.

Mesh generation is identical to the RANS DoE (same blockMesh + snappyHexMesh
settings, validated against Lienhart & Becker at 0% Cd error).

Key differences from RANS case_generator.py:
  - Solver:    pimpleFoam  (transient)
  - Turbulence: kOmegaSSTDES with IDDESDelta
  - Ground BC: noSlip  (stationary, matches DES validation)
  - endTime:   0.45 s physical time
  - Averaging: fieldAverage from t=0.13 s
  - Cores:     10 MPI processes
"""

import subprocess, os
import numpy as np
import pandas as pd
from pathlib import Path

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

DES_CASES_DIR = Path("des_cases")
FREECAD_CMD   = "/Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd"
AHMED_SCRIPT  = str(Path(__file__).parent / "generate_ahmed_freecad.py")
DOCKER_IMAGE  = "opencfd/openfoam-default:latest"

FREESTREAM_U  = 40.0
NU            = 1.5e-5
RHO           = 1.225
N_CORES       = 10

# DES initial conditions — same validated ICs as sandbox DES
_K     = 0.24
_OMEGA = 16000.0
_NUT   = 1.5e-5

# DES time settings
END_TIME       = 0.45
DELTA_T        = 1e-4
MAX_CO         = 5.0
MAX_DELTA_T    = 2.5e-4
WRITE_INTERVAL = 0.05
PURGE_WRITE    = 3
AVG_START      = 0.13


# ─── 10-POINT DES DESIGN MATRIX ───────────────────────────────────────────────

DES_DESIGNS = pd.DataFrame([
    # case_id          slant   diffuser  rh      fr       source
    ("des_case_000",   25.0,   0.0,      50.0,   50.0),   # sandbox baseline (results exist)
    ("des_case_001",   35.483, 11.084,   33.823, 88.385), # DoE case_005 — worst RANS oscillation
    ("des_case_002",   30.892, 15.767,   35.880, 123.069),# DoE case_011 — 2nd worst oscillation
    ("des_case_003",   24.065, 14.593,   76.489, 119.264),# DoE case_029 — 3rd worst, high ride height
    ("des_case_004",   15.370, 9.083,    66.882, 137.748),# DoE case_016 — anchors low-slant Pareto end
    ("des_case_005",   33.893, 17.917,   46.481, 112.346),# DoE case_003 — near critical angle, high diffuser
    ("des_case_006",   28.050, 16.098,   38.201, 55.380), # DoE case_020 — max downforce (Cl=−0.46)
    ("des_case_007",   34.723, 19.445,   55.433, 82.402), # DoE case_026 — near optimum, strong downforce
    ("des_case_008",   37.259, 18.989,   71.389, 103.539),# DoE case_024 — high slant + max diffuser + high RH
    ("des_case_009",   35.874, 20.000,   61.313, 82.825), # RANS optimum — most important single point
], columns=["case_id", "slant_angle", "diffuser_angle", "ride_height", "front_radius"])


# ─── OPENFOAM FILE TEMPLATES ──────────────────────────────────────────────────

def turbulence_properties():
    return """\
FoamFile { version 2.0; format ascii; class dictionary; object turbulenceProperties; }
simulationType  LES;
LES
{
    LESModel        kOmegaSSTDES;
    turbulence      on;
    printCoeffs     on;
    delta           IDDESDelta;
    IDDESDeltaCoeffs { }
}
"""


def control_dict(end_time=END_TIME):
    return f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object controlDict; }}
application     pimpleFoam;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {end_time};
deltaT          {DELTA_T};
adjustTimeStep  yes;
maxCo           {MAX_CO};
maxDeltaT       {MAX_DELTA_T};
writeControl    adjustableRunTime;
writeInterval   {WRITE_INTERVAL};
purgeWrite      {PURGE_WRITE};
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
        writeInterval   5;
        patches         (ahmed_body);
        rho             rhoInf;
        rhoInf          {RHO};
        magUInf         {FREESTREAM_U};
        lRef            1.044;
        Aref            0.056;
        coordinateSystem
        {{
            origin  (0.522 0 0);
            e1      (1 0 0);
            e2      (0 1 0);
        }}
    }}

    fieldAverage
    {{
        type            fieldAverage;
        libs            (fieldFunctionObjects);
        writeControl    writeTime;
        timeStart       {AVG_START};
        fields
        (
            U  {{ mean on; prime2Mean on; base time; }}
            p  {{ mean on; prime2Mean off; base time; }}
        );
    }}
}}
"""


def fv_schemes():
    return """\
FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes      { default backward; }
gradSchemes     { default Gauss linear; grad(U) cellLimited Gauss linear 1; }
divSchemes
{
    default                              none;
    div(phi,U)                           Gauss linearUpwindV grad(U);
    div(phi,k)                           Gauss upwind;
    div(phi,omega)                       Gauss upwind;
    div((nuEff*dev2(T(grad(U)))))        Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
wallDist        { method meshWave; }
"""


def fv_solution():
    return """\
FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-7;
        relTol          0.01;
    }
    pFinal
    {
        $p;
        relTol          0.0;
        tolerance       1e-7;
    }
    Phi { $p; }
    "(U|k|omega)"
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
    }
    "(U|k|omega)Final"
    {
        $U;
        relTol          0.0;
        tolerance       1e-8;
    }
}
PIMPLE
{
    nOuterCorrectors        2;
    nCorrectors             2;
    nNonOrthogonalCorrectors 1;
}
potentialFlow { nNonOrthogonalCorrectors 4; }
cache { grad(U); }
"""


def decompose_par_dict():
    return f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; location "system"; object decomposeParDict; }}
numberOfSubdomains  {N_CORES};
method              scotch;
"""


def initial_conditions():
    u = f"""\
FoamFile {{ version 2.0; format ascii; class volVectorField; object U; }}
dimensions [0 1 -1 0 0 0 0];
internalField uniform ({FREESTREAM_U} 0 0);
boundaryField
{{
    inlet      {{ type fixedValue;  value uniform ({FREESTREAM_U} 0 0); }}
    outlet     {{ type inletOutlet; inletValue uniform (0 0 0); value uniform ({FREESTREAM_U} 0 0); }}
    ground     {{ type noSlip; }}
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
    inlet      {{ type fixedValue; value uniform {_K}; }}
    outlet     {{ type zeroGradient; }}
    ground     {{ type fixedValue; value uniform 0; }}
    top        {{ type symmetry; }}
    side_y_neg {{ type symmetry; }}
    side_y_pos {{ type symmetry; }}
    ahmed_body {{ type fixedValue; value uniform 0; }}
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
internalField uniform {_NUT};
boundaryField
{{
    inlet      {{ type calculated;            value uniform {_NUT}; }}
    outlet     {{ type calculated;            value uniform {_NUT}; }}
    ground     {{ type nutLowReWallFunction;  value uniform 0; }}
    top        {{ type symmetry; }}
    side_y_neg {{ type symmetry; }}
    side_y_pos {{ type symmetry; }}
    ahmed_body {{ type nutLowReWallFunction;  value uniform 0; }}
}}
"""
    return {"U": u, "p": p, "k": k, "omega": omega, "nut": nut}


# ─── RUN SCRIPT ───────────────────────────────────────────────────────────────

def run_script(case_id: str) -> str:
    return f"""\
#!/bin/bash
# Auto-generated DES run script for {case_id}  (n_cores={N_CORES})
set -e
CASE_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="{DOCKER_IMAGE}"

echo "=== {case_id}: meshing ==="

mkdir -p "$CASE_DIR/_field_init"
for f in U p k omega nut; do
    cp "$CASE_DIR/0/$f" "$CASE_DIR/_field_init/$f"
done

docker run --rm --cpus={N_CORES} \\
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

        decomposePar -force > log.decomposePar 2>&1
        mpirun -np {N_CORES} --allow-run-as-root potentialFoam -parallel -initialiseUBCs > log.potentialFoam 2>&1 || true
        mpirun -np {N_CORES} --allow-run-as-root pimpleFoam     -parallel                > log.pimpleFoam 2>&1
        reconstructPar > log.reconstructPar 2>&1
    "

echo "Done: {case_id}"
"""


# ─── STL GENERATION ───────────────────────────────────────────────────────────

def generate_stl(params: dict, case_dir: Path) -> Path:
    stl_path = case_dir / "constant" / "triSurface" / "ahmed_body.stl"
    stl_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "AHMED_SLANT_ANGLE":    str(params["slant_angle"]),
        "AHMED_R_NOSE":         str(params["front_radius"]),
        "AHMED_DIFFUSER_ANGLE": str(params["diffuser_angle"]),
        "AHMED_h":              str(params["ride_height"]),
        "AHMED_OUT":            str(stl_path.resolve()),
        "_AHMED_RUNNING":       "",
    })
    result = subprocess.run(
        [FREECAD_CMD, AHMED_SCRIPT],
        env=env, capture_output=True, text=True
    )
    if result.returncode != 0 or not stl_path.exists():
        raise RuntimeError(f"FreeCAD STL generation failed:\n{result.stdout}\n{result.stderr}")
    return stl_path


# ─── CASE WRITER ──────────────────────────────────────────────────────────────

def write_des_case(case_id: str, params: dict, base_dir: Path) -> Path:
    # Reuse blockMeshDict and snappyHexMeshDict from case_generator — identical settings
    from case_generator import block_mesh_dict, surface_feature_extract_dict, snappy_hex_mesh_dict

    case_dir = base_dir / case_id
    (case_dir / "0").mkdir(parents=True, exist_ok=True)
    (case_dir / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)
    (case_dir / "system").mkdir(parents=True, exist_ok=True)

    # Mesh (shared with RANS)
    (case_dir / "system" / "blockMeshDict").write_text(block_mesh_dict())
    (case_dir / "system" / "surfaceFeatureExtractDict").write_text(surface_feature_extract_dict())
    (case_dir / "system" / "snappyHexMeshDict").write_text(snappy_hex_mesh_dict())

    # DES solver settings
    (case_dir / "system" / "controlDict").write_text(control_dict())
    (case_dir / "system" / "fvSchemes").write_text(fv_schemes())
    (case_dir / "system" / "fvSolution").write_text(fv_solution())
    (case_dir / "system" / "decomposeParDict").write_text(decompose_par_dict())

    # Turbulence model
    (case_dir / "constant" / "turbulenceProperties").write_text(turbulence_properties())

    # Transport properties (same as RANS)
    (case_dir / "constant" / "transportProperties").write_text(
        f"FoamFile {{ version 2.0; format ascii; class dictionary; object transportProperties; }}\n"
        f"transportModel  Newtonian;\nnu              {NU};\n"
    )

    # Initial conditions
    ics = initial_conditions()
    for field, content in ics.items():
        (case_dir / "0" / field).write_text(content)

    # STL geometry
    generate_stl(params, case_dir)

    # Run script
    script = run_script(case_id)
    script_path = case_dir / "run.sh"
    script_path.write_text(script)
    script_path.chmod(0o755)

    return case_dir


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    DES_CASES_DIR.mkdir(exist_ok=True)

    # Save DES design matrix
    dm_path = Path("des_output")
    dm_path.mkdir(exist_ok=True)
    DES_DESIGNS.to_csv(dm_path / "des_design_matrix.csv", index=False)
    print(f"DES design matrix saved → des_output/des_design_matrix.csv")

    # des_case_000 is the sandbox baseline — results already exist, skip generation
    skip = {"des_case_000"}

    print(f"\nGenerating {len(DES_DESIGNS) - len(skip)} DES cases in {DES_CASES_DIR}/\n")
    for _, row in DES_DESIGNS.iterrows():
        cid = row["case_id"]
        if cid in skip:
            print(f"  {cid}  [SKIP — sandbox results already available]")
            continue
        params = {
            "slant_angle":    row["slant_angle"],
            "diffuser_angle": row["diffuser_angle"],
            "ride_height":    row["ride_height"],
            "front_radius":   row["front_radius"],
        }
        write_des_case(cid, params, DES_CASES_DIR)
        print(f"  {cid}  slant={params['slant_angle']:.1f}°  diffuser={params['diffuser_angle']:.1f}°  "
              f"rh={params['ride_height']:.1f}mm  fr={params['front_radius']:.1f}mm")

    print(f"\nDone. Run each case with:  bash des_cases/<case_id>/run.sh")
    print(f"Runtime estimate: ~92 min per case on {N_CORES} cores (~13.8 hrs total for 9 cases)")


if __name__ == "__main__":
    main()
