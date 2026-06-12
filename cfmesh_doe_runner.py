"""
cfmesh_doe_runner.py
====================
Reads doe_output/design_matrix.csv and builds + solves one OpenFOAM case
per row using the validated cfMesh L2 half-plane symmetry pipeline:

  FreeCAD STL → generate_domain_stl.py → cartesianMesh →
  generateBoundaryLayers → renumberMesh → topoSet → subsetMesh →
  decomposePar → simpleFoam (800 iters, 8 cores) → reconstructPar

Cases are written to openfoam_cases/case_XXX/ so post_processor.py
can pick them up unchanged.

Usage:
  python3 cfmesh_doe_runner.py               # run all cases sequentially
  python3 cfmesh_doe_runner.py case_000      # run one specific case
  python3 cfmesh_doe_runner.py --resume      # skip already-completed cases
"""

import os, sys, shutil, subprocess, time, re, importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

# Import case_manager from the same directory as this file
_cm_spec = importlib.util.spec_from_file_location(
    "case_manager", Path(__file__).parent / "case_manager.py")
_cm = importlib.util.module_from_spec(_cm_spec)
_cm_spec.loader.exec_module(_cm)
CaseState    = _cm.CaseState
CaseStatus   = _cm.Status
resume_action = _cm.resume_action

# ─── Configuration ────────────────────────────────────────────────────────────

BASE          = Path(__file__).parent
DESIGN_MATRIX = BASE / "doe_output" / "design_matrix.csv"
CASES_DIR     = BASE / "openfoam_cases"
L2_TEMPLATE   = BASE / "mesh_convergence" / "L2_medium"      # mesh settings template
L2_SYM        = BASE / "mesh_convergence" / "L2_symmetry"    # IC template

FREECAD_CMD   = "/Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd"
AHMED_SCRIPT  = str(BASE / "generate_ahmed_freecad.py")
DOMAIN_SCRIPT = str(BASE / "generate_domain_stl.py")

ESI            = Path("/Volumes/OpenFOAM-v2512")
ESI_BIN        = ESI / "platforms/darwin64ClangDPInt32Opt/bin"
ESI_LIB        = ESI / "platforms/darwin64ClangDPInt32Opt/lib"
USER_LIB       = Path.home() / "OpenFOAM/andrewcoyle-v2512/platforms/darwin64ClangDPInt32Opt/lib"
USER_BIN       = Path.home() / "OpenFOAM/andrewcoyle-v2512/platforms/darwin64ClangDPInt32Opt/bin"

OF_ENV = {
    **os.environ,
    "DYLD_LIBRARY_PATH": f"{USER_LIB}:{ESI_LIB}:{ESI_LIB}/openmpi:{ESI}/env/lib",
    "PATH": f"{USER_BIN}:{ESI_BIN}:{ESI}/env/bin:{os.environ.get('PATH','')}",
    "WM_PROJECT_DIR": str(ESI),
    "FOAM_LIBBIN": str(ESI_LIB),
}

N_CORES       = 8
N_ITERS       = 800
AVG_START     = 600   # average Cd over iters AVG_START..N_ITERS

TOPOSET_DICT = """\
FoamFile { version 2.0; format ascii; class dictionary; object topoSetDict; }
actions
(
    { name halfDomain; type cellSet; action new; source boxToCell;
      min (-6.0 0.0 -0.1); max (20.0 2.1 3.0); }
);
"""

DECOMPOSE_DICT = f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object decomposeParDict; }}
numberOfSubdomains {N_CORES};
method scotch;
"""

MPI_XARGS = f"-x DYLD_LIBRARY_PATH -x PATH"


# ─── OpenFOAM helpers ─────────────────────────────────────────────────────────

def of_run(cmd, cwd, log_path, check=True):
    """Run an OpenFOAM binary, capturing output to log_path."""
    with open(log_path, "w") as fh:
        r = subprocess.run(cmd, cwd=str(cwd), env=OF_ENV,
                           stdout=fh, stderr=subprocess.STDOUT)
    if check and r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed (exit {r.returncode}) — see {log_path}")
    return r.returncode


CFMESH_UTILS = {"cartesianMesh", "generateBoundaryLayers", "cartesian2DMesh"}

def foam_bin(name):
    if name in CFMESH_UTILS:
        return str(USER_BIN / name)
    return str(ESI_BIN / name)


# ─── Step 1: Generate STL ────────────────────────────────────────────────────

def generate_stl(params: dict, case_dir: Path) -> Path:
    """Run FreeCAD headless to create ahmed_body.stl for this design point."""
    stl_path = case_dir / "constant" / "triSurface" / "ahmed_body.stl"
    stl_path.parent.mkdir(parents=True, exist_ok=True)

    env = {**os.environ,
           "AHMED_SLANT_ANGLE":    str(params["slant_angle"]),
           "AHMED_R_NOSE":         str(params["front_radius"]),
           "AHMED_DIFFUSER_ANGLE": str(params["diffuser_angle"]),
           "AHMED_h":              str(params["ride_height"]),
           "AHMED_OUT":            str(stl_path.resolve()),
           "_AHMED_RUNNING":       ""}

    log = case_dir / "log.freecad"
    with open(log, "w") as fh:
        r = subprocess.run([FREECAD_CMD, AHMED_SCRIPT], env=env,
                           stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0 or not stl_path.exists():
        raise RuntimeError(f"FreeCAD failed — see {log}")
    return stl_path


# ─── Step 2: Generate domain STL ─────────────────────────────────────────────

def generate_domain_stl(params: dict, case_dir: Path) -> Path:
    """Combine body STL + domain box → ahmed_domain.stl for cfMesh."""
    body_stl   = case_dir / "constant" / "triSurface" / "ahmed_body.stl"
    domain_stl = case_dir / "constant" / "triSurface" / "ahmed_domain.stl"

    env = {**os.environ, "AHMED_h": str(params["ride_height"])}
    log = case_dir / "log.domain_stl"
    with open(log, "w") as fh:
        r = subprocess.run(
            [sys.executable, DOMAIN_SCRIPT, str(body_stl), str(domain_stl)],
            env=env, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0 or not domain_stl.exists():
        raise RuntimeError(f"generate_domain_stl.py failed — see {log}")
    return domain_stl


# ─── Step 3: Build mesh (cfMesh) ─────────────────────────────────────────────

def copy_mesh_template(case_dir: Path):
    """Copy system/ and constant/turbulenceProperties from L2_medium template."""
    (case_dir / "system").mkdir(exist_ok=True)
    (case_dir / "constant").mkdir(exist_ok=True)

    for f in ["fvSchemes", "fvSolution", "controlDict", "decomposeParDict"]:
        src = L2_TEMPLATE / "system" / f
        if src.exists():
            shutil.copy(src, case_dir / "system" / f)

    for f in ["transportProperties", "turbulenceProperties"]:
        src = L2_TEMPLATE / "constant" / f
        if src.exists():
            shutil.copy(src, case_dir / "constant" / f)

    # Copy cfMesh settings (meshDict)
    shutil.copy(L2_TEMPLATE / "system" / "meshDict",
                case_dir / "system" / "meshDict")


WRITE_INTERVAL = 400   # checkpoint every 400 iters; enables resume mid-solve

def _forcecoeffs_block() -> str:
    return """\
    forceCoeffs
    {
        type            forceCoeffs;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   25;
        patches         (ahmed_body ahmed_legs_base);
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
    }"""


def write_case_controldict(case_dir: Path, start_time: int = 0):
    """Write controlDict. start_time > 0 means resuming from a checkpoint."""
    start_from = "latestTime" if start_time > 0 else "startTime"
    txt = f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object controlDict; }}
application     simpleFoam;
startFrom       {start_from};
startTime       {start_time};
stopAt          endTime;
endTime         {N_ITERS};
deltaT          1;
writeControl    timeStep;
writeInterval   {WRITE_INTERVAL};
purgeWrite      0;
writeFormat     binary;
writePrecision  8;
runTimeModifiable true;
functions
{{
{_forcecoeffs_block()}
}}
"""
    (case_dir / "system" / "controlDict").write_text(txt)


def find_checkpoint(case_dir: Path) -> int:
    """Return the latest saved time directory strictly between 0 and N_ITERS, or 0."""
    times = []
    for d in case_dir.iterdir():
        if d.is_dir() and d.name.isdigit():
            t = int(d.name)
            if 0 < t < N_ITERS and (d / "p").exists():
                times.append(t)
    return max(times) if times else 0


def build_mesh(case_dir: Path):
    """Run cartesianMesh + generateBoundaryLayers + renumberMesh."""
    of_run([foam_bin("cartesianMesh")],        case_dir, case_dir / "log.cartesianMesh")
    of_run([foam_bin("generateBoundaryLayers")], case_dir, case_dir / "log.generateBoundaryLayers")
    of_run([foam_bin("renumberMesh"), "-overwrite"], case_dir, case_dir / "log.renumberMesh")


# ─── Step 4: Cut half-domain ─────────────────────────────────────────────────

def cut_half_domain(case_dir: Path):
    """topoSet + subsetMesh → constant/polyMesh with symmetry patch."""
    (case_dir / "system" / "topoSetDict").write_text(TOPOSET_DICT)
    of_run([foam_bin("topoSet")], case_dir, case_dir / "log.topoSet")

    # Find which time directory subsetMesh writes to (latest time + 1)
    time_dirs_before = {d.name for d in case_dir.iterdir()
                        if d.is_dir() and d.name.isdigit()}

    of_run([foam_bin("subsetMesh"), "halfDomain", "-patch", "symmetry"],
           case_dir, case_dir / "log.subsetMesh")

    time_dirs_after = {d.name for d in case_dir.iterdir()
                       if d.is_dir() and d.name.isdigit()}
    new_dirs = time_dirs_after - time_dirs_before
    if not new_dirs:
        raise RuntimeError("subsetMesh did not create a new time directory")
    subset_time = sorted(new_dirs, key=int)[-1]
    subset_mesh_dir = case_dir / subset_time / "polyMesh"

    # Replace constant/polyMesh with the subset mesh
    dest = case_dir / "constant" / "polyMesh"
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(subset_mesh_dir, dest)

    # Update location tag in boundary file
    bfile = dest / "boundary"
    txt = bfile.read_text()
    txt = re.sub(r'location\s+"[^"]*"', 'location    "constant/polyMesh"', txt)

    # Fix symmetry patch type
    txt = re.sub(
        r'(symmetry\s*\{[^}]*?)type\s+\w+;',
        r'\1type            symmetry;', txt, flags=re.DOTALL)
    txt = re.sub(
        r'(symmetry\s*\{[^}]*?)inGroups\s+1\(\w+\);',
        r'\1inGroups        1(symmetry);', txt, flags=re.DOTALL)
    bfile.write_text(txt)

    # Remove the temporary time directory
    shutil.rmtree(case_dir / subset_time, ignore_errors=True)


# ─── Step 5: Write initial conditions ────────────────────────────────────────

def _patch_names(case_dir: Path) -> list[str]:
    """Read boundary file and return all patch names."""
    bfile = case_dir / "constant" / "polyMesh" / "boundary"
    txt = bfile.read_text()
    return re.findall(r'^\s{4}(\w+)\s*$', txt, re.MULTILINE)


def write_initial_conditions(case_dir: Path):
    """Write 0/ fields with correct BCs for all patches in this mesh."""
    patches = _patch_names(case_dir)
    zero_dir = case_dir / "0"
    zero_dir.mkdir(exist_ok=True)

    U_BCS, P_BCS, K_BCS, OMEGA_BCS, NUT_BCS = {}, {}, {}, {}, {}

    for p in patches:
        if p in ("ahmed_body", "ahmed_legs_base", "defaultFaces"):
            U_BCS[p]     = "type            noSlip;"
            P_BCS[p]     = "type            zeroGradient;"
            K_BCS[p]     = ("type            kLowReWallFunction;\n"
                             "        value           uniform 0;")
            OMEGA_BCS[p] = ("type            omegaWallFunction;\n"
                             "        value           uniform 1e5;")
            NUT_BCS[p]   = ("type            nutUSpaldingWallFunction;\n"
                             "        value           uniform 0;")
        elif p == "inlet":
            U_BCS[p]     = "type            fixedValue;\n        value           uniform (40 0 0);"
            P_BCS[p]     = "type            zeroGradient;"
            K_BCS[p]     = "type            fixedValue;\n        value           uniform 0.24;"
            OMEGA_BCS[p] = "type            fixedValue;\n        value           uniform 1e5;"
            NUT_BCS[p]   = "type            calculated;\n        value           uniform 1.5e-5;"
        elif p == "outlet":
            U_BCS[p]     = ("type            inletOutlet;\n"
                             "        inletValue      uniform (0 0 0);\n"
                             "        value           uniform (40 0 0);")
            P_BCS[p]     = "type            fixedValue;\n        value           uniform 0;"
            K_BCS[p]     = "type            zeroGradient;"
            OMEGA_BCS[p] = "type            zeroGradient;"
            NUT_BCS[p]   = "type            calculated;\n        value           uniform 0;"
        elif p == "ground":
            U_BCS[p]     = "type            movingWallVelocity;\n        value           uniform (40 0 0);"
            P_BCS[p]     = "type            zeroGradient;"
            K_BCS[p]     = ("type            kLowReWallFunction;\n"
                             "        value           uniform 0;")
            OMEGA_BCS[p] = ("type            omegaWallFunction;\n"
                             "        value           uniform 1e5;")
            NUT_BCS[p]   = ("type            nutUSpaldingWallFunction;\n"
                             "        value           uniform 0;")
        elif p == "symmetry":
            for d in (U_BCS, P_BCS, K_BCS, OMEGA_BCS, NUT_BCS):
                d[p] = "type            symmetry;"
        else:
            # top, side1, side2
            U_BCS[p]     = "type            slip;"
            P_BCS[p]     = "type            zeroGradient;"
            K_BCS[p]     = "type            zeroGradient;"
            OMEGA_BCS[p] = "type            zeroGradient;"
            NUT_BCS[p]   = "type            calculated;\n        value           uniform 0;"

    def _bf(bcs):
        lines = []
        for name, bc in bcs.items():
            lines.append(f"    {name}\n    {{\n        {bc}\n    }}")
        return "\n".join(lines)

    U_txt = f"""\
FoamFile {{ version 2.0; format ascii; class volVectorField; object U; }}
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (40 0 0);
boundaryField
{{
{_bf(U_BCS)}
}}
"""
    P_txt = f"""\
FoamFile {{ version 2.0; format ascii; class volScalarField; object p; }}
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{{
{_bf(P_BCS)}
}}
"""
    K_txt = f"""\
FoamFile {{ version 2.0; format ascii; class volScalarField; object k; }}
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0.24;
boundaryField
{{
{_bf(K_BCS)}
}}
"""
    OMEGA_txt = f"""\
FoamFile {{ version 2.0; format ascii; class volScalarField; object omega; }}
dimensions      [0 0 -1 0 0 0 0];
internalField   uniform 1e5;
boundaryField
{{
{_bf(OMEGA_BCS)}
}}
"""
    NUT_txt = f"""\
FoamFile {{ version 2.0; format ascii; class volScalarField; object nut; }}
dimensions      [0 2 -1 0 0 0 0];
internalField   uniform 1.5e-5;
boundaryField
{{
{_bf(NUT_BCS)}
}}
"""
    for fname, content in [("U", U_txt), ("p", P_txt), ("k", K_txt),
                            ("omega", OMEGA_txt), ("nut", NUT_txt)]:
        (zero_dir / fname).write_text(content)


# ─── Step 6: Decompose + solve ────────────────────────────────────────────────

def _run_simpleFoam(case_dir: Path, log_path: Path):
    mpi_cmd = (
        f"mpirun -np {N_CORES} {MPI_XARGS} "
        f"{foam_bin('simpleFoam')} -case {case_dir} -parallel"
    )
    mode = "a" if log_path.exists() else "w"
    with open(log_path, mode) as fh:
        r = subprocess.run(mpi_cmd, shell=True, env=OF_ENV,
                           stdout=fh, stderr=subprocess.STDOUT,
                           cwd=str(case_dir))
    if r.returncode != 0:
        raise RuntimeError(f"simpleFoam failed (exit {r.returncode})")


def solve(case_dir: Path):
    """decomposePar → simpleFoam (parallel) → reconstructPar."""
    (case_dir / "system" / "decomposeParDict").write_text(DECOMPOSE_DICT)
    of_run([foam_bin("decomposePar"), "-force"], case_dir, case_dir / "log.decomposePar")
    _run_simpleFoam(case_dir, case_dir / "log.simpleFoam")
    of_run([foam_bin("reconstructPar"), "-latestTime"],
           case_dir, case_dir / "log.reconstructPar")


def continue_solve(case_dir: Path, checkpoint: int):
    """Resume a simpleFoam run from a saved checkpoint directory."""
    # Update controlDict to start from checkpoint
    write_case_controldict(case_dir, start_time=checkpoint)
    (case_dir / "system" / "decomposeParDict").write_text(DECOMPOSE_DICT)

    # Decompose the checkpoint fields into processor dirs
    # -force overwrites any stale processor dirs; the checkpoint time dir provides fields
    of_run([foam_bin("decomposePar"), "-force", "-fields",
            "-time", str(checkpoint)],
           case_dir, case_dir / "log.decomposePar_resume")

    # Continue the solve (append to existing log)
    _run_simpleFoam(case_dir, case_dir / "log.simpleFoam")
    of_run([foam_bin("reconstructPar"), "-latestTime"],
           case_dir, case_dir / "log.reconstructPar")


# ─── Step 7: Quick Cd/Cl extraction ──────────────────────────────────────────

def extract_results(case_dir: Path) -> dict:
    """Average Cd/Cl over iters AVG_START..N_ITERS from forceCoeffs output."""
    dat = None
    base = case_dir / "postProcessing" / "forceCoeffs"
    if base.exists():
        for td in sorted(base.iterdir()):
            for name in ("coefficient.dat", "forceCoeffs.dat"):
                cand = td / name
                if cand.exists():
                    dat = cand
    if dat is None:
        return {"Cd": None, "Cl": None}

    rows = []
    with open(dat) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    rows.append([float(x) for x in parts])
                except ValueError:
                    pass

    if not rows:
        return {"Cd": None, "Cl": None}

    arr = np.array(rows)
    mask = (arr[:, 0] >= AVG_START) & (arr[:, 0] <= N_ITERS)
    if not mask.any():
        mask = np.ones(len(arr), dtype=bool)
    cd = float(np.mean(arr[mask, 1]))
    cl = float(np.mean(arr[mask, 4]))
    return {"Cd": round(cd, 6), "Cl": round(cl, 6)}


# ─── Main case runner ─────────────────────────────────────────────────────────

def run_case(case_id: str, params: dict, resume: bool = False, cases_dir: Path = None) -> dict:
    """
    Build mesh and solve one case.  State is tracked in <case_dir>/case.status
    so any interruption can be resumed correctly.

    resume=False : always clean-restart (ignore existing state)
    resume=True  : follow the resume_action() state table
    """
    if cases_dir is None:
        cases_dir = CASES_DIR
    case_dir = cases_dir / case_id
    state    = CaseState(case_dir)

    log = case_dir / "log.pipeline"
    case_dir.mkdir(parents=True, exist_ok=True)

    def log_msg(msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {case_id}  {msg}"
        print(line, flush=True)
        with open(log, "a") as f:
            f.write(line + "\n")

    # ── Determine what to do ──────────────────────────────────────────────────
    action = resume_action(case_dir) if resume else "clean_restart"

    if action == "skip":
        res = extract_results(case_dir)
        log_msg(f"SKIP (done)  Cd={res['Cd']}  Cl={res['Cl']}")
        return {**res, "skipped": True}

    if action == "clean_restart":
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True)
        state = CaseState(case_dir)

    t0 = time.time()
    try:
        # ── Mesh phase ────────────────────────────────────────────────────────
        if action in ("clean_restart", "fresh"):
            log_msg(f"STL  slant={params['slant_angle']:.1f}° diff={params['diffuser_angle']:.1f}°"
                    f" rh={params['ride_height']:.0f}mm r={params['front_radius']:.0f}mm")
            state.set(CaseStatus.MESHING)

            generate_stl(params, case_dir)
            log_msg("domain STL")
            generate_domain_stl(params, case_dir)
            log_msg("copy mesh template")
            copy_mesh_template(case_dir)
            write_case_controldict(case_dir)
            log_msg("cartesianMesh + BL")
            build_mesh(case_dir)
            log_msg("topoSet + subsetMesh (half-domain)")
            cut_half_domain(case_dir)

            state.set(CaseStatus.MESH_DONE)
            action = "solve_only"   # fall through to solve

        # ── Solve phase ───────────────────────────────────────────────────────
        if action == "solve_only":
            log_msg("write 0/ ICs")
            write_initial_conditions(case_dir)
            write_case_controldict(case_dir)
            state.set(CaseStatus.SOLVING)
            log_msg(f"simpleFoam {N_ITERS} iters × {N_CORES} cores")
            solve(case_dir)

        elif action == "resume_solve":
            checkpoint = find_checkpoint(case_dir)
            if checkpoint > 0:
                log_msg(f"RESUME solve from checkpoint t={checkpoint}")
                state.set(CaseStatus.SOLVING, f"resuming from t={checkpoint}")
                continue_solve(case_dir, checkpoint)
            else:
                # No checkpoint — re-solve from scratch (mesh stays)
                log_msg("RESUME: no checkpoint found, re-solving from t=0")
                write_initial_conditions(case_dir)
                write_case_controldict(case_dir)
                state.set(CaseStatus.SOLVING)
                solve(case_dir)

        # ── Extract and mark done ─────────────────────────────────────────────
        res = extract_results(case_dir)
        if res["Cd"] is None:
            raise RuntimeError("extract_results returned None — forceCoeffs output missing")

        state.set(CaseStatus.DONE)
        elapsed = time.time() - t0
        f1 = res["Cd"] + (1/3) * res["Cl"]
        log_msg(f"DONE  Cd={res['Cd']}  Cl={res['Cl']}  f={f1:.4f}  t={elapsed:.0f}s")
        return res

    except Exception as e:
        state.set(CaseStatus.FAILED, str(e))
        log_msg(f"FAILED: {e}")
        return {"Cd": None, "Cl": None, "error": str(e)}


def main():
    resume = "--resume" in sys.argv

    # --design-matrix <path> overrides default
    dm_path = DESIGN_MATRIX
    cases_dir = CASES_DIR
    results_dir = BASE / "doe_output"
    for i, a in enumerate(sys.argv[1:], 1):
        if a == "--design-matrix" and i < len(sys.argv):
            dm_path = Path(sys.argv[i + 1]).resolve()
        if a == "--cases-dir" and i < len(sys.argv):
            cases_dir = Path(sys.argv[i + 1]).resolve()
        if a == "--results-dir" and i < len(sys.argv):
            results_dir = Path(sys.argv[i + 1]).resolve()

    dm = pd.read_csv(dm_path)
    cases_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    # Filter to specific cases if given on command line
    targets = [a for a in sys.argv[1:] if a.startswith("case_")]
    if targets:
        dm = dm[dm["case_id"].isin(targets)]

    print(f"\nRunning {len(dm)} cases sequentially ({N_CORES} cores each)...\n")

    results = []
    for _, row in dm.iterrows():
        params = {
            "slant_angle":    float(row["slant_angle"]),
            "diffuser_angle": float(row["diffuser_angle"]),
            "ride_height":    float(row["ride_height"]),
            "front_radius":   float(row["front_radius"]),
        }
        res = run_case(str(row["case_id"]), params, resume=resume, cases_dir=cases_dir)
        results.append({**{"case_id": row["case_id"]}, **params, **res})

    # Write a quick results table
    df = pd.DataFrame(results)
    out = results_dir / "preliminary_results.csv"
    df.to_csv(out, index=False, float_format="%.6f")
    print(f"\nPreliminary results → {out}")

    n_ok = df["Cd"].notna().sum()
    print(f"Completed: {n_ok}/{len(df)} cases")


if __name__ == "__main__":
    main()
