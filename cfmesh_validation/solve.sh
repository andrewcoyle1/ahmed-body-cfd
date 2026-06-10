#!/bin/bash
# solve.sh — Run simpleFoam (RANS kOmegaSST) with auto-scaled parallelism.
# Core count is derived from mesh cell count to keep cells/core in the
# efficient range (50k–150k). Capped at MAX_CORES (P-cores on M4 Pro).
# Requires mesh to exist (run mesh.sh first).

set -euo pipefail

CASE_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── OpenFOAM v2512 environment ────────────────────────────────────────────────
set +eu
source /Volumes/OpenFOAM-v2512/etc/bashrc \
    WM_COMPILER=Clang WM_MPLIB=SYSTEMOPENMPI FOAM_INST_DIR=/Volumes 2>/dev/null
set -eu

ESI=/Volumes/OpenFOAM-v2512
ESI_BIN=$ESI/platforms/darwin64ClangDPInt32Opt/bin
USER_BIN=$HOME/OpenFOAM/andrewcoyle-v2512/platforms/darwin64ClangDPInt32Opt/bin
ESI_LIB=$ESI/platforms/darwin64ClangDPInt32Opt/lib
USER_LIB=$HOME/OpenFOAM/andrewcoyle-v2512/platforms/darwin64ClangDPInt32Opt/lib

export PATH="$USER_BIN:$ESI_BIN:$PATH"
export DYLD_LIBRARY_PATH="$USER_LIB:$ESI_LIB:$ESI_LIB/openmpi:$ESI/env/lib:${DYLD_LIBRARY_PATH:-}"

MPI_XARGS="-x DYLD_LIBRARY_PATH -x PATH"

# ── Auto-scale core count ─────────────────────────────────────────────────────
MAX_CORES=8                  # M4 Pro performance-core count
TARGET_CELLS_PER_CORE=100000 # sweet spot: enough work/core, low comm overhead

# Read cell count — prefer checkMesh log, fall back to polyMesh owner header
N_CELLS=$(grep -E "^\s+cells:" "$CASE_DIR/log.checkMesh" 2>/dev/null \
          | tail -1 | awk '{print $2}')
if [ -z "$N_CELLS" ]; then
    N_CELLS=$(awk '/nCells/{gsub(";","",$2); print $2; exit}' \
              "$CASE_DIR/constant/polyMesh/owner" 2>/dev/null || echo "")
fi

if [ -z "$N_CELLS" ] || [ "$N_CELLS" -le 0 ] 2>/dev/null; then
    echo "WARNING: could not read cell count — defaulting to NP=1 (serial)"
    NP=1
else
    NP=$(( (N_CELLS + TARGET_CELLS_PER_CORE - 1) / TARGET_CELLS_PER_CORE ))
    [ "$NP" -gt "$MAX_CORES" ] && NP=$MAX_CORES
    [ "$NP" -lt 1 ]            && NP=1
fi

echo "=== Core selection ==="
echo "    Cells: ${N_CELLS:-unknown}"
echo "    Target cells/core: $TARGET_CELLS_PER_CORE"
echo "    NP: $NP (max $MAX_CORES)"

# ── Step 1: Decompose mesh ────────────────────────────────────────────────────
# Update decomposeParDict with computed NP before decomposing
sed -i '' "s/^numberOfSubdomains.*/numberOfSubdomains  $NP;/" \
    "$CASE_DIR/system/decomposeParDict"

echo ""
echo "=== [1/4] decomposePar ($NP subdomains, scotch) ==="
"$ESI_BIN/decomposePar" -case "$CASE_DIR" -force \
    2>&1 | tee "$CASE_DIR/log.decomposePar"

# ── Step 2: potentialFoam initialisation (parallel) ───────────────────────────
echo ""
echo "=== [2/4] potentialFoam initialisation ==="
mpirun -np "$NP" $MPI_XARGS \
    "$ESI_BIN/potentialFoam" -case "$CASE_DIR" -parallel -noFunctionObjects \
    2>&1 | tee "$CASE_DIR/log.potentialFoam"

# ── Step 3: simpleFoam (parallel) ────────────────────────────────────────────
echo ""
echo "=== [3/4] simpleFoam (kOmegaSST, $NP cores, max 3000 iterations) ==="
time mpirun -np "$NP" $MPI_XARGS \
    "$ESI_BIN/simpleFoam" -case "$CASE_DIR" -parallel \
    2>&1 | tee "$CASE_DIR/log.simpleFoam"

# ── Step 4: Reconstruct ───────────────────────────────────────────────────────
echo ""
echo "=== [4/4] reconstructPar ==="
"$ESI_BIN/reconstructPar" -case "$CASE_DIR" -latestTime \
    2>&1 | tee "$CASE_DIR/log.reconstructPar"

echo ""
echo "=== Solve complete ==="
echo "    Latest Cd/Cl from forceCoeffs:"
tail -3 "$CASE_DIR/postProcessing/forceCoeffs/0/coefficient.dat" 2>/dev/null || true
