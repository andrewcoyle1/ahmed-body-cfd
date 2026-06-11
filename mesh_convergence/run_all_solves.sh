#!/usr/bin/env bash
# run_all_solves.sh
# Runs simpleFoam on all 4 mesh convergence levels sequentially.
# Cold start (potentialFoam IC) for each level.
# Invoke as: bash ./run_all_solves.sh
#
# Results written to results_summary.txt on completion.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LEVELS=(L1_coarse L2_medium L3_fine L4_veryfine)
TARGET_CELLS_PER_CORE=100000
MAX_CORES=8

# ── Environment ───────────────────────────────────────────────────────────────
set +eu
source /Volumes/OpenFOAM-v2512/etc/bashrc \
    WM_COMPILER=Clang WM_MPLIB=SYSTEMOPENMPI FOAM_INST_DIR=/Volumes 2>/dev/null
set -eu

export PATH="$HOME/OpenFOAM/andrewcoyle-v2512/platforms/darwin64ClangDPInt32Opt/bin:$PATH"
export DYLD_LIBRARY_PATH="$HOME/OpenFOAM/andrewcoyle-v2512/platforms/darwin64ClangDPInt32Opt/lib:/Volumes/OpenFOAM-v2512/platforms/darwin64ClangDPInt32Opt/lib:/Volumes/OpenFOAM-v2512/env/lib:${DYLD_LIBRARY_PATH:-}"

ESI=/Volumes/OpenFOAM-v2512
ESI_LIB=$ESI/platforms/darwin64ClangDPInt32Opt/lib
ESI_BIN=$ESI/platforms/darwin64ClangDPInt32Opt/bin
USER_LIB=$HOME/OpenFOAM/andrewcoyle-v2512/platforms/darwin64ClangDPInt32Opt/lib
USER_BIN=$HOME/OpenFOAM/andrewcoyle-v2512/platforms/darwin64ClangDPInt32Opt/bin

export DYLD_LIBRARY_PATH="$USER_LIB:$ESI_LIB:$ESI_LIB/openmpi:$ESI/env/lib:${DYLD_LIBRARY_PATH:-}"
MPI_XARGS="-x DYLD_LIBRARY_PATH -x PATH"

RESULTS_FILE="$SCRIPT_DIR/results_summary.txt"
> "$RESULTS_FILE"

declare -A LEVEL_CD LEVEL_CL LEVEL_CELLS LEVEL_TIME LEVEL_NP

# ── Loop over levels ──────────────────────────────────────────────────────────
for LEVEL in "${LEVELS[@]}"; do
    CASE_DIR="$SCRIPT_DIR/$LEVEL"
    echo ""
    echo "══════════════════════════════════════════════════════════════════"
    echo "  Solving: $LEVEL"
    echo "══════════════════════════════════════════════════════════════════"

    # Get cell count from checkMesh log
    N_CELLS=$(grep "^    cells:" "$CASE_DIR/log.checkMesh" 2>/dev/null | awk '{print $2}' || echo "0")
    if [ "$N_CELLS" = "0" ]; then
        echo "  ERROR: no checkMesh log found for $LEVEL — skipping"
        continue
    fi
    echo "  Cells: $N_CELLS"

    # Auto-scale NP
    NP=$(( (N_CELLS + TARGET_CELLS_PER_CORE - 1) / TARGET_CELLS_PER_CORE ))
    [ "$NP" -gt "$MAX_CORES" ] && NP=$MAX_CORES
    [ "$NP" -lt 1 ] && NP=1
    echo "  Using $NP cores"
    LEVEL_NP[$LEVEL]=$NP
    LEVEL_CELLS[$LEVEL]=$N_CELLS

    # Patch decomposeParDict
    sed -i '' "s/^numberOfSubdomains.*/numberOfSubdomains  $NP;/" \
        "$CASE_DIR/system/decomposeParDict"

    # Clean stale time dirs and processor dirs
    rm -rf "$CASE_DIR"/processor* "$CASE_DIR"/[1-9]* "$CASE_DIR"/0.[0-9]*

    START_TIME=$(date +%s)

    # decomposePar
    echo "  [1/4] decomposePar"
    "$ESI_BIN/decomposePar" -force -case "$CASE_DIR" \
        > "$CASE_DIR/log.decomposePar" 2>&1

    # potentialFoam (parallel IC)
    echo "  [2/4] potentialFoam"
    mpirun -np "$NP" $MPI_XARGS \
        "$ESI_BIN/potentialFoam" -case "$CASE_DIR" -parallel \
        > "$CASE_DIR/log.potentialFoam" 2>&1

    # simpleFoam
    echo "  [3/4] simpleFoam"
    mpirun -np "$NP" $MPI_XARGS \
        "$ESI_BIN/simpleFoam" -case "$CASE_DIR" -parallel \
        > "$CASE_DIR/log.simpleFoam" 2>&1

    # reconstructPar
    echo "  [4/4] reconstructPar"
    "$ESI_BIN/reconstructPar" -latestTime -case "$CASE_DIR" \
        > "$CASE_DIR/log.reconstructPar" 2>&1

    END_TIME=$(date +%s)
    ELAPSED=$(( END_TIME - START_TIME ))
    LEVEL_TIME[$LEVEL]=$ELAPSED
    echo "  Elapsed: $(( ELAPSED / 60 ))m$(( ELAPSED % 60 ))s"

    # Extract Cd/Cl from forceCoeffs
    COEFF_FILE=$(ls "$CASE_DIR"/postProcessing/forceCoeffs/0/coefficient*.dat 2>/dev/null \
                 | xargs ls -S 2>/dev/null | head -1 || echo "")
    if [ -n "$COEFF_FILE" ]; then
        # Average last 100 samples; Cd col=2 (half-body), Cl col=4
        CD_RAW=$(awk 'NF>1 && !/^#/{print $2}' "$COEFF_FILE" | tail -100 \
                 | awk '{s+=$1; n++} END{print s/n}')
        CL_RAW=$(awk 'NF>1 && !/^#/{print $4}' "$COEFF_FILE" | tail -100 \
                 | awk '{s+=$1; n++} END{print s/n}')
        CD_REAL=$(echo "$CD_RAW / 2" | bc -l | xargs printf "%.4f")
        CL_REAL=$(echo "$CL_RAW / 2" | bc -l | xargs printf "%.4f")
        LEVEL_CD[$LEVEL]=$CD_REAL
        LEVEL_CL[$LEVEL]=$CL_REAL
        echo "  Cd_real = $CD_REAL  Cl_real = $CL_REAL"
    else
        LEVEL_CD[$LEVEL]="n/a"
        LEVEL_CL[$LEVEL]="n/a"
        echo "  WARNING: no forceCoeffs output found"
    fi
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  MESH CONVERGENCE RESULTS"
echo "══════════════════════════════════════════════════════════════════"
printf "  %-15s %10s %6s %8s %8s %10s\n" \
    "Level" "Cells" "Cores" "Cd_real" "Cl_real" "Runtime"
printf "  %-15s %10s %6s %8s %8s %10s\n" \
    "-----" "-----" "-----" "-------" "-------" "-------"

{
echo ""
echo "MESH CONVERGENCE RESULTS — $(date)"
echo "Level           Cells       Cores  Cd_real  Cl_real  Runtime"
echo "--------------------------------------------------------------"
} >> "$RESULTS_FILE"

for LEVEL in "${LEVELS[@]}"; do
    CELLS=${LEVEL_CELLS[$LEVEL]:-"n/a"}
    NP=${LEVEL_NP[$LEVEL]:-"-"}
    CD=${LEVEL_CD[$LEVEL]:-"n/a"}
    CL=${LEVEL_CL[$LEVEL]:-"n/a"}
    if [ -n "${LEVEL_TIME[$LEVEL]:-}" ]; then
        T=${LEVEL_TIME[$LEVEL]}
        TSTR="$(( T / 60 ))m$(( T % 60 ))s"
    else
        TSTR="n/a"
    fi
    printf "  %-15s %10s %6s %8s %8s %10s\n" \
        "$LEVEL" "$CELLS" "$NP" "$CD" "$CL" "$TSTR"
    printf "%-15s %10s %6s %8s %8s %10s\n" \
        "$LEVEL" "$CELLS" "$NP" "$CD" "$CL" "$TSTR" >> "$RESULTS_FILE"
done

echo ""
echo "  Full results saved to: $RESULTS_FILE"
echo "  Reference: Lienhart 2002 Cd=0.299"
