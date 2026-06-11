#!/usr/bin/env bash
# build_all_meshes.sh
# NOTE: invoke as "bash ./build_all_meshes.sh" not "./build_all_meshes.sh"
# macOS SIP strips DYLD_LIBRARY_PATH on shebang exec; bash invocation inherits it.
# Builds cfMesh meshes for all 4 convergence study levels sequentially.
# STLs are symlinked from cfmesh_validation — FreeCAD not re-run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LEVELS=(L1_coarse L2_medium L3_fine L4_veryfine)

# ── OpenFOAM + cfMesh environment (mirrors cfmesh_validation/mesh.sh) ─────────
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

export DYLD_LIBRARY_PATH="$USER_LIB:$ESI_LIB:${DYLD_LIBRARY_PATH:-}"

# ── Loop over levels ──────────────────────────────────────────────────────────
FAILED=()
for LEVEL in "${LEVELS[@]}"; do
    CASE_DIR="$SCRIPT_DIR/$LEVEL"
    echo ""
    echo "══════════════════════════════════════════════════════════════════"
    echo "  Building: $LEVEL"
    echo "══════════════════════════════════════════════════════════════════"

    # Clean any stale polyMesh from a previous run
    rm -rf "$CASE_DIR/constant/polyMesh"

    # Clean stale quality fields
    rm -f "$CASE_DIR"/0/{C,Cx,Cy,Cz,skewness,nonOrthoAngle,minPyrVolume,aspectRatio,cellDeterminant,wallDistance}
    rm -f "$CASE_DIR"/constant/{skewness,nonOrthoAngle,minPyrVolume,aspectRatio}

    echo "--- cartesianMesh ---"
    if ! time "$USER_BIN/cartesianMesh" -case "$CASE_DIR" \
            > "$CASE_DIR/log.cartesianMesh" 2>&1; then
        echo "  FAILED: cartesianMesh for $LEVEL"
        FAILED+=("$LEVEL (cartesianMesh)")
        continue
    fi

    echo "--- generateBoundaryLayers ---"
    if ! time "$USER_BIN/generateBoundaryLayers" -case "$CASE_DIR" \
            > "$CASE_DIR/log.generateBoundaryLayers" 2>&1; then
        echo "  FAILED: generateBoundaryLayers for $LEVEL"
        FAILED+=("$LEVEL (generateBoundaryLayers)")
        continue
    fi

    echo "--- renumberMesh ---"
    "$ESI_BIN/renumberMesh" -overwrite -case "$CASE_DIR" \
        > "$CASE_DIR/log.renumberMesh" 2>&1

    echo "--- checkMesh ---"
    "$ESI_BIN/checkMesh" -case "$CASE_DIR" \
        > "$CASE_DIR/log.checkMesh" 2>&1

    echo ""
    echo "  $LEVEL complete. Key metrics:"
    grep -E "cells:|Max non-orthogonality|average:|Max skewness|Mesh OK|FAILED" \
        "$CASE_DIR/log.checkMesh" | head -10 | sed 's/^/    /'
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  MESH BUILD SUMMARY"
echo "══════════════════════════════════════════════════════════════════"
printf "  %-20s %10s  %s\n" "Level" "Cells" "Quality"
for LEVEL in "${LEVELS[@]}"; do
    LOG="$SCRIPT_DIR/$LEVEL/log.checkMesh"
    if [ -f "$LOG" ]; then
        CELLS=$(grep "cells:" "$LOG" | head -1 | awk '{print $1}' || echo "?")
        STATUS=$(grep -c "Mesh OK" "$LOG" > /dev/null 2>&1 && echo "OK" || echo "FAILED")
        STATUS=$(grep -q "Mesh OK" "$LOG" && echo "OK" || echo "FAILED")
        printf "  %-20s %10s  %s\n" "$LEVEL" "$CELLS" "$STATUS"
    else
        printf "  %-20s %10s  %s\n" "$LEVEL" "-" "not run"
    fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "  FAILURES:"
    for F in "${FAILED[@]}"; do echo "    - $F"; done
    exit 1
fi
echo ""
echo "  All meshes built successfully."
