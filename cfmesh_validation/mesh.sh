#!/bin/bash
# mesh.sh
# Generates the Ahmed body validation mesh (L2 medium, ≈ 1.4 M cells).
# Two-pass BL strategy: cartesianMesh builds the Cartesian background mesh
# without BL insertion, then generateBoundaryLayers adds the prismatic layers.
# This avoids cfMesh's in-mesh BL optimizer limit at 1200:1 aspect ratio.
#
#   1. FreeCAD → ahmed_body_raw.stl  (canonical: 25° slant, 0° diffuser)
#   2. generate_domain_stl.py → ahmed_domain.stl  (body + domain, named patches)
#   3. cartesianMesh         (Cartesian background mesh, no BL)
#   4. generateBoundaryLayers (prismatic BL layers: body only, 3L y1=5e-4m, wall functions)
#   5. renumberMesh -overwrite (fix face ordering + Cuthill-McKee bandwidth reduction)
#   6. checkMesh             (report non-orthogonality, skewness, cell counts)

set -euo pipefail

CASE_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$CASE_DIR")"

# ── Native OpenFOAM v2512 + cfMesh environment ────────────────────────────────
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

FREECAD=/Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd

# ── Step 1: Generate Ahmed body STL ───────────────────────────────────────────
RAW_STL=$CASE_DIR/constant/triSurface/ahmed_body_raw.stl
mkdir -p "$CASE_DIR/constant/triSurface"

echo "=== [1/6] Generating Ahmed body STL ==="
echo "    slant=25°  diffuser=0°  ride_height=50.8mm  nose_radius=100mm"
AHMED_SLANT_ANGLE=25.0  \
AHMED_DIFFUSER_ANGLE=0.0 \
AHMED_h=50.8             \
AHMED_R_NOSE=100.0       \
AHMED_OUT="$RAW_STL"    \
  "$FREECAD" "$PROJECT_DIR/generate_ahmed_freecad.py" 2>&1 | tee "$CASE_DIR/log.freecad"

if [ ! -f "$RAW_STL" ]; then
    echo "ERROR: FreeCAD did not produce $RAW_STL"; exit 1
fi
echo "    OK: $(wc -c < "$RAW_STL") bytes"

# ── Step 2: Build combined domain+body STL ────────────────────────────────────
DOMAIN_STL=$CASE_DIR/constant/triSurface/ahmed_domain.stl

echo ""
echo "=== [2/6] Building combined domain+body STL ==="
python3 "$PROJECT_DIR/generate_domain_stl.py" "$RAW_STL" "$DOMAIN_STL"

# ── Clean stale fields from previous runs ─────────────────────────────────────
rm -f "$CASE_DIR"/0/C "$CASE_DIR"/0/Cx "$CASE_DIR"/0/Cy "$CASE_DIR"/0/Cz
rm -f "$CASE_DIR"/0/skewness "$CASE_DIR"/0/nonOrthoAngle "$CASE_DIR"/0/minPyrVolume
rm -f "$CASE_DIR"/0/aspectRatio "$CASE_DIR"/0/cellDeterminant "$CASE_DIR"/0/wallDistance
rm -f "$CASE_DIR"/constant/skewness "$CASE_DIR"/constant/nonOrthoAngle
rm -f "$CASE_DIR"/constant/minPyrVolume "$CASE_DIR"/constant/aspectRatio

# ── Step 3: cartesianMesh (Cartesian background, no BL) ───────────────────────
echo ""
echo "=== [3/6] Running cartesianMesh (background mesh, BL omitted) ==="
time "$USER_BIN/cartesianMesh" -case "$CASE_DIR" 2>&1 | tee "$CASE_DIR/log.cartesianMesh"

# ── Step 4: generateBoundaryLayers ────────────────────────────────────────────
echo ""
echo "=== [4/6] Running generateBoundaryLayers ==="
time "$USER_BIN/generateBoundaryLayers" -case "$CASE_DIR" 2>&1 | tee "$CASE_DIR/log.generateBoundaryLayers"

# ── Step 5: renumberMesh (fix face ordering + Cuthill-McKee bandwidth reduction)
echo ""
echo "=== [5/6] Running renumberMesh ==="
"$ESI_BIN/renumberMesh" -overwrite -case "$CASE_DIR" 2>&1 | tee "$CASE_DIR/log.renumberMesh"

# ── Step 6: checkMesh ─────────────────────────────────────────────────────────
echo ""
echo "=== [6/6] Running checkMesh ==="
"$ESI_BIN/checkMesh" -case "$CASE_DIR" 2>&1 | tee "$CASE_DIR/log.checkMesh"

echo ""
echo "=== Mesh complete. Key quality metrics above. ==="
echo "    Non-orthogonality target: max < 65°, average < 20°"
echo "    Skewness target: max < 4"
grep -E "Max non-orthogonality|average:|Max skewness|Max aspect|cells:|Mesh OK|Failed" "$CASE_DIR/log.checkMesh" | head -15
