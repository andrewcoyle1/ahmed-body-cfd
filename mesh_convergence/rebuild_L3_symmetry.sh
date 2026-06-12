#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
L3F="$SCRIPT_DIR/L3_fine"
L3S="$SCRIPT_DIR/L3_symmetry"

set +e
source /Volumes/OpenFOAM-v2512/etc/bashrc \
    WM_COMPILER=Clang WM_MPLIB=SYSTEMOPENMPI FOAM_INST_DIR=/Volumes 2>/dev/null
set -e

echo "=== Step 1: copy topoSetDict to L3_fine ==="
cp "$L3S/system/topoSetDict" "$L3F/system/topoSetDict"
cat "$L3F/system/topoSetDict"

echo "=== Step 2: run topoSet on L3_fine ==="
cd "$L3F"
topoSet > log.topoSet 2>&1
tail -5 log.topoSet

echo "=== Step 3: run subsetMesh ==="
subsetMesh halfDomain -patch symmetry > log.subsetMesh 2>&1
tail -10 log.subsetMesh

echo "=== Step 4: check new cell count ==="
# subsetMesh writes to halfDomain/
HALF_DIR="$L3F/halfDomain"
if [ ! -d "$HALF_DIR/constant/polyMesh" ]; then
    echo "ERROR: subsetMesh output not found at $HALF_DIR"
    exit 1
fi
grep "cells:" "$L3F/log.subsetMesh" || true

echo "=== Step 5: replace L3_symmetry polyMesh ==="
rm -rf "$L3S/constant/polyMesh"
cp -r "$HALF_DIR/constant/polyMesh" "$L3S/constant/polyMesh"

echo "=== Step 6: fix boundary — ensure symmetry patch is type symmetry ==="
BFILE="$L3S/constant/polyMesh/boundary"
# Use Python for safe boundary edit
python3 - "$BFILE" <<'PYEOF'
import sys, re

path = sys.argv[1]
with open(path) as f:
    txt = f.read()

# Find symmetry block and fix type
txt = re.sub(
    r'(symmetry\s*\{[^}]*?)type\s+\w+;',
    r'\1type            symmetry;',
    txt, flags=re.DOTALL
)
with open(path, 'w') as f:
    f.write(txt)
print("boundary patched")
PYEOF

# Verify
grep -A4 "^    symmetry" "$BFILE" || grep -A4 "symmetry$" "$BFILE" || echo "(symmetry patch not shown — check manually)"

echo "=== Step 7: rebuild 0/ fields ==="
cd "$L3S"
rm -rf 0
mkdir 0

# Copy fields from _field_init (full-domain uniform ICs) and add symmetry patch
for field in U p k omega nut; do
    SRC="$L3F/_field_init/$field"
    DST="$L3S/0/$field"
    cp "$SRC" "$DST"
done

# Patch all 0/ fields: add symmetry BC (or fix side1→symmetry naming)
# The subsetMesh renames the cut face to 'symmetry'; we need proper BC entries
python3 - "$L3S/0" "$L3F/_field_init" <<'PYEOF'
import sys, os, re

zero_dir = sys.argv[1]

# For each field, read current content and add symmetry BC if not present
symmetry_bcs = {
    'U':     'symmetry { type symmetry; }',
    'p':     'symmetry { type symmetry; }',
    'k':     'symmetry { type symmetry; }',
    'omega': 'symmetry { type symmetry; }',
    'nut':   'symmetry { type symmetry; }',
}

for fname in os.listdir(zero_dir):
    fpath = os.path.join(zero_dir, fname)
    if not os.path.isfile(fpath):
        continue
    with open(fpath) as f:
        txt = f.read()
    # If symmetry BC already present, skip
    if 'symmetry' in txt and 'type symmetry' in txt:
        print(f"  {fname}: symmetry BC already present")
        continue
    # Add symmetry entry before closing brace of boundaryField
    if 'boundaryField' not in txt:
        print(f"  {fname}: no boundaryField, skipping")
        continue
    insert = f'\n    symmetry\n    {{\n        type            symmetry;\n    }}\n'
    # Find last } in boundaryField section
    idx = txt.rfind('\n}')
    if idx == -1:
        print(f"  {fname}: cannot find closing brace")
        continue
    txt = txt[:idx] + insert + txt[idx:]
    with open(fpath, 'w') as f:
        f.write(txt)
    print(f"  {fname}: added symmetry BC")

print("fields patched")
PYEOF

echo "=== Step 8: checkMesh ==="
cd "$L3S"
checkMesh -noTopology > log.checkMesh_new 2>&1
grep -E "cells:|non-orthogonality|symmetry" log.checkMesh_new | head -20

echo "=== Done — ready to solve ==="
