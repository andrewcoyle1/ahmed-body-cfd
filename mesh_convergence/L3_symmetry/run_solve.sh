#!/usr/bin/env bash
set +eu
source /Volumes/OpenFOAM-v2512/etc/bashrc \
    WM_COMPILER=Clang WM_MPLIB=SYSTEMOPENMPI FOAM_INST_DIR=/Volumes 2>/dev/null

ESI=/Volumes/OpenFOAM-v2512
ESI_LIB=$ESI/platforms/darwin64ClangDPInt32Opt/lib
ESI_BIN=$ESI/platforms/darwin64ClangDPInt32Opt/bin
USER_LIB=$HOME/OpenFOAM/andrewcoyle-v2512/platforms/darwin64ClangDPInt32Opt/lib
export DYLD_LIBRARY_PATH="$USER_LIB:$ESI_LIB:$ESI_LIB/openmpi:$ESI/env/lib:${DYLD_LIBRARY_PATH:-}"
MPI_XARGS="-x DYLD_LIBRARY_PATH -x PATH"

CASE=/Users/andrewcoyle/ahmed-body-cfd/mesh_convergence/L3_symmetry
cd "$CASE"

START=$(date +%s)
mpirun -np 8 $MPI_XARGS "$ESI_BIN/simpleFoam" -case "$CASE" -parallel > log.simpleFoam 2>&1
STATUS=$?
END=$(date +%s)
echo "Wall time: $((END-START))s  exit=$STATUS"

"$ESI_BIN/reconstructPar" -latestTime -case "$CASE" > log.reconstructPar 2>&1

echo "=== Mean Cd/Cl iters 1000-1500 ==="
DAT="postProcessing/forceCoeffs/0/coefficient.dat"
awk 'NR>1 && $1>=1000 && $1<=1500 {n++; cd+=$2; cl+=$5}
     END {if(n>0) printf "  Cd=%.4f  Cl=%.4f  f=%.4f  (n=%d)\n", cd/n, cl/n, cd/n+(1.0/3.0)*cl/n, n}' "$DAT"
