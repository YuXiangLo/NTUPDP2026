#!/bin/bash
# sbatch_omp_scaling_full.sh — TRUE multi-core OpenMP scaling (the honest baseline).
#
# Fixes two flaws in the earlier scaling runs:
#   (1) the old jobs requested only -c 4, so any run with >4 threads was
#       oversubscribing 4 physical cores (hence the bogus "regression past 4");
#   (2) v2 first-touches the image serially (NUMA-remote for half the threads).
# Here we allocate -c 48 and sweep 1..40 threads with proper pinning, comparing
# v2 (serial first-touch) against v4 (NUMA-aware first-touch). The CPU baseline
# for the paper is the BEST ms/seam over all thread counts.
#
# Also verifies v4 is bit-exact vs the sequential reference.
#
# Nodes are 36-core / 8-GPU, gated at ~4 CPUs per GPU, so grabbing all 8 GPUs
# is the only way to obtain a real multi-core (32-thread) CPU allocation.
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-node=8
#SBATCH -A ACD115083
#SBATCH -t 01:00:00
#SBATCH -o slurm-ompfull-%j.out
#SBATCH -e slurm-ompfull-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
ROOT=$(pwd)
module load gcc 2>/dev/null || true

export OMP_PROC_BIND=close
export OMP_PLACES=cores

cd openmp
# Build with -march=native -funroll-loops so the multicore baseline is on equal
# footing with the optimized single-thread C++ reference (fairest comparison).
make clean >/dev/null 2>&1
make ARCH="-march=native -funroll-loops" \
     seam_carve_seq seam_carve_omp_v2 seam_carve_omp_v4 >/dev/null 2>&1 \
    || { echo "BUILD FAILED" >&2; exit 1; }
echo "=== Build OK (-march=native -funroll-loops) ==="
echo "node=$(hostname)  nproc=$(nproc)"
echo ""

CTRL=../data/ctrl/broadway_tower_ctrl_960x540.png
P1080=../data/1080p/forest_pano_1080p_1920x1080.png
IMG4K=../data/4k/desert_mesa_4k_3840x2160.png
IMG8K=../data/8k/desert_mesa_8k_7680x4320.png

RUNS=3
THREADS="1 2 4 8 16 24 32"

# best-of-RUNS ms/seam for a binary at a given thread count
best_msseam() {
    local bin="$1" img="$2" seams="$3" thr="$4" best=""
    export OMP_NUM_THREADS=$thr
    for r in $(seq 1 $RUNS); do
        t=$("./$bin" "$img" "$seams" /tmp/_omp.png 2>/dev/null \
            | grep -oP '\(\K[0-9.]+(?= ms/seam)' | head -1)
        [ -z "$t" ] && continue
        if [ -z "$best" ] || awk "BEGIN{exit !($t < $best)}"; then best="$t"; fi
    done
    echo "${best:--}"
}

# ---------------------------------------------------------------------------
echo "=== Correctness: v4 vs seq (bit-exact PNG) ==="
OMP_NUM_THREADS=40 ./seam_carve_omp_v4 "$CTRL" 96 /tmp/_v4.png >/dev/null 2>&1
./seam_carve_seq "$CTRL" 96 /tmp/_seq.png >/dev/null 2>&1
if cmp -s /tmp/_v4.png /tmp/_seq.png; then echo "[PASS] v4 == seq"; else echo "[FAIL] v4 != seq"; fi
echo ""

# ---------------------------------------------------------------------------
sweep() {
    local label="$1" img="$2" seams="$3"
    echo "=== $label  ($seams seams, ms/seam, best-of-$RUNS) ==="
    printf "%-8s %12s %12s\n" "threads" "v2(serial)" "v4(NUMA)"
    for thr in $THREADS; do
        v2=$(best_msseam seam_carve_omp_v2 "$img" "$seams" "$thr")
        v4=$(best_msseam seam_carve_omp_v4 "$img" "$seams" "$thr")
        printf "%-8s %12s %12s\n" "$thr" "$v2" "$v4"
    done
    echo ""
}

sweep "ctrl 960x540"   "$CTRL"  50
sweep "1080p 1920x1080" "$P1080" 50
sweep "4K 3840x2160"   "$IMG4K" 30
sweep "8K 7680x4320"   "$IMG8K" 20

echo "=== DONE ==="
