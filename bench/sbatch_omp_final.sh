#!/bin/bash
# sbatch_omp_final.sh — definitive multicore CPU baseline for the paper.
# v4 (persistent region + NUMA first-touch + ping-pong cost, -march=native),
# all six resolutions, threads 1..32. Reports ms/seam best-of-3 and the best
# thread count per resolution. Single consistent codebase for the whole CPU
# column. Nodes are 36-core/8-GPU gated at ~4 CPU/GPU, so we grab 8 GPUs for 32.
#
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-node=8
#SBATCH -A ACD115083
#SBATCH -t 01:00:00
#SBATCH -o slurm-ompfinal-%j.out
#SBATCH -e slurm-ompfinal-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load gcc 2>/dev/null || true
export OMP_PROC_BIND=close
export OMP_PLACES=cores

cd openmp
make clean >/dev/null 2>&1
make ARCH="-march=native -funroll-loops" seam_carve_seq seam_carve_omp_v4 >/dev/null 2>&1 \
    || { echo "BUILD FAILED" >&2; exit 1; }
echo "=== Build OK (v4, -march=native) ==="
echo "node=$(hostname) nproc=$(nproc)"

# correctness once
OMP_NUM_THREADS=32 ./seam_carve_omp_v4 ../data/ctrl/broadway_tower_ctrl_960x540.png 96 /tmp/_v4.png >/dev/null 2>&1
./seam_carve_seq ../data/ctrl/broadway_tower_ctrl_960x540.png 96 /tmp/_seq.png >/dev/null 2>&1
cmp -s /tmp/_v4.png /tmp/_seq.png && echo "[PASS] v4==seq" || echo "[FAIL] v4!=seq"
echo ""

RUNS=3
THREADS="1 2 4 8 16 24 32"

best_msseam() {
    local img="$1" seams="$2" thr="$3" best=""
    export OMP_NUM_THREADS=$thr
    for r in $(seq 1 $RUNS); do
        t=$(./seam_carve_omp_v4 "$img" "$seams" /tmp/_o.png 2>/dev/null \
            | grep -oP '\(\K[0-9.]+(?= ms/seam)' | head -1)
        [ -z "$t" ] && continue
        if [ -z "$best" ] || awk "BEGIN{exit !($t < $best)}"; then best="$t"; fi
    done
    echo "${best:--}"
}

sweep() {
    local label="$1" img="$2" seams="$3"
    echo "=== $label ($seams seams) ==="
    for thr in $THREADS; do
        printf "  %-3s threads : %s ms/seam\n" "$thr" "$(best_msseam "$img" "$seams" "$thr")"
    done
    echo ""
}

sweep "ctrl 960x540"    ../data/ctrl/broadway_tower_ctrl_960x540.png   50
sweep "1080p 1920x1080" ../data/1080p/forest_pano_1080p_1920x1080.png  50
sweep "2K 2560x1440"    ../data/2k/forest_pano_2k_2560x1440.png        50
sweep "4K 3840x2160"    ../data/4k/desert_mesa_4k_3840x2160.png        30
sweep "6K 6144x3456"    ../data/6k/forest_pano_6k_6144x3456.png        20
sweep "8K 7680x4320"    ../data/8k/desert_mesa_8k_7680x4320.png        20

echo "=== DONE ==="
