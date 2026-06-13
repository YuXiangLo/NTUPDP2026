#!/bin/bash
# sbatch_batch_full.sh — consistent single-vs-batch numbers across all 6 resolutions.
# seam_carve_batch single mode = exact one-seam-per-DP-pass (Tiled+prefetch);
# batch mode = K=60 strip-local seams per DP pass + one batch-remove pass.
#
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus-per-node=1
#SBATCH -A ACD115083
#SBATCH -t 00:30:00
#SBATCH -o slurm-batchfull-%j.out
#SBATCH -e slurm-batchfull-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load cuda 2>/dev/null || true
cd cuda
make seam_carve_batch >/dev/null 2>&1 || { echo BUILD FAILED; exit 1; }
echo "=== Build OK ==="
BT=./seam_carve_batch
RUNS=3
N=300   # 5 batch passes of K=60

best() {  # mode img -> best ms/seam
    local mode="$1" img="$2" best=""
    for r in $(seq 1 $RUNS); do
        t=$($BT "$img" $N /tmp/_b.png "$mode" 2>/dev/null \
            | grep -oP '\(\K[0-9.]+(?= ms/seam)' | head -1)
        [ -z "$t" ] && continue
        if [ -z "$best" ] || awk "BEGIN{exit !($t < $best)}"; then best="$t"; fi
    done
    echo "${best:--}"
}

row() {
    local label="$1" img="$2"
    local s=$(best single "$img"); local b=$(best batch "$img")
    local sp="-"
    [ "$s" != "-" ] && [ "$b" != "-" ] && sp=$(awk "BEGIN{printf \"%.0f\", $s/$b}")
    printf "%-16s  single=%-9s batch=%-9s  speedup=%sx\n" "$label" "$s" "$b" "$sp"
}

echo "=== single vs batch (ms/seam, best-of-$RUNS, N=$N) ==="
row "ctrl 960x540"    ../data/ctrl/broadway_tower_ctrl_960x540.png
row "1080p 1920x1080" ../data/1080p/forest_pano_1080p_1920x1080.png
row "2K 2560x1440"    ../data/2k/forest_pano_2k_2560x1440.png
row "4K 3840x2160"    ../data/4k/desert_mesa_4k_3840x2160.png
row "6K 6144x3456"    ../data/6k/forest_pano_6k_6144x3456.png
row "8K 7680x4320"    ../data/8k/desert_mesa_8k_7680x4320.png
echo "=== DONE ==="
