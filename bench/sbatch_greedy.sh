#!/bin/bash
# sbatch_greedy.sh — non-DP greedy seam carving: speed vs exact Tiled + quality.
#
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus-per-node=1
#SBATCH -A ACD115083
#SBATCH -t 00:30:00
#SBATCH -o slurm-greedy-%j.out
#SBATCH -e slurm-greedy-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load cuda 2>/dev/null || true
cd cuda
make seam_carve_greedy seam_carve_tiled_pf >/dev/null 2>&1 || { echo BUILD FAILED; exit 1; }
echo "=== Build OK ==="
GR=./seam_carve_greedy
PF=./seam_carve_tiled_pf
RUNS=3
N=100

best() {  # bin img -> best ms/seam
    local bin="$1" img="$2" best=""
    for r in $(seq 1 $RUNS); do
        t=$("$bin" "$img" $N /tmp/_g.png 2>/dev/null \
            | grep -oP '\(\K[0-9.]+(?= ms/seam)' | head -1)
        [ -z "$t" ] && continue
        if [ -z "$best" ] || awk "BEGIN{exit !($t < $best)}"; then best="$t"; fi
    done
    echo "${best:--}"
}

row() {
    local label="$1" img="$2"
    local g=$(best $GR "$img"); local e=$(best $PF "$img")
    local sp="-"
    [ "$g" != "-" ] && [ "$e" != "-" ] && sp=$(awk "BEGIN{printf \"%.1f\", $e/$g}")
    printf "%-16s  greedy=%-9s exact=%-9s  greedy-faster=%sx\n" "$label" "$g" "$e" "$sp"
}

echo "=== ms/seam: greedy (non-DP) vs exact Tiled (best-of-$RUNS, N=$N) ==="
row "ctrl 960x540"    ../data/ctrl/broadway_tower_ctrl_960x540.png
row "1080p 1920x1080" ../data/1080p/forest_pano_1080p_1920x1080.png
row "2K 2560x1440"    ../data/2k/forest_pano_2k_2560x1440.png
row "4K 3840x2160"    ../data/4k/desert_mesa_4k_3840x2160.png
row "6K 6144x3456"    ../data/6k/forest_pano_6k_6144x3456.png
row "8K 7680x4320"    ../data/8k/desert_mesa_8k_7680x4320.png

echo ""
echo "=== Quality: carve 200 seams from broadway ctrl, both methods ==="
$GR ../data/ctrl/broadway_tower_ctrl_960x540.png 200 /tmp/broadway_greedy.png  | grep -i time
$PF ../data/ctrl/broadway_tower_ctrl_960x540.png 200 /tmp/broadway_exact.png   | grep -i time
cp /tmp/broadway_greedy.png /tmp/broadway_exact.png "$ROOT_OUT" 2>/dev/null || true
# copy outputs to repo /tmp-equivalent shared path for later inspection
cp /tmp/broadway_greedy.png ../broadway_greedy.png 2>/dev/null || true
cp /tmp/broadway_exact.png  ../broadway_exact.png  2>/dev/null || true
echo "wrote broadway_greedy.png and broadway_exact.png to repo root"
echo "=== DONE ==="
