#!/bin/bash
# sbatch_tiled.sh — correctness + performance sweep for seam_carve_tiled
#
# Phase 1: bit-exact check vs seam_carve_v6 (ctrl, 1080p, 4K)
#          and vs seam_carve_wide (4K, 8K — since v6 fails at 8K)
# Phase 2: ms/seam comparison: v6 / wide / tiled at 4K and 8K
#
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus-per-node=1
#SBATCH -A ACD115083
#SBATCH -t 00:40:00
#SBATCH -o slurm-tiled-%j.out
#SBATCH -e slurm-tiled-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load cuda 2>/dev/null || true

cd cuda
make seam_carve_tiled seam_carve_v6 seam_carve_wide seam_carve >/dev/null 2>&1 \
    || { echo "BUILD FAILED" >&2; exit 1; }
echo "=== Build OK ==="

SEAMS=50
TMP_A=/tmp/tiled_ref.png
TMP_B=/tmp/tiled_out.png

pass() { echo "[PASS] $*"; }
fail() { echo "[FAIL] $*"; }

verify() {
    local lbl="$1" img="$2" seams="$3" ref_bin="$4" test_bin="$5"
    "$ref_bin"  "../$img" "$seams" "$TMP_A" >/dev/null 2>&1
    "$test_bin" "../$img" "$seams" "$TMP_B" >/dev/null 2>&1
    if cmp -s "$TMP_A" "$TMP_B"; then pass "$lbl"; else fail "$lbl"; fi
}

echo ""
echo "=== Phase 1: Correctness ==="
verify "ctrl  vs v6"   data/ctrl/broadway_tower_ctrl_960x540.png   $SEAMS ./seam_carve_v6    ./seam_carve_tiled
verify "1080p vs v6"   data/1080p/forest_pano_1080p_1920x1080.png  $SEAMS ./seam_carve_v6    ./seam_carve_tiled
verify "4K    vs v6"   data/4k/desert_mesa_4k_3840x2160.png        $SEAMS ./seam_carve_v6    ./seam_carve_tiled
verify "4K    vs wide" data/4k/desert_mesa_4k_3840x2160.png        $SEAMS ./seam_carve_wide  ./seam_carve_tiled
verify "8K    vs wide" data/8k/desert_mesa_8k_7680x4320.png        50     ./seam_carve_wide  ./seam_carve_tiled

echo ""
echo "=== Phase 2: Performance (ms/seam, best of 3) ==="
BENCH_SEAMS=100
RUNS=3

bench_img() {
    local label="$1" img="$2" n="$3"
    echo ""
    echo "--- $label (${n} seams) ---"
    for bin in seam_carve_v6 seam_carve_wide seam_carve_tiled; do
        best=""
        for r in $(seq 1 $RUNS); do
            t=$("./$bin" "../$img" "$n" "$TMP_B" 2>/dev/null \
                | grep -oP 'carving time:\s*\K[\d.]+' | head -1)
            [ -z "$t" ] && continue
            ms=$(awk "BEGIN{printf \"%.4f\", $t / $n}")
            if [ -z "$best" ] || awk "BEGIN{exit !($ms < $best)}"; then best="$ms"; fi
        done
        printf "  %-22s  %s ms/seam\n" "$bin" "${best:--}"
    done
}

bench_img "4K (3840x2160)"  data/4k/desert_mesa_4k_3840x2160.png   $BENCH_SEAMS
bench_img "8K (7680x4320)"  data/8k/desert_mesa_8k_7680x4320.png   50

echo ""
echo "=== DONE ==="
