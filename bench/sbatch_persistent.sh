#!/bin/bash
# sbatch_persistent.sh — correctness + performance for seam_carve_persistent
#
# Phase 1: bit-exact correctness vs seam_carve_tiled_pf (K=60, T=64)
# Phase 2: ms/seam at 4K and 8K: tiled_pf vs persistent (both K=60, T=64)
#
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus-per-node=1
#SBATCH -A ACD115083
#SBATCH -t 00:30:00
#SBATCH -o slurm-persistent-%j.out
#SBATCH -e slurm-persistent-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load cuda 2>/dev/null || true

cd cuda
make stb_impl.o seam_carve_persistent seam_carve_tiled >/dev/null 2>&1 \
    || { echo "BUILD FAILED" >&2; exit 1; }

# Build tiled_pf K=60 T=64 inline (not in Makefile as a named target)
nvcc -O3 -std=c++14 -arch=sm_70 -DSTRIP_K=60 -DTILE_T=64 -DNT_TILE=256 \
     seam_carve_tiled_pf.cu stb_impl.o -o tiled_pf_ref 2>/dev/null \
    || { echo "TILED_PF BUILD FAILED" >&2; exit 1; }

echo "=== Persistent kernel: $(date) === node=$(hostname)"

TMP_A=/tmp/pers_ref.png
TMP_B=/tmp/pers_out.png
SEAMS=50

pass() { echo "[PASS] $*"; }
fail() { echo "[FAIL] $*"; }

verify() {
    local lbl="$1" img="$2" seams="$3"
    ./tiled_pf_ref      "../$img" "$seams" "$TMP_A" >/dev/null 2>&1
    ./seam_carve_persistent "../$img" "$seams" "$TMP_B" >/dev/null 2>&1
    if cmp -s "$TMP_A" "$TMP_B"; then pass "$lbl"; else fail "$lbl"; fi
}

echo ""
echo "============================================================"
echo " Phase 1: Correctness (persistent vs tiled_pf K=60 T=64)"
echo "============================================================"
verify "ctrl  (960x540)"    data/ctrl/broadway_tower_ctrl_960x540.png  $SEAMS
verify "1080p (1920x1080)"  data/1080p/forest_pano_1080p_1920x1080.png $SEAMS
verify "4K    (3840x2160)"  data/4k/desert_mesa_4k_3840x2160.png       $SEAMS
verify "8K    (7680x4320)"  data/8k/desert_mesa_8k_7680x4320.png       30

echo ""
echo "============================================================"
echo " Phase 2: Performance (best-of-5, ms/seam)"
echo "============================================================"

RUNS=5
best_ms() {
    local bin="$1" img="$2" n="$3" best=""
    for r in $(seq 1 $RUNS); do
        local t
        t=$("./$bin" "$img" "$n" "$TMP_B" 2>/dev/null \
            | grep -oP 'carving time:\s*\K[\d.]+' | head -1)
        [ -z "$t" ] && continue
        local ms; ms=$(awk "BEGIN{printf \"%.4f\", $t / $n}")
        if [ -z "$best" ] || awk "BEGIN{exit !($ms < $best)}"; then best="$ms"; fi
    done
    echo "${best:--}"
}

I4K="../data/4k/desert_mesa_4k_3840x2160.png"
I8K="../data/8k/desert_mesa_8k_7680x4320.png"

echo ""
echo "--- 4K (3840x2160, 50 seams) ---"
pf4=$(best_ms  tiled_pf_ref           "$I4K" 50)
ps4=$(best_ms  seam_carve_persistent  "$I4K" 50)
printf "  tiled_pf (K=60,T=64)  : %s ms/seam\n" "$pf4"
printf "  persistent (K=60,T=64): %s ms/seam\n" "$ps4"
if [ "$pf4" != "-" ] && [ "$ps4" != "-" ]; then
    gain=$(awk "BEGIN{printf \"%.1f\", ($pf4 - $ps4) / $pf4 * 100}")
    printf "  improvement           : %s%%\n" "$gain"
fi

echo ""
echo "--- 8K (7680x4320, 30 seams) ---"
pf8=$(best_ms  tiled_pf_ref           "$I8K" 30)
ps8=$(best_ms  seam_carve_persistent  "$I8K" 30)
printf "  tiled_pf (K=60,T=64)  : %s ms/seam\n" "$pf8"
printf "  persistent (K=60,T=64): %s ms/seam\n" "$ps8"
if [ "$pf8" != "-" ] && [ "$ps8" != "-" ]; then
    gain=$(awk "BEGIN{printf \"%.1f\", ($pf8 - $ps8) / $pf8 * 100}")
    printf "  improvement           : %s%%\n" "$gain"
fi

echo ""
echo "=== DONE: $(date) ==="
