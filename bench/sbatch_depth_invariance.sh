#!/bin/bash
# sbatch_depth_invariance.sh — does transpose ever ACCELERATE seam carving?
#
# Controlled experiment: the SAME image, carved in two orientations.
#   original  (HxW): vertical-seam DP, serial depth D = H
#   transposed(WxH): vertical-seam DP, serial depth D = W   (= horizontal on orig)
# Total work H*W is identical; only the serial depth D differs.
# If time tracks D (shorter-D orientation always faster), then transpose---which
# swaps D<->P but cannot REDUCE D for a fixed seam direction---cannot accelerate.
#
# Tested for both SINGLE-seam (tiled_pf) and BATCH (seam_carve_batch).
#
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus-per-node=1
#SBATCH -A ACD115083
#SBATCH -t 00:20:00
#SBATCH -o slurm-depthinv-%j.out
#SBATCH -e slurm-depthinv-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
ROOT=$(pwd)
module load cuda 2>/dev/null || true
PY="$ROOT/.venv/bin/python"

# Generate transposed inputs ON THIS NODE (node-local /tmp is not shared).
echo "=== Preparing transposed inputs ==="
$PY - <<PYEOF
from PIL import Image
for src in ["$ROOT/data/4k/desert_mesa_4k_3840x2160.png",
            "$ROOT/data/8k/desert_mesa_8k_7680x4320.png"]:
    im = Image.open(src).convert("RGB").transpose(Image.TRANSPOSE)
    out = "/tmp/" + src.split("/")[-1].replace(".png","_T.png")
    im.save(out); print(out, im.size)
PYEOF

cd cuda
make seam_carve_tiled_pf seam_carve_batch >/dev/null 2>&1 || { echo BUILD FAILED; exit 1; }
echo "=== Build OK ==="

PF=./seam_carve_tiled_pf
BT=./seam_carve_batch
RUNS=3

# best-of-N ms/seam from a binary's "ms/seam" stdout
best_msseam() {
    local bin="$1" img="$2" n="$3" mode="${4:-}"
    local best=""
    for r in $(seq 1 $RUNS); do
        t=$("$bin" "$img" "$n" /tmp/_di.png $mode 2>/dev/null \
            | grep -oP '\(\K[0-9.]+(?= ms/seam)' | head -1)
        [ -z "$t" ] && continue
        if [ -z "$best" ] || awk "BEGIN{exit !($t < $best)}"; then best="$t"; fi
    done
    echo "${best:--}"
}

run_pair() {
    local label="$1" orig="$2" trans="$3" nv="$4" nh="$5"
    echo ""
    echo "=== $label ==="
    echo "  SINGLE-seam (tiled_pf):"
    printf "    original  (D=H): %s ms/seam\n" "$(best_msseam $PF "$orig"  $nv)"
    printf "    transposed(D=W): %s ms/seam\n" "$(best_msseam $PF "$trans" $nh)"
    echo "  BATCH (seam_carve_batch, batch mode):"
    printf "    original  (D=H): %s ms/seam\n" "$(best_msseam $BT "$orig"  $nv batch)"
    printf "    transposed(D=W): %s ms/seam\n" "$(best_msseam $BT "$trans" $nh batch)"
}

# 4K: original 3840x2160 (D=2160) vs transposed 2160x3840 (D=3840)
run_pair "4K desert_mesa (orig D=2160 vs transposed D=3840)" \
    ../data/4k/desert_mesa_4k_3840x2160.png \
    /tmp/desert_mesa_4k_3840x2160_T.png 200 120

# 8K: original 7680x4320 (D=4320) vs transposed 4320x7680 (D=7680)
run_pair "8K desert_mesa (orig D=4320 vs transposed D=7680)" \
    ../data/8k/desert_mesa_8k_7680x4320.png \
    /tmp/desert_mesa_8k_7680x4320_T.png 200 120

echo ""
echo "=== DONE ==="
