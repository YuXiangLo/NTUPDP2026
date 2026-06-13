#!/bin/bash
# sbatch_scaling_2k6k.sh — add 2K and 6K data points for scaling curve.
# Supplements existing ctrl/1080p/4K/8K data with two intermediate sizes.
#
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus-per-node=1
#SBATCH -A ACD115083
#SBATCH -t 00:30:00
#SBATCH -o slurm-scaling-%j.out
#SBATCH -e slurm-scaling-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load cuda 2>/dev/null || true

mkdir -p results

# Check / generate 2K and 6K images if missing

IMG_2K="data/2k/forest_pano_2k_2560x1440.png"
IMG_6K="data/6k/forest_pano_6k_6144x3456.png"

cd cuda
make seam_carve seam_carve_v6 seam_carve_wide seam_carve_tiled seam_carve_v0 \
    seam_carve_cpu seam_carve_cpu_omp >/dev/null 2>&1 \
    || { echo "BUILD FAILED" >&2; exit 1; }

TMP=/tmp/sc_out.png
RUNS=3
SEAMS=50

bench_one() {
    local label="$1" img="$2" bin="$3"
    best=""
    for r in $(seq 1 $RUNS); do
        t=$("./$bin" "../$img" "$SEAMS" "$TMP" 2>/dev/null \
            | grep -oP 'carving time:\s*\K[\d.]+' | head -1)
        [ -z "$t" ] && continue
        ms=$(awk "BEGIN{printf \"%.4f\", $t / $SEAMS}")
        if [ -z "$best" ] || awk "BEGIN{exit !($ms < $best)}"; then best="$ms"; fi
    done
    printf "  %-24s  %-26s  %s ms/seam\n" "$bin" "$label" "${best:--}"
}

echo "=== Scaling 2K + 6K: $(date) ==="
echo "node=$(hostname)  SEAMS=$SEAMS  RUNS=$RUNS (best-of)"
echo ""

for img_info in "$IMG_2K:2K(2560x1440)" "$IMG_6K:6K(6144x3456)"; do
    img="${img_info%%:*}"
    label="${img_info##*:}"
    if [ ! -f "../$img" ]; then echo "SKIP missing: $img"; continue; fi
    echo "--- $label ---"
    for bin in seam_carve_cpu seam_carve_cpu_omp seam_carve_v0 seam_carve seam_carve_v6 seam_carve_wide seam_carve_tiled; do
        bench_one "$label" "$img" "$bin"
    done
    echo ""
done

echo "=== DONE ==="
