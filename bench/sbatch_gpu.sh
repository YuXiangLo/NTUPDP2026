#!/bin/bash
# sbatch_gpu.sh — Single GPU benchmark run, appends one aggregated row to CSV.
#
# Required env vars:
#   IMPL        binary tag, e.g. CUDA_v5
#   BINARY      path to binary, e.g. cuda/seam_carve_v5
#   IMAGE       input PNG, e.g. data/4k/forest_pano_4k_3840x2160.png
#   SEAMS       number of seams, e.g. 500
#   REPS        number of timed repetitions (default 10)
#   OUT_CSV     CSV file to append results to
#   GPU_ARCH    sm_70 or sm_80 (for labelling)
#
# Usage:
#   sbatch --export=ALL,IMPL=CUDA_v5,BINARY=cuda/seam_carve_v5,\
#          IMAGE=data/4k/forest_pano_4k_3840x2160.png,SEAMS=500,\
#          REPS=10,OUT_CSV=results/gpu.csv,GPU_ARCH=sm_70 bench/sbatch_gpu.sh
#
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1
#SBATCH -A ACD115083
#SBATCH -t 00:30:00
#SBATCH -o slurm-bench-%j.out
#SBATCH -e slurm-bench-%j.err

set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$0")/.."
fi

module load cuda 2>/dev/null || true

REPS=${REPS:-10}
WARMUP=2
TMP_OUT=$(mktemp --suffix=.png)
TIMES_FILE=$(mktemp)

# GPU info
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr -s ' ' '_' || echo "unknown")
GPU_ARCH=${GPU_ARCH:-unknown}

echo "=== GPU Benchmark ==="
echo "  IMPL    : $IMPL"
echo "  BINARY  : $BINARY"
echo "  IMAGE   : $IMAGE"
echo "  SEAMS   : $SEAMS"
echo "  REPS    : $REPS  (+ $WARMUP warmup)"
echo "  GPU     : $GPU_NAME ($GPU_ARCH)"
echo "  OUT_CSV : $OUT_CSV"
echo ""

IMG_W=$(identify -format "%w" "$IMAGE" 2>/dev/null || echo 0)
IMG_H=$(identify -format "%h" "$IMAGE" 2>/dev/null || echo 0)
MEGAPIX=$(python3 -c "print(round($IMG_W*$IMG_H/1e6,3))" 2>/dev/null || echo 0)

# Warm up
for i in $(seq 1 $WARMUP); do
    "$BINARY" "$IMAGE" "$SEAMS" "$TMP_OUT" > /dev/null 2>&1 || true
done

# Timed runs — extract "carving time: X.XXX ms" from stdout
for i in $(seq 1 $REPS); do
    MS=$("$BINARY" "$IMAGE" "$SEAMS" "$TMP_OUT" 2>/dev/null \
         | grep -oP 'carving time:\s*\K[\d.]+' | head -1)
    echo "${MS:-0}" >> "$TIMES_FILE"
    echo "  rep $i / $REPS : ${MS} ms"
done

# Compute mean and std via Python (stdlib only, no numpy needed)
python3 - "$TIMES_FILE" "$REPS" <<'PYEOF'
import sys, math, os
times_file, reps = sys.argv[1], int(sys.argv[2])
vals = [float(l.strip()) for l in open(times_file) if l.strip() and float(l.strip()) > 0]
if not vals:
    print("mean=0 std=0"); sys.exit(1)
mean = sum(vals) / len(vals)
variance = sum((x - mean)**2 for x in vals) / len(vals)
std = math.sqrt(variance)
ms_per_seam = mean / int(os.environ.get("SEAMS", 1))
mp = float(os.environ.get("MEGAPIX", 0))
seams = int(os.environ.get("SEAMS", 1))
throughput = (mp * seams) / (mean / 1000.0) if mean > 0 else 0
print("mean={:.4f} std={:.4f} ms_per_seam={:.5f} throughput={:.2f}".format(
    mean, std, ms_per_seam, throughput))
PYEOF

STATS=$(python3 - "$TIMES_FILE" <<'PYEOF'
import sys, math
vals = [float(l.strip()) for l in open(sys.argv[1]) if l.strip() and float(l.strip()) > 0]
if not vals:
    print("0,0,0,0"); sys.exit(0)
mean = sum(vals) / len(vals)
std = math.sqrt(sum((x - mean)**2 for x in vals) / len(vals))
import os
seams = int(os.environ.get("SEAMS", 1))
mp = float(os.environ.get("MEGAPIX", 0))
ms_per_seam = mean / seams
throughput = (mp * seams) / (mean / 1000.0) if mean > 0 else 0
print("{:.4f},{:.4f},{:.5f},{:.2f}".format(mean, std, ms_per_seam, throughput))
PYEOF
)

MEAN=$(echo "$STATS" | cut -d, -f1)
STD=$(echo  "$STATS" | cut -d, -f2)
MPS=$(echo  "$STATS" | cut -d, -f3)
THR=$(echo  "$STATS" | cut -d, -f4)

# Write CSV row (create header if file doesn't exist)
mkdir -p "$(dirname "$OUT_CSV")"
if [ ! -f "$OUT_CSV" ]; then
    echo "impl,device,arch,W,H,megapixels,seams,transfer_mode,n_reps,mean_ms,std_ms,ms_per_seam,throughput_mpix_s" > "$OUT_CSV"
fi

TRANSFER=${TRANSFER_MODE:-resident}
echo "${IMPL},${GPU_NAME},${GPU_ARCH},${IMG_W},${IMG_H},${MEGAPIX},${SEAMS},${TRANSFER},${REPS},${MEAN},${STD},${MPS},${THR}" >> "$OUT_CSV"

echo ""
echo "=== DONE: mean=${MEAN} ms  std=${STD}  ms/seam=${MPS}  throughput=${THR} Mpix/s ==="

rm -f "$TMP_OUT" "$TIMES_FILE"
