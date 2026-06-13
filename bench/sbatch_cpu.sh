#!/bin/bash
# sbatch_cpu.sh — Single CPU/OpenMP benchmark run, appends one aggregated row to CSV.
#
# Required env vars:
#   IMPL        binary tag, e.g. OMP_v2
#   BINARY      path to binary, e.g. openmp/seam_carve_omp_v2
#   IMAGE       input PNG
#   SEAMS       number of seams, e.g. 200
#   REPS        number of timed repetitions (default 10)
#   OMP_THREADS OMP_NUM_THREADS (default: all physical cores on node)
#   OUT_CSV     CSV file to append results to
#
# Usage:
#   sbatch --export=ALL,IMPL=OMP_v2,BINARY=openmp/seam_carve_omp_v2,\
#          IMAGE=data/1080p/broadway_tower_1080p_1920x1080.png,SEAMS=200,\
#          REPS=10,OMP_THREADS=40,OUT_CSV=results/cpu.csv bench/sbatch_cpu.sh
#
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
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

REPS=${REPS:-10}
WARMUP=2

# Detect physical core count; fall back to nproc
PHYS_CORES=$(lscpu 2>/dev/null | grep "^Core(s) per socket:" | awk '{print $4}')
SOCKETS=$(lscpu 2>/dev/null | grep "^Socket(s):" | awk '{print $2}')
if [ -n "$PHYS_CORES" ] && [ -n "$SOCKETS" ]; then
    DEFAULT_THREADS=$((PHYS_CORES * SOCKETS))
else
    DEFAULT_THREADS=$(nproc)
fi
OMP_THREADS=${OMP_THREADS:-$DEFAULT_THREADS}

export OMP_NUM_THREADS=$OMP_THREADS
export OMP_PROC_BIND=close
export OMP_PLACES=cores

TMP_OUT=$(mktemp --suffix=.png)
TIMES_FILE=$(mktemp)

echo "=== CPU Benchmark ==="
echo "  IMPL    : $IMPL"
echo "  BINARY  : $BINARY"
echo "  IMAGE   : $IMAGE"
echo "  SEAMS   : $SEAMS"
echo "  REPS    : $REPS  (+ $WARMUP warmup)"
echo "  THREADS : $OMP_THREADS"
echo "  OUT_CSV : $OUT_CSV"
echo "  NODE    : $(hostname)"
echo ""

IMG_W=$(identify -format "%w" "$IMAGE" 2>/dev/null || echo 0)
IMG_H=$(identify -format "%h" "$IMAGE" 2>/dev/null || echo 0)
MEGAPIX=$(python3 -c "print(round($IMG_W*$IMG_H/1e6,3))" 2>/dev/null || echo 0)
CPU_MODEL=$(lscpu 2>/dev/null | grep "^Model name:" | sed 's/.*: *//' | tr -s ' ' | head -1 || echo "unknown")

# Warm up
for i in $(seq 1 $WARMUP); do
    "$BINARY" "$IMAGE" "$SEAMS" "$TMP_OUT" > /dev/null 2>&1 || true
done

# Timed runs
for i in $(seq 1 $REPS); do
    MS=$("$BINARY" "$IMAGE" "$SEAMS" "$TMP_OUT" 2>/dev/null \
         | grep -oP 'carving time:\s*\K[\d.]+' | head -1)
    echo "${MS:-0}" >> "$TIMES_FILE"
    echo "  rep $i / $REPS : ${MS} ms"
done

STATS=$(python3 - "$TIMES_FILE" <<'PYEOF'
import sys, math, os
vals = [float(l.strip()) for l in open(sys.argv[1]) if l.strip() and float(l.strip()) > 0]
if not vals:
    print("0,0,0,0"); sys.exit(0)
mean = sum(vals) / len(vals)
std = math.sqrt(sum((x - mean)**2 for x in vals) / len(vals))
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

mkdir -p "$(dirname "$OUT_CSV")"
if [ ! -f "$OUT_CSV" ]; then
    echo "impl,device,threads,W,H,megapixels,seams,n_reps,mean_ms,std_ms,ms_per_seam,throughput_mpix_s" > "$OUT_CSV"
fi

# Escape commas in CPU model name
CPU_SAFE=$(echo "$CPU_MODEL" | tr ',' ';')
echo "${IMPL},${CPU_SAFE},${OMP_THREADS},${IMG_W},${IMG_H},${MEGAPIX},${SEAMS},${REPS},${MEAN},${STD},${MPS},${THR}" >> "$OUT_CSV"

echo ""
echo "=== DONE: mean=${MEAN} ms  std=${STD}  ms/seam=${MPS}  throughput=${THR} Mpix/s ==="

rm -f "$TMP_OUT" "$TIMES_FILE"
