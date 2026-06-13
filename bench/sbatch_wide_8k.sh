#!/bin/bash
# sbatch_wide_8k.sh — v2/v6/wide comparison across ctrl/1080p/4k/8k,
# plus correct OMP numbers (seam_carve_cpu_omp with OMP_NUM_THREADS=4).
#
# Purpose:
#   v6 fails at 8K (register overflow at CPT=8×1024 threads).
#   seam_carve_wide is the register-safe fallback (grid-stride, fixed regs).
#   This job gives us (a) the 8K optimized data point using wide, and
#   (b) corrected OMP baselines (previous sbatch_8k.sh didn't request -c).
#
#   sbatch bench/sbatch_wide_8k.sh
#   squeue -u $USER
#   cat slurm-wide8k-<jobid>.out
#
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus-per-node=1
#SBATCH -A ACD115083
#SBATCH -t 00:50:00
#SBATCH -o slurm-wide8k-%j.out
#SBATCH -e slurm-wide8k-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load cuda 2>/dev/null || true

# OMP: 4 threads, bind to cores (corrects sbatch_8k.sh which got 1-thread OMP)
export OMP_NUM_THREADS=4
export OMP_PROC_BIND=close
export OMP_PLACES=cores

mkdir -p results
IMGS="data/ctrl/desert_mesa_ctrl_960x540.png \
      data/1080p/desert_mesa_1080p_1920x1080.png \
      data/4k/desert_mesa_4k_3840x2160.png \
      data/8k/desert_mesa_8k_7680x4320.png"

for i in $IMGS; do
    [ -f "$i" ] || { echo "MISSING IMAGE: $i" >&2; exit 1; }
done

cd cuda
make >/dev/null 2>&1 || { echo "BUILD FAILED" >&2; exit 1; }

PIMGS=""
for i in $IMGS; do PIMGS="$PIMGS ../$i"; done

echo "######## baseline_compare: cpu / cpu_omp(4t) / v0 / v2 / v6 / wide ########"
echo "### OMP_NUM_THREADS=$OMP_NUM_THREADS  (fixed vs sbatch_8k.sh) ###"
echo ""

# v6 will show – at 8K (register overflow); wide succeeds at all sizes.
# Speedup columns at end use v6 by default; check wide column manually for 8K.
SEAMS=10 RUNS=3 \
    BINS="seam_carve_cpu seam_carve_cpu_omp seam_carve_v0 seam_carve seam_carve_v6 seam_carve_wide" \
    bash baseline_compare.sh $PIMGS 2>&1 | tee ../results/wide_8k.md

echo ""
echo "######## bench sweep: v2/v6/wide, 5/10/20%, ctrl->1080p->4k->8k ########"
PCTS="5 10 20" RUNS=3 \
    BINS="seam_carve seam_carve_v6 seam_carve_wide" \
    OUT=../results/sweep_wide_8k.csv \
    bash bench.sh $PIMGS 2>&1 | tee ../results/sweep_wide_8k.log

echo ""
echo "######## DONE — results/wide_8k.md  results/sweep_wide_8k.csv ########"
