#!/bin/bash
# sbatch_strip.sh — tiled DP wall-time vs active-strip count (latency-bound probe).
# Decides whether a cone-restricted incremental DP can speed up single-seam on V100.
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus-per-node=1
#SBATCH -A ACD115083
#SBATCH -t 00:10:00
#SBATCH -o slurm-strip-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load cuda 2>/dev/null || true
cd cuda && make strip_microbench >/dev/null 2>&1 || { echo BUILD FAIL; exit 1; }
echo "=== 8K ==="; ./strip_microbench ../data/8k/desert_mesa_8k_7680x4320.png
echo "=== 4K ==="; ./strip_microbench ../data/4k/desert_mesa_4k_3840x2160.png
