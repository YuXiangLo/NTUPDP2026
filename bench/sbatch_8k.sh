#!/bin/bash
# sbatch_8k.sh — scene-consistent scaling sweep (desert_mesa) ctrl->1080p->4k->8k.
# One GPU allocation runs both paper protocols on the same scene at four
# resolutions, so we (a) confirm v6 runs 8K, (b) get the 8K end-to-end row in
# Table II's format, and (c) get a scene-controlled scaling series for Fig. 5.
#
#   sbatch bench/sbatch_8k.sh
#
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1
#SBATCH -A ACD115083
#SBATCH -t 00:40:00
#SBATCH -o slurm-8k-%j.out
#SBATCH -e slurm-8k-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load cuda 2>/dev/null || true

mkdir -p results
IMGS="data/ctrl/desert_mesa_ctrl_960x540.png \
      data/1080p/desert_mesa_1080p_1920x1080.png \
      data/4k/desert_mesa_4k_3840x2160.png \
      data/8k/desert_mesa_8k_7680x4320.png"

# Verify inputs exist up front so we fail fast instead of mid-run.
for i in $IMGS; do
    [ -f "$i" ] || { echo "MISSING IMAGE: $i" >&2; exit 1; }
done

cd cuda
make >/dev/null 2>&1 || { echo "BUILD FAILED" >&2; exit 1; }

# Scripts run from cuda/, so prefix image paths with ../
PIMGS=""
for i in $IMGS; do PIMGS="$PIMGS ../$i"; done

echo "######## A. baseline_compare — end-to-end, 10 seams, scene=desert_mesa ########"
SEAMS=10 RUNS=3 bash baseline_compare.sh $PIMGS 2>&1 | tee ../results/baseline_desertmesa.md

echo ""
echo "######## B. bench sweep — v2/v5/v6, 5/10/20% width, scene=desert_mesa ########"
PCTS="5 10 20" RUNS=3 BINS="seam_carve seam_carve_v5 seam_carve_v6" \
    OUT=../results/sweep_desertmesa.csv bash bench.sh $PIMGS 2>&1 \
    | tee ../results/sweep_desertmesa.log

echo ""
echo "######## DONE — outputs in results/baseline_desertmesa.md and results/sweep_desertmesa.csv ########"
