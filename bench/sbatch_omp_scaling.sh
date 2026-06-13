#!/bin/bash
# sbatch_omp_scaling.sh — OpenMP thread-scaling sweep for paper Table III / Fig.
#
# Tests seq(1t) / omp_v1 / omp_v2 at 1, 2, 4 threads on ctrl and 1080p images.
# Writes results/omp_scaling.md (markdown table) and results/omp_scaling.csv.
#
# Node limit is 4 CPUs; OMP_NUM_THREADS max = 4.
#
#   sbatch bench/sbatch_omp_scaling.sh
#
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus-per-node=1
#SBATCH -A ACD115083
#SBATCH -t 00:30:00
#SBATCH -o slurm-omp-%j.out
#SBATCH -e slurm-omp-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

export OMP_PROC_BIND=close
export OMP_PLACES=cores

mkdir -p results

RUNS=5
SEAMS=20

IMGS="data/ctrl/broadway_tower_ctrl_960x540.png \
      data/1080p/forest_pano_1080p_1920x1080.png"

for i in $IMGS; do
    [ -f "$i" ] || { echo "MISSING IMAGE: $i" >&2; exit 1; }
done

cd openmp
make >/dev/null 2>&1 || { echo "BUILD FAILED" >&2; exit 1; }

TMP=/tmp/omp_out.png

# Run binary best-of-N, return ms/seam
run_best() {
    local bin="$1" img="$2" threads="$3" best=""
    export OMP_NUM_THREADS=$threads
    for r in $(seq 1 $RUNS); do
        t=$("./$bin" "../$img" $SEAMS "$TMP" 2>/dev/null \
            | grep -oP 'carving time:\s*\K[\d.]+' | head -1)
        [ -z "$t" ] && continue
        if [ -z "$best" ] || awk "BEGIN{exit !($t < $best)}"; then best="$t"; fi
    done
    # convert total ms -> ms/seam
    if [ -n "$best" ]; then
        awk "BEGIN{printf \"%.4f\", $best / $SEAMS}"
    else
        echo "–"
    fi
}

OUT_MD="../results/omp_scaling.md"
OUT_CSV="../results/omp_scaling.csv"

echo "# OpenMP Thread-Scaling Results" > "$OUT_MD"
echo "SEAMS=$SEAMS, RUNS=$RUNS (best-of), broadway_tower/forest_pano ctrl+1080p" >> "$OUT_MD"
echo "" >> "$OUT_MD"

echo "impl,image,threads,ms_per_seam" > "$OUT_CSV"

echo "=== OpenMP Scaling Sweep: $(date) ==="
echo "SEAMS=$SEAMS  RUNS=$RUNS  node=$(hostname)"
echo ""

for img in $IMGS; do
    RES=$(../cuda/seam_carve "../$img" 1 "$TMP" 2>&1 | grep -oP '\d+x\d+' | head -1 || echo "?")
    echo "| impl | threads | ${img##data/*/} ($RES) |" | tee -a "$OUT_MD"
    echo "|---|---|---|" | tee -a "$OUT_MD"

    for BIN in seam_carve_seq seam_carve_omp_v1 seam_carve_omp_v2; do
        [ -x "./$BIN" ] || { echo "missing: $BIN" >&2; continue; }
        if [ "$BIN" = "seam_carve_seq" ]; then
            ms=$(run_best "$BIN" "$img" 1)
            echo "| seq | 1 | $ms ms/seam |" | tee -a "$OUT_MD"
            echo "seq,$img,1,$ms" >> "$OUT_CSV"
        else
            for T in 1 2 4; do
                ms=$(run_best "$BIN" "$img" "$T")
                echo "| $BIN | $T | $ms ms/seam |" | tee -a "$OUT_MD"
                echo "$BIN,$img,$T,$ms" >> "$OUT_CSV"
            done
        fi
    done
    echo "" | tee -a "$OUT_MD"
done

echo ""
echo "=== DONE — results/omp_scaling.md  results/omp_scaling.csv ==="
