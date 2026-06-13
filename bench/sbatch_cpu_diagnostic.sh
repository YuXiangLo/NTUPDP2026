#!/bin/bash
# Quick diagnostic: test CPU at multiple thread counts (1, 4, 8, 16, 32, 40)
# Usage: sbatch bench/sbatch_cpu_diagnostic.sh

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus-per-node=1
#SBATCH -A ACD115083
#SBATCH -t 00:20:00
#SBATCH -o slurm-cpu-diag-%j.out
#SBATCH -e slurm-cpu-diag-%j.err

cd "$SLURM_SUBMIT_DIR" || exit 1

BINARY="/home/u2713124/NTUPDP2026/cuda/seam_carve_cpu_omp"
IMAGE="/home/u2713124/NTUPDP2026/data/ctrl/broadway_tower_ctrl_960x540.png"
SEAMS=50

echo "=== CPU Threading Diagnostic ==="
echo "Image: $IMAGE"
echo "Seams: $SEAMS"
echo "Node: $(hostname)"
echo "Available cores: $(nproc)"
echo ""
echo "Threads | ms/seam | Total Time"
echo "--------|---------|----------"

for THREADS in 1 2 4 8 16 32 40; do
    export OMP_NUM_THREADS=$THREADS
    export OMP_PROC_BIND=close
    export OMP_PLACES=cores
    
    OUTPUT=$("$BINARY" "$IMAGE" "$SEAMS" /tmp/diag.png 2>&1 | grep "carving time")
    MS_PER_SEAM=$(echo "$OUTPUT" | grep -oP '\(\K[0-9.]+(?= ms/seam)')
    TOTAL_MS=$(echo "$OUTPUT" | grep -oP 'time: \K[0-9.]+(?= ms)')
    
    printf "%7d | %7s | %s\n" "$THREADS" "$MS_PER_SEAM" "$TOTAL_MS ms"
done

echo ""
echo "Done. Results show scaling behavior on actual compute node."
