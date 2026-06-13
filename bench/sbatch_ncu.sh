#!/bin/bash
# sbatch_ncu.sh — Nsight Compute profiling for roofline (F5 + T3 data).
# Profiles v5, v6, wide at ctrl (960×540) and 1080p (1920×1080).
# Writes one .ncu-rep file per variant; ncu_table.py --roofline reads them.
#
# Submit from repo root: sbatch bench/sbatch_ncu.sh
#
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1
#SBATCH -A ACD115083
#SBATCH -t 00:30:00
#SBATCH -o slurm-ncu-%j.out
#SBATCH -e slurm-ncu-%j.err

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then cd "$SLURM_SUBMIT_DIR"; else cd "$(dirname "$0")/.."; fi

module load cuda

mkdir -p results/ncu

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
echo "=== NCU Profiling — $(date) ==="
echo "  GPU: $GPU"
echo ""

# Metrics for arithmetic intensity + roofline
METRICS="sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"

profile_one() {
    local label="$1" binary="$2" image="$3" seams="$4"
    local out="results/ncu/${label}"
    echo "--- Profiling $label on $image ($seams seams) ---"
    ncu --set full \
        --metrics "$METRICS" \
        --target-processes all \
        -o "$out" \
        "$binary" "$image" "$seams" /dev/null 2>&1 | tail -5
    echo "  -> $out.ncu-rep"
}

# v5 (baseline)
profile_one "v5_ctrl"   cuda/seam_carve_v5  data/ctrl/broadway_tower_ctrl_960x540.png   10
profile_one "v5_1080p"  cuda/seam_carve_v5  data/1080p/forest_pano_1080p_1920x1080.png  10

# v6 (fused kernel — new contribution)
profile_one "v6_ctrl"   cuda/seam_carve_v6  data/ctrl/broadway_tower_ctrl_960x540.png   10
profile_one "v6_1080p"  cuda/seam_carve_v6  data/1080p/forest_pano_1080p_1920x1080.png  10

# wide (4K-capable)
profile_one "wide_ctrl"  cuda/seam_carve_wide data/ctrl/broadway_tower_ctrl_960x540.png   10
profile_one "wide_4k"    cuda/seam_carve_wide data/4k/desert_mesa_4k_3840x2160.png        10

echo ""
echo "=== Generating roofline tables ==="
for rep in results/ncu/*.ncu-rep; do
    label=$(basename "$rep" .ncu-rep)
    echo "--- $label ---"
    .venv/bin/python cuda/ncu_table.py --roofline "$rep" 2>/dev/null || \
        python3 cuda/ncu_table.py --roofline "$rep" 2>/dev/null || \
        echo "  (ncu_table.py failed — check manually)"
done

echo ""
echo "=== NCU profiling DONE ==="
