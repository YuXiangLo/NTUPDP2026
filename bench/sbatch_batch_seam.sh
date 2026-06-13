#!/bin/bash
#SBATCH -N 1 -n 1 --gpus-per-node 1
#SBATCH -A ACD115083
#SBATCH -t 15
#SBATCH -o slurm-batch_seam-%j.out
#SBATCH -J batch_seam

# Benchmark: approximate batch multi-seam removal vs single-seam Tiled DP.
#
# Single mode: N seams = N × (gray + energy + DP + backtrack + remove)
# Batch  mode: N/K batches × (gray + energy + DP + K_backtracks + sort + batch_remove)
# Expected speedup: ~K× amortisation of the dominant DP cost.

set -e
ROOT=/home/u2713124/NTUPDP2026
cd "$ROOT"

BIN=$ROOT/cuda/seam_carve_batch
CTRL=$ROOT/data/ctrl/broadway_tower_ctrl_960x540.png
IMG4K=$ROOT/data/4k/desert_mesa_4k_3840x2160.png
IMG8K=$ROOT/data/8k/forest_pano_8k_7680x4320.png

K=60   # must match STRIP_K compile constant

echo "=========================================="
echo "Batch multi-seam experiment  (K=${K})"
echo "Binary: ${BIN}"
echo "Date: $(date)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "=========================================="

run_pair() {
    local label="$1"
    local img="$2"
    local n="$3"   # number of seams (should be multiple of K)

    if [ ! -f "$img" ]; then
        echo "SKIP $label — image not found: $img"
        return
    fi

    echo ""
    echo "--- ${label} (${n} seams) ---"
    echo "[single]"
    $BIN "$img" "$n" /dev/null single
    echo "[batch K=${K}]"
    $BIN "$img" "$n" /dev/null batch
}

# ctrl (960×540): remove 60 seams
run_pair "ctrl 960×540"       "$CTRL"  60

# 4K (3840×2160): remove 60 seams (1 batch) and 300 seams (5 batches)
run_pair "4K  3840×2160 n=60"  "$IMG4K"  60
run_pair "4K  3840×2160 n=300" "$IMG4K" 300

# 8K if available
run_pair "8K  7680×4320 n=60"  "$IMG8K"  60

echo ""
echo "Done."
