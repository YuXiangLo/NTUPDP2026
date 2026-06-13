#!/bin/bash
# sbatch_verify_4k.sh — Correctness check: wide vs SEQ reference at 4K.
# SEQ C implementation has no width limit; compare its output to wide's.
#
# Submit from repo root: sbatch bench/sbatch_verify_4k.sh
#
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1
#SBATCH -A ACD115083
#SBATCH -t 00:20:00
#SBATCH -o slurm-verify4k-%j.out
#SBATCH -e slurm-verify4k-%j.err

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then cd "$SLURM_SUBMIT_DIR"; else cd "$(dirname "$0")/.."; fi

module load cuda

echo "=== 4K Correctness Verification — $(date) ==="
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

PASS=0; FAIL=0

verify_pair() {
    local label="$1" gpu_bin="$2" cpu_bin="$3" image="$4" seams="$5"
    local WORK; WORK=$(mktemp -d)
    trap "rm -rf $WORK" RETURN

    printf "[%-20s] %-45s %s seams\n" "$label" "$image" "$seams"

    # GPU output
    local gpu_out="$WORK/gpu.png"
    local gpu_t
    gpu_t=$("$gpu_bin" "$image" "$seams" "$gpu_out" 2>/dev/null \
        | grep -oP 'carving time:\s*\K[\d.]+' | head -1)
    printf "  GPU (%s): %s ms\n" "$gpu_bin" "${gpu_t:-FAILED}"

    # CPU reference (SEQ — single threaded, exact)
    local cpu_out="$WORK/cpu.png"
    local cpu_t
    cpu_t=$("$cpu_bin" "$image" "$seams" "$cpu_out" 2>/dev/null \
        | grep -oP 'carving time:\s*\K[\d.]+' | head -1)
    printf "  CPU (SEQ): %s ms\n" "${cpu_t:-FAILED}"

    if [ ! -f "$gpu_out" ] || [ ! -f "$cpu_out" ]; then
        echo "  RESULT: FAILED (missing output)"; FAIL=$((FAIL+1)); return
    fi

    if diff -q "$gpu_out" "$cpu_out" > /dev/null 2>&1; then
        echo "  RESULT: BIT_EXACT ✓"; PASS=$((PASS+1))
    else
        PSNR=$(compare -metric PSNR "$cpu_out" "$gpu_out" /dev/null 2>&1 || true)
        PAE=$(compare -metric PAE  "$cpu_out" "$gpu_out" /dev/null 2>&1 || true)
        echo "  RESULT: DIFFER  PSNR=${PSNR}  PAE=${PAE}"
        # Pass if PSNR >= 50 dB (spec §4)
        PSNR_VAL=$(echo "$PSNR" | grep -oP '[\d.]+' | head -1)
        if [ -n "$PSNR_VAL" ] && awk "BEGIN{exit !($PSNR_VAL >= 50)}"; then
            echo "  -> WITHIN_TOL (PSNR >= 50 dB) ✓"; PASS=$((PASS+1))
        else
            echo "  -> FAIL (below threshold)"; FAIL=$((FAIL+1))
        fi
    fi
    echo ""
}

# 4K desert (3840×2160)
verify_pair "wide/4k/desert" \
    cuda/seam_carve_wide \
    openmp/seam_carve_seq \
    data/4k/desert_mesa_4k_3840x2160.png 20

# 4K forest (3840×2160)
verify_pair "wide/4k/forest" \
    cuda/seam_carve_wide \
    openmp/seam_carve_seq \
    data/4k/forest_pano_4k_3840x2160.png 20

# v2 at 8K (7680×4320, tiny seam count — just correctness)
verify_pair "v2/8k/desert" \
    cuda/seam_carve \
    openmp/seam_carve_seq \
    data/8k/desert_mesa_8k_7680x4320.png 5

echo "=== FINAL: $PASS PASS / $FAIL FAIL ==="
