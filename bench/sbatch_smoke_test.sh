#!/bin/bash
# sbatch_smoke_test.sh — Run correctness smoke test for new CUDA variants.
# Tests seam_carve_v6, seam_carve_wide, and seam_carve_naive against v5 output
# on the ctrl image (960x540), 20 seams.
#
# Submit from repo root:
#   sbatch bench/sbatch_smoke_test.sh
#
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1
#SBATCH -A ACD115083
#SBATCH -t 00:05:00
#SBATCH -o slurm-smoke-%j.out
#SBATCH -e slurm-smoke-%j.err

# Use $SLURM_SUBMIT_DIR when available (reliable inside sbatch);
# fall back to dirname-based path for direct execution.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$0")/.."
fi

module load cuda

IMAGE=data/ctrl/broadway_tower_ctrl_960x540.png
SEAMS=20
WORK=$(mktemp -d)
trap "rm -rf $WORK" EXIT

echo "=== CUDA Smoke Test — $(date) ==="
echo "  image : $IMAGE"
echo "  seams : $SEAMS"
echo "  GPU   : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

PASS=0
FAIL=0

run_and_check() {
    local label="$1"
    local binary="$2"
    local out="$WORK/${label}.png"
    local ref="$WORK/v5.png"

    if [ ! -f "$binary" ]; then
        printf "[%-12s] SKIP — binary not found: %s\n" "$label" "$binary"
        FAIL=$((FAIL+1))
        return
    fi

    # Run binary; capture stdout, allow non-zero exit
    local stdout
    stdout=$("$binary" "$IMAGE" "$SEAMS" "$out" 2>/dev/null) || true
    local t
    t=$(echo "$stdout" | grep -oP 'carving time:\s*\K[\d.]+' | head -1)
    t=${t:-"?"}

    printf "[%-12s] %s ms  " "$label" "$t"

    if [ "$label" = "v5" ]; then
        echo "(reference)"
        PASS=$((PASS+1))
        return
    fi

    if [ ! -f "$out" ]; then
        echo "FAILED — no output file"
        FAIL=$((FAIL+1))
        return
    fi

    # Bit-exact check first
    if diff -q "$ref" "$out" > /dev/null 2>&1; then
        echo "BIT_EXACT ✓"
        PASS=$((PASS+1))
    else
        # PSNR via ImageMagick
        PSNR=$(compare -metric PSNR "$ref" "$out" /dev/null 2>&1 || true)
        echo "DIFFER  PSNR=${PSNR}"
        FAIL=$((FAIL+1))
    fi
}

run_and_check "v5"    "cuda/seam_carve_v5"
run_and_check "v6"    "cuda/seam_carve_v6"
run_and_check "wide"  "cuda/seam_carve_wide"
run_and_check "naive" "cuda/seam_carve_naive"

echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "=== RESULT: ALL $PASS PASS ==="
else
    echo "=== RESULT: $PASS PASS / $FAIL FAIL ==="
fi
