#!/bin/bash
# sbatch_smoke_4k.sh — Test seam_carve_wide on 4K image (v5 would reject it).
# Also test v6 on 1080p.
#
# Submit from repo root: sbatch bench/sbatch_smoke_4k.sh
#
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1
#SBATCH -A ACD115083
#SBATCH -t 00:10:00
#SBATCH -o slurm-smoke4k-%j.out
#SBATCH -e slurm-smoke4k-%j.err

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then cd "$SLURM_SUBMIT_DIR"; else cd "$(dirname "$0")/.."; fi

module load cuda

echo "=== Extended Smoke Test — $(date) ==="
echo "  GPU : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

PASS=0; FAIL=0

run_test() {
    local label="$1" binary="$2" image="$3" seams="$4" ref_bin="$5"
    local WORK; WORK=$(mktemp -d)

    printf "[%-16s] %-40s  %s seams  " "$label" "$image" "$seams"

    if [ ! -f "$binary" ]; then
        echo "SKIP — binary missing"; FAIL=$((FAIL+1)); rm -rf "$WORK"; return
    fi
    if [ ! -f "$image" ]; then
        echo "SKIP — image missing"; FAIL=$((FAIL+1)); rm -rf "$WORK"; return
    fi

    local out="$WORK/out.png"
    local stdout
    stdout=$("$binary" "$image" "$seams" "$out" 2>/dev/null) || true
    local t; t=$(echo "$stdout" | grep -oP 'carving time:\s*\K[\d.]+' | head -1); t=${t:-"?"}
    printf "%s ms  " "$t"

    if [ -z "$ref_bin" ] || [ ! -f "$ref_bin" ]; then
        echo "OK (no ref comparison)"
        PASS=$((PASS+1)); rm -rf "$WORK"; return
    fi

    # Run reference
    local ref="$WORK/ref.png"
    "$ref_bin" "$image" "$seams" "$ref" > /dev/null 2>&1 || true

    if [ ! -f "$out" ]; then
        echo "FAILED — no output"; FAIL=$((FAIL+1)); rm -rf "$WORK"; return
    fi

    if diff -q "$ref" "$out" > /dev/null 2>&1; then
        echo "BIT_EXACT ✓"; PASS=$((PASS+1))
    else
        PSNR=$(compare -metric PSNR "$ref" "$out" /dev/null 2>&1 || true)
        echo "DIFFER  PSNR=${PSNR}"; FAIL=$((FAIL+1))
    fi
    rm -rf "$WORK"
}

# 1080p: wide vs v5
run_test "wide/1080p"  cuda/seam_carve_wide \
    data/1080p/forest_pano_1080p_1920x1080.png  192  cuda/seam_carve_v5

# 1080p: v6 vs v5
run_test "v6/1080p"    cuda/seam_carve_v6 \
    data/1080p/forest_pano_1080p_1920x1080.png  192  cuda/seam_carve_v5

# 4K: wide only (v5 would error out at W=3840 > 2048)
run_test "wide/4k"     cuda/seam_carve_wide \
    data/4k/desert_mesa_4k_3840x2160.png        384  ""

# 4K: naive (measures transfer overhead at 4K — only supported up to W=2048; expect rejection)
# Note: naive has W<=2048 limit too, so this is expected to fail cleanly
run_test "naive/4k(exp-fail)" cuda/seam_carve_naive \
    data/4k/desert_mesa_4k_3840x2160.png        384  ""

echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "=== ALL $PASS PASS ==="
else
    echo "=== $PASS PASS / $FAIL FAIL ==="
fi
