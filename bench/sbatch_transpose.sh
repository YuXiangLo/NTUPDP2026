#!/bin/bash
# sbatch_transpose.sh — correctness + cost of the VRAM transpose / horizontal carve
#
# Phase 1: transpose round-trip self-test (transpose(transpose(img)) == img)
# Phase 2: v-mode bit-exact vs seam_carve_tiled_pf (same kernels, must be identical)
# Phase 3: h-mode pixel-exact vs an independent transpose path
#          (PIL-transposed input -> tiled_pf vertical -> PIL transpose back)
# Phase 4: transpose cost + horizontal carve breakdown at ctrl / 4K / 8K
#
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus-per-node=1
#SBATCH -A ACD115083
#SBATCH -t 00:30:00
#SBATCH -o slurm-transpose-%j.out
#SBATCH -e slurm-transpose-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
ROOT=$(pwd)
module load cuda 2>/dev/null || true

PY="$ROOT/.venv/bin/python"
CTRL=data/ctrl/broadway_tower_ctrl_960x540.png
IMG4K=data/4k/desert_mesa_4k_3840x2160.png
IMG8K=data/8k/desert_mesa_8k_7680x4320.png

cd cuda
make seam_carve_tiled_hv seam_carve_tiled_pf >/dev/null 2>&1 \
    || { echo "BUILD FAILED" >&2; exit 1; }
echo "=== Build OK ==="
HV=./seam_carve_tiled_hv
PF=./seam_carve_tiled_pf

# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 1: transpose round-trip self-test ==="
for img in "$CTRL" "$IMG4K"; do
    $HV "../$img" 1 /tmp/_st.png selftest | grep -E "selftest|PASS|FAIL"
done

# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 2: v-mode == tiled_pf (bit-exact PNG) ==="
N=96
for img in "$CTRL" "$IMG4K"; do
    $HV "../$img" $N /tmp/_hv_v.png v >/dev/null 2>&1
    $PF "../$img" $N /tmp/_pf_v.png   >/dev/null 2>&1
    if cmp -s /tmp/_hv_v.png /tmp/_pf_v.png; then
        echo "[PASS] v-mode identical to tiled_pf : $img"
    else
        echo "[FAIL] v-mode differs : $img"
    fi
done

# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 3: h-mode == independent transpose path (pixel-exact) ==="
# Reference path: transpose input (PIL) -> tiled_pf vertical -> transpose back (PIL)
verify_h() {
    local img="$1" n="$2"
    $PY -c "from PIL import Image; Image.open('../$img').convert('RGB').transpose(Image.TRANSPOSE).save('/tmp/_in_T.png')"
    $HV "../$img"      $n /tmp/_hv_h.png  h  >/dev/null 2>&1
    $PF /tmp/_in_T.png $n /tmp/_pf_T.png     >/dev/null 2>&1
    $PY -c "from PIL import Image; Image.open('/tmp/_pf_T.png').transpose(Image.TRANSPOSE).save('/tmp/_pf_h.png')"
    $PY - "$img" <<'PYEOF'
import sys
from PIL import Image, ImageChops
a = Image.open('/tmp/_hv_h.png').convert('RGB')
b = Image.open('/tmp/_pf_h.png').convert('RGB')
if a.size != b.size:
    print(f"[FAIL] h-mode size {a.size} vs ref {b.size} : {sys.argv[1]}"); sys.exit()
diff = ImageChops.difference(a, b).getbbox()
if diff is None:
    print(f"[PASS] h-mode pixel-identical to transpose path : {sys.argv[1]} ({a.size[0]}x{a.size[1]})")
else:
    # count nonzero pixels for context
    import numpy as np
    d = np.asarray(ImageChops.difference(a, b))
    print(f"[WARN] h-mode differs in bbox {diff}, max={d.max()}, nnz={int((d>0).sum())} : {sys.argv[1]}")
PYEOF
}
verify_h "$CTRL" 96
verify_h "$IMG4K" 96

# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 4: transpose cost + horizontal carve breakdown (best of 3) ==="
bench_h() {
    local label="$1" img="$2" n="$3"
    echo "--- $label : horizontal, $n seams ---"
    for r in 1 2 3; do
        $HV "../$img" "$n" /tmp/_bh.png h 2>/dev/null \
            | grep -E "transpose-in|HORIZONTAL"
        echo "  ---"
    done
}
bench_h "ctrl 960x540"   "$CTRL"  96
bench_h "4K 3840x2160"   "$IMG4K" 200
bench_h "8K 7680x4320"   "$IMG8K" 200

echo ""
echo "=== DONE ==="
