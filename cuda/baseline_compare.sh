#!/usr/bin/env bash
# baseline_compare.sh - end-to-end speedup of our optimized CUDA vs the NAIVE
# CUDA baseline, all in one toolchain (no torch/python; the server's python is
# 3.6 and can't run benchmark.py).
#
# seam_carve_v0 is the naive baseline: the DP is one kernel launch per row
# (H launches/seam, full global round-trips, no fusion) -- it mirrors the
# original PyTorch CUDA_v0's per-row launches. v2 fuses the whole DP into one
# kernel; v5/v6 then optimize that. The v6-vs-v0 column is the real headline:
# the total win over the naive starting point.
#
# Uses a FIXED, modest seam count (default 10) because v0 is so slow that the
# 5/10/20%-of-width sweep in bench.sh would take many minutes on the big image.
#
#   module load cuda
#   make
#   srun -N 1 -n 1 --gpus-per-node 1 -A ACD115083 -t 30 \
#       bash baseline_compare.sh ../Broadway_tower_edit.jpg ../input1.jpg ../input.jpg
#
# Tunables:  SEAMS=10  RUNS=3  BINS="seam_carve_v0 seam_carve seam_carve_v5 seam_carve_v6"
set -u

SEAMS="${SEAMS:-10}"
RUNS="${RUNS:-3}"
BINS="${BINS:-seam_carve_v0 seam_carve seam_carve_v5 seam_carve_v6}"
BINDIR="${BINDIR:-.}"
TMP="${TMPDIR:-/tmp}/seam_base_out.png"

if [ "$#" -lt 1 ]; then
    echo "usage: bash baseline_compare.sh <image1> [image2 ...]" >&2
    exit 1
fi

ver_label() {
    case "$1" in
        seam_carve_v0) echo "v0(naive)" ;;
        seam_carve)    echo "v2" ;;
        seam_carve_v5) echo "v5" ;;
        seam_carve_v6) echo "v6" ;;
        *)             echo "$1" ;;
    esac
}

for b in $BINS; do
    [ -x "$BINDIR/$b" ] || { echo "missing binary: $BINDIR/$b (run make)" >&2; exit 1; }
done

# best-of-RUNS ms/seam for one binary on one image, or "" if it fails.
bestbin() {
    local bin="$1" img="$2" best="" t
    for r in $(seq 1 "$RUNS"); do
        t=$("$BINDIR/$bin" "$img" "$SEAMS" "$TMP" 2>/dev/null \
            | sed -nE 's/.*\(([0-9.]+) ms\/seam\).*/\1/p' | head -1)
        [ -z "$t" ] && continue
        if [ -z "$best" ] || awk "BEGIN{exit !($t < $best)}"; then best="$t"; fi
    done
    echo "$best"
}

# header
hdr="| image | res |"
for b in $BINS; do hdr="$hdr $(ver_label "$b") |"; done
hdr="$hdr v6 vs v0 |"
echo
echo "$hdr"
sep="|---|---|"; for b in $BINS; do sep="$sep---|"; done; sep="$sep---|"
echo "$sep"

for img in "$@"; do
    [ -f "$img" ] || { echo "!! missing $img" >&2; continue; }
    res=$("$BINDIR/seam_carve" "$img" 1 "$TMP" 2>&1 | sed -nE 's/.*: ([0-9]+x[0-9]+),.*/\1/p' | head -1)

    row="| $img | ${res:-?} |"
    v0_ms=""; v6_ms=""
    for b in $BINS; do
        ms=$(bestbin "$b" "$img")
        [ "$b" = "seam_carve_v0" ] && v0_ms="$ms"
        [ "$b" = "seam_carve_v6" ] && v6_ms="$ms"
        row="$row ${ms:-–} |"
    done
    sp=$(awk -v a="$v0_ms" -v b="$v6_ms" 'BEGIN{ if(a=="" || b=="")exit; printf "%.1fx", a/b }')
    row="$row ${sp:-–} |"
    echo "$row"
done

echo
echo "(ms/seam, best-of-$RUNS, SEAMS=$SEAMS; last column = v0/v6)"
