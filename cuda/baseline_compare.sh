#!/usr/bin/env bash
# baseline_compare.sh - end-to-end speedup of our standalone CUDA (v2/v6) vs the
# project's original PyTorch baselines (the naive per-row CUDA_v0 etc.).
#
# For each image it runs benchmark.py ONCE (which times PyTorch + every
# registered CUDA_v* version on the GPU) and then our v2/v6 binaries, all at the
# same seam count, and prints a markdown table with v6's speedup over each
# baseline. This is the "real" headline number: v6 vs the naive starting point,
# not just v6 vs our already-fused v2.
#
# Run on a GPU node (torch needs CUDA). benchmark.py lives one dir up.
#
#   module load cuda
#   make
#   srun -N 1 -n 1 --gpus-per-node 1 -A ACD115083 -t 30 \
#       bash baseline_compare.sh ../Broadway_tower_edit.jpg ../input1.jpg ../input.jpg
#
# Tunables:
#   SEAMS=10        seams removed (same for torch baselines and our binaries)
#   RUNS=3          benchmark.py runs (mean) ; our binaries use best-of RUNS too
#   PY=python       python launcher for benchmark.py
#   BENCH=../benchmark.py
set -u

SEAMS="${SEAMS:-10}"
RUNS="${RUNS:-3}"
PY="${PY:-python}"
BENCH="${BENCH:-../benchmark.py}"
TMP="${TMPDIR:-/tmp}/seam_base_out.png"

if [ "$#" -lt 1 ]; then
    echo "usage: bash baseline_compare.sh <image1> [image2 ...]" >&2
    exit 1
fi

# ms/seam from a benchmark.py "<label> mean=<sec>s ..." line, or empty if absent.
secs_to_msps() { awk -v s="$1" -v n="$SEAMS" 'BEGIN{ if(s=="")exit; printf "%.4f", s/n*1000 }'; }

printf '\n| image | res | PyTorch | CUDA_v0 | CUDA_v1 | v2 | v6 | v6 vs CUDA_v0 |\n'
printf '|---|---|---|---|---|---|---|---|\n'

for img in "$@"; do
    [ -f "$img" ] || { echo "!! missing $img" >&2; continue; }

    # One benchmark.py run -> PyTorch + all registered CUDA_v* versions.
    bout=$("$PY" "$BENCH" --image "$img" --seams "$SEAMS" --runs "$RUNS" \
                 --warmup 1 --torch-device cuda --skip-cpu 2>&1)
    py_s=$(echo "$bout"  | grep -E '^PyTorch'        | sed -nE 's/.*mean=([0-9.]+)s.*/\1/p' | head -1)
    v0_s=$(echo "$bout"  | grep 'CUDA (CUDA_v0)'      | sed -nE 's/.*mean=([0-9.]+)s.*/\1/p' | head -1)
    v1_s=$(echo "$bout"  | grep 'CUDA (CUDA_v1)'      | sed -nE 's/.*mean=([0-9.]+)s.*/\1/p' | head -1)
    if [ -z "$py_s$v0_s$v1_s" ]; then
        echo "!! benchmark.py produced no timings for $img:" >&2
        echo "$bout" | tail -8 >&2
    fi
    py_ms=$(secs_to_msps "$py_s")
    v0_ms=$(secs_to_msps "$v0_s")
    v1_ms=$(secs_to_msps "$v1_s")

    # Our standalone binaries (best-of RUNS), parse "(X ms/seam)".
    bestbin() {
        local bin="$1" best="" t
        for r in $(seq 1 "$RUNS"); do
            t=$(./"$bin" "$img" "$SEAMS" "$TMP" 2>/dev/null \
                | sed -nE 's/.*\(([0-9.]+) ms\/seam\).*/\1/p' | head -1)
            [ -z "$t" ] && continue
            if [ -z "$best" ] || awk "BEGIN{exit !($t < $best)}"; then best="$t"; fi
        done
        echo "$best"
    }
    res=$(./seam_carve "$img" 1 "$TMP" 2>&1 | sed -nE 's/.*: ([0-9]+x[0-9]+),.*/\1/p' | head -1)
    v2_ms=$(bestbin seam_carve)
    v6_ms=$(bestbin seam_carve_v6)

    sp=$(awk -v b="$v0_s" -v v="$v6_ms" -v n="$SEAMS" \
        'BEGIN{ if(b=="" || v=="")exit; printf "%.1fx", (b/n*1000)/v }')

    printf '| %s | %s | %s | %s | %s | %s | %s | %s |\n' \
        "$img" "${res:-?}" "${py_ms:-–}" "${v0_ms:-–}" "${v1_ms:-–}" \
        "${v2_ms:-–}" "${v6_ms:-–}" "${sp:-–}"
done

echo
echo "(all numbers are ms/seam; speedup column = CUDA_v0 / v6)"
