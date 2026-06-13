#!/bin/bash
# sbatch_tiled_sweep.sh — K/T sweep + prefetch comparison + Fig.3 ctrl/1080p data
#
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus-per-node=1
#SBATCH -A ACD115083
#SBATCH -t 01:30:00
#SBATCH -o slurm-sweep-%j.out
#SBATCH -e slurm-sweep-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load cuda 2>/dev/null || true

mkdir -p results
cd cuda   # all paths below are relative to cuda/

make stb_impl.o seam_carve_v6 seam_carve_wide seam_carve_cpu >/dev/null 2>&1 \
    || { echo "BASE BUILD FAILED" >&2; exit 1; }
echo "=== Tiled K/T Sweep: $(date) === node=$(hostname)"

TMP=/tmp/sw_out.png
RUNS=3

# best_ms BIN IMG_PATH_FROM_CUDA_DIR N_SEAMS
# img path must be relative to cuda/ (e.g. ../data/4k/...)
best_ms() {
    local bin="$1" img="$2" n="$3" best=""
    for r in $(seq 1 $RUNS); do
        local t
        t=$("./$bin" "$img" "$n" "$TMP" 2>/dev/null \
            | grep -oP 'carving time:\s*\K[\d.]+' | head -1)
        [ -z "$t" ] && continue
        local ms; ms=$(awk "BEGIN{printf \"%.4f\", $t / $n}")
        if [ -z "$best" ] || awk "BEGIN{exit !($ms < $best)}"; then best="$ms"; fi
    done
    echo "${best:--}"
}

# build_tiled K T — produces tiled_KK_TT and tiled_pf_KK_TT in cuda/
build_tiled() {
    local K="$1" T="$2" ok=1
    nvcc -O3 -std=c++14 -arch=sm_70 -DSTRIP_K=$K -DTILE_T=$T -DNT_TILE=256 \
         seam_carve_tiled.cu stb_impl.o -o tiled_K${K}_T${T} 2>/tmp/nvcc_err.txt \
         || { echo "  BUILD FAILED tiled K=$K T=$T: $(cat /tmp/nvcc_err.txt)"; ok=0; }
    nvcc -O3 -std=c++14 -arch=sm_70 -DSTRIP_K=$K -DTILE_T=$T -DNT_TILE=256 \
         seam_carve_tiled_pf.cu stb_impl.o -o tiled_pf_K${K}_T${T} 2>/tmp/nvcc_err.txt \
         || { echo "  BUILD FAILED tiled_pf K=$K T=$T: $(cat /tmp/nvcc_err.txt)"; ok=0; }
    return $ok
}

I4K="../data/4k/desert_mesa_4k_3840x2160.png"
I8K="../data/8k/desert_mesa_8k_7680x4320.png"
N4=50; N8=30

echo ""
echo "============================================================"
echo " Phase 1: K/T grid  (Tiled vs Tiled-pf, 4K and 8K)"
echo "============================================================"

printf "\n%-5s %-5s | %10s %10s | %10s %10s\n" K T "4K nopf" "4K pf" "8K nopf" "8K pf"
printf "%-5s %-5s | %10s %10s | %10s %10s\n"  - - "ms/seam" "ms/seam" "ms/seam" "ms/seam"

OUT_SWEEP="../results/tiled_sweep.md"
{
    printf "| K  | T   | 4K nopf | 4K pf | 8K nopf | 8K pf |\n"
    printf "|---|---|---|---|---|---|\n"
} > "$OUT_SWEEP"

for K in 40 60 80; do
    for T in 32 64 128; do
        printf "  Building K=%s T=%s ..." "$K" "$T"
        build_tiled "$K" "$T"

        m4n=$(best_ms "tiled_K${K}_T${T}"    "$I4K" $N4)
        m4p=$(best_ms "tiled_pf_K${K}_T${T}" "$I4K" $N4)
        m8n=$(best_ms "tiled_K${K}_T${T}"    "$I8K" $N8)
        m8p=$(best_ms "tiled_pf_K${K}_T${T}" "$I8K" $N8)

        printf "\r%-5s %-5s | %10s %10s | %10s %10s\n" "$K" "$T" "$m4n" "$m4p" "$m8n" "$m8p"
        printf "| %s | %s | %s | %s | %s | %s |\n" "$K" "$T" "$m4n" "$m4p" "$m8n" "$m8p" >> "$OUT_SWEEP"
    done
done

echo ""
echo "Sweep written to $OUT_SWEEP"

echo ""
echo "============================================================"
echo " Phase 2: Fig. 3 data — ctrl / 1080p / 2K / 4K / 6K / 8K"
echo "============================================================"

OUT_FIG3="../results/tiled_fig3.md"
{
    echo "# Tiled-pf (K=60,T=64) vs Templated vs CPU — $(date)"
    printf "| res | cpu_seq | tiled_pf | prev_best | speedup_cpu |\n"
    printf "|---|---|---|---|---|\n"
} > "$OUT_FIG3"

BIN_PF="tiled_pf_K60_T64"

declare -A RIMGS=(
    [ctrl]="../data/ctrl/broadway_tower_ctrl_960x540.png"
    [1080p]="../data/1080p/forest_pano_1080p_1920x1080.png"
    [2K]="../data/2k/forest_pano_2k_2560x1440.png"
    [4K]="../data/4k/desert_mesa_4k_3840x2160.png"
    [6K]="../data/6k/forest_pano_6k_6144x3456.png"
    [8K]="../data/8k/desert_mesa_8k_7680x4320.png"
)

for lbl in ctrl 1080p 2K 4K 6K 8K; do
    img="${RIMGS[$lbl]}"
    [ -f "$img" ] || { echo "SKIP $lbl (missing $img)"; continue; }
    echo "  Benchmarking $lbl ..."

    # CPU seq (best of RUNS)
    ms_cpu=$(best_ms "seam_carve_cpu" "$img" 50)

    # Tiled-pf K=60 T=64
    ms_pf=$(best_ms "$BIN_PF" "$img" 50)

    # Previous best: Templated (v6) for <=4K, GridStride (wide) for 6K/8K
    if [ "$lbl" = "6K" ] || [ "$lbl" = "8K" ]; then
        ms_prev=$(best_ms "seam_carve_wide" "$img" 50)
        prev_label="GridStride"
    else
        ms_prev=$(best_ms "seam_carve_v6" "$img" 50)
        prev_label="Templated"
    fi

    if [ "$ms_pf" != "-" ] && [ "$ms_cpu" != "-" ]; then
        spd=$(awk "BEGIN{printf \"%.1f\", $ms_cpu / $ms_pf}")x
    else spd="-"; fi

    printf "| %-5s | %-10s | %-10s | %-10s (%s) | %-8s |\n" \
        "$lbl" "$ms_cpu" "$ms_pf" "$ms_prev" "$prev_label" "$spd" \
        | tee -a "$OUT_FIG3"
done

echo ""
echo "Fig3 written to $OUT_FIG3"
echo "=== ALL DONE: $(date) ==="
