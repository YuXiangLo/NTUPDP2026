#!/bin/bash
# sbatch_incr.sh — validate exact incremental seam carving + measure speedup.
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gpus-per-node=1
#SBATCH -A ACD115083
#SBATCH -t 00:20:00
#SBATCH -o slurm-incr-%j.out
#SBATCH -e slurm-incr-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load cuda 2>/dev/null || true
cd cuda
g++ -O3 -std=c++14 -march=native seam_carve_cpu_incr.cpp stb_impl.o -o seam_carve_cpu_incr || { echo BUILD FAIL; exit 1; }
echo "=== Build OK ==="
B=./seam_carve_cpu_incr

echo "=== Correctness: incr output == full output (bit-exact PNG) ==="
for img in ../data/ctrl/broadway_tower_ctrl_960x540.png ../data/1080p/forest_pano_1080p_1920x1080.png; do
  $B "$img" 96 /tmp/_full.png full >/dev/null 2>&1
  $B "$img" 96 /tmp/_incr.png incr >/dev/null 2>&1
  if cmp -s /tmp/_full.png /tmp/_incr.png; then echo "[PASS] $img"; else echo "[FAIL] $img"; fi
done

echo ""
echo "=== Speedup: incr vs full (ms/seam, single-thread) ==="
bench() {
  local label="$1" img="$2" n="$3"
  local f=$($B "$img" $n /tmp/_o.png full 2>/dev/null | grep -oP '\(\K[0-9.]+(?= ms/seam)')
  local i=$($B "$img" $n /tmp/_o.png incr 2>/dev/null | grep -oP '\(\K[0-9.]+(?= ms/seam)')
  local sp=$(awk "BEGIN{printf \"%.2f\", $f/$i}")
  printf "%-16s full=%-9s incr=%-9s  speedup=%sx\n" "$label" "$f" "$i" "$sp"
}
bench "ctrl 960x540"    ../data/ctrl/broadway_tower_ctrl_960x540.png   100
bench "1080p 1920x1080" ../data/1080p/forest_pano_1080p_1920x1080.png  100
bench "4K 3840x2160"    ../data/4k/desert_mesa_4k_3840x2160.png        60
echo "=== DONE ==="
