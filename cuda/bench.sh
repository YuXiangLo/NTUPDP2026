#!/usr/bin/env bash
# bench.sh - sweep seam-carving versions over several images x seam-ratios.
#
# Runs each version (v2 / v4 / v5) on every image at several seam counts
# (a fixed % of the image width), best-of-N runs, and writes a tidy CSV.
# Use bench_table.py to turn the CSV into a markdown report table.
#
# Usage (run INSIDE one GPU allocation so all the work shares one srun):
#
#   module load cuda
#   make                                   # build seam_carve, _v4, _v5
#   srun -N 1 -n 1 --gpus-per-node 1 -A ACD115083 -t 20 \
#       bash bench.sh small.jpg medium.jpg large.jpg
#
#   python3 bench_table.py results.csv     # render the report table
#
# Tunables via env:
#   PCTS="5 10 20"   seam counts as % of each image's width
#   RUNS=3           runs per data point (the fastest is kept)
#   BINS="seam_carve seam_carve_v4 seam_carve_v5"
#   OUT=results.csv  output CSV path
#   BINDIR=.         where the binaries live
set -u

PCTS="${PCTS:-5 10 20}"
RUNS="${RUNS:-3}"
BINS="${BINS:-seam_carve seam_carve_v4 seam_carve_v5}"
OUT="${OUT:-results.csv}"
BINDIR="${BINDIR:-.}"
TMP="${TMPDIR:-/tmp}/seam_bench_out.png"

if [ "$#" -lt 1 ]; then
    echo "usage: bash bench.sh <image1> [image2 ...]" >&2
    echo "  (give 3 different-resolution images for the report)" >&2
    exit 1
fi
IMAGES="$@"

# map binary name -> short version label for the CSV
ver_label() {
    case "$1" in
        seam_carve)    echo "v2" ;;
        seam_carve_v4) echo "v4" ;;
        seam_carve_v5) echo "v5" ;;
        *)             echo "$1" ;;
    esac
}

# check binaries exist up front
for b in $BINS; do
    if [ ! -x "$BINDIR/$b" ]; then
        echo "missing binary: $BINDIR/$b  (run 'make' first)" >&2
        exit 1
    fi
done

echo "image,width,height,pct,seams,version,best_total_ms,ms_per_seam,runs" > "$OUT"

for img in $IMAGES; do
    if [ ! -f "$img" ]; then
        echo "!! skipping missing image: $img" >&2
        continue
    fi

    # Probe W x H by carving a single seam (cheap) and parsing the load line.
    probe_bin=$(echo $BINS | awk '{print $1}')
    probe=$("$BINDIR/$probe_bin" "$img" 1 "$TMP" 2>&1)
    dims=$(echo "$probe" | sed -nE 's/.*: ([0-9]+)x([0-9]+),.*/\1 \2/p' | head -1)
    W=$(echo "$dims" | awk '{print $1}')
    Hh=$(echo "$dims" | awk '{print $2}')
    if [ -z "${W:-}" ]; then
        echo "!! could not read dimensions for $img, skipping" >&2
        echo "$probe" >&2
        continue
    fi
    echo "== $img : ${W}x${Hh} =="
    if [ "$W" -gt 2048 ]; then
        echo "   !! width $W > 2048: v4/v5 will fail on this image (v2 only)" >&2
    fi

    for pct in $PCTS; do
        seams=$(awk -v w="$W" -v p="$pct" 'BEGIN{s=int(w*p/100+0.5); if(s<1)s=1; if(s>=w)s=w-1; print s}')

        for b in $BINS; do
            vl=$(ver_label "$b")
            best_total=""
            best_per=""
            for r in $(seq 1 "$RUNS"); do
                out=$("$BINDIR/$b" "$img" "$seams" "$TMP" 2>&1)
                line=$(echo "$out" | grep "GPU carving time")
                if [ -z "$line" ]; then
                    echo "   !! $vl no timing on $img seams=$seams:" >&2
                    echo "$out" | tail -3 >&2
                    continue
                fi
                t=$(echo "$line" | sed -nE 's/.*time: ([0-9.]+) ms.*/\1/p')
                p=$(echo "$line" | sed -nE 's/.*\(([0-9.]+) ms\/seam\).*/\1/p')
                if [ -z "$best_total" ] || awk "BEGIN{exit !($t < $best_total)}"; then
                    best_total="$t"; best_per="$p"
                fi
            done
            if [ -z "$best_total" ]; then
                echo "   $vl  pct=${pct}% seams=$seams  -> FAILED" >&2
                continue
            fi
            printf '   %-3s pct=%-3s seams=%-4s  best=%8s ms  (%s ms/seam)\n' \
                "$vl" "${pct}%" "$seams" "$best_total" "$best_per"
            echo "$img,$W,$Hh,$pct,$seams,$vl,$best_total,$best_per,$RUNS" >> "$OUT"
        done
    done
done

echo
echo "wrote $OUT"
echo "render table:  python3 bench_table.py $OUT"
