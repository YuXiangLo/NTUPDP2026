#!/usr/bin/env bash
# gen_figures.sh - render the paper's real before/seams/after figure assets.
# Run from cuda/ after `make seam_vis`. Pure CPU; no GPU/srun needed.
#
#   make seam_vis
#   bash gen_figures.sh                       # defaults: Broadway, 200 seams
#   IMG=../input1.jpg SEAMS=300 bash gen_figures.sh
#
# Output structure (paths are what paper/seam_carving.tex \includegraphics reads):
#   paper/figures/broadway_orig.jpg     copy of the original image
#   paper/figures/broadway_seams.png    original with removed seams drawn red
#   paper/figures/broadway_carved.png   image after removing SEAMS seams
set -u

IMG="${IMG:-../Broadway_tower_edit.jpg}"
SEAMS="${SEAMS:-200}"
NAME="${NAME:-broadway}"
OUTDIR="${OUTDIR:-../paper/figures}"

[ -x ./seam_vis ] || { echo "missing ./seam_vis (run: make seam_vis)" >&2; exit 1; }
[ -f "$IMG" ]     || { echo "missing image: $IMG" >&2; exit 1; }
mkdir -p "$OUTDIR"

cp "$IMG" "$OUTDIR/${NAME}_orig.jpg"
./seam_vis "$IMG" "$SEAMS" "$OUTDIR/${NAME}_seams.png" "$OUTDIR/${NAME}_carved.png"

echo
echo "figures written to $OUTDIR :"
echo "  ${NAME}_orig.jpg   ${NAME}_seams.png   ${NAME}_carved.png"
