#!/usr/bin/env python3
"""
make_dataset.py — Produce standardized benchmark images from raw source images.

Usage:
    python3 bench/make_dataset.py [--src images/] [--out data/] [--dry-run]

Pipeline per source image:
  1. If portrait (H > W), rotate 90 deg to landscape.
  2. Center-crop to 16:9.
  3. Downsample (Lanczos) to each target resolution — only if source is larger
     (never upscale; blurry upscaled images would invalidate quality claims).
  4. Write PNG to data/<class>/<content>_<WxH>.png.
  5. Append a row to data/MANIFEST.csv.

Target resolutions (spec §2.2):
    ctrl  : 960 x 540
    1080p : 1920 x 1080
    4k    : 3840 x 2160
    8k    : 7680 x 4320

Requires: ImageMagick (convert, identify) — no Python image libraries needed.
"""

from __future__ import print_function
import argparse
import csv
import hashlib
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Target definitions
# ---------------------------------------------------------------------------
TARGETS = [
    ("ctrl",  960,  540),
    ("1080p", 1920, 1080),
    ("4k",    3840, 2160),
    ("8k",    7680, 4320),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd, dry_run=False):
    if dry_run:
        print("[DRY]", " ".join(cmd))
        return
    subprocess.check_call(cmd)


def identify_dimensions(path):
    """Return (W, H) of image using ImageMagick identify."""
    out = subprocess.check_output(
        ["identify", "-format", "%w %h", path + "[0]"],
        stderr=subprocess.DEVNULL
    ).decode().strip().split()
    return int(out[0]), int(out[1])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def source_name(path):
    """Extract content label from filename, e.g. 'broadway_tower'."""
    base = os.path.splitext(os.path.basename(path))[0]
    # Strip trailing _WxH if present
    parts = base.rsplit("_", 1)
    if len(parts) == 2 and "x" in parts[1]:
        return parts[0]
    return base


def make_16x9_crop(W, H):
    """Return (crop_w, crop_h, off_x, off_y) for a 16:9 center crop."""
    if W * 9 >= H * 16:
        # wider than 16:9 — fit by height
        cw = int(round(H * 16.0 / 9.0))
        # make even
        cw = cw - (cw % 2)
        ch = H - (H % 2)
        ox = (W - cw) // 2
        oy = 0
    else:
        # taller than 16:9 — fit by width
        cw = W - (W % 2)
        ch = int(round(W * 9.0 / 16.0))
        ch = ch - (ch % 2)
        ox = 0
        oy = (H - ch) // 2
    return cw, ch, ox, oy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="images", help="Source image directory")
    ap.add_argument("--out", default="data",   help="Output directory")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_dir = args.src
    out_dir = args.out

    sources = sorted([
        os.path.join(src_dir, f) for f in os.listdir(src_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
        and not f.startswith(".")
    ])
    if not sources:
        print("No source images found in", src_dir)
        sys.exit(1)

    manifest_path = os.path.join(out_dir, "MANIFEST.csv")
    manifest_rows = []

    for src in sources:
        W, H = identify_dimensions(src)
        label = source_name(src)
        print("\n[SRC] {} {}x{}".format(src, W, H))

        # Step 1: rotate portrait to landscape
        rotated = False
        if H > W:
            W, H = H, W
            rotated = True
            print("  => rotated to landscape {}x{}".format(W, H))

        # Step 2: compute 16:9 crop
        cw, ch, ox, oy = make_16x9_crop(W, H)
        print("  => 16:9 crop {}x{} +{}+{}".format(cw, ch, ox, oy))

        # Build ImageMagick crop+rotate geometry string
        # Applied in single convert call per target to avoid temp files.
        for cls, tw, th in TARGETS:
            if cw < tw or ch < th:
                print("  [SKIP] {} ({}x{} < {}x{})".format(cls, cw, ch, tw, th))
                continue

            cls_dir = os.path.join(out_dir, cls)
            if not args.dry_run:
                os.makedirs(cls_dir, exist_ok=True)

            out_name = "{}_{}_{}x{}.png".format(label, cls, tw, th)
            out_path = os.path.join(cls_dir, out_name)

            cmd = ["convert", src]
            if rotated:
                cmd += ["-rotate", "90"]
            cmd += [
                "-gravity", "NorthWest",
                "-crop", "{}x{}+{}+{}".format(cw, ch, ox, oy),
                "+repage",
                "-resize", "{}x{}!".format(tw, th),
                "-filter", "Lanczos",
                "-quality", "95",
                out_path,
            ]

            print("  => {} {}x{}  ->  {}".format(cls, tw, th, out_path))
            run(cmd, dry_run=args.dry_run)

            if not args.dry_run and os.path.exists(out_path):
                digest = sha256(out_path)
                manifest_rows.append({
                    "src": os.path.basename(src),
                    "src_WxH": "{}x{}".format(
                        W if not rotated else H,
                        H if not rotated else W
                    ),
                    "rotated": rotated,
                    "crop": "{}x{}+{}+{}".format(cw, ch, ox, oy),
                    "class": cls,
                    "out": out_path,
                    "W": tw,
                    "H": th,
                    "sha256": digest,
                })

    if not args.dry_run and manifest_rows:
        fieldnames = ["src", "src_WxH", "rotated", "crop", "class", "out", "W", "H", "sha256"]
        with open(manifest_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(manifest_rows)
        print("\nWrote", len(manifest_rows), "entries to", manifest_path)
    elif args.dry_run:
        print("\n[DRY RUN] no files written.")


if __name__ == "__main__":
    main()
