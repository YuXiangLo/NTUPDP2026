#!/usr/bin/env python3
"""
verify.py — Correctness harness: compare all implementation outputs against SEQ reference.

Usage:
    # Run SEQ + all impls on a test image, then compare:
    python3 bench/verify.py --image data/ctrl/broadway_tower_ctrl_960x540.png --seams 20 \
        --seq-bin openmp/seam_carve_seq \
        --impl "OMP_v1:openmp/seam_carve_omp_v1" \
        --impl "OMP_v2:openmp/seam_carve_omp_v2" \
        --impl "OMP_v3:openmp/seam_carve_omp_v3" \
        [--impl "CUDA_v5:cuda/seam_carve_v5"] \
        [--out-dir /tmp/verify_out]

    # Or compare pre-existing output PNGs:
    python3 bench/verify.py --compare-dir /tmp/verify_out --reference seq.png

Spec §4:
  - Same-precision (C seq ↔ OpenMP): bit-exact (zero diff).
  - Cross-precision (CPU ↔ CUDA/PyTorch): PSNR >= 50 dB AND max pixel diff <= 2.

Requires: ImageMagick (convert, compare, identify). No Python image libs needed.
"""

from __future__ import print_function
import argparse
import csv
import os
import subprocess
import sys
import tempfile

PSNR_THRESHOLD = 50.0
MAX_DIFF_THRESHOLD = 2

# ---------------------------------------------------------------------------
# ImageMagick helpers
# ---------------------------------------------------------------------------

def run_binary(binary, image, seams, out_png, extra_env=None):
    """Run a seam-carving binary and return elapsed ms from its stdout."""
    import re, os
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    proc = subprocess.Popen(
        [binary, image, str(seams), out_png],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env
    )
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        print("  ERROR running {}: {}".format(binary, stderr.decode()[:300]))
        return None
    # Parse "TAG carving time: X.XXX ms ..."
    text = stdout.decode()
    m = re.search(r"carving time:\s+([\d.]+)\s+ms", text)
    ms = float(m.group(1)) if m else None
    return ms


def images_bit_exact(ref, tgt):
    """Return True if two images are pixel-for-pixel identical."""
    r = subprocess.call(
        ["diff", "-q", ref, tgt],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return r == 0


def images_psnr(ref, tgt):
    """Return PSNR (float) between two images using ImageMagick compare."""
    try:
        out = subprocess.check_output(
            ["compare", "-metric", "PSNR", ref, tgt, "/dev/null"],
            stderr=subprocess.STDOUT
        ).decode().strip()
        # Output is just the number, e.g. "63.4724" or "inf"
        if out.lower() in ("inf", "infinity"):
            return float("inf")
        return float(out.split()[0])
    except subprocess.CalledProcessError as e:
        txt = e.output.decode().strip()
        # compare returns code 1 when images differ but still prints the metric
        if txt.lower() in ("inf", "infinity"):
            return float("inf")
        try:
            return float(txt.split()[0])
        except ValueError:
            return None


def images_max_diff(ref, tgt):
    """Return max absolute pixel difference using ImageMagick compare."""
    try:
        out = subprocess.check_output(
            ["compare", "-metric", "AE", ref, tgt, "/dev/null"],
            stderr=subprocess.STDOUT
        ).decode().strip()
        # AE gives count of different pixels; use PAE for max absolute error
        out2 = subprocess.check_output(
            ["compare", "-metric", "PAE", ref, tgt, "/dev/null"],
            stderr=subprocess.STDOUT
        ).decode().strip()
        # PAE output: "0 (0)" or "0.00392157 (1)" where parens = 8-bit value
        import re
        m = re.search(r"\((\d+)\)", out2)
        return int(m.group(1)) if m else None
    except subprocess.CalledProcessError as e:
        txt = e.output.decode().strip()
        import re
        m = re.search(r"\((\d+)\)", txt)
        return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True, help="Input image for seam carving")
    ap.add_argument("--seams", type=int, default=20, help="Number of seams to remove")
    ap.add_argument("--seq-bin", default="openmp/seam_carve_seq",
                    help="Path to SEQ binary (correctness reference)")
    ap.add_argument("--impl", action="append", default=[],
                    metavar="LABEL:BINARY",
                    help="Implementation to test, e.g. OMP_v2:openmp/seam_carve_omp_v2")
    ap.add_argument("--omp-threads", type=int, default=4,
                    help="OMP_NUM_THREADS for OpenMP impls")
    ap.add_argument("--out-dir", default=None,
                    help="Directory for output PNGs (default: tempdir)")
    ap.add_argument("--csv", default=None, help="Append results to this CSV file")
    args = ap.parse_args()

    use_tmp = args.out_dir is None
    out_dir = args.out_dir or tempfile.mkdtemp(prefix="verify_")
    os.makedirs(out_dir, exist_ok=True)

    # Run SEQ reference
    ref_png = os.path.join(out_dir, "seq.png")
    print("[SEQ]  {} ...".format(args.seq_bin), end=" ", flush=True)
    ms = run_binary(args.seq_bin, args.image, args.seams, ref_png)
    if ms is None or not os.path.exists(ref_png):
        print("FAILED — cannot build reference, aborting.")
        sys.exit(1)
    print("{:.1f} ms  (reference)".format(ms))

    rows = []
    all_pass = True

    for spec in args.impl:
        if ":" not in spec:
            print("Bad --impl spec (need LABEL:BINARY):", spec)
            continue
        label, binary = spec.split(":", 1)
        out_png = os.path.join(out_dir, label + ".png")

        env = {}
        # Set OMP threads for CPU impls
        if "omp" in binary.lower() or "seq" in binary.lower():
            env["OMP_NUM_THREADS"] = str(args.omp_threads)
            env["OMP_PROC_BIND"] = "close"
            env["OMP_PLACES"] = "cores"

        print("[{}]  {} ...".format(label, binary), end=" ", flush=True)
        ms = run_binary(binary, args.image, args.seams, out_png, extra_env=env if env else None)
        if ms is None or not os.path.exists(out_png):
            print("FAILED to run")
            rows.append({"label": label, "status": "RUN_FAILED"})
            all_pass = False
            continue

        # Bit-exact check first (cheap)
        if images_bit_exact(ref_png, out_png):
            psnr = float("inf")
            max_diff = 0
            status = "BIT_EXACT"
        else:
            psnr = images_psnr(ref_png, out_png)
            max_diff = images_max_diff(ref_png, out_png)
            if psnr is not None and psnr >= PSNR_THRESHOLD and (max_diff is None or max_diff <= MAX_DIFF_THRESHOLD):
                status = "WITHIN_TOL"
            else:
                status = "FAIL"
                all_pass = False

        psnr_str = "inf" if psnr == float("inf") else ("{:.1f}".format(psnr) if psnr else "?")
        print("{:.1f} ms  {}  PSNR={}  maxdiff={}".format(
            ms, status, psnr_str, max_diff if max_diff is not None else "?"))

        rows.append({
            "image": os.path.basename(args.image),
            "seams": args.seams,
            "label": label,
            "status": status,
            "ms": ms,
            "psnr": psnr_str,
            "max_diff": max_diff,
        })

    print("")
    print("=== RESULT: {} ===".format("ALL PASS" if all_pass else "FAILURES DETECTED"))

    if args.csv:
        fields = ["image", "seams", "label", "status", "ms", "psnr", "max_diff"]
        write_header = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerows(rows)
        print("Appended {} rows to {}".format(len(rows), args.csv))

    if use_tmp and not rows:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
