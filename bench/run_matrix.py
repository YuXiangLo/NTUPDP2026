#!/usr/bin/env python3
"""
run_matrix.py — Submit the full benchmark matrix to SLURM.

Each (impl, image, seam_count) tuple becomes one sbatch job.  Results
accumulate in results/gpu.csv and results/cpu.csv, ready for plot.py.

Usage:
    python3 bench/run_matrix.py [options]

Options:
    --dry-run       Print sbatch commands but do not submit
    --gpu-only      Submit only GPU jobs
    --cpu-only      Submit only CPU jobs
    --impl LABEL    Only run the named impl (repeatable)
    --res LABEL     Only run the named resolution class (ctrl,1080p,4k,8k)
    --reps N        Timed repetitions per job (default 10)
    --threads N     OMP_NUM_THREADS for CPU jobs (default 40)
    --arch ARCH     GPU arch label for GPU jobs (default sm_70)
    --out-dir DIR   Directory for result CSVs (default results/)

Matrix definition (edit the MATRIX dict below to add/remove variants):
    CPU impls : SEQ, OMP_v1, OMP_v2, OMP_v3
    GPU impls : CUDA_v2, CUDA_v4, CUDA_v5, CUDA_v6, CUDA_wide, CUDA_naive

Resolution classes and seam counts defined in SEAMS_PER_CLASS.
"""

from __future__ import print_function
import argparse
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Matrix definition
# ---------------------------------------------------------------------------

# impl_label -> relative binary path
CPU_IMPLS = {
    "SEQ":    "openmp/seam_carve_seq",
    "OMP_v1": "openmp/seam_carve_omp_v1",
    "OMP_v2": "openmp/seam_carve_omp_v2",
    "OMP_v3": "openmp/seam_carve_omp_v3",
}

GPU_IMPLS = {
    "CUDA_v2":    "cuda/seam_carve",     # binary is seam_carve (not seam_carve_v2)
    "CUDA_v4":    "cuda/seam_carve_v4",
    "CUDA_v5":    "cuda/seam_carve_v5",
    "CUDA_v6":    "cuda/seam_carve_v6",      # fused energy+DP (new)
    "CUDA_wide":  "cuda/seam_carve_wide",    # v5 variant, no width limit
    "CUDA_naive": "cuda/seam_carve_naive",   # per-seam H2D/D2H baseline
}

# Resolution classes -> image filenames to include
# Keys must match subdirectory names in data/
RES_IMAGES = {
    "ctrl":  [
        "broadway_tower_ctrl_960x540.png",
        "desert_mesa_ctrl_960x540.png",
        "forest_pano_ctrl_960x540.png",
        "golden_gate_ctrl_960x540.png",
        "iceland_waterfall_ctrl_960x540.png",
    ],
    "1080p": [
        "desert_mesa_1080p_1920x1080.png",
        "forest_pano_1080p_1920x1080.png",
        "golden_gate_1080p_1920x1080.png",
        "iceland_waterfall_1080p_1920x1080.png",
    ],
    "4k": [
        "desert_mesa_4k_3840x2160.png",
        "forest_pano_4k_3840x2160.png",
    ],
    "8k": [
        "desert_mesa_8k_7680x4320.png",
        "forest_pano_8k_7680x4320.png",
    ],
}

# Seam count per resolution class: remove 10% of width
SEAMS_PER_CLASS = {
    "ctrl":  96,    # 10% of 960
    "1080p": 192,   # 10% of 1920
    "4k":    384,   # 10% of 3840
    "8k":    768,   # 10% of 7680
}

# GPU impls that only support width <= 2048 (2-column-per-thread DP kernel).
# CUDA_v2 and CUDA_wide use grid-stride and support any width.
WIDTH_LIMITED = {"CUDA_v4", "CUDA_v5", "CUDA_v6", "CUDA_naive"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SLURM_ACCOUNT = "ACD115083"

def sbatch_gpu(impl, binary, image_path, seams, reps, out_csv, gpu_arch,
               dry_run=False, transfer_mode="resident"):
    env_exports = ",".join([
        "ALL",
        "IMPL={}".format(impl),
        "BINARY={}".format(binary),
        "IMAGE={}".format(image_path),
        "SEAMS={}".format(seams),
        "REPS={}".format(reps),
        "OUT_CSV={}".format(out_csv),
        "GPU_ARCH={}".format(gpu_arch),
        "TRANSFER_MODE={}".format(transfer_mode),
        "MEGAPIX=0",   # computed inside sbatch script
    ])
    cmd = [
        "sbatch",
        "--export={}".format(env_exports),
        "bench/sbatch_gpu.sh",
    ]
    if dry_run:
        print("[DRY GPU]", " ".join(cmd))
        return None
    result = subprocess.check_output(cmd).decode().strip()
    print("  submitted GPU job:", result)
    return result


def sbatch_cpu(impl, binary, image_path, seams, reps, omp_threads, out_csv,
               dry_run=False):
    env_exports = ",".join([
        "ALL",
        "IMPL={}".format(impl),
        "BINARY={}".format(binary),
        "IMAGE={}".format(image_path),
        "SEAMS={}".format(seams),
        "REPS={}".format(reps),
        "OMP_THREADS={}".format(omp_threads),
        "OUT_CSV={}".format(out_csv),
        "MEGAPIX=0",
    ])
    cmd = [
        "sbatch",
        "--export={}".format(env_exports),
        "bench/sbatch_cpu.sh",
    ]
    if dry_run:
        print("[DRY CPU]", " ".join(cmd))
        return None
    result = subprocess.check_output(cmd).decode().strip()
    print("  submitted CPU job:", result)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run",   action="store_true")
    ap.add_argument("--gpu-only",  action="store_true")
    ap.add_argument("--cpu-only",  action="store_true")
    ap.add_argument("--impl",      action="append", default=[], metavar="LABEL",
                    help="Limit to specific impl(s)")
    ap.add_argument("--res",       action="append", default=[], metavar="CLASS",
                    help="Limit to specific resolution class(es): ctrl,1080p,4k,8k")
    ap.add_argument("--reps",      type=int, default=10)
    ap.add_argument("--threads",   type=int, default=40)
    ap.add_argument("--arch",      default="sm_70",
                    help="GPU arch label for labelling (sm_70 or sm_80)")
    ap.add_argument("--out-dir",   default="results")
    ap.add_argument("--data-dir",  default="data")
    ap.add_argument("--force-wide", action="store_true",
                    help="Submit width-limited GPU impls on 4k/8k anyway")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    gpu_csv = os.path.join(args.out_dir, "gpu.csv")
    cpu_csv = os.path.join(args.out_dir, "cpu.csv")

    target_impls = set(args.impl) if args.impl else None
    target_res   = set(args.res)  if args.res  else None

    n_gpu = 0
    n_cpu = 0

    res_classes = list(RES_IMAGES.keys())
    if target_res:
        res_classes = [r for r in res_classes if r in target_res]

    # ---- GPU jobs ----
    if not args.cpu_only:
        print("\n=== Submitting GPU jobs ===")
        gpu_impls = GPU_IMPLS
        if target_impls:
            gpu_impls = {k: v for k, v in gpu_impls.items() if k in target_impls}

        for impl, binary in sorted(gpu_impls.items()):
            # Check binary exists
            if not os.path.isfile(binary):
                print("  [SKIP] {} — binary not found: {}".format(impl, binary))
                continue

            for res in res_classes:
                seams = SEAMS_PER_CLASS[res]
                for img_name in RES_IMAGES[res]:
                    img_path = os.path.join(args.data_dir, res, img_name)
                    if not os.path.isfile(img_path):
                        print("  [SKIP] {} — image not found: {}".format(impl, img_path))
                        continue

                    # Skip width-limited impls on wide images
                    W = int(img_name.split("x")[0].rsplit("_", 1)[-1]) if "x" in img_name else 0
                    if impl in WIDTH_LIMITED and W > 2048 and not args.force_wide:
                        print("  [SKIP] {} on {} — width {} > 2048 limit".format(
                            impl, img_name, W))
                        continue

                    print("  {} | {} | {} seams".format(impl, img_path, seams))
                    sbatch_gpu(impl, binary, img_path, seams, args.reps,
                               gpu_csv, args.arch, dry_run=args.dry_run)
                    n_gpu += 1

                    # Also run naive transfer variant for CUDA_v5/v6 at 1080p+
                    if impl in ("CUDA_v5", "CUDA_v6") and res in ("1080p", "4k"):
                        sbatch_gpu(impl + "_naive_xfer", binary, img_path,
                                   seams, args.reps, gpu_csv, args.arch,
                                   dry_run=args.dry_run, transfer_mode="per_seam")
                        n_gpu += 1

    # ---- CPU jobs ----
    if not args.gpu_only:
        print("\n=== Submitting CPU jobs ===")
        cpu_impls = CPU_IMPLS
        if target_impls:
            cpu_impls = {k: v for k, v in cpu_impls.items() if k in target_impls}

        for impl, binary in sorted(cpu_impls.items()):
            if not os.path.isfile(binary):
                print("  [SKIP] {} — binary not found: {}".format(impl, binary))
                continue

            for res in res_classes:
                # CPU benchmarks only up to 4k (8k is unreasonably slow)
                if res == "8k":
                    continue
                seams = SEAMS_PER_CLASS[res]
                for img_name in RES_IMAGES[res]:
                    img_path = os.path.join(args.data_dir, res, img_name)
                    if not os.path.isfile(img_path):
                        print("  [SKIP] {} — image not found: {}".format(impl, img_path))
                        continue

                    print("  {} | {} | {} seams | {} threads".format(
                        impl, img_path, seams, args.threads))
                    sbatch_cpu(impl, binary, img_path, seams, args.reps,
                               args.threads, cpu_csv, dry_run=args.dry_run)
                    n_cpu += 1

    print("\n=== Summary ===")
    print("  GPU jobs submitted : {}".format(n_gpu))
    print("  CPU jobs submitted : {}".format(n_cpu))
    if not args.dry_run:
        print("  GPU results -> {}".format(gpu_csv))
        print("  CPU results -> {}".format(cpu_csv))
        print("")
        print("Monitor with: squeue -u $USER")
        print("Plot with:    python3 bench/plot.py --gpu-csv {} --cpu-csv {}".format(
            gpu_csv, cpu_csv))


if __name__ == "__main__":
    main()
