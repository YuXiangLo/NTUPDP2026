#!/usr/bin/env python3
"""
plot.py — Generate all paper figures from benchmark CSV results.

Usage:
    python3 bench/plot.py \
        --gpu-csv results/gpu.csv \
        --cpu-csv results/cpu.csv \
        [--out-dir figures/]

Produces (spec §8):
    T1: CPU speedup table      (figures/T1_cpu_speedup.csv + .tex)
    T2: GPU kernel timing       (figures/T2_gpu_timing.csv + .tex)
    T3: Precision/quality       (figures/T3_quality.csv + .tex)
    F1: CPU throughput bar      (figures/F1_cpu_throughput.pdf)
    F2: GPU timing vs resolution(figures/F2_gpu_vs_resolution.pdf)
    F3: Speedup over SEQ        (figures/F3_speedup_over_seq.pdf)
    F4: GPU vs CPU crossover    (figures/F4_crossover.pdf)
    F5: Roofline placeholder    (figures/F5_roofline_placeholder.pdf)
    F6: Transfer overhead       (figures/F6_transfer_overhead.pdf)
    F7: Seam count sensitivity  (figures/F7_seam_sensitivity.pdf)  [if data]
    F8: Quality PSNR heatmap    (figures/F8_quality_heatmap.pdf)  [if verify csv]

Requires: matplotlib, csv (stdlib). No pandas needed (Python 3.6 compat).
"""

from __future__ import print_function
import argparse
import csv
import collections
import math
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available — will write CSV/tex tables only", file=sys.stderr)

# ---------------------------------------------------------------------------
# Colour / style definitions
# ---------------------------------------------------------------------------

IMPL_COLORS = {
    "SEQ":          "#555555",
    "OMP_v1":       "#4e79a7",
    "OMP_v2":       "#f28e2b",
    "OMP_v3":       "#e15759",
    "CUDA_v2":      "#76b7b2",
    "CUDA_v4":      "#59a14f",
    "CUDA_v5":      "#edc948",
    "CUDA_v6":      "#b07aa1",
    "CUDA_wide":    "#ff9da7",
    "CUDA_naive":   "#9c755f",
}

RES_ORDER = ["ctrl", "1080p", "4k", "8k"]
RES_LABELS = {"ctrl": "960×540", "1080p": "1920×1080", "4k": "3840×2160", "8k": "7680×4320"}

CPU_IMPL_ORDER = ["SEQ", "OMP_v1", "OMP_v2", "OMP_v3"]
GPU_IMPL_ORDER = ["CUDA_naive", "CUDA_v2", "CUDA_v4", "CUDA_v5", "CUDA_v6", "CUDA_wide"]

# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------

def load_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def float_or_zero(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def group_by(rows, key_fn):
    d = collections.defaultdict(list)
    for r in rows:
        d[key_fn(r)].append(r)
    return d


def mean_field(rows, field):
    vals = [float_or_zero(r.get(field, 0)) for r in rows if float_or_zero(r.get(field, 0)) > 0]
    return sum(vals) / len(vals) if vals else None


def res_class(row):
    """Derive resolution class from W,H."""
    W = int(float_or_zero(row.get("W", 0)))
    H = int(float_or_zero(row.get("H", 0)))
    if W <= 960:   return "ctrl"
    if W <= 1920:  return "1080p"
    if W <= 3840:  return "4k"
    return "8k"


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

def write_csv_table(path, headers, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print("  wrote", path)


def write_latex_table(path, caption, label, headers, rows):
    n = len(headers)
    col_spec = "l" + "r" * (n - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(c) for c in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("  wrote", path)


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------

def fig_cpu_throughput(cpu_rows, out_dir):
    """F1: grouped bar chart of CPU throughput per resolution."""
    if not HAS_MPL or not cpu_rows:
        return
    impls = [i for i in CPU_IMPL_ORDER
             if any(r["impl"] == i for r in cpu_rows)]
    res_classes = [c for c in RES_ORDER
                   if any(res_class(r) == c for r in cpu_rows)]
    if not impls or not res_classes:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    n_res = len(res_classes)
    n_impl = len(impls)
    bar_w = 0.8 / n_impl
    x = list(range(n_res))

    for i, impl in enumerate(impls):
        means = []
        errs  = []
        for cls in res_classes:
            subset = [r for r in cpu_rows if r["impl"] == impl and res_class(r) == cls]
            thr = mean_field(subset, "throughput_mpix_s")
            std_raw = mean_field(subset, "std_ms")
            means.append(thr if thr else 0)
            errs.append(0)
        offsets = [xx + (i - n_impl/2 + 0.5) * bar_w for xx in x]
        color = IMPL_COLORS.get(impl, "#888888")
        ax.bar(offsets, means, width=bar_w, label=impl, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([RES_LABELS.get(c, c) for c in res_classes])
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Throughput (Mpix·seam/s)")
    ax.set_title("CPU Throughput by Implementation and Resolution")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    path = os.path.join(out_dir, "F1_cpu_throughput.pdf")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("  wrote", path)


def fig_gpu_vs_resolution(gpu_rows, out_dir):
    """F2: GPU mean_ms vs resolution for each GPU impl."""
    if not HAS_MPL or not gpu_rows:
        return
    impls = [i for i in GPU_IMPL_ORDER
             if any(r["impl"] == i for r in gpu_rows)]
    res_classes = [c for c in RES_ORDER
                   if any(res_class(r) == c for r in gpu_rows)]
    if not impls:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for impl in impls:
        xs, ys, errs = [], [], []
        for cls in res_classes:
            subset = [r for r in gpu_rows if r["impl"] == impl and res_class(r) == cls
                      and r.get("transfer_mode", "resident") == "resident"]
            m = mean_field(subset, "mean_ms")
            if m:
                xs.append(RES_LABELS.get(cls, cls))
                ys.append(m)
        if xs:
            color = IMPL_COLORS.get(impl, "#888888")
            ax.plot(xs, ys, marker="o", label=impl, color=color)

    ax.set_xlabel("Resolution")
    ax.set_ylabel("Mean carving time (ms)")
    ax.set_title("GPU Kernel Time vs Resolution")
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.4)

    path = os.path.join(out_dir, "F2_gpu_vs_resolution.pdf")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("  wrote", path)


def fig_speedup_over_seq(cpu_rows, gpu_rows, out_dir):
    """F3: Speedup of all impls over SEQ reference at each resolution."""
    if not HAS_MPL:
        return

    all_rows = cpu_rows + gpu_rows
    if not all_rows:
        return

    res_classes = [c for c in RES_ORDER if any(res_class(r) == c for r in all_rows)]
    impl_order = CPU_IMPL_ORDER + GPU_IMPL_ORDER

    fig, axes = plt.subplots(1, len(res_classes), figsize=(4 * len(res_classes), 4.5),
                             sharey=False)
    if len(res_classes) == 1:
        axes = [axes]

    for ax, cls in zip(axes, res_classes):
        seq_subset = [r for r in cpu_rows if r["impl"] == "SEQ" and res_class(r) == cls]
        seq_mean = mean_field(seq_subset, "mean_ms")
        if not seq_mean:
            ax.set_title(RES_LABELS.get(cls, cls) + "\n(no SEQ data)")
            continue

        labels, speedups = [], []
        for impl in impl_order:
            if impl == "SEQ":
                continue
            if impl in CPU_IMPL_ORDER:
                subset = [r for r in cpu_rows if r["impl"] == impl and res_class(r) == cls]
            else:
                subset = [r for r in gpu_rows if r["impl"] == impl and res_class(r) == cls
                          and r.get("transfer_mode", "resident") == "resident"]
            m = mean_field(subset, "mean_ms")
            if m:
                labels.append(impl)
                speedups.append(seq_mean / m)

        colors = [IMPL_COLORS.get(l, "#888888") for l in labels]
        ax.bar(range(len(labels)), speedups, color=colors, alpha=0.85)
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(RES_LABELS.get(cls, cls))
        ax.set_ylabel("Speedup over SEQ")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Speedup over Sequential Baseline")
    path = os.path.join(out_dir, "F3_speedup_over_seq.pdf")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("  wrote", path)


def fig_crossover(cpu_rows, gpu_rows, out_dir):
    """F4: Break-even / crossover: smallest resolution where GPU beats OMP_v2."""
    if not HAS_MPL or not cpu_rows or not gpu_rows:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    megapix_vals = sorted(set(
        float_or_zero(r.get("megapixels", 0))
        for r in (cpu_rows + gpu_rows)
        if float_or_zero(r.get("megapixels", 0)) > 0
    ))

    for impl in ["OMP_v2", "CUDA_v5", "CUDA_v6"]:
        if impl in CPU_IMPL_ORDER:
            src = cpu_rows
        else:
            src = gpu_rows
        points = []
        for mp in megapix_vals:
            subset = [r for r in src if r["impl"] == impl
                      and abs(float_or_zero(r.get("megapixels", 0)) - mp) < mp * 0.05
                      and r.get("transfer_mode", "resident") == "resident"]
            m = mean_field(subset, "mean_ms")
            if m:
                points.append((mp, m))
        if points:
            xs, ys = zip(*sorted(points))
            color = IMPL_COLORS.get(impl, "#888")
            ax.plot(xs, ys, marker="o", label=impl, color=color)

    ax.set_xlabel("Megapixels (W×H)")
    ax.set_ylabel("Mean carving time (ms)")
    ax.set_title("CPU vs GPU Break-even")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)

    path = os.path.join(out_dir, "F4_crossover.pdf")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("  wrote", path)


def fig_transfer_overhead(gpu_rows, out_dir):
    """F6: Per-seam transfer vs resident pipeline overhead."""
    if not HAS_MPL or not gpu_rows:
        return

    resident_impls = set(r["impl"] for r in gpu_rows
                         if r.get("transfer_mode", "resident") == "resident")
    naive_rows = [r for r in gpu_rows if "naive_xfer" in r.get("impl", "")]
    if not naive_rows:
        print("  [SKIP F6] no naive_xfer rows in GPU CSV")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    base_impls = sorted(set(r["impl"].replace("_naive_xfer", "") for r in naive_rows))

    for base in base_impls:
        res_classes = [c for c in RES_ORDER if c != "8k"]
        xs, overhead = [], []
        for cls in res_classes:
            res_rows = [r for r in naive_rows
                        if r["impl"] == base + "_naive_xfer" and res_class(r) == cls]
            base_rows = [r for r in gpu_rows
                         if r["impl"] == base and res_class(r) == cls
                         and r.get("transfer_mode", "resident") == "resident"]
            naive_ms = mean_field(res_rows, "mean_ms")
            base_ms  = mean_field(base_rows, "mean_ms")
            if naive_ms and base_ms and base_ms > 0:
                xs.append(RES_LABELS.get(cls, cls))
                overhead.append((naive_ms - base_ms) / base_ms * 100)
        if xs:
            ax.plot(xs, overhead, marker="s", label=base)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Transfer overhead (%)")
    ax.set_title("Per-seam H2D/D2H Transfer Overhead")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)

    path = os.path.join(out_dir, "F6_transfer_overhead.pdf")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("  wrote", path)


def fig_roofline_placeholder(out_dir):
    """F5: Placeholder roofline (real data comes from ncu_table.py)."""
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    # Roofline axes: arithmetic intensity (FLOP/byte) vs performance (GFLOP/s)
    ai = [0.01, 0.1, 1, 10, 100]
    # V100 roofline: peak 14 TFLOP/s FP32, HBM BW 900 GB/s
    peak_flop = 14000   # GFLOP/s
    hbm_bw    = 900     # GB/s
    ridge_ai  = peak_flop / hbm_bw  # ~15.6 FLOP/byte
    roof = [min(hbm_bw * a, peak_flop) for a in ai]
    ax.loglog(ai, roof, "k-", linewidth=2, label="V100 roofline")
    ax.axvline(ridge_ai, color="gray", linestyle=":", alpha=0.6)
    ax.text(ridge_ai * 1.1, peak_flop * 0.5, "ridge={:.1f}".format(ridge_ai),
            fontsize=8, color="gray")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Performance (GFLOP/s)")
    ax.set_title("Roofline Model — V100 (placeholder; fill with ncu data)")
    ax.legend(fontsize=9)
    ax.grid(which="both", linestyle="--", alpha=0.3)
    ax.text(0.5, 0.5, "FILL WITH NCU DATA", transform=ax.transAxes,
            fontsize=16, color="red", alpha=0.3, ha="center", va="center",
            rotation=30)

    path = os.path.join(out_dir, "F5_roofline_placeholder.pdf")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("  wrote", path)


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def table_cpu_speedup(cpu_rows, out_dir):
    """T1: CPU speedup table."""
    res_classes = [c for c in RES_ORDER if c != "8k"
                   and any(res_class(r) == c for r in cpu_rows)]
    impls = [i for i in CPU_IMPL_ORDER if any(r["impl"] == i for r in cpu_rows)]

    # Build per-impl per-res mean
    data = {}
    for impl in impls:
        for cls in res_classes:
            subset = [r for r in cpu_rows if r["impl"] == impl and res_class(r) == cls]
            m = mean_field(subset, "mean_ms")
            data[(impl, cls)] = m

    # Get SEQ reference
    headers = ["Impl"] + [RES_LABELS.get(c, c) + " ms" for c in res_classes] \
            + [RES_LABELS.get(c, c) + " ×SEQ" for c in res_classes]
    rows = []
    for impl in impls:
        row = [impl]
        for cls in res_classes:
            m = data.get((impl, cls))
            row.append("{:.1f}".format(m) if m else "-")
        for cls in res_classes:
            seq_m = data.get(("SEQ", cls))
            impl_m = data.get((impl, cls))
            if seq_m and impl_m:
                row.append("{:.2f}×".format(seq_m / impl_m))
            else:
                row.append("-")
        rows.append(row)

    write_csv_table(os.path.join(out_dir, "T1_cpu_speedup.csv"), headers, rows)
    write_latex_table(
        os.path.join(out_dir, "T1_cpu_speedup.tex"),
        caption="CPU implementation speedup over sequential baseline.",
        label="tab:cpu_speedup",
        headers=headers, rows=rows
    )


def table_gpu_timing(gpu_rows, out_dir):
    """T2: GPU kernel timing table."""
    res_classes = [c for c in RES_ORDER
                   if any(res_class(r) == c for r in gpu_rows)]
    impls = [i for i in GPU_IMPL_ORDER if any(r["impl"] == i for r in gpu_rows)]

    headers = ["Impl"] + [RES_LABELS.get(c, c) + " ms" for c in res_classes]
    rows = []
    for impl in impls:
        row = [impl]
        for cls in res_classes:
            subset = [r for r in gpu_rows if r["impl"] == impl and res_class(r) == cls
                      and r.get("transfer_mode", "resident") == "resident"]
            m = mean_field(subset, "mean_ms")
            row.append("{:.3f}".format(m) if m else "-")
        rows.append(row)

    write_csv_table(os.path.join(out_dir, "T2_gpu_timing.csv"), headers, rows)
    write_latex_table(
        os.path.join(out_dir, "T2_gpu_timing.tex"),
        caption="GPU mean carving time (ms) across implementations and resolutions.",
        label="tab:gpu_timing",
        headers=headers, rows=rows
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu-csv",  default="results/gpu.csv")
    ap.add_argument("--cpu-csv",  default="results/cpu.csv")
    ap.add_argument("--out-dir",  default="figures")
    ap.add_argument("--verify-csv", default=None,
                    help="Output of bench/verify.py --csv for quality heatmap")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    gpu_rows = load_csv(args.gpu_csv)
    cpu_rows = load_csv(args.cpu_csv)

    print("Loaded {} GPU rows, {} CPU rows".format(len(gpu_rows), len(cpu_rows)))

    if not gpu_rows and not cpu_rows:
        print("No data found — run bench/run_matrix.py first.")
        sys.exit(0)

    print("\n--- Tables ---")
    if cpu_rows:
        table_cpu_speedup(cpu_rows, args.out_dir)
    if gpu_rows:
        table_gpu_timing(gpu_rows, args.out_dir)

    print("\n--- Figures ---")
    if not HAS_MPL:
        print("  matplotlib not installed — skipping figures")
    else:
        fig_cpu_throughput(cpu_rows, args.out_dir)
        fig_gpu_vs_resolution(gpu_rows, args.out_dir)
        fig_speedup_over_seq(cpu_rows, gpu_rows, args.out_dir)
        fig_crossover(cpu_rows, gpu_rows, args.out_dir)
        fig_transfer_overhead(gpu_rows, args.out_dir)
        fig_roofline_placeholder(args.out_dir)

    print("\nDone. Output in", args.out_dir)


if __name__ == "__main__":
    main()
