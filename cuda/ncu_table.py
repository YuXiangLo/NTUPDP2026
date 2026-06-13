#!/usr/bin/env python3
"""Print a tidy per-kernel table from an Nsight Compute report.

Usage:
    python3 ncu_table.py report.ncu-rep          # report file (calls ncu)
    python3 ncu_table.py metrics.csv             # already-exported CSV
    python3 ncu_table.py --roofline report.ncu-rep W H  # print roofline point

Shows per-kernel:
    Duration (ms), SM%, Mem%, Occ%, Arithmetic Intensity (AI, FLOP/byte),
    Achieved GFLOP/s, Roofline bound (compute or memory).

Arithmetic Intensity formula (spec §6):
    AI = flop_count / dram_bytes_accessed
       = sm__sass_thread_inst_executed_op_fadd_pred_on.sum
         + sm__sass_thread_inst_executed_op_fmul_pred_on.sum
         + sm__sass_thread_inst_executed_op_ffma_pred_on.sum * 2
       / l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum
         + l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum
         (HBM bytes = dram__bytes_read.sum + dram__bytes_write.sum)

Hardware roof lines (V100 sm_70, from spec):
    FP32 peak : 14 TFLOP/s = 14000 GFLOP/s
    HBM2 BW   : 900 GB/s
    Ridge AI  : 14000 / 900 ≈ 15.6 FLOP/byte

Needs `ncu` in PATH (run `module load cuda` first) when given a .ncu-rep.
Profile with:
    ncu --set full --metrics \
        sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
        sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
        sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
        dram__bytes_read.sum,dram__bytes_write.sum \
        -o report_v5 ./seam_carve_v5 data/ctrl/broadway_tower_ctrl_960x540.png 10
"""
import csv
import io
import math
import subprocess
import sys

# V100 hardware constants (sm_70)
V100_PEAK_FLOP_GS = 14000.0  # GFLOP/s (FP32)
V100_HBM_BW_GBS   = 900.0    # GB/s

COLS = [
    ("Kernel Name", "kernel"),
    ("gpu__time_duration.sum", "Dur(ms)"),
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "SM%"),
    ("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "Mem%"),
    ("sm__warps_active.avg.pct_of_peak_sustained_active", "Occ%"),
]

# Metrics needed for AI / roofline
AI_FLOP_COLS = [
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",   # counts as 2 FLOP
]
AI_BW_COLS = [
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
]


def load_rows(path):
    if path.endswith(".csv"):
        with open(path, newline="") as f:
            return list(csv.reader(f))
    cmd = ["ncu", "--import", path, "--csv", "--page", "raw"]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True)
    except FileNotFoundError:
        sys.exit("error: `ncu` not found in PATH (run `module load cuda` first).")
    if proc.returncode != 0:
        sys.exit("error: ncu import failed:\n" + (proc.stderr or ""))
    return list(csv.reader(io.StringIO(proc.stdout)))


def safe_float(row, header, col):
    try:
        return float(row[header.index(col)].replace(",", ""))
    except (ValueError, IndexError):
        return None


def compute_ai(row, header):
    """Return (AI FLOP/byte, achieved_GFLOPS, bound_label) or (None, None, None)."""
    # FLOP count
    fadd = safe_float(row, header, AI_FLOP_COLS[0])
    fmul = safe_float(row, header, AI_FLOP_COLS[1])
    ffma = safe_float(row, header, AI_FLOP_COLS[2])
    # HBM bytes
    dr   = safe_float(row, header, AI_BW_COLS[0])
    dw   = safe_float(row, header, AI_BW_COLS[1])

    if None in (fadd, fmul, ffma, dr, dw):
        return None, None, None

    flops = fadd + fmul + ffma * 2.0
    dram_bytes = dr + dw
    if dram_bytes <= 0 or flops <= 0:
        return None, None, None

    ai = flops / dram_bytes  # FLOP/byte

    # Duration in nanoseconds (ncu reports in ns)
    dur_ns = safe_float(row, header, "gpu__time_duration.sum")
    if dur_ns and dur_ns > 0:
        achieved_gflops = flops / dur_ns  # GFLOP/s (FLOP/ns = GFLOP/s)
    else:
        achieved_gflops = None

    # Roofline bound: memory if AI < ridge_point, compute otherwise
    ridge = V100_PEAK_FLOP_GS / V100_HBM_BW_GBS
    bound = "MEM" if ai < ridge else "COMP"
    roofline_perf = min(V100_PEAK_FLOP_GS, V100_HBM_BW_GBS * ai)

    return ai, achieved_gflops, bound


def main():
    # Simple arg parsing (avoid argparse for Python 3.6 compat)
    args = sys.argv[1:]
    show_roofline = "--roofline" in args
    if show_roofline:
        args = [a for a in args if a != "--roofline"]

    if len(args) < 1:
        sys.exit("usage: python3 ncu_table.py [--roofline] <report.ncu-rep | metrics.csv>")

    rows = load_rows(args[0])
    if len(rows) < 3:
        sys.exit("error: no kernel rows found in the report.")
    header = rows[0]

    try:
        idx = {name: header.index(name) for name, _ in COLS}
    except ValueError as e:
        sys.exit("error: expected metric column missing (%s). "
                 "Re-profile with `--set full`." % e)

    has_ai_cols = all(c in header for c in AI_FLOP_COLS + AI_BW_COLS)

    if has_ai_cols:
        fmt = "%-28s%10s%7s%7s%7s%8s%10s%6s"
        print(fmt % ("kernel", "Dur(ms)", "SM%", "Mem%", "Occ%", "AI", "GFLOP/s", "Bound"))
    else:
        fmt = "%-28s%10s%7s%7s%7s"
        print(fmt % ("kernel", "Dur(ms)", "SM%", "Mem%", "Occ%"))
        print("  (AI/roofline columns unavailable: re-profile with --set full "
              "and the metrics listed in the script header)")

    stall_cols = [
        (c, c.split("stalled_")[1].split("_per_issue")[0])
        for c in header
        if c.startswith("smsp__average_warps_issue_stalled_")
        and c.endswith("_per_issue_active.ratio")
    ]

    rows_by_kernel = []
    seen = set()
    for r in rows[2:]:
        if len(r) <= max(idx.values()):
            continue
        kernel = r[idx["Kernel Name"]].split("(")[0]
        if not kernel or kernel in seen:
            continue
        seen.add(kernel)
        rows_by_kernel.append((kernel, r))

        base_vals = tuple(r[idx[name]] for name, _ in COLS[1:])
        if has_ai_cols:
            ai, gflops, bound = compute_ai(r, header)
            ai_s    = "%.3f" % ai    if ai    is not None else "-"
            gf_s    = "%.1f"  % gflops if gflops is not None else "-"
            bound_s = bound if bound else "-"
            print(fmt % ((kernel,) + base_vals + (ai_s, gf_s, bound_s)))
        else:
            print(fmt % ((kernel,) + base_vals))

    if stall_cols:
        print("\ntop warp-stall reasons (cycles/active, higher = bigger bottleneck):")
        for kernel, r in rows_by_kernel:
            vals = []
            for col, name in stall_cols:
                try:
                    vals.append((float(r[header.index(col)]), name))
                except (ValueError, IndexError):
                    pass
            vals.sort(reverse=True)
            top = "  ".join("%s=%.2f" % (n, v) for v, n in vals[:4])
            print("  %-28s %s" % (kernel, top))

    if show_roofline and has_ai_cols:
        print("\n--- Roofline summary (V100: %.0f GFLOP/s peak, %.0f GB/s HBM) ---" %
              (V100_PEAK_FLOP_GS, V100_HBM_BW_GBS))
        ridge = V100_PEAK_FLOP_GS / V100_HBM_BW_GBS
        print("  Ridge point: %.2f FLOP/byte" % ridge)
        for kernel, r in rows_by_kernel:
            ai, gflops, bound = compute_ai(r, header)
            if ai is None:
                continue
            roof = min(V100_PEAK_FLOP_GS, V100_HBM_BW_GBS * ai)
            eff  = (gflops / roof * 100.0) if gflops and roof > 0 else 0.0
            print("  %-28s AI=%.3f  achieved=%.1f  roofline=%.1f  eff=%.1f%%  [%s]" %
                  (kernel, ai, gflops if gflops else 0, roof, eff, bound))


if __name__ == "__main__":
    main()
