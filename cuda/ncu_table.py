#!/usr/bin/env python3
"""Print a tidy per-kernel table from an Nsight Compute report.

Usage:
    python3 ncu_table.py report_v3.ncu-rep        # report file (calls ncu)
    python3 ncu_table.py m.csv                     # already-exported CSV

Shows, for each distinct kernel (first occurrence = largest width pass):
    Duration (ms), SM throughput %, Memory throughput %, Achieved occupancy %.

Needs `ncu` in PATH (run `module load cuda` first) when given a .ncu-rep.
Profile beforehand with e.g.:
    ncu --set full -o report_v3 ./seam_carve_v3 ../Broadway_tower_edit.jpg 10 out3.png
"""
import csv
import io
import subprocess
import sys

COLS = [
    ("Kernel Name", "kernel"),
    ("gpu__time_duration.sum", "Dur(ms)"),
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "SM%"),
    ("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "Mem%"),
    ("sm__warps_active.avg.pct_of_peak_sustained_active", "Occ%"),
]


def load_rows(path):
    if path.endswith(".csv"):
        with open(path, newline="") as f:
            return list(csv.reader(f))
    # Otherwise treat it as an ncu report and shell out to convert it.
    cmd = ["ncu", "--import", path, "--csv", "--page", "raw"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except FileNotFoundError:
        sys.exit("error: `ncu` not found in PATH (run `module load cuda` first).")
    except subprocess.CalledProcessError as e:
        sys.exit("error: ncu import failed:\n" + (e.stderr or ""))
    return list(csv.reader(io.StringIO(out)))


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python3 ncu_table.py <report.ncu-rep | metrics.csv>")
    rows = load_rows(sys.argv[1])
    if len(rows) < 3:
        sys.exit("error: no kernel rows found in the report.")
    header = rows[0]
    try:
        idx = {name: header.index(name) for name, _ in COLS}
    except ValueError as e:
        sys.exit("error: expected metric column missing (%s). "
                 "Re-profile with `--set full`." % e)

    fmt = "%-22s%10s%8s%8s%8s"
    print(fmt % tuple(label for _, label in COLS))
    seen = set()
    for r in rows[2:]:  # rows[1] is the units row
        if len(r) <= max(idx.values()):
            continue
        kernel = r[idx["Kernel Name"]].split("(")[0]
        if not kernel or kernel in seen:
            continue
        seen.add(kernel)
        print(fmt % tuple(r[idx[name]] for name, _ in COLS))


if __name__ == "__main__":
    main()
