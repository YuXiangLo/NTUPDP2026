#!/usr/bin/env python3
# bench_table.py - render bench.sh's results.csv as report-ready markdown tables.
#
#   python3 bench_table.py results.csv
#
# Prints, per image, a table of (seam-ratio x version) wall-clock + ms/seam,
# with the speedup of each version vs the v2 baseline. python3.6-safe.

import sys
import csv
from collections import OrderedDict


def main():
    if len(sys.argv) < 2:
        print("usage: python3 bench_table.py <results.csv>", file=sys.stderr)
        return 1
    path = sys.argv[1]

    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        print("no rows in %s" % path, file=sys.stderr)
        return 1

    # versions present, in a stable preferred order
    order = ["v2", "v4", "v5", "v6"]
    versions = [v for v in order if any(r["version"] == v for r in rows)]
    versions += sorted({r["version"] for r in rows} - set(versions))

    # group by image -> (pct,seams) -> version -> (total_ms, ms_per_seam)
    images = OrderedDict()
    for r in rows:
        img = r["image"]
        key = (int(r["pct"]), int(r["seams"]))
        images.setdefault(img, {}).setdefault(key, {})[r["version"]] = (
            float(r["best_total_ms"]),
            float(r["ms_per_seam"]),
        )

    base = versions[0]  # baseline for speedup (v2 if present)

    for img, data in images.items():
        anyrow = next(iter(data.values()))
        w = h = "?"
        for rr in rows:
            if rr["image"] == img:
                w, h = rr["width"], rr["height"]
                break
        print("\n### %s  (%sx%s)\n" % (img, w, h))

        head = ["seams (ratio)"]
        for v in versions:
            head.append("%s ms/seam" % v)
        for v in versions[1:]:
            head.append("%s vs %s" % (v, base))
        print("| " + " | ".join(head) + " |")
        print("|" + "|".join(["---"] * len(head)) + "|")

        for (pct, seams) in sorted(data.keys()):
            cells = data[(pct, seams)]
            line = ["%d (%d%%)" % (seams, pct)]
            base_per = cells.get(base, (None, None))[1]
            for v in versions:
                tot, per = cells.get(v, (None, None))
                line.append("%.4f" % per if per is not None else "-")
            for v in versions[1:]:
                per = cells.get(v, (None, None))[1]
                if per and base_per:
                    line.append("%.2fx" % (base_per / per))
                else:
                    line.append("-")
            print("| " + " | ".join(line) + " |")

    # compact cross-image summary at the best version
    best_ver = versions[-1]
    print("\n### summary — %s vs %s (ms/seam, best-of-runs)\n" % (best_ver, base))
    print("| image | resolution | seam-ratio | %s | %s | speedup |"
          % (base, best_ver))
    print("|---|---|---|---|---|---|")
    for img, data in images.items():
        w = h = "?"
        for rr in rows:
            if rr["image"] == img:
                w, h = rr["width"], rr["height"]
                break
        for (pct, seams) in sorted(data.keys()):
            cells = data[(pct, seams)]
            b = cells.get(base, (None, None))[1]
            x = cells.get(best_ver, (None, None))[1]
            sp = ("%.2fx" % (b / x)) if (b and x) else "-"
            print("| %s | %sx%s | %d%% | %s | %s | %s |" % (
                img, w, h, pct,
                "%.4f" % b if b else "-",
                "%.4f" % x if x else "-",
                sp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
