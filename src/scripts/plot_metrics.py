#!/usr/bin/env python3
"""
Plot each metric (each unique `tag` in the CSV) in its own subplot.
CSV is expected to have columns: step, tag, value, wall_time

Examples
--------
python plot_metrics_grid.py --csv runs/metrics.csv
python plot_metrics_grid.py --csv runs/metrics.csv --x step --sharex --rolling 9
python plot_metrics_grid.py --csv runs/metrics.csv --include train eval --out runs/metrics_grid.png
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser(description="Plot each metric in its own subplot from a metrics.csv file.")
    ap.add_argument("--csv", type=Path, required=True, help="Path to metrics CSV (columns: step, tag, value, wall_time)")
    ap.add_argument("--out", type=Path, default=None, help="Output image path. Defaults to <csv_stem>_grid.png")
    ap.add_argument("--x", choices=["step", "wall_time"], default="step", help="X-axis: step or wall_time")
    ap.add_argument("--include", nargs="*", default=None,
                    help="Only include tags containing any of these substrings (case-sensitive)")
    ap.add_argument("--exclude", nargs="*", default=None,
                    help="Exclude tags containing any of these substrings (case-sensitive)")
    ap.add_argument("--rolling", type=int, default=0,
                    help="Centered rolling window size for smoothing (0 disables)")
    ap.add_argument("--sharex", action="store_true", help="Share X axis across subplots")
    ap.add_argument("--sharey", action="store_true", help="Share Y axis across subplots")
    ap.add_argument("--figsize", type=float, nargs=2, default=None,
                    help="Figure size in inches: W H (defaults scale with grid size)")
    ap.add_argument("--title", default=None, help="Optional figure title")
    return ap.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.csv)
    required = {"tag", "value", "step", "wall_time"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    # Collect unique tags and apply include/exclude filters
    tags = sorted(map(str, df["tag"].unique()))
    if args.include:
        tags = [t for t in tags if any(s in t for s in args.include)]
    if args.exclude:
        tags = [t for t in tags if not any(s in t for s in args.exclude)]
    if not tags:
        raise ValueError("No tags to plot after filtering.")

    n = len(tags)
    # Choose a grid that is roughly square but with at most 4 columns by default
    cols = min(4, max(1, math.ceil(math.sqrt(n))))
    rows = math.ceil(n / cols)

    if args.figsize is None:
        # Scale size with number of rows/cols
        figsize = (4.0 * cols, 3.0 * rows)
    else:
        figsize = tuple(args.figsize)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=120,
                             sharex=args.sharex, sharey=args.sharey)
    axes = np.atleast_1d(axes).ravel()

    xcol = args.x

    for i, tag in enumerate(tags):
        ax = axes[i]
        sub = df[df["tag"] == tag].sort_values(xcol)
        x = sub[xcol].to_numpy()
        y = sub["value"].to_numpy()

        if args.rolling and args.rolling > 1 and len(y) >= 2:
            # Centered rolling mean; keep edges reasonable
            y = pd.Series(y).rolling(window=args.rolling, center=True,
                                      min_periods=max(1, args.rolling // 2)).mean().to_numpy()

        ax.plot(x, y, linewidth=1.5)
        ax.set_title(tag)
        ax.grid(True, alpha=0.3)

    # Remove any extra axes if grid > number of tags
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Label axes (avoid over-labeling by setting only bottom row/left col when shared)
    for r in range(rows):
        for c in range(cols):
            k = r * cols + c
            if k >= n:
                continue
            ax = axes[k]
            if (not args.sharex) or r == rows - 1:
                ax.set_xlabel(xcol)
            if (not args.sharey) or c == 0:
                ax.set_ylabel("value")

    if args.title:
        fig.suptitle(args.title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    else:
        plt.tight_layout()

    out = args.out or args.csv.with_name(args.csv.stem + "_grid.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved plot to: {out}")


if __name__ == "__main__":
    main()
