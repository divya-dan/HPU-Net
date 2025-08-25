# src/scripts/plot_metrics.py
#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)  # columns: step, tag, value, wall_time
    # quick panels
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=120)
    ax = axes.ravel()

    def plot_tags(ax, prefix):
        sub = df[df["tag"].str.startswith(prefix)].copy()
        for t, g in sub.groupby("tag"):
            g = g.sort_values("step")
            ax.plot(g["step"], g["value"], label=t.split("/", 1)[-1])
        ax.set_title(prefix); ax.set_xlabel("step"); ax.legend(loc="best"); ax.grid(True, alpha=0.3)

    plot_tags(ax[0], "train")
    plot_tags(ax[1], "eval")

    # a couple of individual lines if present
    for i, tag in enumerate(["train/loss", "train/recon", "train/kl", "eval/HungIoU"]):
        sub = df[df["tag"] == tag]
        if not sub.empty:
            sub = sub.sort_values("step")
            ax[i//2].plot(sub["step"], sub["value"], linewidth=2.0)

    plt.tight_layout()
    out = args.out or args.csv.with_name("metrics_plot.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved plot to: {out}")

if __name__ == "__main__":
    main()
