from __future__ import annotations
import csv, os, time
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


class Logger:
    """
    Tidy, append-only metrics:
        step, tag, value, wall_time
    - CSV at <outdir>/metrics.csv
    - optional TensorBoard
    - summary figure via save_summary_figure()
    """
    def __init__(self, outdir: str | os.PathLike, use_tensorboard: bool = True, tb_subdir: str = "tb"):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.outdir / "metrics.csv"
        self._fh = open(self.csv_path, "a", newline="")
        self._writer = csv.writer(self._fh)
        if self.csv_path.stat().st_size == 0:
            self._writer.writerow(["step", "tag", "value", "wall_time"])
            self._fh.flush()

        self.tb = None
        if use_tensorboard and SummaryWriter is not None:
            self.tb = SummaryWriter(log_dir=str(self.outdir / tb_subdir))

    @staticmethod
    def _now() -> float:
        import time as _time
        return _time.time()

    def log_scalars(self, step: int, scalars: dict[str, float]):
        """Log multiple metrics at the same 'step'."""
        wt = self._now()
        s = int(step)
        for tag, val in scalars.items():
            try:
                v = float(val)
            except Exception:
                continue
            self._writer.writerow([s, str(tag), v, wt])
            if self.tb is not None:
                self.tb.add_scalar(str(tag), v, s)
        self._fh.flush()

    # ---------- NEW: single-image summary ----------
    def save_summary_figure(self, out_path: str | os.PathLike, ema: float | None = 0.95):
        """
        Create a 2x3 dashboard PNG with common training/eval curves.
        - ema: exponential moving average coefficient (0..1). Use 0/None to disable.
        """
        import pandas as pd
        import matplotlib.pyplot as plt

        out_path = Path(out_path)
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            raise FileNotFoundError(f"No metrics at {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        if df.empty:
            raise RuntimeError("metrics.csv is empty")

        # coerce types and sort
        df["step"] = pd.to_numeric(df["step"], errors="coerce").fillna(0).astype(int)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.sort_values(["tag", "step"])
        tags = set(df["tag"].unique())

        def smooth(series):
            if not ema or ema <= 0 or ema >= 1:
                return series
            # EMA: y_t = ema*y_{t-1} + (1-ema)*x_t
            alpha = 1.0 - float(ema)
            return series.ewm(alpha=alpha, adjust=False).mean()

        # helper to draw a group of tags onto one axis
        def draw(ax, want_tags, title, ylabel=None, ylim=None):
            shown = False
            for t in want_tags:
                if t not in tags:  # skip gracefully
                    continue
                sub = df[df["tag"] == t]
                if sub.empty: 
                    continue
                y = smooth(sub["value"])
                ax.plot(sub["step"], y, label=t.split("/", 1)[-1])
                shown = True
            if shown:
                ax.set_title(title)
                if ylabel: ax.set_ylabel(ylabel)
                ax.set_xlabel("step")
                if ylim: ax.set_ylim(*ylim)
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best", fontsize=9)
            else:
                ax.axis("off")  # nothing to show

        # layout: 2 rows x 3 cols
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), dpi=140)
        ax00, ax01, ax02, ax10, ax11, ax12 = axes.ravel()

        # top row
        draw(ax00, ["train/loss", "train/recon", "train/kl"], "Train: loss/recon/kl", ylabel="value")
        draw(ax01, ["train/lr"], "Train: learning rate", ylabel="lr")
        # HPU-specific (ignored for sPU if not present)
        draw(ax02, ["train/lambda", "train/C", "train/C_bar"], "HPU GECO (if present)")

        # bottom row
        draw(ax10, ["eval/IoU_rec", "eval/HungIoU"], "Eval: IoU metrics", ylabel="IoU", ylim=(0, 1))
        draw(ax11, ["eval/GED2"], "Eval: GED\u00b2", ylabel="GED\u00b2")
        draw(ax12, ["eval/num"], "Eval: evaluated images", ylabel="#")

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    def close(self):
        try:
            if self.tb is not None:
                self.tb.flush(); self.tb.close()
        finally:
            try:
                self._fh.flush(); self._fh.close()
            except Exception:
                pass
