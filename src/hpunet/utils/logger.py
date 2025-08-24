from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import csv

class Logger:
    """
    Lightweight logger:
      • Always writes scalars to <outdir>/logs.csv
      • If TensorBoard is installed, also writes to <outdir> via SummaryWriter
    """
    def __init__(self, outdir: Path | str, use_tensorboard: bool = True):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        # CSV
        self.csv_path = self.outdir / "logs.csv"
        self._csv_file = self.csv_path.open("a", newline="")
        self._csv_writer: Optional[csv.DictWriter] = None
        self._header_written = self.csv_path.stat().st_size > 0  # append mode

        # TensorBoard (optional)
        self.tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore
                self.tb = SummaryWriter(log_dir=str(self.outdir))
            except Exception:
                self.tb = None  # silently fall back to CSV-only

    def log_scalars(self, step: int, scalars: Dict[str, float]):
        # CSV header (once)
        if not self._header_written:
            header = ["step"] + list(scalars.keys())
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=header)
            self._csv_writer.writeheader()
            self._header_written = True
        elif self._csv_writer is None:
            # header existed already; infer fieldnames from first row we write now
            header = ["step"] + list(scalars.keys())
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=header)

        row = {"step": int(step)}
        row.update({k: float(v) for k, v in scalars.items()})
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        # TensorBoard
        if self.tb is not None:
            for k, v in scalars.items():
                try:
                    self.tb.add_scalar(k, float(v), global_step=int(step))
                except Exception:
                    pass

    def close(self):
        try:
            if self.tb is not None:
                self.tb.flush()
                self.tb.close()
        finally:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception:
                pass
