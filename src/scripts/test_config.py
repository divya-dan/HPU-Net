#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hpunet.utils.config import load_config

def main():
    root = Path(__file__).resolve().parents[2]
    cfg_spu = load_config(root / "configs/train_spu_lidc.json")
    cfg_hpu = load_config(root / "configs/train_hpu_lidc.json")

    print("SPU:", {k: getattr(cfg_spu, k) for k in ["total_steps","batch_size","lr","use_topk","recon_strategy"]})
    print("HPU:", {k: getattr(cfg_hpu, k) for k in ["total_steps","batch_size","lr","use_topk","recon_strategy"]})
    print("HPU GECO:", cfg_hpu.geco)

    # Example: override a field at runtime
    cfg_override = load_config(root / "configs/train_spu_lidc.json", overrides={"batch_size": 16, "lr": 5e-5})
    print("SPU overrides:", {"batch_size": cfg_override.batch_size, "lr": cfg_override.lr})

if __name__ == "__main__":
    main()
