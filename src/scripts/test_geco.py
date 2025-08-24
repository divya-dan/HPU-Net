#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from hpunet.losses.geco import GECO, GECOConfig

def main():
    cfg = GECOConfig(kappa=0.05, alpha=0.99, lambda_init=1.0, step_size=1.0)
    geco = GECO(cfg).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Simulate a few recon values drifting toward kappa
    recon_vals = [0.20, 0.15, 0.10, 0.07, 0.05, 0.04, 0.03]
    kl = torch.tensor(0.01)

    for t, r in enumerate(recon_vals, 1):
        r_t = torch.tensor(r, dtype=torch.float32, device=geco._device)
        info = geco.step(r_t)
        L = geco.lagrangian(r_t, kl)
        print(f"t={t:02d}  recon={r:.3f}  λ={info['lambda']:.4f}  C={info['C']:.4f}  C̄={info['C_bar']:.4f}  L={float(L.item()):.4f}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
