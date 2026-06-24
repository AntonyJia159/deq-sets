"""Disentangle: is short reach on roman-empire forced by the Jacobian penalty, or chosen by
the task?  Soft jac_reg penalizes ||J||_F^2, which over-taxes the benign diagonal eps*I
(N*eps^2 term, N~22k) -- at jac_gamma=1.0 the model crushed eps 0.40->0.15 and never grew s,
landing at rho(J)~0.5 (huge unused contraction margin).

Probe jac_gamma on ONE split (fast). If, as gamma->0, the model pushes s up / rho(J)->1 and
accuracy climbs, the penalty was the culprit (use a lighter / rho-targeted penalty). If reach
and accuracy stay flat even unpenalized, the bottleneck is the FAGCN cell's expressiveness,
not well-posedness -- the more important finding.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.fagcn_deq_jacsweep
"""

import time

import numpy as np
import torch

from experiments.fagcn_deq_smoke import load
from experiments.fagcn_deq_train import FAGCNDEQ, CFG, DEV, run_split

GAMMAS = [0.0, 0.01, 0.1, 1.0]
SPLIT = 0


def main():
    print(f"device = {DEV}")
    X, y, edges, deg, masks, K = load("roman_empire")
    X, y, edges, deg = X.to(DEV), y.to(DEV), edges.to(DEV), deg.to(DEV)
    tr = torch.tensor(masks["train_masks"][SPLIT].astype(bool)).to(DEV)
    va = torch.tensor(masks["val_masks"][SPLIT].astype(bool)).to(DEV)
    te = torch.tensor(masks["test_masks"][SPLIT].astype(bool)).to(DEV)
    print(f"roman_empire split {SPLIT}  (MLP floor 0.644, hard-clamp DEQ 0.622)\n")
    for g in GAMMAS:
        cfg = dict(CFG); cfg["jac_gamma"] = g
        t0 = time.time()
        a, d = run_split(X.shape[1], K, edges, deg, X, y, tr, va, te, cfg)
        print(f"  jac_gamma {g:<5}: test {a:.3f}  "
              f"[eps {d['eps']:.2f}  s {d['s']:.2f}  rho(J) {d['rho']:.3f}]  "
              f"({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
