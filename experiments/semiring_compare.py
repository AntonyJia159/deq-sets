"""THE SEMIRING 2x2: does the aggregator have to match the task's ALGEBRA?

Why the earlier expressivity comparison was a wash: manifold/spectral operators are FUNCTIONS OF THE
LAPLACIAN g(L) -> LINEAR functionals of X; the task (L^k X)^2 is phi(linear aggregate), which any
linear-propagation model with a nonlinear head computes. Both our MPNN (SUM aggregation) and the
InstantGNN-linear incumbent live in the SUM-PRODUCT semiring, so they tie on any spectral task.

The escape is an ORDER STATISTIC (max), which is NOT a function of any linear aggregate -> the
TROPICAL (max-plus) semiring (Viterbi/Bellman-Ford/DTW; neural algorithmic reasoning). Prediction:

  task \ aggregator     sum (linear semiring)        max (tropical)
  spectral (Laplacian)  GOOD                         (ok/worse)
  tropical (max-reach)  FAILS (sum can't be a max)   GOOD

Diagonal dominance => the aggregator must MATCH the task's algebra. The linear-push incumbent is
stuck in the SUM row forever (max-plus has no linear push); our framework spans BOTH rows and stays
edit-local in each (max-plus with discount gamma<1 is a sup-norm contraction). So a real, well-
motivated expressivity win that ALSO explains the earlier wash.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.semiring_compare
"""

import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.broyden_synthetic import grid_graph
from experiments.aniso_teacher import AnisoTeacher
from experiments.mpnn_deq import MPNNDEQ, CFG
from experiments.maintenance_compare import LinearPPRDEQ

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, D_FEAT, R, N_SPLITS = 40, 16, 4, 3
TASKS = [("laplacian", 2, "spectral (linear)"), ("maxreach", 1, "tropical (max-plus)")]


def splits(N, seed):
    g = torch.Generator().manual_seed(seed); p = torch.randperm(N, generator=g)
    tr = torch.zeros(N, dtype=torch.bool); va = tr.clone(); te = tr.clone()
    tr[p[: N // 2]] = True; va[p[N // 2: 3 * N // 4]] = True; te[p[3 * N // 4:]] = True
    return tr.to(DEV), va.to(DEV), te.to(DEV)


def r2(out, t, m):
    o = out[m].squeeze(-1); tm = t[m]
    return (1 - ((o - tm) ** 2).sum() / (((tm - tm.mean()) ** 2).sum() + 1e-9)).item()


def train(Cls, cfg, edges, deg, X, t, tr, va, te):
    torch.manual_seed(0)
    m = Cls(D_FEAT, 1, edges, deg, cfg).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    bv, bt = -1e9, 0.0
    for e in range(cfg["epochs"]):
        m.train(); opt.zero_grad()
        out, reg = m(X, jac=True)
        (F.mse_loss(out[tr].squeeze(-1), t[tr]) + cfg["jac_gamma"] * reg).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if e % 10 == 0:
            m.eval()
            with torch.no_grad():
                out, _ = m(X)
            v = r2(out, t, va)
            if v > bv:
                bv, bt = v, r2(out, t, te)
    return bt


def main():
    print(f"device = {DEV}")
    edges, deg, N = grid_graph(L)
    edges, deg = edges.to(DEV), deg.to(DEV)
    print(f"grid {L}x{L}={N}, {N_SPLITS} splits -> test R^2\n", flush=True)

    models = [("InstantGNN-linear (sum)", LinearPPRDEQ, dict(CFG)),
              ("MPNN-DEQ  sum-agg",        MPNNDEQ,      dict(CFG, agg="sum")),
              ("MPNN-DEQ  MAX-agg",        MPNNDEQ,      dict(CFG, agg="max"))]
    results = {}
    for tname, k, tlabel in TASKS:
        teacher = AnisoTeacher(edges, deg, N, d_feat=D_FEAT, R=R, k=k, seed=0, target=tname)
        teacher.generate()
        X, t = teacher.X, teacher.s
        print(f"== task: {tlabel} [{tname} k={k}] ==", flush=True)
        for mname, Cls, cfg in models:
            t0 = time.time()
            v = [train(Cls, cfg, edges, deg, X, t, *splits(N, s)) for s in range(N_SPLITS)]
            results[(tname, mname)] = (np.mean(v), np.std(v))
            print(f"   {mname:<26} R^2 {np.mean(v):+.3f} +- {np.std(v):.3f}  ({time.time()-t0:.0f}s)",
                  flush=True)
        print(flush=True)

    print("=== 2x2 SUMMARY (test R^2) ===", flush=True)
    print(f"{'model \\ task':<26} {'spectral':>12} {'tropical':>12}", flush=True)
    for mname in [m[0] for m in models]:
        sp = results[("laplacian", mname)][0]; tr_ = results[("maxreach", mname)][0]
        print(f"{mname:<26} {sp:>12.3f} {tr_:>12.3f}", flush=True)
    print("\nREAD: if sum-models win spectral & MAX-agg wins tropical => the aggregator must match the "
          "task's ALGEBRA; the linear incumbent is locked in the sum row, ours spans both.", flush=True)


if __name__ == "__main__":
    main()
