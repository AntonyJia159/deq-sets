"""Decisive test: does going to EQUILIBRIUM buy expressivity over a finite K-step unroll of the SAME
cell, on a LOCAL (reach-k) target? Theory says no -- reach = edit-length = xi, so an edit-local
(contractive) equilibrium is just a depth-~xi weight-tied net; K>=reach Picard steps should already
match equilibrium. If they tie, the equilibrium's value is NOT local expressivity (it is maintenance
+ characterization). We also run an untied K-layer GIN as the practical interleaved finite-depth
baseline (more expressive per layer, but NO fixed point to warm-start -> costlier to maintain).

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.unroll_vs_eq
"""

import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.broyden_synthetic import grid_graph
from experiments.aniso_teacher import AnisoTeacher
from experiments.mpnn_deq import MPNNDEQ, CFG as MPNN_CFG
from experiments.cora_deletion import renorm_sparse

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, D_FEAT, R, N_SPLITS = 56, 16, 4, 3
TARGETS = [("laplacian", 1), ("laplacian", 2)]


def splits(N, seed):
    g = torch.Generator().manual_seed(seed)
    p = torch.randperm(N, generator=g)
    tr = torch.zeros(N, dtype=torch.bool); va = tr.clone(); te = tr.clone()
    tr[p[: N // 2]] = True; va[p[N // 2: 3 * N // 4]] = True; te[p[3 * N // 4:]] = True
    return tr.to(DEV), va.to(DEV), te.to(DEV)


def r2(out, t, m):
    o = out[m].squeeze(-1); tm = t[m]
    return (1 - ((o - tm) ** 2).sum() / (((tm - tm.mean()) ** 2).sum() + 1e-9)).item()


def train_mpnn(edges, deg, X, t, tr, va, te, f_iter):
    """Same MPNN cell; f_iter controls forward iterations (small=finite unroll, 40=equilibrium)."""
    cfg = dict(MPNN_CFG, f_max_iter=f_iter, f_tol=1e-9)   # tiny tol => exactly f_iter Picard steps
    torch.manual_seed(0)
    m = MPNNDEQ(D_FEAT, 1, edges, deg, cfg).to(DEV)
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


class GIN(nn.Module):
    """Untied K-layer interleaved GNN (GIN-style): h <- MLP((1+eps) h + Ahat h), per layer. The
    practical interleaved finite-depth baseline -- but no fixed point => no warm-start maintenance."""
    def __init__(self, d_in, h, K, drop=0.5):
        super().__init__()
        self.enc = nn.Linear(d_in, h)
        self.eps = nn.Parameter(torch.zeros(K))
        self.mlps = nn.ModuleList(
            nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h)) for _ in range(K))
        self.out = nn.Linear(h, 1)
        self.drop = drop

    def forward(self, X, Ahat):
        h = self.enc(F.dropout(X, self.drop, self.training))
        for k, mlp in enumerate(self.mlps):
            h = mlp((1 + self.eps[k]) * h + torch.sparse.mm(Ahat, h))
        return self.out(F.dropout(h, self.drop, self.training))


def train_gin(X, t, Ahat, tr, va, te, K):
    torch.manual_seed(0)
    m = GIN(D_FEAT, 64, K).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=5e-3)
    bv, bt = -1e9, 0.0
    for e in range(300):
        m.train(); opt.zero_grad()
        F.mse_loss(m(X, Ahat)[tr].squeeze(-1), t[tr]).backward(); opt.step()
        if e % 10 == 0:
            m.eval()
            with torch.no_grad():
                out = m(X, Ahat)
            v = r2(out, t, va)
            if v > bv:
                bv, bt = v, r2(out, t, te)
    return bt


def main():
    print(f"device = {DEV}")
    edges, deg, N = grid_graph(L)
    edges, deg = edges.to(DEV), deg.to(DEV)
    A = sp.coo_matrix((np.ones(edges.shape[1]), (edges[0].cpu().numpy(), edges[1].cpu().numpy())),
                      shape=(N, N))
    Ahat = renorm_sparse(A).to(DEV)
    print(f"grid {L}x{L}={N}, {N_SPLITS} splits -> test R^2\n")
    for name, k in TARGETS:
        teacher = AnisoTeacher(edges, deg, N, d_feat=D_FEAT, R=R, k=k, seed=0, target=name)
        teacher.generate()
        X, t = teacher.X, teacher.s
        print(f"== {name} k={k} (reach {k}) ==", flush=True)
        for fi in (1, 2, 3, 5, 40):
            t0 = time.time()
            v = [train_mpnn(edges, deg, X, t, *splits(N, s), f_iter=fi) for s in range(N_SPLITS)]
            tag = "equilibrium" if fi == 40 else f"unroll K={fi}"
            print(f"   MPNN {tag:<14} R^2 {np.mean(v):+.3f} +- {np.std(v):.3f}  ({time.time()-t0:.0f}s)",
                  flush=True)
        for K in (k, k + 1, 3):
            t0 = time.time()
            v = [train_gin(X, t, Ahat, *splits(N, s), K=K) for s in range(N_SPLITS)]
            print(f"   GIN  untied K={K:<7} R^2 {np.mean(v):+.3f} +- {np.std(v):.3f}  ({time.time()-t0:.0f}s)",
                  flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
