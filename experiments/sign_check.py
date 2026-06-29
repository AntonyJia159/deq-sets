"""Does the edit-local PROPAGATE-then-MLP class (SIGN) match our equilibrium on the local targets?

The honest finite-depth baseline we were missing. SIGN precomputes a LINEAR multi-hop bank
[X, AX, ..., A^k X] (edit-local: finite receptive field) then applies a per-node MLP. Prediction:
  laplacian/biharmonic (L^k psi)^2 : L^k psi is IN the linear bank -> MLP squares it -> SIGN MATCHES us
  nbr_sq A(psi^2)                  : psi^2 not in a LINEAR bank -> SIGN FAILS (square is pre-prop)
If SIGN ties us on laplacian/biharmonic, those LOCAL targets do NOT separate the equilibrium from
finite-depth -- exactly the concern. (reach = edit-length = xi for an equilibrium, so an edit-local
equilibrium is short-reach == a depth-matched MPNN.)

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.sign_check
"""

import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.broyden_synthetic import grid_graph
from experiments.aniso_teacher import AnisoTeacher
from experiments.cora_deletion import renorm_sparse

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, D_FEAT, R, HOPS, N_SPLITS = 56, 16, 4, 3, 3
TARGETS = [("nbr_sq", 1), ("laplacian", 1), ("laplacian", 2)]


class SIGN(nn.Module):
    """prop-then-MLP: linear multi-hop bank [X,AX,...,A^HOPS X] then a DEEP per-node MLP (NONLINEAR).
    depth = number of hidden layers; deeper -> better approximates the quadratic-form target, which
    is the fair test of whether the DEQ's edge is architectural or just 'more depth'."""
    def __init__(self, d_in, h, depth=1, hops=HOPS, drop=0.5):
        super().__init__()
        self.hops, self.drop, self._cache = hops, drop, None
        dims = [d_in * (hops + 1)] + [h] * depth
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(depth))
        self.out = nn.Linear(h, 1)

    def forward(self, X, Ahat):
        if self._cache is None:
            S = [X]
            for _ in range(self.hops):
                S.append(torch.sparse.mm(Ahat, S[-1]))
            self._cache = torch.cat(S, dim=1).detach()
        h = self._cache
        for lin in self.layers:
            h = F.relu(lin(F.dropout(h, self.drop, self.training)))
        return self.out(h)


def splits(N, seed):
    g = torch.Generator().manual_seed(seed)
    p = torch.randperm(N, generator=g)
    tr = torch.zeros(N, dtype=torch.bool); va = tr.clone(); te = tr.clone()
    tr[p[: N // 2]] = True; va[p[N // 2: 3 * N // 4]] = True; te[p[3 * N // 4:]] = True
    return tr.to(DEV), va.to(DEV), te.to(DEV)


def r2(out, t, m):
    o = out[m].squeeze(-1); tm = t[m]
    return (1 - ((o - tm) ** 2).sum() / (((tm - tm.mean()) ** 2).sum() + 1e-9)).item()


def train(X, t, Ahat, tr, va, te, depth=1, epochs=300):
    m = SIGN(D_FEAT, 128, depth=depth).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=5e-4)
    bv, bt = -1e9, 0.0
    for e in range(epochs):
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
    print(f"grid {L}x{L}={N}, SIGN hops={HOPS} (prop-then-MLP)\n")
    for name, k in TARGETS:
        teacher = AnisoTeacher(edges, deg, N, d_feat=D_FEAT, R=R, k=k, seed=0, target=name)
        teacher.generate()
        X, t = teacher.X, teacher.s
        label = f"{name}" + (f" k={k}" if name == "laplacian" else "")
        for depth in (1, 2, 3):
            t0 = time.time()
            v = [train(X, t, Ahat, *splits(N, s), depth=depth) for s in range(N_SPLITS)]
            print(f"  SIGN(depth {depth}) on {label:<14} R^2 {np.mean(v):+.3f} +- {np.std(v):.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
