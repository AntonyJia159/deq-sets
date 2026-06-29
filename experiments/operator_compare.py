"""Who can learn the local-operator teacher family? FAGCN-DEQ vs general MPNN-DEQ vs baselines.

Targets (aniso_teacher.py), all leak-light + local + nonlinear, regressed by test R^2:
  nbr_sq      s_i = sum_r a_r mean_{j~i}(X_j w_r)^2          (square-then-average; reach 1)
  laplacian   s_i = sum_r a_r ((L   psi_r)_i)^2             (Dirichlet energy;  propagate-then-square, reach 1)
  biharmonic  s_i = sum_r a_r ((L^2 psi_r)_i)^2             (thin-plate Delta^2; propagate-then-square, reach 2)

Predictions:
  SGC / Linear-HP (linear)   ~0 on all (cannot square)        -> the beyond-LINEAR control
  APPNP (square-then-prop)   partial on nbr_sq, ~0 on lap/biharm (wrong ORDER for propagate-then-square)
  MLP (features only)        ~0 (nbr_sq leak-free; lap/biharm only a small center fraction)
  FAGCN-DEQ                  good where the op is square-of-a-LINEAR-aggregate
  MPNN-DEQ (per-edge nonlin) >= FAGCN, and the gap should widen with operator order (biharmonic)

This is the conversion-principle demo: two different cells, both cast as edit-local equilibria.
Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.operator_compare
"""

import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.broyden_synthetic import grid_graph
from experiments.aniso_teacher import AnisoTeacher
from experiments.hetero_headtohead import MLP, SGC, APPNP
from experiments.fagcn_deq_mlp import FAGCNDEQMLP, CFG as FAGCN_CFG
from experiments.mpnn_deq import MPNNDEQ, CFG as MPNN_CFG
from experiments.cora_deletion import renorm_sparse

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, D_FEAT, R = 56, 16, 4
N_SPLITS = 3
TARGETS = [("nbr_sq", 1), ("laplacian", 1), ("laplacian", 2)]   # (name, operator order k)
FAGCN = dict(FAGCN_CFG, drop_in=0.1, drop_out=0.5, edge_drop=0.3, wd=5e-3, jac_gamma=0.5, epochs=200)


def splits(N, seed):
    g = torch.Generator().manual_seed(seed)
    p = torch.randperm(N, generator=g)
    tr = torch.zeros(N, dtype=torch.bool); va = tr.clone(); te = tr.clone()
    tr[p[: N // 2]] = True; va[p[N // 2: 3 * N // 4]] = True; te[p[3 * N // 4:]] = True
    return tr.to(DEV), va.to(DEV), te.to(DEV)


def r2(out, t, m):
    o = out[m].squeeze(-1); tm = t[m]
    return (1 - ((o - tm) ** 2).sum() / (((tm - tm.mean()) ** 2).sum() + 1e-9)).item()


def train_graph(build, X, t, Ahat, tr, va, te, epochs=200):
    m = build().to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=5e-4)
    bv, bt = -1e9, 0.0
    for e in range(epochs):
        m.train(); opt.zero_grad()
        F.mse_loss(m(X, Ahat, None)[tr].squeeze(-1), t[tr]).backward(); opt.step()
        if e % 10 == 0:
            m.eval()
            with torch.no_grad():
                out = m(X, Ahat, None)
            v = r2(out, t, va)
            if v > bv:
                bv, bt = v, r2(out, t, te)
    return bt


def train_linear_hp(feat, t, tr, va, te, epochs=300):
    lin = nn.Linear(feat.shape[1], 1).to(DEV)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=5e-4)
    bv, bt = -1e9, 0.0
    for e in range(epochs):
        lin.train(); opt.zero_grad()
        F.mse_loss(lin(feat)[tr].squeeze(-1), t[tr]).backward(); opt.step()
        if e % 10 == 0:
            lin.eval()
            with torch.no_grad():
                out = lin(feat)
            v = r2(out, t, va)
            if v > bv:
                bv, bt = v, r2(out, t, te)
    return bt


def train_deq(Cell, cfg, edges, deg, X, t, tr, va, te):
    torch.manual_seed(0)
    m = Cell(X.shape[1], 1, edges, deg, cfg).to(DEV)
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
    return bt, m.spectral_radius(X)


def hp_bank(X, Ahat, hops=3):
    S = [X]
    for _ in range(hops):
        S.append(torch.sparse.mm(Ahat, S[-1]))
    hi = [S[i] - S[i + 1] for i in range(hops)]
    return torch.cat(S + hi, dim=1)


def main():
    print(f"device = {DEV}")
    edges, deg, N = grid_graph(L)
    edges, deg = edges.to(DEV), deg.to(DEV)
    A = sp.coo_matrix((np.ones(edges.shape[1]), (edges[0].cpu().numpy(), edges[1].cpu().numpy())),
                      shape=(N, N))
    Ahat = renorm_sparse(A).to(DEV)
    print(f"grid {L}x{L}={N}, d_feat={D_FEAT}, R={R}, {N_SPLITS} splits -> test R^2\n")

    for name, k in TARGETS:
        teacher = AnisoTeacher(edges, deg, N, d_feat=D_FEAT, R=R, k=k, seed=0, target=name)
        teacher.generate()
        X, t = teacher.X, teacher.s
        feat = hp_bank(X, Ahat)
        label = f"{name}" + (f" k={k}" if name == "laplacian" else "")
        print(f"== target {label} (reach {k}) ==", flush=True)

        def run(fn):
            v = [fn(s) for s in range(N_SPLITS)]
            return np.mean(v), np.std(v)

        for nm, build in [("MLP (no graph)", lambda: MLP(D_FEAT, 64, 1)),
                          ("SGC (linear)", lambda: SGC(D_FEAT, 1)),
                          ("APPNP (MLP+prop)", lambda: APPNP(D_FEAT, 64, 1))]:
            t0 = time.time()
            mu, sd = run(lambda s: train_graph(build, X, t, Ahat, *splits(N, s)))
            print(f"   {nm:<18} R^2 {mu:+.3f} +- {sd:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        t0 = time.time()
        mu, sd = run(lambda s: train_linear_hp(feat, t, *splits(N, s)))
        print(f"   {'Linear-HP':<18} R^2 {mu:+.3f} +- {sd:.3f}  ({time.time()-t0:.0f}s)", flush=True)

        for nm, Cell, cfg in [("FAGCN-DEQ", FAGCNDEQMLP, FAGCN), ("MPNN-DEQ", MPNNDEQ, MPNN_CFG)]:
            t0 = time.time()
            res = [train_deq(Cell, cfg, edges, deg, X, t, *splits(N, s)) for s in range(N_SPLITS)]
            r, rho = np.mean([x[0] for x in res]), np.mean([x[1] for x in res])
            sd = np.std([x[0] for x in res])
            print(f"   {nm:<18} R^2 {r:+.3f} +- {sd:.3f}  [rho(J) {rho:.2f}]  ({time.time()-t0:.0f}s)",
                  flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
