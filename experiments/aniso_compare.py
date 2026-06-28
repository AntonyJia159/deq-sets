"""Step 1: who can learn the anisotropic-diffusion teacher? (REGRESSION on the continuous energy)

Teacher (aniso_teacher.py) target = sum_r a_r || B_dx^r (A_hat^t X) ||^2 : a (t+1)-hop-local,
anisotropic, high-pass, NONLINEAR (squared) energy. We regress the standardized scalar and report
test R^2 (a continuous target -> R^2 is far more sensitive than 5-way binning).

Predictions by construction:
  MLP (features only)        R^2 ~ 0  (label needs the graph)
  SGC / APPNP (low-pass lin) R^2 ~ 0  (low-pass kills the derivative; linear cannot square)
  Linear-HP (lin on multi-hop hi+lo bank) R^2 ~ 0  (has high-pass features but a LINEAR readout
                             cannot square -> isolates "beyond-linear", not just "beyond-low-pass")
  FAGCN-DEQ (ours)           R^2 high (signed/high-pass + nonlinear equilibrium)

Random 50/25/25 splits, 3 seeds. Run:
  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.aniso_compare
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
from experiments.fagcn_deq_mlp import FAGCNDEQMLP, CFG as DEQ_CFG
from experiments.cora_deletion import renorm_sparse

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, D_FEAT, R, K_HOPS = 30, 16, 4, 3
N_SPLITS = 3
DEQ = dict(DEQ_CFG, drop_in=0.1, drop_out=0.3, edge_drop=0.1, epochs=150)


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


def train_deq(edges, deg, d_in, X, t, tr, va, te):
    torch.manual_seed(0)
    m = FAGCNDEQMLP(d_in, 1, edges, deg, DEQ).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=DEQ["lr"], weight_decay=DEQ["wd"])
    bv, bt = -1e9, 0.0
    for e in range(DEQ["epochs"]):
        m.train(); opt.zero_grad()
        out, reg = m(X, jac=True)
        (F.mse_loss(out[tr].squeeze(-1), t[tr]) + DEQ["jac_gamma"] * reg).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if e % 10 == 0:
            m.eval()
            with torch.no_grad():
                out, _ = m(X)
            v = r2(out, t, va)
            if v > bv:
                bv, bt = v, r2(out, t, te)
    return bt


def hp_bank(X, Ahat):
    S = [X]
    for _ in range(K_HOPS):
        S.append(torch.sparse.mm(Ahat, S[-1]))
    hi = [S[i] - S[i + 1] for i in range(K_HOPS)]
    return torch.cat(S + hi, dim=1)


def main():
    print(f"device = {DEV}")
    edges, deg, N = grid_graph(L)
    edges, deg = edges.to(DEV), deg.to(DEV)
    teacher = AnisoTeacher(edges, deg, N, d_feat=D_FEAT, R=R, k=K_HOPS, seed=0)
    teacher.generate()
    X, t = teacher.X, teacher.s                              # regress the standardized energy
    A = sp.coo_matrix((np.ones(edges.shape[1]), (edges[0].cpu().numpy(), edges[1].cpu().numpy())),
                      shape=(N, N))
    Ahat = renorm_sparse(A).to(DEV)
    feat = hp_bank(X, Ahat)
    print(f"teacher: grid {L}x{L}={N}, d_feat={D_FEAT}, R={R}, k={K_HOPS} hops -> regress energy (R^2)\n")

    rows = {}
    builders = {"MLP (no graph)": lambda: MLP(D_FEAT, 64, 1),
                "SGC (low-pass lin)": lambda: SGC(D_FEAT, 1),
                "APPNP (low-pass lin)": lambda: APPNP(D_FEAT, 64, 1)}
    for name, b in builders.items():
        v, t0 = [], time.time()
        for s in range(N_SPLITS):
            torch.manual_seed(s)
            tr, va, te = splits(N, s)
            v.append(train_graph(b, X, t, Ahat, tr, va, te))
        rows[name] = np.mean(v)
        print(f"  {name:<22} R^2 {np.mean(v):+.3f} +- {np.std(v):.3f}  ({time.time()-t0:.0f}s)", flush=True)
    v, t0 = [], time.time()
    for s in range(N_SPLITS):
        tr, va, te = splits(N, s)
        v.append(train_linear_hp(feat, t, tr, va, te))
    rows["Linear-HP"] = np.mean(v)
    print(f"  {'Linear-HP (lin hi+lo)':<22} R^2 {np.mean(v):+.3f} +- {np.std(v):.3f}  ({time.time()-t0:.0f}s)",
          flush=True)
    v, t0 = [], time.time()
    for s in range(N_SPLITS):
        tr, va, te = splits(N, s)
        v.append(train_deq(edges, deg, D_FEAT, X, t, tr, va, te))
    rows["FAGCN-DEQ (ours)"] = np.mean(v)
    print(f"  {'FAGCN-DEQ (ours)':<22} R^2 {np.mean(v):+.3f} +- {np.std(v):.3f}  ({time.time()-t0:.0f}s)",
          flush=True)

    best_lin = max(rows["SGC (low-pass lin)"], rows["APPNP (low-pass lin)"], rows["Linear-HP"])
    print(f"\n  ours R^2 {rows['FAGCN-DEQ (ours)']:.3f} | best-linear R^2 {best_lin:.3f} | "
          f"MLP R^2 {rows['MLP (no graph)']:.3f}", flush=True)


if __name__ == "__main__":
    main()
