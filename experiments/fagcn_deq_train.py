"""Regularized training of the recurrentized FAGCN-DEQ (state-dependent signed attention).

Smoke (fagcn_deq_smoke.py) overfit badly (train 0.98 / test 0.34). This adds the standard
GNN regularization knobs and best-val early stopping, and reports test acc over the dataset's
standard splits so we can compare against the linear baselines (chameleon SGC 0.426, squirrel
SGC 0.402, MLP 0.396/0.367 from baseline_hetero.py).

Regularization knobs (reported at run time):
  - input feature dropout      (bag-of-words features are high-dim -> strong dropout helps)
  - dropout on z* before readout
  - DropEdge during training   (randomly drop edges from the attention each forward)
  - weight decay
  - best-validation early stopping (report test at best val, not final)

CONTRACTION CONTROL -- soft, not hard.  Earlier we hard-clamped (eps, s) so eps + s < 1.
That caps reach at ~1 hop (screening length s/(1-eps)), which on roman-empire (high-diameter
syntactic graph, MLP floor 0.644, graph-SOTA ~0.89) leaves the long-range signal unused.
Instead we now make eps and s *learnable* (softplus, so the model picks the reach it wants)
and hold well-posedness with a SOFT penalty on the Jacobian: jac_gamma * ||J||_F^2 estimated
via TorchDEQ's Hutchinson `jac_reg` at the equilibrium (Bai et al. 2021, Stabilizing
Equilibrium Models by Jacobian Regularization). We log the learned (eps, s) and the spectral
radius rho(J) (TorchDEQ `power_method`) so we can see (a) whether the model pushes reach up and
(b) whether it stays contractive (rho(J) < 1) on its own.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.fagcn_deq_train
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdeq import get_deq
from torchdeq.loss import jac_reg, power_method

from experiments.fagcn_deq_smoke import load

DEV = "cuda" if torch.cuda.is_available() else "cpu"

CFG = dict(d=64, eps0=0.4, s0=0.3, jac_gamma=1.0,         # eps0,s0 = init; jac_gamma = soft penalty
           drop_in=0.6, drop_out=0.5, edge_drop=0.3,
           lr=1e-2, wd=1e-3, epochs=200, f_max_iter=40, f_tol=1e-4)
DATASETS = ["roman_empire"]                     # the discriminating battleground; run alone first
N_SPLITS = 5


def _inv_softplus(y):
    return math.log(math.expm1(y))             # raw r s.t. softplus(r) = y


class FAGCNDEQ(nn.Module):
    def __init__(self, d_in, k, edges, deg, cfg):
        super().__init__()
        d = cfg["d"]
        self.enc = nn.Linear(d_in, d)
        self.att = nn.Linear(2 * d, 1, bias=False)
        self.ro = nn.Linear(d, k)
        # learnable, kept positive via softplus; no hard eps + s < 1 clamp
        self.eps_raw = nn.Parameter(torch.tensor(_inv_softplus(cfg["eps0"])))
        self.s_raw = nn.Parameter(torch.tensor(_inv_softplus(cfg["s0"])))
        self.jac_gamma = cfg["jac_gamma"]
        self.drop_in, self.drop_out, self.edge_drop = cfg["drop_in"], cfg["drop_out"], cfg["edge_drop"]
        self.register_buffer("edges", edges)
        self.register_buffer("norm", 1.0 / torch.sqrt(deg[edges[0]] * deg[edges[1]]))
        self.deq = get_deq(f_solver="fixed_point_iter", f_max_iter=cfg["f_max_iter"],
                           f_tol=cfg["f_tol"])

    @property
    def eps(self):
        return F.softplus(self.eps_raw)

    @property
    def s(self):
        return F.softplus(self.s_raw)

    def aggregate(self, z):
        dst, src, norm = self.edges[0], self.edges[1], self.norm
        if self.training and self.edge_drop > 0:                 # DropEdge
            keep = torch.rand(dst.shape[0], device=z.device) > self.edge_drop
            dst, src, norm = dst[keep], src[keep], norm[keep]
        alpha = torch.tanh(self.att(torch.cat([z[dst], z[src]], dim=-1)).squeeze(-1))
        out = torch.zeros_like(z)
        out.index_add_(0, dst, (alpha * norm).unsqueeze(-1) * z[src])
        return out

    def forward(self, X, jac=False):
        h0 = self.enc(F.dropout(X, self.drop_in, self.training))

        def f(z):
            return self.eps * z + h0 + self.s * self.aggregate(z)

        z = self.deq(f, torch.zeros_like(h0))[0][-1]
        reg = z.new_zeros(())
        if jac and self.jac_gamma > 0:                  # soft Jacobian penalty at equilibrium
            z0 = z.detach().requires_grad_(True)
            reg = jac_reg(f(z0), z0, vecs=1)
        z = F.dropout(z, self.drop_out, self.training)
        return self.ro(z), reg

    @torch.no_grad()
    def spectral_radius(self, X):
        """rho(J) at the equilibrium via power iteration (well-posedness probe)."""
        h0 = self.enc(X)

        def f(z):
            return self.eps * z + h0 + self.s * self.aggregate(z)

        z = self.deq(f, torch.zeros_like(h0))[0][-1]
        with torch.enable_grad():
            z0 = z.detach().requires_grad_(True)
            _, rho = power_method(f(z0), z0, n_iters=30)
        return rho.max().item()


def run_split(d_in, K, edges, deg, X, y, tr, va, te, cfg):
    torch.manual_seed(0)
    model = FAGCNDEQ(d_in, K, edges, deg, cfg).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    best_va, best_te = 0.0, 0.0
    for ep in range(cfg["epochs"]):
        model.train(); opt.zero_grad()
        out, reg = model(X, jac=True)
        loss = F.cross_entropy(out[tr], y[tr]) + cfg["jac_gamma"] * reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                out, _ = model(X)
                va_acc = (out.argmax(1)[va] == y[va]).float().mean().item()
                te_acc = (out.argmax(1)[te] == y[te]).float().mean().item()
            if va_acc > best_va:
                best_va, best_te = va_acc, te_acc
    model.eval()
    diag = dict(eps=model.eps.item(), s=model.s.item(), rho=model.spectral_radius(X))
    return best_te, diag


def main():
    print(f"device = {DEV}")
    print("config:", CFG, "\n")
    for ds in DATASETS:
        X, y, edges, deg, masks, K = load(ds)
        X, y, edges, deg = X.to(DEV), y.to(DEV), edges.to(DEV), deg.to(DEV)
        import time
        accs = []
        for s in range(N_SPLITS):
            tr = torch.tensor(masks["train_masks"][s].astype(bool)).to(DEV)
            va = torch.tensor(masks["val_masks"][s].astype(bool)).to(DEV)
            te = torch.tensor(masks["test_masks"][s].astype(bool)).to(DEV)
            t0 = time.time()
            a, d = run_split(X.shape[1], K, edges, deg, X, y, tr, va, te, CFG)
            accs.append(a)
            print(f"  {ds} split {s}: test {a:.3f}  "
                  f"[eps {d['eps']:.2f}  s {d['s']:.2f}  rho(J) {d['rho']:.3f}]  "
                  f"({time.time()-t0:.1f}s)", flush=True)
        print(f"{ds:<20} FAGCN-DEQ test {np.mean(accs):.3f} +- {np.std(accs):.3f}", flush=True)


if __name__ == "__main__":
    main()
