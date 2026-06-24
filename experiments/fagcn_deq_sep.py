"""FAGCN-DEQ v2: GAT-sep-style local expressivity inside a *genuinely contractive* equilibrium.

Diagnosis from fagcn_deq_train.py: jac_gamma=0 still gave ~0.62 (below the 0.644 MLP floor),
so well-posedness was never the binding constraint -- LOCAL EXPRESSIVITY is. On roman-empire a
purely *local* model (GAT-sep) already reaches 0.888; global attention only adds the last ~3.5pt.
So the win is reachable inside our locality-preserving (=maintainable) paradigm -- we just have
to make the per-hop cell as expressive as GAT-sep, while keeping the map contractive.

What v2 adds over v1 (all local, all maintainable):
  - MULTI-HEAD signed attention (H heads), not a single scalar attention head.
  - learned value transform W_v and output projection W_o, spectral-normalized to 1.
  - EGO-SEPARATION AT THE READOUT: classify on concat[z*, h0] -> floor at >= MLP.

CONTRACTION IS NOW ENFORCED, not hoped for.  An earlier unconstrained version hit 0.690 but with
rho(J) = 2.5-4.6 and eps+s -> 1.7: it had *abandoned* contraction and become a truncated 40-layer
net, which voids the maintenance thesis (no attracting fixed point). So we now:
  - HARD-CAP the linear part: [eps, s, slack] = budget * softmax(.), so eps + s <= budget < 1
    (value path SN'd to 1, |alpha|<=1, rho(M)<=1  =>  linear-part spectral radius <= eps+s).
  - SOFT-control the attention-gradient term (which the value-path SN does NOT bound) with a
    jac_reg penalty on ||J||_F, and SN the attention projection W_p as well.
  - REPORT rho(J): a result only counts if rho(J) < 1.  The honest question this run answers is
    "what is the ceiling of a *truly contractive* (=maintainable) local equilibrium on roman?"

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.fagcn_deq_sep
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdeq import get_deq
from torchdeq.loss import jac_reg, power_method

from experiments.fagcn_deq_smoke import load

DEV = "cuda" if torch.cuda.is_available() else "cpu"

CFG = dict(d=64, heads=4, budget=0.9, jac_gamma=1.0,        # budget = hard cap on eps+s (<1)
           drop_in=0.6, drop_out=0.5, edge_drop=0.3,
           lr=1e-2, wd=1e-3, epochs=200, f_max_iter=40, f_tol=1e-4)
DATASETS = ["roman_empire"]
N_SPLITS = 5


def _sn(W):
    """Spectral-normalize a weight matrix to operator norm 1 (exact, d is small)."""
    return W / torch.linalg.matrix_norm(W, ord=2)


class FAGCNDEQSep(nn.Module):
    def __init__(self, d_in, k, edges, deg, cfg):
        super().__init__()
        d, H = cfg["d"], cfg["heads"]
        assert d % H == 0
        self.d, self.H, self.dh = d, H, d // H
        self.budget, self.jac_gamma = cfg["budget"], cfg["jac_gamma"]
        self.enc = nn.Linear(d_in, d)
        self.Wp = nn.Linear(d, d, bias=False)              # attention projection (SN'd in forward)
        self.Wv = nn.Linear(d, d, bias=False)              # value transform     (SN'd in forward)
        self.Wo = nn.Linear(d, d, bias=False)              # output projection   (SN'd in forward)
        self.att = nn.Parameter(torch.randn(H, 2 * self.dh) * 0.1)
        self.head = nn.Sequential(                          # ego-sep readout on concat[z*, h0]
            nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(cfg["drop_out"]), nn.Linear(d, k))
        # [eps, s, slack] = budget * softmax(mix)  =>  eps + s <= budget < 1  (hard contraction cap)
        self.mix = nn.Parameter(torch.log(torch.tensor([0.4, 0.3, 0.2])))
        self.drop_in, self.edge_drop = cfg["drop_in"], cfg["edge_drop"]
        self.register_buffer("edges", edges)
        self.register_buffer("norm", 1.0 / torch.sqrt(deg[edges[0]] * deg[edges[1]]))
        self.deq = get_deq(f_solver="fixed_point_iter", f_max_iter=cfg["f_max_iter"],
                           f_tol=cfg["f_tol"])

    @property
    def _eps_s(self):
        w = self.budget * F.softmax(self.mix, dim=0)
        return w[0], w[1]

    @property
    def eps(self):
        return self._eps_s[0]

    @property
    def s(self):
        return self._eps_s[1]

    def _mha(self, z, Wp_n, Wv_n, Wo_n):
        N, H, dh = z.shape[0], self.H, self.dh
        dst, src, norm = self.edges[0], self.edges[1], self.norm
        if self.training and self.edge_drop > 0:                  # DropEdge
            keep = torch.rand(dst.shape[0], device=z.device) > self.edge_drop
            dst, src, norm = dst[keep], src[keep], norm[keep]
        p = (z @ Wp_n.t()).view(N, H, dh)
        v = (z @ Wv_n.t()).view(N, H, dh)
        e_in = torch.cat([p[dst], p[src]], dim=-1)               # (E, H, 2dh)
        alpha = torch.tanh((e_in * self.att).sum(-1))            # (E, H)  signed, in [-1,1]
        msg = (alpha * norm.unsqueeze(-1)).unsqueeze(-1) * v[src]  # (E, H, dh)
        out = torch.zeros(N, H, dh, device=z.device, dtype=z.dtype)
        out.index_add_(0, dst, msg)
        return (out.view(N, self.d)) @ Wo_n.t()

    def _make_f(self, h0):
        Wp_n, Wv_n, Wo_n = _sn(self.Wp.weight), _sn(self.Wv.weight), _sn(self.Wo.weight)
        eps, s = self._eps_s

        def f(z):
            return eps * z + h0 + s * self._mha(z, Wp_n, Wv_n, Wo_n)
        return f

    def forward(self, X, jac=False):
        h0 = self.enc(F.dropout(X, self.drop_in, self.training))
        f = self._make_f(h0)
        z = self.deq(f, torch.zeros_like(h0))[0][-1]
        reg = z.new_zeros(())
        if jac and self.jac_gamma > 0:
            z0 = z.detach().requires_grad_(True)
            reg = jac_reg(f(z0), z0, vecs=1)
        return self.head(torch.cat([z, h0], dim=-1)), reg

    @torch.no_grad()
    def spectral_radius(self, X):
        f = self._make_f(self.enc(X))
        z = self.deq(f, torch.zeros_like(self.enc(X)))[0][-1]
        with torch.enable_grad():
            z0 = z.detach().requires_grad_(True)
            _, rho = power_method(f(z0), z0, n_iters=30)
        return rho.max().item()


def run_split(d_in, K, edges, deg, X, y, tr, va, te, cfg):
    torch.manual_seed(0)
    model = FAGCNDEQSep(d_in, K, edges, deg, cfg).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    best_va, best_te = 0.0, 0.0
    for ep in range(cfg["epochs"]):
        model.train(); opt.zero_grad()
        out, reg = model(X, jac=True)
        (F.cross_entropy(out[tr], y[tr]) + cfg["jac_gamma"] * reg).backward()
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
        print(f"{ds:<20} FAGCN-DEQ-sep test {np.mean(accs):.3f} +- {np.std(accs):.3f}", flush=True)


if __name__ == "__main__":
    main()
