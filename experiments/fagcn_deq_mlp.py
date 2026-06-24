"""FAGCN-DEQ v3: nonlinear (IGNN-class) equilibrium with an ego-separated MLP update.

Motivation (vs v2 fagcn_deq_sep.py, which hit 0.682 contractively on roman_empire):
v2's map z' = eps*z + h0 + s*MHA(z) is LINEAR in the state (only the scalar attention coeff is
nonlinear) -> closer to a signed APPNP than to a real nonlinear GNN. IGNN's expressivity comes
from a nonlinearity applied to the aggregated update EVERY iteration. So v3:
  - DROPS the scalar ego term eps*z (algebraically redundant: in the resolvent it only rescales
    s_eff = s/(1-eps); the model drove eps->0.10 on its own; ego is preserved at the readout and
    via the h0 injection).
  - adds a REAL MLP-style projection inside the loop, RESPECTING EGO-SEPARATION: ego (z_i) and
    neighbor aggregate (nbr_i) enter as SEPARATE halves of a concat, then a 2-layer MLP
    g_i = W2 phi(W1 [z_i || nbr_i]).  W1's columns act on ego vs neighbor distinctly -> never the
    destructive shared-average GCN does.

Exact update (per iter; h0 = Enc(X) fixed):
  alpha_ij^h = tanh(a_h . [ (Wp z_i)^h || (Wp z_j)^h ])                       in [-1,1]
  nbr_i      = Wo * concat_h( sum_{j in N(i)} alpha_ij^h/sqrt(d_i d_j) (Wv z_j)^h )
  u_i        = [ z_i || nbr_i ]                                              (ego || neighbor)
  g_i        = W2 phi(W1 u_i + b1)                                           (ReLU MLP)
  z_i'       = h0_i + s * g_i
  readout    = Head([ z*_i || h0_i ])                                        (ego-sep at readout)

Contraction: Wp,Wv,Wo,W1,W2 spectral-normed to <=1; phi=ReLU 1-Lipschitz; concat-Jacobian [I;J_nbr]
has norm <= sqrt(1+||J_nbr||^2) <= sqrt(2), so Lip(f) <= sqrt(2)*s. Single knob s = s_max*sigmoid(.)
+ jac_reg on ||J||_F (controls the attention-gradient term SN doesn't bound) + power_method probe.
A result only counts if rho(J) < 1 (else it's a truncated deep net, not an equilibrium).

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.fagcn_deq_mlp
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

CFG = dict(d=128, heads=4, mlp_hidden=128, s_max=0.5, jac_gamma=1.0,
           drop_in=0.5, drop_out=0.5, edge_drop=0.3,
           lr=1e-2, wd=1e-3, epochs=200, f_max_iter=40, f_tol=1e-4)
DATASETS = ["roman_empire"]
N_SPLITS = 5


def _sn(W):
    """Spectral-normalize to operator norm 1 (exact; works for non-square too)."""
    return W / torch.linalg.matrix_norm(W, ord=2)


class FAGCNDEQMLP(nn.Module):
    def __init__(self, d_in, k, edges, deg, cfg):
        super().__init__()
        d, H, m = cfg["d"], cfg["heads"], cfg["mlp_hidden"]
        assert d % H == 0
        self.d, self.H, self.dh = d, H, d // H
        self.s_max, self.jac_gamma = cfg["s_max"], cfg["jac_gamma"]
        self.enc = nn.Linear(d_in, d)
        self.Wp = nn.Linear(d, d, bias=False)              # attention projection (SN'd)
        self.Wv = nn.Linear(d, d, bias=False)              # value transform     (SN'd)
        self.Wo = nn.Linear(d, d, bias=False)              # head mix            (SN'd)
        self.att = nn.Parameter(torch.randn(H, 2 * self.dh) * 0.1)
        self.W1 = nn.Linear(2 * d, m)                      # ego||neighbor -> hidden (SN'd)
        self.W2 = nn.Linear(m, d, bias=False)             # hidden -> d             (SN'd)
        self.head = nn.Sequential(                          # ego-sep readout on concat[z*, h0]
            nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(cfg["drop_out"]), nn.Linear(d, k))
        self.s_raw = nn.Parameter(torch.tensor(0.4))       # s = s_max*sigmoid(.) ~ 0.3 init
        self.drop_in, self.edge_drop = cfg["drop_in"], cfg["edge_drop"]
        self.register_buffer("edges", edges)
        self.register_buffer("norm", 1.0 / torch.sqrt(deg[edges[0]] * deg[edges[1]]))
        self.deq = get_deq(f_solver="fixed_point_iter", f_max_iter=cfg["f_max_iter"],
                           f_tol=cfg["f_tol"])

    @property
    def s(self):
        return self.s_max * torch.sigmoid(self.s_raw)

    def _mha(self, z, Wp_n, Wv_n, Wo_n):
        N, H, dh = z.shape[0], self.H, self.dh
        dst, src, norm = self.edges[0], self.edges[1], self.norm
        if self.training and self.edge_drop > 0:                  # DropEdge
            keep = torch.rand(dst.shape[0], device=z.device) > self.edge_drop
            dst, src, norm = dst[keep], src[keep], norm[keep]
        p = (z @ Wp_n.t()).view(N, H, dh)
        v = (z @ Wv_n.t()).view(N, H, dh)
        alpha = torch.tanh((torch.cat([p[dst], p[src]], dim=-1) * self.att).sum(-1))  # (E,H)
        msg = (alpha * norm.unsqueeze(-1)).unsqueeze(-1) * v[src]                      # (E,H,dh)
        out = torch.zeros(N, H, dh, device=z.device, dtype=z.dtype)
        out.index_add_(0, dst, msg)
        return (out.view(N, self.d)) @ Wo_n.t()

    def _make_f(self, h0):
        Wp_n, Wv_n, Wo_n = _sn(self.Wp.weight), _sn(self.Wv.weight), _sn(self.Wo.weight)
        W1_n, W2_n = _sn(self.W1.weight), _sn(self.W2.weight)
        s = self.s

        def f(z):
            nbr = self._mha(z, Wp_n, Wv_n, Wo_n)
            u = torch.cat([z, nbr], dim=-1)                       # ego || neighbor
            g = F.relu(u @ W1_n.t() + self.W1.bias) @ W2_n.t()    # ego-sep MLP
            return h0 + s * g
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
        h0 = self.enc(X)
        f = self._make_f(h0)
        z = self.deq(f, torch.zeros_like(h0))[0][-1]
        with torch.enable_grad():
            z0 = z.detach().requires_grad_(True)
            _, rho = power_method(f(z0), z0, n_iters=30)
        return rho.max().item()


def run_split(d_in, K, edges, deg, X, y, tr, va, te, cfg):
    torch.manual_seed(0)
    model = FAGCNDEQMLP(d_in, K, edges, deg, cfg).to(DEV)
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
    diag = dict(s=model.s.item(), rho=model.spectral_radius(X))
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
            print(f"  {ds} split {s}: test {a:.3f}  [s {d['s']:.2f}  rho(J) {d['rho']:.3f}]  "
                  f"({time.time()-t0:.1f}s)", flush=True)
        print(f"{ds:<20} FAGCN-DEQ-mlp test {np.mean(accs):.3f} +- {np.std(accs):.3f}", flush=True)


if __name__ == "__main__":
    main()
