"""(2) Soft cell + Broyden on roman-empire: is the non-contractive equilibrium real & accurate?

Hard variant (fagcn_deq_mlp.py, spectral-normed, Picard) tops out ~0.687 contractively; the
s_max sweep showed accuracy only rises as rho(J) crosses 1, but those rho>1 points were
Picard-TRUNCATED (no real fixed point). This is the SOFT variant of the Jacobian-control axis:
  - FREE weights (no spectral norm), LayerNorm INSIDE the cell (Bai et al. note LN matters for
    DEQ stability; we try post-norm on the ego||neighbor features).
  - solved with BROYDEN (quasi-Newton), which -- unlike Picard -- can find genuine fixed points
    with rho(J) > 1, as long as (I-J) is well-conditioned (sigma_min(I-J) away from 0).
  - CRITICAL: present the graph as (1, N, d) so TorchDEQ's batch_flatten treats it as ONE coupled
    system (the bug that made every prior Broyden run NaN; see broyden_synthetic.py).
  - light jac_reg for TRAINING stability only (keeps the forward solvable early on).

Gating questions: (a) does Broyden converge at roman scale (rel-residual), (b) what test acc does
the REAL equilibrium reach (does 0.70-0.72 become real?), (c) where is rho(J)? sigma_min(I-J) is
not formable at N~22k -- deferred to the locality experiment (3) via measured edit-response decay.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.fagcn_deq_broyden
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

CFG = dict(d=64, heads=4, mlp_hidden=64, s_init=1.0, jac_gamma=0.1,
           drop_in=0.5, drop_out=0.5, edge_drop=0.3,
           lr=5e-3, wd=1e-3, epochs=120, f_max_iter=30, f_tol=1e-4)
DATASETS = ["roman_empire"]
N_SPLITS = 1                                       # first run: one split, gauge convergence/acc/speed


def _inv_softplus(y):
    import math
    return math.log(math.expm1(y))


class SoftFAGCN(nn.Module):
    """Free-weight, LayerNorm-inside cell; no spectral norm. f operates on (1, N, d)."""
    def __init__(self, d_in, k, edges, deg, cfg):
        super().__init__()
        d, H, m = cfg["d"], cfg["heads"], cfg["mlp_hidden"]
        assert d % H == 0
        self.d, self.H, self.dh = d, H, d // H
        self.jac_gamma = cfg["jac_gamma"]
        self.enc = nn.Linear(d_in, d)
        self.Wp = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.att = nn.Parameter(torch.randn(H, 2 * self.dh) * 0.1)
        self.ln = nn.LayerNorm(2 * d)                       # post-norm on [ego || neighbor]
        self.W1 = nn.Linear(2 * d, m)
        self.W2 = nn.Linear(m, d, bias=False)
        self.head = nn.Sequential(
            nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(cfg["drop_out"]), nn.Linear(d, k))
        self.s_raw = nn.Parameter(torch.tensor(_inv_softplus(cfg["s_init"])))
        self.drop_in, self.edge_drop = cfg["drop_in"], cfg["edge_drop"]
        self.register_buffer("edges", edges)
        self.register_buffer("norm", 1.0 / torch.sqrt(deg[edges[0]] * deg[edges[1]]))
        self.deq = get_deq(f_solver="broyden", f_max_iter=cfg["f_max_iter"], f_tol=cfg["f_tol"])

    @property
    def s(self):
        return F.softplus(self.s_raw)

    def _mha(self, z, edges, norm):                        # z: (N, d)
        N, H, dh = z.shape[0], self.H, self.dh
        dst, src = edges[0], edges[1]
        if self.training and self.edge_drop > 0:
            keep = torch.rand(dst.shape[0], device=z.device) > self.edge_drop
            dst, src, norm = dst[keep], src[keep], norm[keep]
        p = self.Wp(z).view(N, H, dh)
        v = self.Wv(z).view(N, H, dh)
        alpha = torch.tanh((torch.cat([p[dst], p[src]], dim=-1) * self.att).sum(-1))
        msg = (alpha * norm.unsqueeze(-1)).unsqueeze(-1) * v[src]
        out = torch.zeros(N, H, dh, device=z.device, dtype=z.dtype)
        out.index_add_(0, dst, msg)
        return self.Wo(out.view(N, self.d))

    def _make_f(self, h0, edges, norm):                    # h0: (N, d)
        s = self.s

        def f(z):                                          # z: (1, N, d)
            zf = z[0]
            u = self.ln(torch.cat([zf, self._mha(zf, edges, norm)], dim=-1))
            g = self.W2(F.relu(self.W1(u)))
            return (h0 + s * g).unsqueeze(0)
        return f

    def forward(self, X, jac=False):
        h0 = self.enc(F.dropout(X, self.drop_in, self.training))
        f = self._make_f(h0, self.edges, self.norm)
        z0 = torch.zeros(1, *h0.shape, device=h0.device)
        z = self.deq(f, z0)[0][-1]
        reg = z.new_zeros(())
        if jac and self.jac_gamma > 0:
            z0d = z.detach().requires_grad_(True)
            reg = jac_reg(f(z0d), z0d, vecs=1)
        return self.head(torch.cat([z[0], h0], dim=-1)), reg

    @torch.no_grad()
    def diagnose(self, X):
        h0 = self.enc(X)
        f = self._make_f(h0, self.edges, self.norm)
        z0 = torch.zeros(1, *h0.shape, device=h0.device)
        z = self.deq(f, z0)[0][-1]
        resid = ((f(z) - z).norm() / (z.norm() + 1e-9)).item()
        with torch.enable_grad():
            z0d = z.detach().requires_grad_(True)
            _, rho = power_method(f(z0d), z0d, n_iters=30)
        return resid, rho.max().item()


def run_split(d_in, K, edges, deg, X, y, tr, va, te, cfg):
    torch.manual_seed(0)
    model = SoftFAGCN(d_in, K, edges, deg, cfg).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    best_va, best_te = 0.0, 0.0
    for ep in range(cfg["epochs"]):
        model.train(); opt.zero_grad()
        out, reg = model(X, jac=True)
        loss = F.cross_entropy(out[tr], y[tr]) + cfg["jac_gamma"] * reg
        if not torch.isfinite(loss):
            print(f"    epoch {ep}: non-finite loss (Broyden likely diverged) -- stopping", flush=True)
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                out, _ = model(X)
                va_acc = (out.argmax(1)[va] == y[va]).float().mean().item()
                te_acc = (out.argmax(1)[te] == y[te]).float().mean().item()
            resid, rho = model.diagnose(X)
            if va_acc > best_va:
                best_va, best_te = va_acc, te_acc
            print(f"    ep {ep:>3}: val {va_acc:.3f} test {te_acc:.3f}  "
                  f"[s {model.s.item():.2f} rho(J) {rho:.3f} broyden-resid {resid:.1e}]", flush=True)
    return best_te, model


def main():
    print(f"device = {DEV}")
    print("config:", CFG, "\n")
    for ds in DATASETS:
        X, y, edges, deg, masks, K = load(ds)
        X, y, edges, deg = X.to(DEV), y.to(DEV), edges.to(DEV), deg.to(DEV)
        print(f"{ds}: {X.shape[0]} nodes, {X.shape[1]} feat, {K} classes\n")
        for s in range(N_SPLITS):
            tr = torch.tensor(masks["train_masks"][s].astype(bool)).to(DEV)
            va = torch.tensor(masks["val_masks"][s].astype(bool)).to(DEV)
            te = torch.tensor(masks["test_masks"][s].astype(bool)).to(DEV)
            t0 = time.time()
            a, _ = run_split(X.shape[1], K, edges, deg, X, y, tr, va, te, CFG)
            print(f"  {ds} split {s}: best test {a:.3f}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
