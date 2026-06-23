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

Contraction control unchanged from smoke: (eps, s) with eps + s < 1; solver verified to converge.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.fagcn_deq_train
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdeq import get_deq

from experiments.fagcn_deq_smoke import load

DEV = "cuda" if torch.cuda.is_available() else "cpu"

CFG = dict(d=64, eps=0.4, s=0.3, drop_in=0.6, drop_out=0.5, edge_drop=0.3,
           lr=1e-2, wd=1e-3, epochs=200, f_max_iter=30, f_tol=1e-4)  # s=0.3 -> solve ~10 iters
DATASETS = ["roman_empire"]                     # the discriminating battleground; run alone first
N_SPLITS = 5


class FAGCNDEQ(nn.Module):
    def __init__(self, d_in, k, edges, deg, cfg):
        super().__init__()
        d = cfg["d"]
        self.enc = nn.Linear(d_in, d)
        self.att = nn.Linear(2 * d, 1, bias=False)
        self.ro = nn.Linear(d, k)
        self.eps, self.s = cfg["eps"], cfg["s"]
        self.drop_in, self.drop_out, self.edge_drop = cfg["drop_in"], cfg["drop_out"], cfg["edge_drop"]
        self.register_buffer("edges", edges)
        self.register_buffer("norm", 1.0 / torch.sqrt(deg[edges[0]] * deg[edges[1]]))
        self.deq = get_deq(f_solver="fixed_point_iter", f_max_iter=cfg["f_max_iter"],
                           f_tol=cfg["f_tol"])

    def aggregate(self, z):
        dst, src, norm = self.edges[0], self.edges[1], self.norm
        if self.training and self.edge_drop > 0:                 # DropEdge
            keep = torch.rand(dst.shape[0], device=z.device) > self.edge_drop
            dst, src, norm = dst[keep], src[keep], norm[keep]
        alpha = torch.tanh(self.att(torch.cat([z[dst], z[src]], dim=-1)).squeeze(-1))
        out = torch.zeros_like(z)
        out.index_add_(0, dst, (alpha * norm).unsqueeze(-1) * z[src])
        return out

    def forward(self, X):
        h0 = self.enc(F.dropout(X, self.drop_in, self.training))

        def f(z):
            return self.eps * z + h0 + self.s * self.aggregate(z)

        z = self.deq(f, torch.zeros_like(h0))[0][-1]
        z = F.dropout(z, self.drop_out, self.training)
        return self.ro(z)


def run_split(d_in, K, edges, deg, X, y, tr, va, te, cfg):
    torch.manual_seed(0)
    model = FAGCNDEQ(d_in, K, edges, deg, cfg).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    best_va, best_te = 0.0, 0.0
    for ep in range(cfg["epochs"]):
        model.train(); opt.zero_grad()
        out = model(X)
        F.cross_entropy(out[tr], y[tr]).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                out = model(X)
                va_acc = (out.argmax(1)[va] == y[va]).float().mean().item()
                te_acc = (out.argmax(1)[te] == y[te]).float().mean().item()
            if va_acc > best_va:
                best_va, best_te = va_acc, te_acc
    return best_te


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
            a = run_split(X.shape[1], K, edges, deg, X, y, tr, va, te, CFG)
            accs.append(a)
            print(f"  {ds} split {s}: test {a:.3f}  ({time.time()-t0:.1f}s)", flush=True)
        print(f"{ds:<20} FAGCN-DEQ test {np.mean(accs):.3f} +- {np.std(accs):.3f}", flush=True)


if __name__ == "__main__":
    main()
