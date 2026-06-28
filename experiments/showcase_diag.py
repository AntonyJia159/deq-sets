"""Showcase search: which Platonov task has a large BEYOND-LINEAR graph margin (and is local)?

roman-empire is adversarial to us: its useful graph signal is long-range, so accuracy beyond ~0.68
needs reach that fights locality. We want a task where the graph margin is large AND the signal is
local (so short screening is optimal and our maintenance is a feature). Test the datasets we never
ran -- minesweeper (a 100x100 GRID: local, nonlinear/constraint), tolokers, questions -- comparing
the feature floor (MLP) and LINEAR propagation (SGC/APPNP) against our contractive nonlinear cell.

These are BINARY tasks scored by ROC-AUC (class-imbalanced), so we report AUC, not accuracy.

Favorability = our_cell well above BOTH MLP and SGC/APPNP (beyond-linear is genuinely needed) on a
LOCAL graph (minesweeper = grid; tolokers/questions = social/small-world, less local).

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.showcase_diag
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata

from experiments.hetero_headtohead import load as load_sparse, MLP, SGC, APPNP
from experiments.fagcn_deq_smoke import load as load_edges
from experiments.fagcn_deq_mlp import FAGCNDEQMLP, CFG as DEQ_CFG
from experiments.cora_deletion import renorm_sparse

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DATASETS = ["minesweeper", "tolokers", "questions"]
N_SPLITS = 3


def auc(scores, labels):
    r = rankdata(scores)
    pos = labels == 1
    npos, nneg = pos.sum(), (~pos).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def score(out, y, mask, binary):
    if binary:
        s = torch.softmax(out, 1)[:, 1].detach().cpu().numpy()
        m = mask.detach().cpu().numpy()
        return auc(s[m], y.detach().cpu().numpy()[m])
    return (out.argmax(1)[mask] == y[mask]).float().mean().item()


def train_linear(build, X, y, Ahat, tr, va, te, binary, epochs=200):
    m = build().to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=5e-4)
    bv, bt = -1.0, 0.0
    for e in range(epochs):
        m.train(); opt.zero_grad()
        out = m(X, Ahat, None)
        F.cross_entropy(out[tr], y[tr]).backward(); opt.step()
        if e % 10 == 0:
            m.eval()
            with torch.no_grad():
                out = m(X, Ahat, None)
            v = score(out, y, va, binary)
            if v > bv:
                bv, bt = v, score(out, y, te, binary)
    return bt


def train_deq(edges, deg, d_in, K, X, y, tr, va, te, binary, epochs=120):
    torch.manual_seed(0)
    m = FAGCNDEQMLP(d_in, K, edges, deg, DEQ_CFG).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=DEQ_CFG["lr"], weight_decay=DEQ_CFG["wd"])
    bv, bt = -1.0, 0.0
    for e in range(epochs):
        m.train(); opt.zero_grad()
        out, reg = m(X, jac=True)
        (F.cross_entropy(out[tr], y[tr]) + DEQ_CFG["jac_gamma"] * reg).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if e % 10 == 0:
            m.eval()
            with torch.no_grad():
                out, _ = m(X)
            v = score(out, y, va, binary)
            if v > bv:
                bv, bt = v, score(out, y, te, binary)
    return bt


def main():
    print(f"device = {DEV}\n")
    for ds in DATASETS:
        Xs, ys, A, masks, K = load_sparse(ds)
        X, y, edges, deg, _, _ = load_edges(ds)
        binary = (K == 2)
        X, y, edges, deg = X.to(DEV), y.to(DEV), edges.to(DEV), deg.to(DEV)
        Ahat = renorm_sparse(A).to(DEV)
        d_in = X.shape[1]
        metric = "AUC" if binary else "acc"
        print(f"===== {ds}: {X.shape[0]} nodes, {d_in} feat, {K} classes, "
              f"{A.nnz // 2} edges -> metric={metric} =====", flush=True)
        builders = {
            "MLP (no graph)": lambda: MLP(d_in, 64, K),
            "SGC (linear)": lambda: SGC(d_in, K),
            "APPNP (linear)": lambda: APPNP(d_in, 64, K),
        }
        res = {}
        for name, build in builders.items():
            vals, t0 = [], time.time()
            for s in range(N_SPLITS):
                torch.manual_seed(s); np.random.seed(s)
                tr = torch.tensor(masks["train_masks"][s].astype(bool)).to(DEV)
                va = torch.tensor(masks["val_masks"][s].astype(bool)).to(DEV)
                te = torch.tensor(masks["test_masks"][s].astype(bool)).to(DEV)
                vals.append(train_linear(build, X, y, Ahat, tr, va, te, binary))
            res[name] = np.mean(vals)
            print(f"  {name:<16} {metric} {np.mean(vals):.3f} +- {np.std(vals):.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        # our contractive nonlinear cell
        vals, t0 = [], time.time()
        for s in range(N_SPLITS):
            tr = torch.tensor(masks["train_masks"][s].astype(bool)).to(DEV)
            va = torch.tensor(masks["val_masks"][s].astype(bool)).to(DEV)
            te = torch.tensor(masks["test_masks"][s].astype(bool)).to(DEV)
            vals.append(train_deq(edges, deg, d_in, K, X, y, tr, va, te, binary))
        res["FAGCN-DEQ (ours)"] = np.mean(vals)
        print(f"  {'FAGCN-DEQ (ours)':<16} {metric} {np.mean(vals):.3f} +- {np.std(vals):.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        margin_mlp = res["FAGCN-DEQ (ours)"] - res["MLP (no graph)"]
        margin_lin = res["FAGCN-DEQ (ours)"] - max(res["SGC (linear)"], res["APPNP (linear)"])
        print(f"  --> margin over MLP {margin_mlp:+.3f} | over best-linear {margin_lin:+.3f}\n",
              flush=True)


if __name__ == "__main__":
    main()
