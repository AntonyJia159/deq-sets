"""Cora: accuracy vs near-SOTA, and the ring-truncation (approximate-unlearning) tradeoff.

Two quantifications the deletion study still owed.

PART A -- accuracy anchor. How much accuracy does the deletion-enabling construction cost
vs a strong feedforward GNN? We train a tuned 2-layer GCN (the standard Cora anchor, ~0.81)
and our clipped-SN graph-DEQ across a kappa sweep (kappa = the contraction cap; training
saturates it so rho~kappa). As kappa->1 the DEQ should approach a normal GNN's accuracy, but
that is exactly where locality (and thus cheap deletion) dies. Literature reference points
(test acc, public split): GCN 0.815, GAT 0.830, APPNP 0.840, GCNII 0.855.

PART B -- ring truncation = a SECOND knob, orthogonal to kappa. kappa trades accuracy for
locality at the MODEL level. On a FIXED model, re-solving a ring SMALLER than the exact radius
R* trades exactness for SPEED at deletion time (approximate unlearning). We sweep R below R*
and measure: nodes touched (speed), residual error vs the exact deletion (inexactness), and --
the operational cost -- how many node PREDICTIONS flip vs the exact deletion.

Run:  D:\\deq-venv\\Scripts\\python.exe -m experiments.cora_perf_tradeoff
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.cora_smoke import download, load_cora
from experiments.cora_deq_solvers import GraphDEQ
from experiments.cora_deletion import renorm_sparse, solve, bfs_hops, pick_targets

SOTA_REF = {"GCN (lit)": 0.815, "GAT (lit)": 0.830, "APPNP (lit)": 0.840, "GCNII (lit)": 0.855}


class GCN(nn.Module):
    """Standard 2-layer GCN with dropout -- the near-SOTA-ish feedforward anchor."""

    def __init__(self, d_in, d_h, k, dropout=0.5):
        super().__init__()
        self.l1 = nn.Linear(d_in, d_h)
        self.l2 = nn.Linear(d_h, k)
        self.drop = dropout

    def forward(self, Ahat, X):
        h = F.dropout(X, self.drop, self.training)
        h = F.relu(torch.sparse.mm(Ahat, self.l1(h)))
        h = F.dropout(h, self.drop, self.training)
        return torch.sparse.mm(Ahat, self.l2(h))


def train_acc(model, Ahat, X, y, itr, ite, epochs, lr=1e-2, wd=5e-4, is_deq=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        out = model(Ahat, X)[0] if is_deq else model(Ahat, X)
        F.cross_entropy(out[itr], y[itr]).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)   # IGNN uses grad clip
        opt.step()
    model.eval()
    with torch.no_grad():
        out = model(Ahat, X)[0] if is_deq else model(Ahat, X)
        return (out.argmax(1)[ite] == y[ite]).float().mean().item()


def main():
    download()
    adj, X, labels_oh, idx_train, idx_test = load_cora()
    adj = adj.tocsr()
    y = torch.tensor(labels_oh.argmax(1), dtype=torch.long)
    X_t = torch.tensor(X)
    itr, ite = torch.tensor(idx_train), torch.tensor(idx_test)
    K, F_in = labels_oh.shape[1], X.shape[1]
    Ahat = renorm_sparse(adj)

    print("=" * 60)
    print("PART A -- accuracy: deletion-enabling DEQ vs strong GCN")
    print("=" * 60)
    torch.manual_seed(0); np.random.seed(0)
    gcn = GCN(F_in, 64, K, dropout=0.5)
    gcn_acc = train_acc(gcn, Ahat, X_t, y, itr, ite, epochs=200)
    print(f"{'model':<22}{'eff rho':>9}{'test acc':>10}")
    print(f"{'GCN (ours, tuned)':<22}{'-':>9}{gcn_acc:>10.3f}")
    for name, a in SOTA_REF.items():
        print(f"{name:<22}{'-':>9}{a:>10.3f}")
    print("  " + "-" * 35)

    deq_models = {}
    for kappa in [0.5, 0.7, 0.9, 0.99]:
        torch.manual_seed(0); np.random.seed(0)
        # dropout HURTS this DEQ (feature dropout destabilizes the equilibrium); its best
        # config is none -- GCN keeps its standard dropout, so this is each model's best shot.
        m = GraphDEQ(F_in, 64, K, "fixed_point_iter", norm="clipped", kappa=kappa, dropout=0.0)
        acc = train_acc(m, Ahat, X_t, y, itr, ite, epochs=150, is_deq=True)
        with torch.no_grad():
            eff = (m.s * torch.linalg.matrix_norm(m._Wc(), ord=2)).item()
        deq_models[kappa] = m
        print(f"{'DEQ kappa=' + str(kappa):<22}{eff:>9.3f}{acc:>10.3f}")

    print("\n" + "=" * 60)
    print("PART B -- ring truncation: speed vs exactness on a FIXED model")
    print("=" * 60)
    kappa_b = 0.9
    model = deq_models[kappa_b]
    model.eval()
    with torch.no_grad():
        Wc = model._Wc().detach(); s = model.s
        h0 = model.enc(X_t).detach()
        readout = model.readout
        z_old, _ = solve(Ahat, Wc, h0, s)
    deg = np.asarray(adj.sum(1)).ravel()
    G = nx.from_scipy_sparse_array(adj)
    targets = pick_targets(deg)
    print(f"fixed model: DEQ kappa={kappa_b} (eff rho "
          f"{(s * torch.linalg.matrix_norm(Wc, ord=2)).item():.3f}). "
          f"Sweep ring R below the exact radius R*:\n")

    for name in ["median", "hub"]:
        v = targets[name]
        adj_del = adj.tolil(); adj_del[v, :] = 0; adj_del[:, v] = 0
        Ahat_del = renorm_sparse(adj_del.tocsr())
        with torch.no_grad():
            z_new, _ = solve(Ahat_del, Wc, h0, s, z0=z_old)       # exact deletion (gold)
            pred_exact = readout(z_new).argmax(1)
        hops = bfs_hops(G, v, adj.shape[0])
        finite = torch.tensor(np.isfinite(hops))
        maxR = int(np.nanmax(hops[np.isfinite(hops)]))

        print(f"--- {name} node (deg {int(deg[v])}) ---")
        print(f"{'R':>3}{'touched':>9}{'N/touched':>11}{'resid err':>11}"
              f"{'pred flips':>11}{'test flips':>11}")
        for R in range(1, min(maxR, 8) + 1):
            restrict = torch.tensor(hops <= R)
            with torch.no_grad():
                z_tr, _ = solve(Ahat_del, Wc, h0, s, z0=z_old, restrict=restrict)
                err = (z_tr - z_new)[finite].abs().max().item()
                pred_tr = readout(z_tr).argmax(1)
                flips = int((pred_tr != pred_exact)[finite].sum())
                test_flips = int((pred_tr[ite] != pred_exact[ite]).sum())
            tag = "  <- exact" if err < 1e-4 else ""
            print(f"{R:>3}{int(restrict.sum()):>9}"
                  f"{adj.shape[0] / int(restrict.sum()):>11.1f}{err:>11.1e}"
                  f"{flips:>11}{test_flips:>11}{tag}")
        print()


if __name__ == "__main__":
    main()
