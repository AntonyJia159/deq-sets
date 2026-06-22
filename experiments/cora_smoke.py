"""Cora smoke test: load the real dataset and confirm a contractive graph-DEQ behaves.

Step zero before the channel-1 deletion experiment on Cora. Checks, in order:
  1. download + parse the standard Planetoid Cora split (no PyG; urllib + scipy);
  2. sanity-check shapes / class balance / graph stats against known Cora values;
  3. confirm the contractive DEQ map actually CONVERGES on this graph (residual, iters);
  4. confirm it reaches sane node-classification accuracy (Cora GCN ~0.80; >0.70 is fine here).

If any of these is off, the deletion experiment built on top would be meaningless.

Run:  D:\\deq-venv\\Scripts\\python.exe experiments\\cora_smoke.py
"""

import os
import pickle
import urllib.request

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "cora")
BASE = "https://raw.githubusercontent.com/tkipf/gcn/master/gcn/data/"
NAMES = ["x", "y", "tx", "ty", "allx", "ally", "graph"]

# known-good Cora facts to assert against (catches a silently-wrong parse)
EXPECT_NODES, EXPECT_FEAT, EXPECT_CLASSES = 2708, 1433, 7


def download():
    os.makedirs(DATA, exist_ok=True)
    files = [f"ind.cora.{n}" for n in NAMES] + ["ind.cora.test.index"]
    for f in files:
        dst = os.path.join(DATA, f)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        url = BASE + f
        print(f"  downloading {f} ...", flush=True)
        urllib.request.urlretrieve(url, dst)
    print("  all files present.")


def _load(name):
    with open(os.path.join(DATA, f"ind.cora.{name}"), "rb") as fh:
        return pickle.load(fh, encoding="latin1")


def load_cora():
    """Standard Planetoid assembly (mirrors tkipf/gcn load_data)."""
    x, y, tx, ty, allx, ally, graph = (_load(n) for n in NAMES)
    test_idx = np.array(
        [int(l) for l in open(os.path.join(DATA, "ind.cora.test.index"))])
    test_idx_range = np.sort(test_idx)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx, :] = features[test_idx_range, :]      # reorder test rows
    labels = np.vstack((ally, ty))
    labels[test_idx, :] = labels[test_idx_range, :]

    import networkx as nx
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    idx_train = np.arange(len(ally) - tx.shape[0] - 500)     # first ~140 labelled
    idx_train = np.arange(140)
    idx_test = test_idx_range
    return adj, np.asarray(features.todense(), np.float32), labels, idx_train, idx_test


def normalize_adj(adj):
    """Symmetric-normalized adjacency with self-loops; spectral radius <= 1."""
    A = adj + sp.eye(adj.shape[0])
    deg = np.asarray(A.sum(1)).ravel()
    dinv = sp.diags(1.0 / np.sqrt(deg))
    return torch.tensor(np.asarray((dinv @ A @ dinv).todense()), dtype=torch.float32)


class GraphDEQ(nn.Module):
    """z <- tanh( s * Ahat @ (z W) + Enc(x) ), s*||W||<1 -> contraction -> unique fp."""

    def __init__(self, d_in, d_lat, k, iters=50, tol=1e-5):
        super().__init__()
        self.enc = nn.Linear(d_in, d_lat)
        self.W = nn.Parameter(torch.randn(d_lat, d_lat) * 0.1)
        self.readout = nn.Linear(d_lat, k)
        self.s, self.iters, self.tol = 0.9, iters, tol

    def _Wc(self):
        return self.W / (torch.linalg.matrix_norm(self.W, ord=2) + 1e-6)

    def solve(self, Ahat, X, track=False):
        h0 = self.enc(X)
        Wc = self._Wc()
        z = torch.zeros_like(h0)
        last = None
        for it in range(1, self.iters + 1):
            zn = torch.tanh(self.s * (Ahat @ (z @ Wc)) + h0)
            last = (zn - z).norm().item() / (z.norm().item() + 1e-8)
            z = zn
            if track and last < self.tol:
                return z, it, last
        return (z, self.iters, last) if track else z

    def forward(self, Ahat, X):
        return self.readout(self.solve(Ahat, X))


def main():
    torch.manual_seed(0); np.random.seed(0)
    print("[1] download")
    download()

    print("[2] parse + sanity checks")
    adj, X, labels_oh, idx_train, idx_test = load_cora()
    y = torch.tensor(labels_oh.argmax(1), dtype=torch.long)
    N, F_in = X.shape
    K = labels_oh.shape[1]
    print(f"    nodes={N} feat={F_in} classes={K} "
          f"edges={int(adj.sum() / 2)} train={len(idx_train)} test={len(idx_test)}")
    assert (N, F_in, K) == (EXPECT_NODES, EXPECT_FEAT, EXPECT_CLASSES), "Cora parse mismatch!"
    assert labels_oh.sum(1).min() == 1, "labels not one-hot"
    counts = np.bincount(y.numpy(), minlength=K)
    print(f"    class counts: {counts.tolist()}  (degree mean={adj.sum(1).mean():.2f} "
          f"max={int(adj.sum(1).max())})")

    Ahat = normalize_adj(adj)
    X_t = torch.tensor(X)
    idx_train_t = torch.tensor(idx_train); idx_test_t = torch.tensor(idx_test)

    print("[3] convergence check (random weights, no training)")
    model = GraphDEQ(F_in, 64, K)
    with torch.no_grad():
        _, iters0, res0 = model.solve(Ahat, X_t, track=True)
    print(f"    random-weight solve: {iters0} iters, final rel-residual {res0:.2e} "
          f"({'CONVERGED' if res0 < 1e-4 else 'did NOT converge'})")

    print("[4] train + accuracy")
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    for ep in range(120):
        opt.zero_grad()
        out = model(Ahat, X_t)
        loss = F.cross_entropy(out[idx_train_t], y[idx_train_t])
        loss.backward(); opt.step()
        if (ep + 1) % 40 == 0:
            with torch.no_grad():
                pred = model(Ahat, X_t).argmax(1)
                tr = (pred[idx_train_t] == y[idx_train_t]).float().mean().item()
                te = (pred[idx_test_t] == y[idx_test_t]).float().mean().item()
            print(f"    epoch {ep+1:3d}  loss {loss.item():.3f}  "
                  f"train {tr:.3f}  test {te:.3f}")

    with torch.no_grad():
        _, itersF, resF = model.solve(Ahat, X_t, track=True)
        pred = model(Ahat, X_t).argmax(1)
        te = (pred[idx_test_t] == y[idx_test_t]).float().mean().item()
    verdict = "GREEN" if (te > 0.70 and resF < 1e-4) else "CHECK"
    print(f"\n    trained solve: {itersF} iters, residual {resF:.2e};  test acc {te:.3f}")
    print(f"    >>> SMOKE: {verdict}  (want test>0.70 and converged)")


if __name__ == "__main__":
    main()
