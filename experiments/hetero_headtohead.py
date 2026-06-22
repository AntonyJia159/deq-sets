"""Heterophily head-to-head: does a contractive NONLINEAR/attention equilibrium beat
LINEAR propagation (the InstantGNN expressiveness class) where smoothing fails?

Motivation: InstantGNN & co. get cheap local incremental maintenance from LINEARITY
(generalized PageRank propagation). We get it from CONTRACTION -- a strictly larger space
that admits attention / feature-dependent neighbor weighting. On homophilous graphs (Cora)
linear smoothing is already near-optimal so the flexibility is invisible. Heterophily is the
battleground: the right neighbor weighting depends on FEATURES, which linear propagation
(structure-only weights) cannot express but attention can.

Models (all contraction-safe DEQs use spectral-normed W, s*||W||<1):
  MLP            -- no graph (the heterophily floor)
  SGC (k=2)      -- linear smoothing (InstantGNN class, fixed weights)
  APPNP (K=10)   -- linear PPR propagation + MLP head (InstantGNN class)
  DEQ-mean       -- our basic contractive equilibrium (still low-pass / smoothing)
  DEQ-attn-ego   -- contractive equilibrium with input-conditioned attention + an ego term
                    (feature-dependent weighting + ego/neighbor separation; the 'beef-up')

Data: Platonov et al. 2023 filtered heterophily graphs (fixes Chameleon/Squirrel leakage),
with the repo's standard splits. Run:
  D:\\deq-venv\\Scripts\\python.exe -m experiments.hetero_headtohead
"""

import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.cora_deletion import renorm_sparse

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "hetero")
DATASETS = ["chameleon_filtered", "squirrel_filtered"]
N_SPLITS = 5
EPOCHS = 200


def load(name):
    d = np.load(os.path.join(DATA, f"{name}.npz"))
    X = torch.tensor(d["node_features"], dtype=torch.float32)
    y = torch.tensor(d["node_labels"], dtype=torch.long)
    e = d["edges"]
    N = X.shape[0]
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(N, N))
    A = A + A.T
    A.data[:] = 1.0
    masks = {}
    for split in ["train_masks", "val_masks", "test_masks"]:
        m = d[split]
        masks[split] = m if m.shape[0] != N else m.T   # -> (num_splits, N)
    return X, y, A.tocsr(), masks, int(y.max()) + 1


# --------------------------------------------------------------------- models

class MLP(nn.Module):
    def __init__(self, d_in, h, k, drop=0.5):
        super().__init__()
        self.l1, self.l2, self.drop = nn.Linear(d_in, h), nn.Linear(h, k), drop

    def forward(self, X, Ahat, Amask):
        h = F.dropout(F.relu(self.l1(F.dropout(X, self.drop, self.training))),
                      self.drop, self.training)
        return self.l2(h)


class SGC(nn.Module):
    def __init__(self, d_in, k, hops=2):
        super().__init__()
        self.lin, self.hops, self._cache = nn.Linear(d_in, k), hops, None

    def forward(self, X, Ahat, Amask):
        if self._cache is None:
            S = X
            for _ in range(self.hops):
                S = torch.sparse.mm(Ahat, S)
            self._cache = S.detach()
        return self.lin(self._cache)


class APPNP(nn.Module):
    def __init__(self, d_in, h, k, K=10, alpha=0.1, drop=0.5):
        super().__init__()
        self.l1, self.l2 = nn.Linear(d_in, h), nn.Linear(h, k)
        self.K, self.alpha, self.drop = K, alpha, drop

    def forward(self, X, Ahat, Amask):
        h = F.dropout(F.relu(self.l1(F.dropout(X, self.drop, self.training))),
                      self.drop, self.training)
        h = self.l2(h)
        z = h
        for _ in range(self.K):
            z = (1 - self.alpha) * torch.sparse.mm(Ahat, z) + self.alpha * h
        return z


class DEQMean(nn.Module):
    def __init__(self, d_in, d, k, s=0.9, iters=40, drop=0.5):
        super().__init__()
        self.enc, self.W, self.ro = nn.Linear(d_in, d), nn.Parameter(torch.randn(d, d) * .1), nn.Linear(d, k)
        self.s, self.iters, self.drop = s, iters, drop

    def _Wc(self):
        return self.W / (torch.linalg.matrix_norm(self.W, 2) + 1e-6)

    def forward(self, X, Ahat, Amask):
        h0 = self.enc(F.dropout(X, self.drop, self.training)); Wc = self._Wc()
        z = torch.zeros_like(h0)
        for _ in range(self.iters):
            z = torch.tanh(self.s * torch.sparse.mm(Ahat, z @ Wc) + h0)
        return self.ro(z)


class DEQAttnEgo(nn.Module):
    """Contractive equilibrium with input-conditioned attention + ego term.
    z_i <- tanh( s*(0.5 z_i W_s + 0.5 sum_j a_ij z_j W_n) + h0_i ), a from h0 (fixed)."""

    def __init__(self, d_in, d, k, s=0.7, iters=40, drop=0.5):
        super().__init__()
        self.enc = nn.Linear(d_in, d)
        self.Ws = nn.Parameter(torch.randn(d, d) * .1)
        self.Wn = nn.Parameter(torch.randn(d, d) * .1)
        self.att = nn.Linear(2 * d, 1)
        self.ro = nn.Linear(d, k)
        self.s, self.iters, self.drop = s, iters, drop

    def _nrm(self, W):
        return W / (torch.linalg.matrix_norm(W, 2) + 1e-6)

    def forward(self, X, Ahat, Amask):
        h0 = self.enc(F.dropout(X, self.drop, self.training))
        N, d = h0.shape
        # input-conditioned attention over existing edges (dense, masked) -- fixed across iters
        hi = h0.unsqueeze(1).expand(N, N, d)
        hj = h0.unsqueeze(0).expand(N, N, d)
        e = F.leaky_relu(self.att(torch.cat([hi, hj], -1)).squeeze(-1), 0.2)
        e = e.masked_fill(Amask == 0, float("-inf"))
        A_att = torch.softmax(e, dim=1)                       # row-stochastic over neighbors
        Ws, Wn = self._nrm(self.Ws), self._nrm(self.Wn)
        z = torch.zeros_like(h0)
        for _ in range(self.iters):
            agg = 0.5 * (z @ Ws) + 0.5 * (A_att @ (z @ Wn))
            z = torch.tanh(self.s * agg + h0)
        return self.ro(z)


# ----------------------------------------------------------------- train/eval

def run_split(model, X, y, Ahat, Amask, tr, va, te, epochs=EPOCHS):
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    best_va, best_te = 0.0, 0.0
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        out = model(X, Ahat, Amask)
        F.cross_entropy(out[tr], y[tr]).backward()
        opt.step()
        if _ % 5 == 0:
            model.eval()
            with torch.no_grad():
                out = model(X, Ahat, Amask)
                va_acc = (out.argmax(1)[va] == y[va]).float().mean().item()
                te_acc = (out.argmax(1)[te] == y[te]).float().mean().item()
            if va_acc > best_va:
                best_va, best_te = va_acc, te_acc
    return best_te


def main():
    for ds in DATASETS:
        X, y, A, masks, K = load(ds)
        Ahat = renorm_sparse(A)
        Amask = torch.tensor(np.asarray((A + sp.eye(A.shape[0])).todense()) > 0)
        d_in = X.shape[1]
        homophily_note = "(filtered heterophily benchmark)"
        print(f"\n===== {ds} {homophily_note}: {X.shape[0]} nodes, {d_in} feat, "
              f"{K} classes, {A.nnz // 2} edges =====")
        builders = {
            "MLP": lambda: MLP(d_in, 64, K),
            "SGC (linear)": lambda: SGC(d_in, K),
            "APPNP (linear)": lambda: APPNP(d_in, 64, K),
            "DEQ-mean": lambda: DEQMean(d_in, 64, K),
            "DEQ-attn-ego": lambda: DEQAttnEgo(d_in, 64, K),
        }
        print(f"{'model':<16}{'test acc (mean+-std over splits)':>34}")
        for name, build in builders.items():
            accs = []
            for sp_i in range(N_SPLITS):
                torch.manual_seed(sp_i); np.random.seed(sp_i)
                tr = torch.tensor(masks["train_masks"][sp_i].astype(bool))
                va = torch.tensor(masks["val_masks"][sp_i].astype(bool))
                te = torch.tensor(masks["test_masks"][sp_i].astype(bool))
                m = build()
                accs.append(run_split(m, X, y, Ahat, Amask, tr, va, te))
            print(f"{name:<16}{np.mean(accs):>22.3f} +- {np.std(accs):.3f}")


if __name__ == "__main__":
    main()
