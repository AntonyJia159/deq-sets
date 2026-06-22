"""Smoke test: recurrentized FAGCN as a state-dependent signed-attention DEQ.

Map (ZJ's form), solved to equilibrium:
    z' = eps*z + Ax + sum_{j in N(i)} (alpha_ij / sqrt(d_i d_j)) z_j ,
    alpha_ij = tanh( g^T [z_i || z_j] )            (STATE-DEPENDENT: from current z)

Ax is the per-step input injection (mandatory for a DEQ). alpha is signed (tanh) -> can do
high-pass / heterophily. Solved with TorchDEQ (fixed_point_iter + PhantomGrad, the default).

This is just a smoke test: does the state-dependent signed map (a) converge from zeros, and
(b) train (loss down, acc up)? Contraction is NOT guaranteed for state-dependent attention
(this is the regime where we saw multistability before) -- here we only check it runs and the
solver residual behaves. Contraction control is (eps, s); s scales the aggregation.

Run:  D:\\deq-venv\\Scripts\\python.exe -m experiments.fagcn_deq_smoke
"""

import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdeq import get_deq

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "hetero")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(name):
    d = np.load(os.path.join(DATA, f"{name}.npz"))
    X = torch.tensor(d["node_features"], dtype=torch.float32)
    y = torch.tensor(d["node_labels"], dtype=torch.long)
    e = d["edges"]
    N = X.shape[0]
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(N, N))
    A = (A + A.T); A.data[:] = 1.0
    A = A.tocoo()
    edges = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)   # (2, E) symmetric
    deg = torch.tensor(np.asarray(sp.csr_matrix((np.ones(A.nnz), (A.row, A.col)),
                       shape=(N, N)).sum(1)).ravel(), dtype=torch.float32).clamp(min=1)
    m = d["train_masks"]; mt = m if m.shape[0] != N else m.T
    masks = {k: (d[k] if d[k].shape[0] != N else d[k].T) for k in
             ["train_masks", "val_masks", "test_masks"]}
    return X, y, edges, deg, masks, int(y.max()) + 1


class FAGCNDEQ(nn.Module):
    def __init__(self, d_in, d, k, edges, deg, eps=0.4, s=0.3):
        super().__init__()
        self.enc = nn.Linear(d_in, d)
        self.att = nn.Linear(2 * d, 1, bias=False)
        self.ro = nn.Linear(d, k)
        self.eps, self.s = eps, s
        self.register_buffer("edges", edges)
        self.register_buffer("norm", 1.0 / torch.sqrt(deg[edges[0]] * deg[edges[1]]))
        self.deq = get_deq(f_solver="fixed_point_iter", f_max_iter=60, f_tol=1e-4)

    def aggregate(self, z):
        dst, src = self.edges[0], self.edges[1]          # message j(src) -> i(dst)
        alpha = torch.tanh(self.att(torch.cat([z[dst], z[src]], dim=-1)).squeeze(-1))
        coef = (alpha * self.norm).unsqueeze(-1)          # (E,1) signed, normalized
        out = torch.zeros_like(z)
        out.index_add_(0, dst, coef * z[src])
        return out

    def forward(self, X):
        h0 = self.enc(X)

        def f(z):
            return self.eps * z + h0 + self.s * self.aggregate(z)

        z0 = torch.zeros_like(h0)
        z_out, info = self.deq(f, z0)
        z = z_out[-1]
        return self.ro(z), info


def main():
    print(f"device = {DEV}")
    X, y, edges, deg, masks, K = load("chameleon_filtered")
    X, y, edges, deg = X.to(DEV), y.to(DEV), edges.to(DEV), deg.to(DEV)
    print(f"chameleon_filtered: {X.shape[0]} nodes, {X.shape[1]} feat, {K} classes, "
          f"{edges.shape[1] // 2} undirected edges")

    tr = torch.tensor(masks["train_masks"][0].astype(bool)).to(DEV)
    va = torch.tensor(masks["val_masks"][0].astype(bool)).to(DEV)
    te = torch.tensor(masks["test_masks"][0].astype(bool)).to(DEV)

    torch.manual_seed(0)
    model = FAGCNDEQ(X.shape[1], 64, K, edges, deg).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    for ep in range(120):
        model.train(); opt.zero_grad()
        out, info = model(X)
        loss = F.cross_entropy(out[tr], y[tr])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if ep % 20 == 0 or ep == 119:
            model.eval()
            with torch.no_grad():
                out, info = model(X)
                tra = (out.argmax(1)[tr] == y[tr]).float().mean().item()
                vaa = (out.argmax(1)[va] == y[va]).float().mean().item()
                tea = (out.argmax(1)[te] == y[te]).float().mean().item()
            rel = info.get("rel_lowest", info.get("rel_trace", torch.tensor(float("nan"))))
            rel = float(rel.mean()) if torch.is_tensor(rel) else float(rel)
            nstep = info.get("nstep", torch.tensor(float("nan")))
            nstep = float(nstep.mean()) if torch.is_tensor(nstep) else float(nstep)
            print(f"ep {ep:3d}  loss {loss.item():.3f}  "
                  f"train {tra:.3f} val {vaa:.3f} test {tea:.3f}  "
                  f"| solver: nstep {nstep:.0f}, rel_res {rel:.1e}")

    print("\nsmoke verdict: did it (a) solve (rel_res small) and (b) train (loss down)?")


if __name__ == "__main__":
    main()
