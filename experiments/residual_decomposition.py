"""Where does the membership residual live? Per-channel MIA on a contractive graph-DEQ.

Question (Geng-endorsed unlearning thread): when a DEQ is trained transductively on a
graph, the deleted/queried node's information can leak through several distinct channels.
This script measures, for ONE trained DEQ, how much membership signal each channel carries:

  output channel   -- loss value / prediction margin at the node       (mixed, classic MIA)
  WEIGHT channel   -- || grad_theta loss(node) ||                       (white-box; the residual
                                                                         baked into trained weights)
  DYNAMICAL chan.  -- iterations for the node's residual to converge    (DEQ-SPECIFIC new surface;
                                                                         a feedforward net has no analog)

A node is a "member" iff it was in the training loss. Members and non-members are drawn
identically (random label mask), so AUC > 0.5 reflects MEMORIZATION, not distribution shift.
Label noise is injected on a fraction of members -> those are the most-memorized, most-attackable
nodes (Feldman long-tail), giving the attack something to find.

The point is the *decomposition*: which channel's AUC is above chance tells us where the
attackable residual sits -- in the weights (would need GNNDelete/GIF-style weight surgery to
remove) vs in the recomputable equilibrium (removed for free by re-solving on the smaller graph).

Run:  D:\\deq-venv\\Scripts\\python.exe experiments\\residual_decomposition.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

DEV = "cpu"  # small graph; CPU is fine and keeps per-node grad loops simple
torch.manual_seed(0)
np.random.seed(0)

N = 400          # nodes
K = 4            # classes / communities
D_IN = 16        # input feature dim
D_LAT = 32       # latent dim
P_IN, P_OUT = 0.06, 0.004   # SBM intra/inter edge prob
LABEL_FRAC = 0.5            # fraction of nodes that are training MEMBERS
NOISE_FRAC = 0.30          # fraction of members with flipped labels (memorization bait)
SOLVE_ITERS = 60
TOL = 1e-5
EPOCHS = 400


# ----------------------------------------------------------------- data: planted SBM

def make_sbm():
    comm = np.random.randint(0, K, size=N)
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            p = P_IN if comm[i] == comm[j] else P_OUT
            if np.random.rand() < p:
                A[i, j] = A[j, i] = 1.0
    # features: class-correlated mean + noise (so the task is learnable but not trivial)
    centers = np.random.randn(K, D_IN) * 1.5
    X = centers[comm] + np.random.randn(N, D_IN) * 1.0
    # symmetric normalized adjacency with self-loops: spectral radius <= 1
    A = A + np.eye(N, dtype=np.float32)
    deg = A.sum(1)
    Dinv = np.diag(1.0 / np.sqrt(deg))
    Ahat = Dinv @ A @ Dinv
    return (torch.tensor(Ahat), torch.tensor(X, dtype=torch.float32),
            torch.tensor(comm, dtype=torch.long))


# --------------------------------------------------------------- contractive graph-DEQ

class GraphDEQ(nn.Module):
    """z <- tanh( s * Ahat @ (z W) + Enc(x) ).  Spectral-normalizing W with s<1 makes the
    map a provable contraction (||Ahat||<=1, tanh 1-Lipschitz) -> unique fixed point."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(D_IN, D_LAT)
        self.W = nn.Parameter(torch.randn(D_LAT, D_LAT) * 0.1)
        self.readout = nn.Linear(D_LAT, K)
        self.s = 0.9

    def _Wc(self):
        # contraction factor < 1: rescale W so s * ||W||_2 < 1
        sigma = torch.linalg.matrix_norm(self.W, ord=2)
        return self.W / (sigma + 1e-6)

    def solve(self, Ahat, X, track=False):
        h0 = self.enc(X)
        Wc = self._Wc()
        z = torch.zeros_like(h0)
        node_iter = torch.full((X.shape[0],), SOLVE_ITERS, dtype=torch.float32)
        done = torch.zeros(X.shape[0], dtype=torch.bool)
        for it in range(1, SOLVE_ITERS + 1):
            zn = torch.tanh(self.s * (Ahat @ (z @ Wc)) + h0)
            if track:
                r = (zn - z).norm(dim=1)                  # per-node residual
                newly = (~done) & (r < TOL)
                node_iter[newly] = it
                done = done | newly
            z = zn
        return (z, node_iter) if track else z

    def forward(self, Ahat, X):
        return self.readout(self.solve(Ahat, X))


class FeedforwardGCN(nn.Module):
    """L-layer GCN with RESIDUAL connections (so depth is trainable / avoids oversmoothing,
    letting it actually reach train_acc ~1.0 for a MATCHED-FIT comparison). Distinct weight
    per layer -> the comparator against the DEQ's single weight-tied matrix."""

    def __init__(self, layers=6, width=D_LAT):
        super().__init__()
        self.enc = nn.Linear(D_IN, width)
        self.gcn = nn.ParameterList(
            [nn.Parameter(torch.randn(width, width) * 0.1) for _ in range(layers)])
        self.readout = nn.Linear(width, K)

    def forward(self, Ahat, X):
        h = torch.tanh(self.enc(X))
        for W in self.gcn:
            h = h + torch.tanh(Ahat @ (h @ W))   # residual: trainable depth
        return self.readout(h)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def train_model(model, Ahat, X, y_train, member, epochs=EPOCHS, lr=5e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(Ahat, X)[member], y_train[member])
        loss.backward()
        opt.step()
    model.eval()


def weight_grad_norms(model, Ahat, X, y, idx, chunk=40):
    """Per-node || grad_theta CE(node) || for all probe nodes, vectorized via torch.func.
    vmap(grad(.)) batches the per-node backward passes; chunked to bound memory (each chunk
    holds `chunk` parallel forward graphs of the unrolled solve)."""
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def loss_at(p, i):
        logits = functional_call(model, (p, buffers), (Ahat, X))
        row = logits.index_select(0, i.reshape(1))           # (1, K), vmap-safe
        tgt = y.index_select(0, i.reshape(1))                # (1,)
        return F.cross_entropy(row, tgt)

    gfn = vmap(grad(loss_at), in_dims=(None, 0))
    norms = []
    for c in idx.split(chunk):
        g = gfn(params, c)                                   # name -> (len(c), *param_shape)
        flat = torch.cat([v.reshape(len(c), -1) for v in g.values()], dim=1)
        norms.append(flat.norm(dim=1).detach())
    return torch.cat(norms)


def attack_model(model, Ahat, X, y_true, y_train, member):
    """Returns per-channel membership AUC dict + accuracies (paired across models)."""
    with torch.no_grad():
        logits = model(Ahat, X)
    train_acc = (logits[member].argmax(1) == y_train[member]).float().mean().item()
    test_acc = (logits[~member].argmax(1) == y_true[~member]).float().mean().item()

    p = logits.softmax(1)
    top2 = p.topk(2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1])
    mlab = member.numpy().astype(int)

    probe_idx = torch.cat([torch.where(member)[0][:120], torch.where(~member)[0][:120]])
    gradnorm = weight_grad_norms(model, Ahat, X, y_train, probe_idx)
    plab = member[probe_idx].numpy().astype(int)

    return {
        "params": n_params(model), "train_acc": train_acc, "test_acc": test_acc,
        "margin_auc": auc(margin.numpy(), mlab),
        "weight_auc": auc(-gradnorm.numpy(), plab),
    }


# ----------------------------------------------------------------------- AUC (no sklearn)

def auc(scores, labels):
    """Mann-Whitney AUC: P(score[member] > score[non-member]). labels: 1=member."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = s.argsort()
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average-rank tie handling
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt); start = cum - cnt
    avg = (start + cum + 1) / 2.0
    ranks = avg[inv]
    R = ranks[y == 1].sum()
    return (R - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


# ------------------------------------------------------------------------------- run

def run_once(seed, noise_frac):
    """One paired trial: same graph/split, DEQ vs residual feedforward GCN."""
    torch.manual_seed(seed); np.random.seed(seed)
    Ahat, X, y_true = make_sbm()
    member = torch.zeros(N, dtype=torch.bool)
    member[torch.randperm(N)[: int(LABEL_FRAC * N)]] = True
    y_train = y_true.clone()
    noisy = member & (torch.rand(N) < noise_frac)
    if noisy.any():
        y_train[noisy] = (y_train[noisy] + torch.randint(1, K, (int(noisy.sum()),))) % K

    deq = GraphDEQ().to(DEV)
    ff = FeedforwardGCN(layers=6, width=64).to(DEV)   # enough capacity to reach train_acc~1
    train_model(deq, Ahat, X, y_train, member, epochs=EPOCHS)
    train_model(ff, Ahat, X, y_train, member, epochs=800)
    return (attack_model(deq, Ahat, X, y_true, y_train, member),
            attack_model(ff, Ahat, X, y_true, y_train, member))


def agg(rows, key):
    v = np.array([r[key] for r in rows])
    return v.mean(), v.std()


def report(deqs, ffs, title):
    print(f"\n=== {title} ===")
    print(f"{'':14}{'DEQ (weight-tied)':>22}{'FF-GCN (resid, 6x64)':>22}")
    for key, lab in [("params", "params"), ("train_acc", "train acc"),
                     ("test_acc", "test acc"),
                     ("margin_auc", "OUTPUT auc"), ("weight_auc", "WEIGHT auc")]:
        dm, ds = agg(deqs, key); fm, fs = agg(ffs, key)
        if key == "params":
            print(f"{lab:14}{dm:22.0f}{fm:22.0f}")
        else:
            print(f"{lab:14}{dm:>12.3f}+-{ds:<7.3f}{fm:>12.3f}+-{fs:<7.3f}")
    print(f"  -> WEIGHT leak DEQ {agg(deqs,'weight_auc')[0]:.3f} vs "
          f"FF {agg(ffs,'weight_auc')[0]:.3f}  "
          f"(fit gap: DEQ {agg(deqs,'train_acc')[0]:.2f} / FF {agg(ffs,'train_acc')[0]:.2f})")


def main():
    SEEDS = [0, 1, 2, 3, 4]
    for nf, title in [(0.30, "NOISY labels (memorization bait)"),
                      (0.00, "CLEAN labels (baseline)")]:
        deqs, ffs = [], []
        for s in SEEDS:
            d, f = run_once(s, nf)
            deqs.append(d); ffs.append(f)
        report(deqs, ffs, title)


if __name__ == "__main__":
    main()
