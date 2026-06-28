"""A local anisotropic-diffusion TEACHER that generates a beyond-linear, k-hop-local node task.

Motivation: DGN (Directional Graph Networks, Beaini et al. 2020, arXiv:2010.02863) and Graph
Anisotropic Diffusion (Elhag/Corso/Staerk/Bronstein 2022, arXiv:2205.00354) build anisotropic
HIGH-PASS aggregation from the graph Laplacian. Their key operator is the directional DERIVATIVE
    B_dx = F_hat - diag( sum_j F_hat[:,j] ),   F_hat = L1-row-normalized field F,
a 1-hop, signed (high-pass) directional operator. In DGN/GAD the field F = grad(phi) is the
gradient of a LOW-FREQUENCY Laplacian eigenvector (the Fiedler vector) -- which is GLOBAL: deleting
a node shifts the whole eigenvector, so the anisotropy (and the target) would be globally coupled,
breaking edit-locality.

We keep the operator B_dx but replace the global eigenvector field with the gradient of a RANDOM
per-node potential psi ~ N(0,1). Same anisotropic high-pass operator; LOCAL field (deleting a node
only changes grad(psi) on incident edges). A bank of such random fields gives PER-CHANNEL anisotropy
once combined by a random channel-mixer (GAD's "concat anisotropic filters -> MLP" pattern).

Teacher (k local layers, then a random MLP readout):
    bank = [ I, A_hat(low-pass), B_dx^(1..R)(aniso high-pass from random potentials) ]   # all 1-hop
    H_0 = X (random Gaussian features)
    H_{l+1} = tanh( concat_o[ O H_l ] @ W_l )          # W_l random: per-channel anisotropic mix
    label = argmax( relu(H_k @ Wa) @ Wb )              # random nonlinear readout
By construction: features alone are uninformative (random X -> MLP at chance); the signal is
high-pass + anisotropic (low-pass SGC/APPNP cannot match); it is nonlinear (needs >linear filter);
and it is k-hop LOCAL (every operator is 1-hop), so screening length k is optimal -> maintainable.

This module = the teacher + a label generator + a quick sanity (class balance, feature-only probe).
The MLP/SGC/APPNP/ours comparison and the edit-locality/regeneration probe are the next scripts.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.aniso_teacher
"""

import numpy as np
import torch

from experiments.broyden_synthetic import grid_graph

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def sym_norm_adj(edges, deg, N):
    vals = 1.0 / torch.sqrt(deg[edges[0]] * deg[edges[1]])
    return torch.sparse_coo_tensor(edges, vals, (N, N)).coalesce()


def directional_derivative(edges, N, psi):
    """DGN B_dx from a node potential psi: field F[i,j]=psi[j]-psi[i] (edge i<-j), L1-row-normalized,
    minus the centering diagonal. edges=[dst,src]; entry (row=dst i, col=src j)."""
    dst, src = edges[0], edges[1]
    f = psi[src] - psi[dst]                                   # F[i,j] = psi[j] - psi[i]
    denom = torch.zeros(N, device=psi.device).index_add_(0, dst, f.abs()) + 1e-6
    fhat = f / denom[dst]                                     # L1-row-normalized field
    colsum = torch.zeros(N, device=psi.device).index_add_(0, src, fhat)   # sum_j F_hat[:,i]
    idx = torch.cat([edges, torch.arange(N, device=psi.device).repeat(2, 1)], dim=1)
    vals = torch.cat([fhat, -colsum])                        # off-diag F_hat + diag -colsum
    return torch.sparse_coo_tensor(idx, vals, (N, N)).coalesce()


def build_bank(edges, deg, N, R, gen):
    """Operator bank: identity (passthrough, handled in forward), A_hat, and R random B_dx."""
    bank = [sym_norm_adj(edges, deg, N)]
    for _ in range(R):
        psi = torch.randn(N, generator=gen, device=edges.device)
        bank.append(directional_derivative(edges, N, psi))
    return bank


class AnisoTeacher:
    """Fixed random local-anisotropic-diffusion teacher -> k-hop-local nonlinear node labels."""
    def __init__(self, edges, deg, N, d_feat=16, d_hid=16, R=4, k=3, K=5, seed=0):
        self.edges, self.N, self.d_feat, self.K = edges, N, d_feat, K
        g = torch.Generator(device=edges.device).manual_seed(seed)
        self.bank = build_bank(edges, deg, N, R, g)
        n_op = 1 + len(self.bank)                              # +1 for identity passthrough
        self.Ws, dims = [], [d_feat] + [d_hid] * k
        for l in range(k):
            fan = dims[l] * n_op
            self.Ws.append(torch.randn(fan, dims[l + 1], generator=g, device=edges.device) / fan ** 0.5)
        self.Wa = torch.randn(d_hid, d_hid, generator=g, device=edges.device) / d_hid ** 0.5
        self.wb = torch.randn(d_hid, generator=g, device=edges.device) / d_hid ** 0.5  # -> scalar
        self.gen = g

    @torch.no_grad()
    def generate(self):
        X = torch.randn(self.N, self.d_feat, generator=self.gen, device=self.edges.device)
        H = X
        for W in self.Ws:
            Z = torch.cat([H] + [torch.sparse.mm(O, H) for O in self.bank], dim=1)
            H = torch.tanh(Z @ W)
        s = torch.relu(H @ self.Wa) @ self.wb                  # scalar teacher output per node
        # quantile-bin the scalar into K BALANCED classes (removes the argmax class-collapse)
        q = torch.quantile(s, torch.linspace(0, 1, self.K + 1, device=s.device)[1:-1])
        y = torch.bucketize(s, q)
        return X, y


def feature_only_probe(X, y, K, epochs=300):
    """Linear logistic on features alone -> should be ~chance (label needs the graph)."""
    torch.manual_seed(0)
    lin = torch.nn.Linear(X.shape[1], K).to(X.device)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    n = X.shape[0]
    tr = torch.zeros(n, dtype=torch.bool, device=X.device); tr[: n // 2] = True
    for _ in range(epochs):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(lin(X)[tr], y[tr]).backward(); opt.step()
    with torch.no_grad():
        pred = lin(X).argmax(1)
    return (pred[~tr] == y[~tr]).float().mean().item()


def main():
    print(f"device = {DEV}")
    L, d_feat, R, k, K = 30, 16, 4, 3, 5
    edges, deg, N = grid_graph(L)
    edges, deg = edges.to(DEV), deg.to(DEV)
    teacher = AnisoTeacher(edges, deg, N, d_feat=d_feat, R=R, k=k, K=K, seed=0)
    X, y = teacher.generate()
    counts = torch.bincount(y, minlength=K).cpu().numpy()
    print(f"grid {L}x{L}={N} nodes | teacher: R={R} random fields, k={k} hops, K={K} classes")
    print(f"class balance: {counts}  (chance acc = {1.0/K:.3f})")
    feat_acc = feature_only_probe(X, y, K)
    print(f"feature-only logistic test acc: {feat_acc:.3f}  "
          f"(should be ~chance {1.0/K:.3f} => label needs the graph)")
    print("\nteacher OK: random features + local anisotropic high-pass diffusion -> k-hop-local labels.")


if __name__ == "__main__":
    main()
