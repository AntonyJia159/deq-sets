"""A local anisotropic-diffusion TEACHER that generates a beyond-linear, k-hop-local node task.

Motivation: DGN (Directional Graph Networks, Beaini et al. 2020, arXiv:2010.02863) and Graph
Anisotropic Diffusion (Elhag/Corso/Staerk/Bronstein 2022, arXiv:2205.00354) build anisotropic
HIGH-PASS aggregation from the graph Laplacian. Their key operator is the directional DERIVATIVE
    B_dx = F_hat - diag( sum_j F_hat[:,j] ),   F_hat = L1-row-normalized field F,
a 1-hop, signed (high-pass) directional operator. In DGN/GAD the field F = grad(phi) is the
gradient of a LOW-FREQUENCY Laplacian eigenvector -- which is GLOBAL (a deletion shifts the whole
eigenvector -> breaks edit-locality) AND positional (a hidden signal the student can't observe ->
an unpredictable label component that caps every model).

Our adaptation, on both counts: the field is the gradient of a random *feature* projection,
psi = X @ w_r (w_r ~ N(0,1)^d). This is (a) LOCAL (1-hop op; a deletion only changes grad(psi) on
incident edges) and (b) LEARNABLE (a function of the observed features X, exactly the kind of
signed feature-difference weighting FAGCN's attention can represent). A bank of such random fields
gives PER-CHANNEL anisotropy once combined by a random channel-mixer (GAD's concat-filters->MLP).

Teacher (k local layers, then a random nonlinear readout, quantile-binned to balanced classes):
    bank = [ A_hat(low-pass), B_dx^(1..R)(aniso high-pass, feature-gradient fields) ]   # all 1-hop
    H_0 = X (random Gaussian features)
    H_{l+1} = tanh( concat[ H_l, {O H_l : O in bank} ] @ W_l )     # W_l random channel-mixer
    s = relu(H_k @ Wa) @ wb ;  y = quantile_bin(s, K)              # balanced K-class labels
By construction: features alone are uninformative (label needs neighbor aggregation -> MLP ~chance);
the signal is high-pass + anisotropic (low-pass SGC/APPNP cannot match); it is nonlinear (a linear
filter, even multi-hop high-pass, cannot match); and it is k-hop LOCAL, so screening length k is
optimal -> maintainable. generate() accepts an edges/deg override (with X and thresholds fixed) so
an edited graph can be re-labelled for the edit-locality / regeneration probe.

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
    minus the centering diagonal. edges=[dst,src]; entry at (row=dst i, col=src j)."""
    dst, src = edges[0], edges[1]
    f = psi[src] - psi[dst]                                   # F[i,j] = psi[j] - psi[i]
    denom = torch.zeros(N, device=psi.device).index_add_(0, dst, f.abs()) + 1e-6
    fhat = f / denom[dst]                                     # L1-row-normalized field
    colsum = torch.zeros(N, device=psi.device).index_add_(0, src, fhat)   # sum_j F_hat[:,i]
    idx = torch.cat([edges, torch.arange(N, device=psi.device).repeat(2, 1)], dim=1)
    vals = torch.cat([fhat, -colsum])                        # off-diag F_hat + diag -colsum
    return torch.sparse_coo_tensor(idx, vals, (N, N)).coalesce()


class AnisoTeacher:
    """Fixed random local-anisotropic-diffusion teacher -> k-hop-local nonlinear node labels.
    Field source = random feature projections (learnable + local)."""
    def __init__(self, edges, deg, N, d_feat=16, R=4, k=3, K=5, seed=0):
        self.edges, self.deg, self.N = edges, deg, N
        self.d_feat, self.K, self.k = d_feat, K, k
        self.t = max(0, k - 1)                                # linear-diffusion hops; reach = t + 1
        g = torch.Generator(device=edges.device).manual_seed(seed)
        self.proj = [torch.randn(d_feat, generator=g, device=edges.device) for _ in range(R)]
        self.a = torch.randn(R, generator=g, device=edges.device)      # field combination weights
        self.X = torch.randn(N, d_feat, generator=g, device=edges.device)
        self.thresholds = None

    @torch.no_grad()
    def generate(self, edges=None, deg=None):
        """label = quantile-bin( sum_r a_r * || B_dx^r ( A_hat^t X ) ||^2 ).
        Structured + SHALLOW so it is learnable: t linear-diffusion hops (GAD's linear step), then
        anisotropic directional derivatives, then a NONLINEAR (squared-energy) readout. High-pass
        (derivative) + anisotropic (random directional fields) + nonlinear (square) + (t+1)-hop local;
        a linear readout cannot match the square, low-pass cannot keep the derivative."""
        edges = self.edges if edges is None else edges
        deg = self.deg if deg is None else deg
        Ahat = sym_norm_adj(edges, deg, self.N)
        Xd = self.X
        for _ in range(self.t):                               # linear anisotropy-free diffusion
            Xd = torch.sparse.mm(Ahat, Xd)
        s = torch.zeros(self.N, device=edges.device)
        for w, a in zip(self.proj, self.a):
            D = torch.sparse.mm(directional_derivative(edges, self.N, self.X @ w), Xd)
            s = s + a * (D ** 2).sum(1)                       # squared directional high-freq energy
        self.s = (s - s.mean()) / (s.std() + 1e-6)            # standardized continuous target (regression)
        if self.thresholds is None:                           # fix thresholds from the base graph
            self.thresholds = torch.quantile(
                s, torch.linspace(0, 1, self.K + 1, device=s.device)[1:-1])
        y = torch.bucketize(s, self.thresholds)
        return self.X, y


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
    print(f"grid {L}x{L}={N} nodes | teacher: R={R} feature-gradient fields, k={k} hops, K={K} classes")
    print(f"class balance: {counts}  (chance acc = {1.0/K:.3f})")
    feat_acc = feature_only_probe(X, y, K)
    print(f"feature-only logistic test acc: {feat_acc:.3f}  "
          f"(should be ~chance {1.0/K:.3f} => label needs the graph)")
    print("\nteacher OK: random features + local anisotropic high-pass diffusion -> k-hop-local labels.")


if __name__ == "__main__":
    main()
