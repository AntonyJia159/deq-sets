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


def neighbor_mean_adj(edges, deg, N):
    """Row-normalized adjacency M = D^-1 A, NO self-loop: (M psi)_i = mean_{j~i} psi_j.
    Pure neighbor average -- node i's OWN value never enters. A target built from M and a
    feature projection psi=X@w depends only on {X_j : j~i}, never X_i, so it is INVISIBLE to a
    features-only MLP (leak-free) -- unlike the directional derivative, whose field weights use
    psi_i and so leak the center back into the label."""
    vals = 1.0 / deg[edges[0]].clamp(min=1.0)
    return torch.sparse_coo_tensor(edges, vals, (N, N)).coalesce()


def directional_derivative(edges, N, psi, diag=True):
    """DGN B_dx from a node potential psi: field F[i,j]=psi[j]-psi[i] (edge i<-j), L1-row-normalized,
    minus the centering diagonal. edges=[dst,src]; entry at (row=dst i, col=src j).
    diag=False drops the centering diagonal -> a PURELY OFF-DIAGONAL (neighbor-only) signed operator:
    (B_off X)_i = sum_{j~i} fhat_ij X_j depends only on neighbors' features, never X_i. The field
    WEIGHTS fhat_ij ~ (psi_j-psi_i) are still signed/anisotropic/high-pass (teacher-fixed constants),
    so this stays high-pass while being leak-free for a features-only MLP."""
    dst, src = edges[0], edges[1]
    f = psi[src] - psi[dst]                                   # F[i,j] = psi[j] - psi[i]
    denom = torch.zeros(N, device=psi.device).index_add_(0, dst, f.abs()) + 1e-6
    fhat = f / denom[dst]                                     # L1-row-normalized field
    if not diag:
        return torch.sparse_coo_tensor(edges, fhat, (N, N)).coalesce()
    colsum = torch.zeros(N, device=psi.device).index_add_(0, src, fhat)   # sum_j F_hat[:,i]
    idx = torch.cat([edges, torch.arange(N, device=psi.device).repeat(2, 1)], dim=1)
    vals = torch.cat([fhat, -colsum])                        # off-diag F_hat + diag -colsum
    return torch.sparse_coo_tensor(idx, vals, (N, N)).coalesce()


class AnisoTeacher:
    """Fixed random local high-pass teacher -> local nonlinear node labels.
    Field source = random feature projections (learnable + local). Two targets:
      'laplacian': spectral-convolution energy s_i = sum_r a_r ((L^k psi_r)_i)^2, L = I - Ahat the
          normalized Laplacian, psi_r = X w_r. k=1 = Dirichlet/Laplacian energy (reach 1), k=2 =
          BIHARMONIC (Delta^2, thin-plate; reach 2), ... -> a principled reach hierarchy = the
          locality<->reach knob. PROPAGATE(L^k) then square: linear can't square; APPNP squares
          BEFORE propagating (wrong order) so it also fails; aggregate-then-nonlinear cells win.
          (Small MLP center-leak: L differences from the center, so the pure psi_i^2 fraction is
          features-visible -- bounded and measured, not engineered away.)
      'nbr_sq': leak-free 1-hop NEIGHBOR SECOND-MOMENT -- s_i = sum_r a_r mean_{j~i}(X_j w_r)^2.
          Node-LOCAL nonlinearity (square of each node's own projection) then a plain neighbor average:
          leak-free (no X_i), nonlinear (beats every linear baseline), and EXPRESSIBLE by our cell
          (each node squares its projection in z_j, the next equilibrium layer averages neighbor z).
          Smooth/monotone (no mean^2 cancellation), so stable to fit. Reach = 1 (maximally local).
      'aniso_local': leak-free 1-hop ANISOTROPIC squared energy --
          s_i = sum_r a_r || sum_{j~i} fhat^r_ij X_j ||^2, fhat = off-diagonal directional field.
          Square of a neighbor-AGGREGATE (no X_i term): leak-free for MLP, high-pass/anisotropic
          (signed directional weights), AND expressible by our aggregate-then-nonlinear cell
          (unlike neighbor-variance, which needs a per-edge square). Reach = 1 (maximally local).
      'variance': leak-free 1-hop NEIGHBOR-VARIANCE s_i = sum_r a_r Var_{j~i}(X_j w_r). Also
          leak-free, but Var = mean(psi^2) - mean(psi)^2 needs squaring EACH neighbor before the
          sum -> a per-edge nonlinearity our aggregate-then-nonlinear cell cannot form in one step.
      'aniso': the original DGN-style directional squared-energy WITH the centering diagonal +
          k-hop diffusion (anisotropic, k-hop, but its diagonal/diffusion leak psi_i into MLP)."""
    def __init__(self, edges, deg, N, d_feat=16, R=4, k=3, K=5, seed=0, target="nbr_sq"):
        self.edges, self.deg, self.N = edges, deg, N
        self.d_feat, self.K, self.k = d_feat, K, k
        self.t = max(0, k - 1)                                # linear-diffusion hops; reach = t + 1
        self.target = target
        g = torch.Generator(device=edges.device).manual_seed(seed)
        self.proj = [torch.randn(d_feat, generator=g, device=edges.device) for _ in range(R)]
        self.a = torch.randn(R, generator=g, device=edges.device)      # field combination weights
        self.X = torch.randn(N, d_feat, generator=g, device=edges.device)
        self.thresholds = None

    @torch.no_grad()
    def _energy(self, edges, deg):
        if self.target == "laplacian":                       # (L^k psi)^2 : Laplacian/biharmonic energy
            Ahat = sym_norm_adj(edges, deg, self.N)           # L = I - Ahat (normalized Laplacian)
            s = torch.zeros(self.N, device=edges.device)      # k=1 Laplacian, k=2 biharmonic, ... reach=k
            for w, a in zip(self.proj, self.a):
                f = (self.X @ w)[:, None]
                for _ in range(self.k):                       # apply L: Lf = f - Ahat f (1 hop each)
                    f = f - torch.sparse.mm(Ahat, f)
                s = s + a * f.squeeze(1) ** 2                 # PROPAGATE(L^k) THEN square (vs APPNP's
            return s                                          # square-then-propagate -> defeats APPNP)
        if self.target == "nbr_sq":                          # leak-free 1-hop neighbor second-moment
            M = neighbor_mean_adj(edges, deg, self.N)         # s_i = sum_r a_r mean_{j~i}(X_j w_r)^2
            s = torch.zeros(self.N, device=edges.device)
            for w, a in zip(self.proj, self.a):
                psi2 = (self.X @ w) ** 2                      # NODE-LOCAL nonlinearity (squarable in z_j)
                s = s + a * torch.sparse.mm(M, psi2[:, None]).squeeze(1)   # then neighbor-averaged
            return s
        if self.target == "aniso_local":                     # leak-free 1-hop anisotropic energy
            s = torch.zeros(self.N, device=edges.device)
            for w, a in zip(self.proj, self.a):
                B = directional_derivative(edges, self.N, self.X @ w, diag=False)  # neighbor-only
                D = torch.sparse.mm(B, self.X)               # square of a pure-neighbor aggregate
                s = s + a * (D ** 2).sum(1)
            return s
        if self.target == "variance":                        # leak-free 1-hop neighbor variance
            M = neighbor_mean_adj(edges, deg, self.N)
            s = torch.zeros(self.N, device=edges.device)
            for w, a in zip(self.proj, self.a):
                psi = self.X @ w                             # neighbor-only field source
                var = torch.sparse.mm(M, (psi ** 2)[:, None]).squeeze(1) \
                    - torch.sparse.mm(M, psi[:, None]).squeeze(1) ** 2
                s = s + a * var                              # E[psi^2|nbr] - E[psi|nbr]^2 >= 0
            return s
        Ahat = sym_norm_adj(edges, deg, self.N)              # original anisotropic directional energy
        Xd = self.X
        for _ in range(self.t):
            Xd = torch.sparse.mm(Ahat, Xd)
        s = torch.zeros(self.N, device=edges.device)
        for w, a in zip(self.proj, self.a):
            D = torch.sparse.mm(directional_derivative(edges, self.N, self.X @ w), Xd)
            s = s + a * (D ** 2).sum(1)
        return s

    @torch.no_grad()
    def generate(self, edges=None, deg=None):
        """Continuous standardized energy self.s (regression target) + quantile-binned labels y."""
        edges = self.edges if edges is None else edges
        deg = self.deg if deg is None else deg
        s = self._energy(edges, deg)
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
    L, d_feat, R, k, K = 40, 16, 4, 3, 5
    edges, deg, N = grid_graph(L)
    edges, deg = edges.to(DEV), deg.to(DEV)
    teacher = AnisoTeacher(edges, deg, N, d_feat=d_feat, R=R, k=k, K=K, seed=0, target="nbr_sq")
    X, y = teacher.generate()
    counts = torch.bincount(y, minlength=K).cpu().numpy()
    print(f"grid {L}x{L}={N} nodes | teacher: variance, R={R} feature fields, K={K} classes")
    print(f"class balance: {counts}  (chance acc = {1.0/K:.3f})")
    feat_acc = feature_only_probe(X, y, K)
    print(f"feature-only logistic test acc: {feat_acc:.3f}  "
          f"(should be ~chance {1.0/K:.3f} => label needs the graph)")
    print("\nteacher OK: random features + local anisotropic high-pass diffusion -> k-hop-local labels.")


if __name__ == "__main__":
    main()
