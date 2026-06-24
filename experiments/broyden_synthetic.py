"""Solver sanity on a small synthetic grid: does Broyden find fixed points the contractive
Picard regime can't, when we only SOFT-constrain?

Context: every prior attempt to use Anderson/Broyden on our graph/set DEQs failed (stagnate/NaN).
Reframing (memory): that was an artifact of FORCING strong uniform contraction (rho pinned ~0.9,
single-mode geometric decay -> degenerate residual history for quasi-Newton). And maintainability
does NOT need rho<1 -- it needs (I-J) well-conditioned (Demko-Moss-Smith), which is exactly the
regime where Broyden works. So the prediction: relax the hard spectral cap, let the operator be
mildly NON-contractive (rho>1 but spectrum away from +1), and Broyden/Anderson should converge
where Picard diverges.

This script: fixed grid graph + nonlinear cell f(z) = h0 + scale * Wo(relu(W1[z || M z])), free
(un-normalized) random weights. Scan `scale` so rho(J) sweeps through 1, and for each report the
final relative residual ||f(z)-z||/||z|| of fixed_point_iter (Picard) vs anderson vs broyden.
Expectation: Picard residual blows up once rho>1; Broyden/Anderson stay ~1e-6.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.broyden_synthetic
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdeq import get_deq
from torchdeq.loss import power_method

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def grid_graph(L):
    """2D L x L grid: symmetric edge list (2,E)=[dst,src] and degree vector."""
    idx = lambda r, c: r * L + c
    dst, src = [], []
    for r in range(L):
        for c in range(L):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < L and 0 <= cc < L:
                    dst.append(idx(r, c)); src.append(idx(rr, cc))
    edges = torch.tensor([dst, src], dtype=torch.long)
    N = L * L
    deg = torch.zeros(N).index_add_(0, edges[0], torch.ones(edges.shape[1]))
    return edges, deg, N


class Cell(nn.Module):
    """f(z) = h0 + scale * Wo(relu(W1[z || M z])); free weights (no spectral norm).

    Works on z of shape (1, N, d): a SINGLE batch element whose N*d coords are one coupled
    system -- required so TorchDEQ's broyden/power_method (which batch on dim 0) treat the whole
    graph as one Jacobian, not N independent per-node blocks.
    """
    def __init__(self, d, edges, deg, scale):
        super().__init__()
        self.W1 = nn.Linear(2 * d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.scale = scale
        self.register_buffer("edges", edges)
        self.register_buffer("norm", 1.0 / torch.sqrt(deg[edges[0]] * deg[edges[1]]))

    def agg(self, z):                              # z: (N, d)
        out = torch.zeros_like(z)
        out.index_add_(0, self.edges[0], self.norm.unsqueeze(-1) * z[self.edges[1]])
        return out

    def make_f(self, h0):                          # h0: (N, d)
        def f(z):                                  # z: (1, N, d)
            zf = z[0]
            g = self.Wo(F.relu(self.W1(torch.cat([zf, self.agg(zf)], dim=-1))))
            return (h0 + self.scale * g).unsqueeze(0)
        return f


def rho_at(f, z_ref):
    z0 = z_ref.detach().requires_grad_(True)
    with torch.enable_grad():
        _, rho = power_method(f(z0), z0, n_iters=60)
    return rho.max().item()


def residual(f, z_init, solver):
    try:
        deq = get_deq(f_solver=solver, f_max_iter=200, f_tol=1e-9)
        z = deq(f, z_init)[0][-1]
        r = (f(z) - z).norm() / (z.norm() + 1e-9)
        v = r.item()
        return "NaN" if not np.isfinite(v) else f"{v:.1e}"
    except Exception as e:
        return f"ERR({type(e).__name__})"


def main():
    print(f"device = {DEV}")
    L, d = 30, 16
    edges, deg, N = grid_graph(L)
    edges, deg = edges.to(DEV), deg.to(DEV)
    h0 = (torch.randn(N, d) * 0.5).to(DEV)
    h0b = h0.unsqueeze(0)                           # (1, N, d) reference for rho probe
    print(f"grid {L}x{L} = {N} nodes, d={d}, {edges.shape[1]} directed edges\n")
    print(f"{'scale':>6} {'rho(J)':>8} | {'picard':>10} {'anderson':>10} {'broyden':>10}")
    print("-" * 52)
    for scale in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        torch.manual_seed(0)                       # same weights, only scale differs
        cell = Cell(d, edges, deg, scale).to(DEV)
        f = cell.make_f(h0)
        z0 = torch.zeros(1, N, d, device=DEV)
        rho = rho_at(f, h0b)
        res = {s: residual(f, z0, s) for s in ("fixed_point_iter", "anderson", "broyden")}
        edge = "  <- non-contractive" if rho >= 1.0 else ""
        print(f"{scale:>6} {rho:>8.3f} | {res['fixed_point_iter']:>10} "
              f"{res['anderson']:>10} {res['broyden']:>10}{edge}", flush=True)
    print("\n(residual = ||f(z)-z||/||z|| at the solver's output; lower = found the fixed point)")


if __name__ == "__main__":
    main()
