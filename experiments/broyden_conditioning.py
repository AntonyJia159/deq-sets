"""(1) What actually predicts solver/maintainability breakdown: sigma_min(I-J), not rho(J).

Follow-up to broyden_synthetic.py. There we saw Picard fail at rho>1 while Broyden kept solving.
Theory (Demko-Moss-Smith / Faber) says the governing quantity is NOT rho(J) but the distance of
the spectrum from +1 == conditioning of (I-J) == sigma_min(I-J). And for a NON-symmetric J
(complex spectrum) the honest scalar is sigma_min(I-J) itself, not min|1-lambda| (eigenvalues can
mislead for non-normal matrices).

This script: small grid (dense Jacobian is formable), nonlinear free-weight cell, scan `scale`
WELL past where Broyden worked, and report at each scale:
  rho(J)          = max |eigenvalue|            (the WRONG probe)
  min|1-lambda|   = eigenvalue distance from +1 (right idea, but eig-based -> shaky if non-normal)
  sigma_min(I-J)  = smallest singular value     (the RIGHT, symmetry-agnostic probe)
  kappa(I-J)      = cond number
  + residuals of picard / anderson / broyden.

Prediction: Broyden keeps solving (and a real fixed point exists) as long as sigma_min(I-J) is
bounded away from 0 -- EVEN when rho(J)>1 -- and breaks only as sigma_min(I-J) -> 0 (spectrum
reaching +1), regardless of how large rho already is.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.broyden_conditioning
"""

import numpy as np
import torch
from torchdeq import get_deq

from experiments.broyden_synthetic import grid_graph, Cell, DEV


def solve(f, z0, solver):
    try:
        deq = get_deq(f_solver=solver, f_max_iter=300, f_tol=1e-10)
        z = deq(f, z0)[0][-1]
        r = ((f(z) - z).norm() / (z.norm() + 1e-9)).item()
        return z, (r if np.isfinite(r) else np.inf)
    except Exception:
        return None, np.inf


def dense_jacobian(f, z_ref):
    """Full Jacobian df/dz at z_ref (shape (1,N,d)) as an (n,n) matrix, n=N*d."""
    flat = z_ref.reshape(-1).clone()

    def f_flat(v):
        return f(v.view_as(z_ref)).reshape(-1)

    J = torch.autograd.functional.jacobian(f_flat, flat, vectorize=False)
    return J.double()


def main():
    print(f"device = {DEV}")
    L, d = 8, 8
    edges, deg, N = grid_graph(L)
    edges, deg = edges.to(DEV), deg.to(DEV)
    n = N * d
    h0 = (torch.randn(N, d) * 0.5).to(DEV)
    print(f"grid {L}x{L} = {N} nodes, d={d}, jacobian dim n={n}\n")
    hdr = (f"{'scale':>6} {'rho(J)':>8} {'min|1-l|':>9} {'sgmin(I-J)':>11} {'kappa':>9} | "
           f"{'picard':>9} {'anderson':>9} {'broyden':>9}")
    print(hdr); print("-" * len(hdr))
    for scale in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        torch.manual_seed(0)                       # same weights, only scale differs
        cell = Cell(d, edges, deg, scale).to(DEV)
        f = cell.make_f(h0)
        z0 = torch.zeros(1, N, d, device=DEV)
        res = {}
        zsol = {}
        for s in ("fixed_point_iter", "anderson", "broyden"):
            zsol[s], res[s] = solve(f, z0, s)
        # evaluate J at a real fixed point if Broyden found one, else at h0 (consistent reference)
        zb = zsol["broyden"]
        z_ref = zb if (zb is not None and res["broyden"] < 1e-3) else h0.unsqueeze(0)
        J = dense_jacobian(f, z_ref)
        eig = torch.linalg.eigvals(J)
        rho = eig.abs().max().item()
        gap_eig = (1.0 - eig).abs().min().item()
        sv = torch.linalg.svdvals(torch.eye(n, dtype=torch.double, device=J.device) - J)
        sgmin, kappa = sv.min().item(), (sv.max() / sv.min()).item()

        def fmt(x):
            return f"{x:.1e}" if np.isfinite(x) else "  diverge"
        print(f"{scale:>6} {rho:>8.3f} {gap_eig:>9.3f} {sgmin:>11.3e} {kappa:>9.1e} | "
              f"{fmt(res['fixed_point_iter']):>9} {fmt(res['anderson']):>9} "
              f"{fmt(res['broyden']):>9}", flush=True)
    print("\nPrediction: broyden tracks sigma_min(I-J)->0 (spectrum reaching +1), NOT rho>1.")


if __name__ == "__main__":
    main()
