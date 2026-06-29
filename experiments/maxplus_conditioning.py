"""Does the linear conditioning theory (sigma_min / field-of-values via Chebyshev-Faber) predict
edit-locality of a TROPICAL (max-aggregation) equilibrium -- or does non-smoothness break it?

Report #7: DMS/Benzi-Golub bound |(I-J)^{-1}_ij| <= C q^{d(i,j)} via best-polynomial (Faber)
approximation of g(z)=1/(1-z) on a region containing spec(J)/W(J) but EXCLUDING the pole z=1.
=> rho(J)>1 is fine; locality holds iff the spectrum/field-of-values avoids +1, rate set by
dist(1, W(J)). This rests on a FIXED linear J. For max-agg, J is a SUBGRADIENT (argmax routing)
that is locally constant then jumps. So the frozen-J Faber prediction should hold TO FIRST ORDER
-- unless an edit SWITCHES the routing a lot. We measure both.

Small 8x8 grid (N=64, d=16 -> 1024-dim state, dense Jacobian formable). Train the max-agg cell on
the tropical (max-reach) task, form J at the fixed point, and compare candidate scalars to the
MEASURED screening length, plus count argmax-routing switches caused by a deletion.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.maxplus_conditioning
"""

import numpy as np
import torch
import torch.nn.functional as F

from experiments.broyden_synthetic import grid_graph
from experiments.aniso_teacher import AnisoTeacher
from experiments.mpnn_deq import MPNNDEQ, CFG, _sn
from experiments.fagcn_deq_locality import delete_node, build_adj, bfs_hops

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, D_FEAT, R, d = 8, 8, 3, 16
MAXHOP = 8
AGG = "logsumexp"      # "max" (hard, non-smooth subgradient) | "logsumexp" (SMOOTH, learnable beta)


@torch.no_grad()
def solve(f, z, tol=1e-9, maxit=1000):
    for k in range(1, maxit + 1):
        zn = f(z); r = (zn - z).norm() / (zn.norm() + 1e-12); z = zn
        if r < tol:
            return z, k
    return z, maxit


@torch.no_grad()
def argmax_routing(model, z, edges, norm):
    """For max-agg: which neighbor wins the max at each (node, channel). Returns the winning src
    index per (dst, channel), for counting routing switches."""
    Wm_n, Wm2_n = _sn(model.Wm.weight), _sn(model.Wm2.weight)
    dst, src = edges[0], edges[1]
    e = torch.cat([z[dst], z[src]], dim=-1)
    m = F.relu(e @ Wm_n.t() + model.Wm.bias) @ Wm2_n.t()         # (E, d)
    N = z.shape[0]
    win = torch.full((N, model.d), -1, dtype=torch.long, device=z.device)
    best = torch.full((N, model.d), -1e30, device=z.device)
    for k in range(dst.shape[0]):                                 # small graph -> simple loop is fine
        better = m[k] > best[dst[k]]
        best[dst[k]] = torch.where(better, m[k], best[dst[k]])
        win[dst[k]] = torch.where(better, torch.full_like(win[dst[k]], src[k]), win[dst[k]])
    return win                                                    # (N, d) winning src per channel


def main():
    print(f"device = {DEV}")
    edges, deg, N = grid_graph(L)
    edges, deg = edges.to(DEV), deg.to(DEV)
    teacher = AnisoTeacher(edges, deg, N, d_feat=D_FEAT, R=R, k=1, seed=0, target="maxreach")
    teacher.generate()
    X, t = teacher.X, teacher.s
    cfg = dict(CFG, d=d, msg_hidden=d, mlp_hidden=d, agg=AGG, drop_in=0.0, drop_out=0.0,
               edge_drop=0.0, jac_gamma=0.0)
    torch.manual_seed(0)
    model = MPNNDEQ(D_FEAT, 1, edges, deg, cfg).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    allm = torch.ones(N, dtype=torch.bool, device=DEV)
    for _ in range(300):                                          # fit (diagnostic, not generalization)
        model.train(); opt.zero_grad()
        out, _ = model(X)
        F.mse_loss(out[allm].squeeze(-1), t[allm]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        h0 = model.enc(X)
    f0 = model._make_f(h0)
    z0 = torch.zeros(N, d, device=DEV)
    z_star, it = solve(f0, z0)
    rate = (((f0(z_star) - z_star).norm() / z_star.norm())).item()
    btag = f", beta={F.softplus(model.beta_raw).item():.2f}" if AGG == "logsumexp" else ""
    print(f"grid {L}x{L}={N}, d={d}, agg={AGG}{btag}; tropical fit; Picard {it} iters, resid {rate:.1e}",
          flush=True)

    # dense Jacobian (subgradient) at the fixed point
    def f_flat(zf):
        return model._make_f(h0)(zf.view(N, d)).reshape(-1)
    J = torch.autograd.functional.jacobian(f_flat, z_star.reshape(-1)).detach()    # (Nd, Nd)
    I = torch.eye(J.shape[0], device=J.device)
    eig = torch.linalg.eigvals(J)
    rho = eig.abs().max().item()
    dist_spec = (1.0 - eig).abs().min().item()                   # dist(1, spectrum)
    smin = torch.linalg.svdvals(I - J).min().item()              # sigma_min(I-J) = 1/||(I-J)^-1||_2
    Jinf = J.abs().sum(1).max().item()                           # ||J||_inf
    ImJ_inv = torch.linalg.inv(I - J)
    inv_inf = ImJ_inv.abs().sum(1).max().item()                  # ||(I-J)^-1||_inf
    print(f"\nrho(J) {rho:.3f}   min|1-lambda| {dist_spec:.3f}   sigma_min(I-J) {smin:.3e}   "
          f"||J||_inf {Jinf:.3f}   ||(I-J)^-1||_inf {inv_inf:.2f}", flush=True)

    # measured edit-response decay + routing switches
    adj = build_adj(edges, N)
    idx = lambda r, c: r * L + c
    targets = [idx(3, 3), idx(4, 4), idx(2, 5)]
    by_hop = {h: [] for h in range(MAXHOP + 1)}
    sw_near, sw_far = [], []
    base_route = argmax_routing(model, z_star, edges, model.norm)
    for v in targets:
        ee, ne = delete_node(v, edges, N)
        zw, _ = solve(model._make_f(h0, ee, ne), z_star)
        dz = (zw - z_star).norm(dim=1).cpu().numpy()
        hops = bfs_hops(adj, v, N)
        new_route = argmax_routing(model, zw, ee, ne)
        switched = (new_route != base_route).any(dim=1).cpu().numpy()   # node changed any channel's argmax
        for h in range(MAXHOP + 1):
            mm = hops == h
            if mm.any():
                by_hop[h].extend(dz[mm].tolist())
        sw_near.append(switched[(hops >= 0) & (hops <= 2)].mean())
        sw_far.append(switched[hops >= 3].mean())
    print("\nhop   mean||dz||     nodes", flush=True)
    hs, zs = [], []
    for h in range(MAXHOP + 1):
        if by_hop[h]:
            mz = np.mean(by_hop[h])
            print(f"{h:>3}  {mz:>11.3e}  {len(by_hop[h]):>6}", flush=True)
            if h >= 1 and mz > 1e-9:
                hs.append(h); zs.append(mz)
    xi = (-1.0 / np.polyfit(hs, np.log(zs), 1)[0]) if len(hs) >= 2 else float("nan")
    # Faber/DMS predicted rate from the pole-distance: q ~ (sqrt(k)-1)/(sqrt(k)+1), k=||.||*sigma scale
    print(f"\nMEASURED screening length xi ~ {xi:.2f} hops", flush=True)
    print(f"routing switches: near (<=2 hops) {np.mean(sw_near):.0%} | far (>=3 hops) {np.mean(sw_far):.0%}",
          flush=True)
    print("\nREAD: if spec avoids +1 (min|1-lambda| healthy) and far-field routing switches ~0, the "
          "FROZEN-J Faber theory should predict xi (linear conditioning extends to tropical); if far "
          "switches are large, non-smoothness breaks it and xi is a nonlinear effect.", flush=True)


if __name__ == "__main__":
    main()
