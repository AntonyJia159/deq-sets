"""THE MAINTENANCE DEMO: warm-start local re-solve of a NONLINEAR equilibrium after a graph edit.

The contribution made concrete. Train the general MPNN-DEQ on the (local, beyond-linear) Laplacian
operator task -- a model the LINEAR editable incumbents (InstantGNN/SIGN/APPNP) cannot express
(operator_compare: SIGN 0.33 / APPNP 0.19 / SGC 0.01 vs ours ~0.7). Then DELETE a node and maintain
the equilibrium by WARM-STARTING from the old fixed point and re-solving. We measure the editing
triple + cost:
  RELIABILITY/exactness : warm-start re-solve == cold (from-zero) re-solve  (path-independence;
                          warm-start is FREE correctness, not an approximation)
  LOCALITY              : ||z* change|| and prediction change DECAY with graph distance -> xi
  COST                  : warm-start iters << cold iters; truncation radius (ring where predictions
                          actually move) << graph diameter

This is the nonlinear-equilibrium generalization of InstantGNN's linear incremental update, valid
because (I-J) is well-conditioned (sigma_min away from 0), NOT because of contraction-as-such.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.maintenance_demo
"""

import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.broyden_synthetic import grid_graph
from experiments.aniso_teacher import AnisoTeacher
from experiments.mpnn_deq import MPNNDEQ, CFG
from experiments.fagcn_deq_locality import delete_node, build_adj, bfs_hops

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, D_FEAT, R, K_OP = 40, 16, 4, 2          # K_OP=2 -> biharmonic teacher (reach 2)
MAXHOP = 14


def r2(out, t, m):
    o = out[m].squeeze(-1); tm = t[m]
    return (1 - ((o - tm) ** 2).sum() / (((tm - tm.mean()) ** 2).sum() + 1e-9)).item()


def train(model, X, t, tr, va, epochs):
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    best_va, state = -1e9, None
    for e in range(epochs):
        model.train(); opt.zero_grad()
        out, reg = model(X, jac=True)
        (F.mse_loss(out[tr].squeeze(-1), t[tr]) + CFG["jac_gamma"] * reg).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if e % 10 == 0:
            model.eval()
            with torch.no_grad():
                out, _ = model(X)
            v = r2(out, t, va)
            if v > best_va:
                best_va = v
                state = {k: x.detach().clone() for k, x in model.state_dict().items()}
    model.load_state_dict(state)
    model.eval()
    return best_va


@torch.no_grad()
def solve(f, z, tol=1e-7, maxit=300):
    """Picard fixed-point with iteration count (cell is contractive -> converges)."""
    for k in range(1, maxit + 1):
        zn = f(z)
        r = (zn - z).norm() / (zn.norm() + 1e-9)
        z = zn
        if r < tol:
            return z, k
    return z, maxit


def main():
    print(f"device = {DEV}")
    edges, deg, N = grid_graph(L)
    edges, deg = edges.to(DEV), deg.to(DEV)
    teacher = AnisoTeacher(edges, deg, N, d_feat=D_FEAT, R=R, k=K_OP, seed=0, target="laplacian")
    teacher.generate()
    X, t = teacher.X, teacher.s
    g = torch.Generator().manual_seed(0); p = torch.randperm(N, generator=g)
    tr = torch.zeros(N, dtype=torch.bool); va = tr.clone()
    tr[p[: N // 2]] = True; va[p[N // 2: 3 * N // 4]] = True
    tr, va = tr.to(DEV), va.to(DEV)

    print(f"grid {L}x{L}={N}, teacher=laplacian k={K_OP} (reach {K_OP}); training MPNN-DEQ ...",
          flush=True)
    torch.manual_seed(0)
    model = MPNNDEQ(D_FEAT, 1, edges, deg, CFG).to(DEV)
    t0 = time.time()
    vr = train(model, X, t, tr, va, CFG["epochs"])
    rho = model.spectral_radius(X)
    print(f"  trained: val R^2 {vr:.3f}  rho(J) {rho:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    print("  (linear editable incumbents on this task: SIGN 0.33 / APPNP 0.19 / SGC 0.01 -- we are "
          "the EXPRESSIVE member of the editable class)\n", flush=True)

    with torch.no_grad():
        h0 = model.enc(X)                                   # graph-independent input injection

    @torch.no_grad()
    def pred(z):
        return model.head(torch.cat([z, h0], dim=-1)).squeeze(-1)

    z0 = torch.zeros(N, model.d, device=DEV)
    f_full = model._make_f(h0)
    z_star, it_full = solve(f_full, z0)
    resid = ((f_full(z_star) - z_star).norm() / z_star.norm()).item()
    pred_star = pred(z_star)
    print(f"full equilibrium: {it_full} Picard iters from cold, rel-resid {resid:.1e}\n", flush=True)

    adj = build_adj(edges, N)
    idx = lambda r, c: r * L + c
    targets = [idx(L // 2, L // 2), idx(L // 2, L // 4), idx(L // 4, L // 4), idx(3, 3)]  # interior

    by_hop_z, by_hop_p = {h: [] for h in range(MAXHOP + 1)}, {h: [] for h in range(MAXHOP + 1)}
    exact_gaps, warm_its, cold_its, trunc_radii = [], [], [], []
    print("--- per-deletion warm-start re-solve ---", flush=True)
    print(f"{'node':>6} {'deg':>4} | {'warmIt':>6} {'coldIt':>6} {'exact(warm-cold)':>17} "
          f"{'predRingR':>10}", flush=True)
    for v in targets:
        ee, ne = delete_node(v, edges, N)
        f_e = model._make_f(h0, ee, ne)
        z_warm, it_w = solve(f_e, z_star)                   # warm-start from old fixed point
        z_cold, it_c = solve(f_e, z0)                       # cold from zero
        gap = ((z_warm - z_cold).norm() / (z_warm.norm() + 1e-9)).item()
        dz = (z_warm - z_star).norm(dim=1).cpu().numpy()
        dp = (pred(z_warm) - pred_star).abs().cpu().numpy()
        hops = bfs_hops(adj, v, N)
        ptol = 1e-3 * float(np.abs(pred_star.cpu().numpy()).mean() + 1e-9)
        ring = 0
        for h in range(MAXHOP + 1):
            m = hops == h
            if m.any():
                by_hop_z[h].extend(dz[m].tolist()); by_hop_p[h].extend(dp[m].tolist())
                if dp[m].mean() > ptol:
                    ring = h
        exact_gaps.append(gap); warm_its.append(it_w); cold_its.append(it_c); trunc_radii.append(ring)
        print(f"{v:>6} {int(deg[v]):>4} | {it_w:>6} {it_c:>6} {gap:>17.1e} {ring:>10}", flush=True)

    print("\n--- edit-response decay vs graph distance (mean over deletions) ---", flush=True)
    print(f"{'hop':>4} {'mean||dz||':>12} {'mean|dpred|':>12} {'nodes':>7}", flush=True)
    hs, zs = [], []
    for h in range(MAXHOP + 1):
        if by_hop_z[h]:
            mz, mp = np.mean(by_hop_z[h]), np.mean(by_hop_p[h])
            print(f"{h:>4} {mz:>12.3e} {mp:>12.3e} {len(by_hop_z[h]):>7}", flush=True)
            if h >= 1 and mz > resid * 5:
                hs.append(h); zs.append(mz)
    if len(hs) >= 2:
        slope, _ = np.polyfit(hs, np.log(zs), 1)
        xi = -1.0 / slope if slope < 0 else np.inf
        print(f"\nscreening length xi ~ {xi:.2f} hops (log-slope {slope:.3f})", flush=True)
    print(f"\nSUMMARY  exactness warm==cold: {np.mean(exact_gaps):.1e}  (path-independent => "
          f"warm-start is free correctness)", flush=True)
    print(f"         cost: warm {np.mean(warm_its):.1f} iters vs cold {np.mean(cold_its):.1f} iters "
          f"({np.mean(cold_its)/max(np.mean(warm_its),1e-9):.1f}x); pred ring radius "
          f"{np.mean(trunc_radii):.1f} hops << diameter {2*(L-1)}", flush=True)


if __name__ == "__main__":
    main()
