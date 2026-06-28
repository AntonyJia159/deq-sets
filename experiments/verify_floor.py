"""Verification: is the ~2.5e-2 conditioning floor (warm-vs-cold basin gap == far-field plateau)
on the non-contractive equilibrium benign or problematic?

Four checks (all forward-solve only; no Jacobian SVD):
  1. SOLVER- vs CONDITIONING-limited: basin gap at standard vs tighter Broyden budget. Shrinks ->
     solver artifact (benign); persists -> genuine near-critical mode.
  2. PREDICTION FLIPS (task-level benignity): does the latent floor change predicted LABELS
     (warm-start re-solve vs cold full recompute), overall and by hop distance from the edit?
     Far-field flips ~0 => benign for the task regardless of latent slop.
  3. MODE characterization: the warm-cold difference vector ~ v_min (the smallest-singular direction
     of (I-J)). Report its per-node profile (uniform/global vs localized) and cosine similarity
     ACROSS deletions (one fixed intrinsic mode vs edit-specific).
  4. CONTRACTIVE CONTRAST: same probe on a contractive (high jac_gamma, rho<1) instance of the
     SAME architecture -> expect basin ~1e-6 and no floor (the floor is the price of rho>1).

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.verify_floor
"""

import time

import numpy as np
import torch
from torchdeq import get_deq

from experiments.fagcn_deq_broyden import SoftFAGCN, CFG, DEV, load, run_split
from experiments.fagcn_deq_locality import delete_node, build_adj, bfs_hops, MAXHOP

TARGETS_PCT = [25, 50, 75, 90, 97]
DEG_FOR_TARGET = None  # filled at runtime


def solver(maxit, tol):
    return get_deq(f_solver="broyden", f_max_iter=maxit, f_tol=tol)


def equilibrium(model, h0, e, n, zinit, deq):
    f = model._make_f(h0, e, n)
    return deq(f, zinit)[0][-1]                              # (1, N, d)


def predict(model, z, h0):
    return model.head(torch.cat([z[0], h0], dim=-1)).argmax(1)


def probe_model(tag, model, X, deg, adj, N, d):
    model.eval()
    with torch.no_grad():
        h0 = model.enc(X)
    std = solver(80, 1e-6)
    tight = solver(120, 1e-8)
    z0 = torch.zeros(1, N, d, device=DEV)
    with torch.no_grad():
        z_star = equilibrium(model, h0, model.edges, model.norm, z0, std)
        f0 = model._make_f(h0, model.edges, model.norm)
        resid = ((f0(z_star) - z_star).norm() / z_star.norm()).item()
    _, rho = model.diagnose(X)
    print(f"\n===== {tag}: rho(J) {rho:.3f}  resid {resid:.1e} =====", flush=True)

    degnp = deg.cpu().numpy()
    order = np.argsort(degnp)
    targets = [int(order[min(N - 1, int(p / 100 * N))]) for p in TARGETS_PCT]

    gap_vecs = []
    print(f"{'node':>7} {'deg':>4} | {'basin_std':>10} {'basin_tight':>11} | "
          f"{'flip_all':>8} {'flip_far':>8} | {'gap_near':>8} {'gap_far':>8}")
    for v in targets:
        ee, ne = delete_node(v, model.edges, N)
        hops = bfs_hops(adj, v, N)
        with torch.no_grad():
            zw = equilibrium(model, h0, ee, ne, z_star, std)
            zc = equilibrium(model, h0, ee, ne, z0, std)
            zw_t = equilibrium(model, h0, ee, ne, z_star, tight)
            zc_t = equilibrium(model, h0, ee, ne, z0, tight)
            basin_std = ((zw - zc).norm() / (zw.norm() + 1e-9)).item()
            basin_tight = ((zw_t - zc_t).norm() / (zw_t.norm() + 1e-9)).item()
            pw, pc = predict(model, zw, h0), predict(model, zc, h0)
            diff = (pw != pc).cpu().numpy()
            far = hops >= 6
            flip_all = diff[hops >= 1].mean() if (hops >= 1).any() else 0.0
            flip_far = diff[far].mean() if far.any() else 0.0
            gap = (zw[0] - zc[0]).norm(dim=1).cpu().numpy()      # per-node basin-gap profile
            gnear = gap[(hops >= 1) & (hops <= 2)].mean()
            gfar = gap[far].mean() if far.any() else np.nan
            gap_vecs.append((zw[0] - zc[0]).reshape(-1).cpu())
        print(f"{v:>7} {int(degnp[v]):>4} | {basin_std:>10.1e} {basin_tight:>11.1e} | "
              f"{flip_all:>8.4f} {flip_far:>8.4f} | {gnear:>8.1e} {gfar:>8.1e}", flush=True)

    # cross-deletion cosine of basin-gap vectors: one fixed intrinsic mode vs edit-specific?
    if len(gap_vecs) >= 2:
        G = torch.stack([g / (g.norm() + 1e-9) for g in gap_vecs])
        C = (G @ G.t())
        offdiag = C[~torch.eye(len(G), dtype=bool)]
        print(f"  basin-gap vector cosine across deletions: mean {offdiag.mean():.3f} "
              f"(|.|->1 = one fixed mode; ->0 = edit-specific)", flush=True)


def main():
    print(f"device = {DEV}")
    X, y, edges, deg, masks, K = load("roman_empire")
    X, y, edges, deg = X.to(DEV), y.to(DEV), edges.to(DEV), deg.to(DEV)
    N, d = X.shape[0], CFG["d"]
    tr = torch.tensor(masks["train_masks"][0].astype(bool)).to(DEV)
    va = torch.tensor(masks["val_masks"][0].astype(bool)).to(DEV)
    te = torch.tensor(masks["test_masks"][0].astype(bool)).to(DEV)
    adj = build_adj(edges, N)

    # Model A: non-contractive (the one with the floor)
    print("\n--- training MODEL A: non-contractive (jac_gamma=0.1) ---", flush=True)
    t0 = time.time()
    accA, mA = run_split(X.shape[1], K, edges, deg, X, y, tr, va, te, dict(CFG, epochs=100))
    print(f"model A trained: test {accA:.3f} ({time.time()-t0:.0f}s)", flush=True)
    probe_model("MODEL A (non-contractive)", mA, X, deg, adj, N, d)

    # Model B: contractive contrast (strong jac_reg + small s init -> rho<1)
    print("\n--- training MODEL B: contractive contrast (jac_gamma=3.0, s_init=0.5) ---", flush=True)
    t0 = time.time()
    accB, mB = run_split(X.shape[1], K, edges, deg, X, y, tr, va, te,
                         dict(CFG, epochs=80, jac_gamma=3.0, s_init=0.5))
    print(f"model B trained: test {accB:.3f} ({time.time()-t0:.0f}s)", flush=True)
    probe_model("MODEL B (contractive)", mB, X, deg, adj, N, d)

    print("\nVERDICT GUIDE: floor benign if (1) basin_tight << basin_std (solver-limited), "
          "and/or (2) flip_far ~ 0 (task-irrelevant), and (3) gap_near>>gap_far (truly local); "
          "problematic if basin persists under tighter solve AND flip_far > 0. Model B should show "
          "basin ~ 1e-6 with no floor, isolating the cost to rho>1.", flush=True)


if __name__ == "__main__":
    main()
