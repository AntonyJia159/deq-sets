"""(3) The maintainability proof: is the NON-CONTRACTIVE Broyden equilibrium still edit-local?

We have a real non-contractive equilibrium on roman (fagcn_deq_broyden.py, rho(J)~1.3, solved by
Broyden). The whole broadened thesis rests on one measurable question: does an edit's influence
decay with graph distance even though rho>1? Theory says yes IFF (I-J) is well-conditioned
(spectrum away from +1); contraction is not required. Here we MEASURE it.

Procedure (on the trained model, eval mode):
  1. Solve the full-graph equilibrium z* (tight Broyden).
  2. For several deleted nodes v (stratified by degree):
       - rebuild the graph without v (drop incident edges + recompute symmetric-norm degrees),
       - WARM-START Broyden from z* -> new equilibrium z*',
       - measure the response ||z*'_i - z*_i|| vs BFS hop distance from v.
     Fit the screening length xi; success = xi << graph diameter (PRACTICAL locality, not just
     existence). This is the analogue of cora_deletion.py but on a non-contractive equilibrium.
  3. WARM vs COLD: re-solve the edited graph from z* and from zero; the gap tests
     path-independence (unique basin) vs multistability at rho>1.

rho(J) is reported (confirms non-contractive). sigma_min(I-J) is not dense-formable at N~22k; the
MEASURED screening length is the ground truth here, and sigma_min was validated as its predictor on
the small grid (broyden_conditioning.py).

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.fagcn_deq_locality
"""

import time
from collections import deque

import numpy as np
import torch
from torchdeq import get_deq

from experiments.fagcn_deq_broyden import SoftFAGCN, CFG, DEV, load, run_split

MAXHOP = 12
N_TARGETS_PCT = [5, 25, 45, 55, 65, 80, 90, 95, 97, 99]   # degree percentiles -> stratified targets
N_BASIN = 4                                               # how many targets also get a cold re-solve


def delete_node(v, edges, N):
    """Edited graph without node v: drop incident edges, recompute symmetric-norm degrees."""
    mask = (edges[0] != v) & (edges[1] != v)
    ee = edges[:, mask]
    deg_e = torch.zeros(N, device=edges.device).index_add_(
        0, ee[0], torch.ones(ee.shape[1], device=edges.device))
    deg_e = deg_e.clamp(min=1.0)
    ne = 1.0 / torch.sqrt(deg_e[ee[0]] * deg_e[ee[1]])
    return ee, ne


def build_adj(edges, N):
    adj = [[] for _ in range(N)]
    e = edges.cpu().numpy()
    for k in range(e.shape[1]):
        adj[e[0, k]].append(e[1, k])
    return adj


def bfs_hops(adj, v, N, maxhop=MAXHOP):
    hops = np.full(N, -1, dtype=int)
    hops[v] = 0
    dq = deque([v])
    while dq:
        u = dq.popleft()
        if hops[u] >= maxhop:
            continue
        for w in adj[u]:
            if hops[w] == -1:
                hops[w] = hops[u] + 1
                dq.append(w)
    return hops


def main():
    print(f"device = {DEV}")
    cfg = dict(CFG, epochs=100)
    print("config:", cfg, "\n")
    X, y, edges, deg, masks, K = load("roman_empire")
    X, y, edges, deg = X.to(DEV), y.to(DEV), edges.to(DEV), deg.to(DEV)
    N, d = X.shape[0], cfg["d"]
    tr = torch.tensor(masks["train_masks"][0].astype(bool)).to(DEV)
    va = torch.tensor(masks["val_masks"][0].astype(bool)).to(DEV)
    te = torch.tensor(masks["test_masks"][0].astype(bool)).to(DEV)

    print("--- training soft cell with Broyden ---", flush=True)
    t0 = time.time()
    acc, model = run_split(X.shape[1], K, edges, deg, X, y, tr, va, te, cfg)
    model.eval()
    print(f"trained: test {acc:.3f}  ({time.time()-t0:.0f}s)\n", flush=True)

    # tight probe solver + helper
    probe = get_deq(f_solver="broyden", f_max_iter=80, f_tol=1e-6)
    with torch.no_grad():
        h0 = model.enc(X)                                  # graph-independent input injection

    def solve(e, n, zinit):
        f = model._make_f(h0, e, n)
        return probe(f, zinit)[0][-1]                      # (1, N, d)

    z_zero = torch.zeros(1, N, d, device=DEV)
    with torch.no_grad():
        z_star = solve(model.edges, model.norm, z_zero)
        f0 = model._make_f(h0, model.edges, model.norm)
        resid0 = ((f0(z_star) - z_star).norm() / z_star.norm()).item()
    _, rho = model.diagnose(X)
    znorm = z_star[0].norm(dim=1).mean().item()
    print(f"full-graph equilibrium: rho(J) {rho:.3f}  broyden rel-resid {resid0:.1e}  "
          f"mean||z*_i|| {znorm:.3f}", flush=True)
    print(f"  (rho>1 => non-contractive; floor for the decay ~ resid x ||z*|| ~ {resid0*znorm:.1e})\n",
          flush=True)

    adj = build_adj(model.edges, N)
    degnp = deg.cpu().numpy()
    order = np.argsort(degnp)
    targets = [int(order[min(N - 1, int(p / 100 * N))]) for p in N_TARGETS_PCT]

    by_hop = {h: [] for h in range(1, MAXHOP + 1)}
    basin_gaps = []
    print("--- per-deletion probes (warm-start re-solve) ---", flush=True)
    print(f"{'node':>7} {'deg':>4} | {'d=1':>9} {'d=2':>9} {'d=4':>9} {'d=6':>9} {'basin':>8}")
    for i, v in enumerate(targets):
        ee, ne = delete_node(v, model.edges, N)
        with torch.no_grad():
            z_warm = solve(ee, ne, z_star)
            delta = (z_warm[0] - z_star[0]).norm(dim=1).cpu().numpy()
        hops = bfs_hops(adj, v, N)
        for h in range(1, MAXHOP + 1):
            d_h = delta[hops == h]
            if d_h.size:
                by_hop[h].extend(d_h.tolist())
        near = {h: (delta[hops == h].mean() if (hops == h).any() else np.nan) for h in (1, 2, 4, 6)}
        bstr = ""
        if i < N_BASIN:
            with torch.no_grad():
                z_cold = solve(ee, ne, z_zero)
                bgap = ((z_warm - z_cold).norm() / (z_warm.norm() + 1e-9)).item()
            basin_gaps.append(bgap)
            bstr = f"{bgap:.1e}"
        print(f"{v:>7} {int(degnp[v]):>4} | {near[1]:>9.2e} {near[2]:>9.2e} "
              f"{near[4]:>9.2e} {near[6]:>9.2e} {bstr:>8}", flush=True)

    # aggregate decay + screening-length fit
    print("\n--- aggregate edit-response decay vs graph distance ---", flush=True)
    print(f"{'hop':>4} {'mean||delta||':>14} {'nodes':>8}")
    hs, ys = [], []
    for h in range(1, MAXHOP + 1):
        vals = np.array(by_hop[h])
        if vals.size:
            m = vals.mean()
            print(f"{h:>4} {m:>14.3e} {vals.size:>8}", flush=True)
            hs.append(h); ys.append(m)
    hs, ys = np.array(hs), np.array(ys)
    floor = resid0 * znorm * 3
    fit_mask = ys > floor
    if fit_mask.sum() >= 2:
        slope, intercept = np.polyfit(hs[fit_mask], np.log(ys[fit_mask]), 1)
        xi = -1.0 / slope if slope < 0 else np.inf
        print(f"\nfit over hops with mean||delta|| > 3*floor:  log-slope {slope:.3f}  "
              f"=> screening length xi ~ {xi:.2f} hops", flush=True)
    else:
        print("\nnot enough above-floor points to fit a screening length", flush=True)
    if basin_gaps:
        print(f"warm-vs-cold basin gap (path-independence): mean {np.mean(basin_gaps):.1e} "
              f"(small => unique basin / no hysteresis at rho>1)", flush=True)
    print("\nVerdict: if xi << roman diameter (~large; high-diameter syntactic graph), a "
          "non-contractive equilibrium is genuinely EDIT-LOCAL => maintainable beyond contraction.",
          flush=True)


if __name__ == "__main__":
    main()
