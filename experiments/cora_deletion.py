"""Channel-1 deletion on Cora: does exact node deletion stay LOCAL on a real graph?

Ports the grid deletion result (Report #5) onto a real, degree-heterogeneous graph.
A node is deleted (its edges cut, graph renormalized) and the equilibrium re-solved with
FROZEN weights -- this is channel-1 unlearning (remove the input/context element exactly).

Measures, per deleted node:
  - influence(i) = ||z_new(i) - z_old(i)||  vs  GRAPH DISTANCE (BFS hops) from the deletion;
  - truncated warm-start: re-solve only nodes within R hops (freeze the far field at z_old),
    and the error vs the exact full re-solve -> is truncation exact, and how big must R be;
  - nodes touched at the truncation radius = the locality COST.

Two axes the grid could not show:
  - DEGREE STRATA (leaf / median / hub): we expect deleting a hub to be far less local.
  - kappa sweep (contraction cap): since training SATURATES the clip (rho -> kappa), kappa is
    our screening knob. Smaller kappa = shorter screening = cheaper exact deletion, but costs
    accuracy. This is the expressiveness<->locality tradeoff, made concrete.

Run:  D:\\deq-venv\\Scripts\\python.exe -m experiments.cora_deletion
"""

import os

import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.cora_smoke import download, load_cora
from experiments.cora_deq_solvers import GraphDEQ

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figs")
SOLVE_TOL, SOLVE_MAX = 1e-7, 400
KAPPAS = [0.5, 0.9]
TRUNC_EPS = 1e-4          # truncated-resolve counts as "exact" below this max error


def renorm_sparse(adj):
    """Symmetric-normalized adjacency with self-loops -> torch sparse (spectral radius <=1)."""
    A = (adj + sp.eye(adj.shape[0])).tocoo()
    deg = np.asarray(A.sum(1)).ravel()
    dinv = 1.0 / np.sqrt(deg)
    vals = A.data * dinv[A.row] * dinv[A.col]
    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(vals, dtype=torch.float32),
                                   A.shape).coalesce()


def solve(Ahat_sp, Wc, h0, s, z0=None, restrict=None, tol=SOLVE_TOL, maxit=SOLVE_MAX):
    """Picard solve of z = tanh(s*Ahat@(z W) + h0). restrict (bool mask): only these nodes
    update; the rest stay frozen at z0 (the truncated re-solve)."""
    z = torch.zeros_like(h0) if z0 is None else z0.clone()
    for it in range(1, maxit + 1):
        zn = torch.tanh(s * torch.sparse.mm(Ahat_sp, z @ Wc) + h0)
        if restrict is not None:
            zn[~restrict] = z[~restrict]
        r = (zn - z).norm().item() / (z.norm().item() + 1e-9)
        z = zn
        if r < tol:
            break
    return z, it


def bfs_hops(G, src, N):
    d = nx.single_source_shortest_path_length(G, src)
    out = np.full(N, np.inf)
    for k, v in d.items():
        out[k] = v
    return out


def pick_targets(deg):
    """One representative node per degree stratum: leaf / median / hub."""
    order = np.argsort(deg)
    leaf = int(order[deg[order].searchsorted(1)]) if (deg == 1).any() else int(order[0])
    median = int(order[len(order) // 2])
    hub = int(np.argmax(deg))
    return {"leaf": leaf, "median": median, "hub": hub}


def analyze(v, adj, Ahat_full, Wc, h0, s, z_old, G):
    N = adj.shape[0]
    adj_del = adj.tolil(); adj_del[v, :] = 0; adj_del[:, v] = 0
    Ahat_del = renorm_sparse(adj_del.tocsr())
    z_new, _ = solve(Ahat_del, Wc, h0, s, z0=z_old)               # exact full deletion

    infl = (z_new - z_old).norm(dim=1)
    infl[v] = 0.0
    hops = bfs_hops(G, v, N)
    inf_np = infl.numpy()

    maxhop = int(np.nanmax(hops[np.isfinite(hops)]))
    prof = [inf_np[hops == d].mean() if (hops == d).any() else 0.0
            for d in range(1, min(maxhop, 8) + 1)]

    # truncated re-solve: grow R until error vs exact full deletion is below TRUNC_EPS
    R_star, touched_star, err_star = None, N, None
    for R in range(1, min(maxhop, 12) + 1):
        restrict = torch.tensor(hops <= R)
        z_tr, _ = solve(Ahat_del, Wc, h0, s, z0=z_old, restrict=restrict)
        err = (z_tr - z_new)[torch.isfinite(torch.tensor(hops))].abs().max().item()
        if R_star is None and err < TRUNC_EPS:
            R_star, touched_star, err_star = R, int(restrict.sum()), err
            break
    if R_star is None:
        restrict = torch.tensor(np.isfinite(hops))
        R_star, touched_star, err_star = maxhop, int(restrict.sum()), err
    return {"deg": int(adj[v].sum()), "prof": prof, "R": R_star,
            "touched": touched_star, "N": N, "trunc_err": err_star}


def main():
    download()
    adj, X, labels_oh, idx_train, idx_test = load_cora()
    adj = adj.tocsr()
    y = torch.tensor(labels_oh.argmax(1), dtype=torch.long)
    X_t = torch.tensor(X)
    itr, ite = torch.tensor(idx_train), torch.tensor(idx_test)
    deg = np.asarray(adj.sum(1)).ravel()
    G = nx.from_scipy_sparse_array(adj)
    targets = pick_targets(deg)
    Ahat_dense = torch.tensor(np.asarray((renorm_sparse(adj).to_dense())))
    print(f"Cora: {adj.shape[0]} nodes; targets "
          f"{ {k: f'{v}(deg{int(deg[v])})' for k, v in targets.items()} }\n")

    fig_done = False
    for kappa in KAPPAS:
        torch.manual_seed(0); np.random.seed(0)
        model = GraphDEQ(X.shape[1], 64, labels_oh.shape[1], "fixed_point_iter",
                         norm="clipped", kappa=kappa)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
        for ep in range(120):
            opt.zero_grad()
            out, _, _ = model(Ahat_dense, X_t)
            F.cross_entropy(out[itr], y[itr]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            Wc = model._Wc().detach(); s = model.s
            h0 = model.enc(X_t).detach()
            Ahat_sp = renorm_sparse(adj)
            z_old, n_it = solve(Ahat_sp, Wc, h0, s)
            acc = (model.readout(z_old).argmax(1)[ite] == y[ite]).float().mean().item()
            eff = (s * torch.linalg.matrix_norm(Wc, ord=2)).item()

        print(f"=== kappa={kappa}  (eff contraction {eff:.3f}, test acc {acc:.3f}, "
              f"z_old solved in {n_it} iters) ===")
        print(f"{'stratum':<8}{'deg':>5}{'R*(eps)':>9}{'touched':>9}"
              f"{'N/touched':>11}{'trunc_err':>11}  decay-by-hop")
        for name, v in targets.items():
            with torch.no_grad():
                r = analyze(v, adj, Ahat_sp, Wc, h0, s, z_old, G)
            prof = " ".join(f"{p:.1e}" for p in r["prof"])
            print(f"{name:<8}{r['deg']:>5}{r['R']:>9}{r['touched']:>9}"
                  f"{r['N'] / r['touched']:>11.1f}{r['trunc_err']:>11.1e}  {prof}")

            if not fig_done:
                hops = bfs_hops(G, v, adj.shape[0])
                plt.figure("decay")
                xs = np.arange(1, len(r["prof"]) + 1)
                plt.semilogy(xs, np.clip(r["prof"], 1e-12, None), "o-", label=f"{name} (deg {r['deg']})")
        if not fig_done:
            plt.figure("decay")
            plt.xlabel("graph distance from deletion (BFS hops)")
            plt.ylabel("mean influence  ||z_new - z_old||")
            plt.title(f"Cora deletion-influence decay (kappa={kappa})")
            plt.legend(); plt.grid(True, alpha=0.3)
            os.makedirs(FIG_DIR, exist_ok=True)
            p = os.path.join(FIG_DIR, "cora_deletion_decay.png")
            plt.savefig(p, dpi=130, bbox_inches="tight")
            print(f"\n  figure -> {p}\n")
            fig_done = True


if __name__ == "__main__":
    main()
