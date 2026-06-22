"""Pixel-deletion on a grid: visualizing and measuring the deletion-influence field.

A grid graph is the cleanest possible testbed for the unlearning story, because a
mean-aggregation equilibrium on a 4-connected grid is a *screened harmonic* problem
whose Green's function is known: a point perturbation decays geometrically with a
screening length set entirely by the contraction factor.

We use the fixed-point map (no learned weights -- the structure is the point):

    z_i  <-  (1 - alpha) * x_i  +  alpha * mean_{j in N(i)} z_j

`alpha` in (0, 1) is the contraction factor (== spectral radius of the iteration,
since the row-normalized adjacency has spectral radius 1). alpha -> 1 is the
unscreened/harmonic limit; alpha < 1 screens.

DELETION = remove a node from the graph (drop its row/col, renormalize its old
neighbors) and re-solve. The quantity of interest is the *deletion-influence field*

    influence(i) = | z_new(i) - z_old(i) |

i.e. how much every other node's representation moved because one node was deleted.

What this script measures
-------------------------
1. DECAY / SCREENING. influence(i) vs graph distance d from the deleted node.
   Geometric decay (constant ratio < 1) => exponential screening; we fit the
   screening length and the truncation radius R(eps).
2. SCREENING IS SET BY CONTRACTION. Sweep alpha; smaller alpha => shorter ξ.
3. R(eps) IS N-INDEPENDENT (in the contractive regime). Grow the grid; if the
   truncation radius stays flat, exact deletion costs O(1) in N, not O(N).
4. TRUNCATION IS EXACT TO THE TAIL. Re-solve only within radius R of the deletion
   (far field frozen at z_old) and confirm the error vs the full re-solve is the
   truncation tail, not a different fixed point -- this is where path-independence
   (unique fixed point) does its work: warm/cold/partial all land on the same z.
5. HEATMAP. influence(i) reshaped to H x W -- the visual of the screening length.

Run:  python experiments/grid_deletion.py
Figures land in reports/figs/.
"""

import os

import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figs")
TOL = 1e-11
MAXIT = 40000


# ---------------------------------------------------------------- grid + solver

def build_grid(H, W):
    """4-connected grid adjacency as a sparse 0/1 matrix."""
    rows, cols = [], []
    for r in range(H):
        for c in range(W):
            i = r * W + c
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    rows.append(i)
                    cols.append(rr * W + cc)
    n = H * W
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def _iteration_operator(A, alpha, drop):
    """Return (M, keep) for z <- (1-alpha)x + M z, with `drop` removed from the graph."""
    n = A.shape[0]
    keep = np.ones(n, dtype=bool)
    Ak = A
    if drop is not None:
        keep[drop] = False
        d = np.ones(n)
        d[drop] = 0.0
        Ak = A.multiply(d).multiply(d[:, None]).tocsr()  # cut all edges to/from drop
    deg = np.asarray(Ak.sum(1)).ravel()
    deg[deg == 0] = 1.0
    M = alpha * (sp.diags(1.0 / deg) @ Ak)
    return M, keep


def solve(A, alpha, x, drop=None, z0=None, restrict=None, tol=TOL, maxit=MAXIT):
    """Iterate the fixed-point map to convergence.

    drop     -- node index removed from the graph (the deletion), or None.
    z0       -- warm-start initialization (else zeros).
    restrict -- boolean mask of nodes allowed to update; others frozen at z0.
                Used for the *truncated* re-solve. Requires z0.
    Returns (z, n_iter).
    """
    M, keep = _iteration_operator(A, alpha, drop)
    b = (1.0 - alpha) * x
    z = np.zeros(A.shape[0]) if z0 is None else z0.copy()
    if restrict is not None:
        assert z0 is not None, "restrict needs a z0 (frozen far field)"
    for it in range(maxit):
        zn = b + M @ z
        zn[~keep] = 0.0
        if restrict is not None:
            zn[~restrict] = z[~restrict]  # freeze the far field
        r = np.max(np.abs(zn - z))
        z = zn
        if r < tol:
            break
    return z, it + 1


# ------------------------------------------------------------- distance helpers

def chebyshev_from(H, W, node):
    """Graph-distance proxy: Chebyshev (Linf) distance of every pixel to `node`."""
    cr, cc = node // W, node % W
    rr = np.abs(np.arange(H)[:, None] - cr)
    cc_ = np.abs(np.arange(W)[None, :] - cc)
    return np.maximum(rr, cc_).ravel()


def decay_profile(influence, dist, rmax):
    """Mean influence at each integer distance 1..rmax-1."""
    return np.array([influence[dist == d].mean() if (dist == d).any() else 0.0
                     for d in range(1, rmax)])


def truncation_radius(prof, eps=1e-4):
    """First distance where influence falls below eps * near-field peak."""
    peak = prof[0] if len(prof) else 0.0
    for d, v in enumerate(prof):
        if v < eps * peak:
            return d + 1
    return len(prof)


def screening_length(prof):
    """Fit influence ~ exp(-d / xi) over the reliable (non-tiny) part; return xi."""
    mask = prof > prof[0] * 1e-6
    d = np.arange(1, len(prof) + 1)[mask]
    if len(d) < 3:
        return float("nan")
    slope = np.polyfit(d, np.log(prof[mask]), 1)[0]
    return -1.0 / slope if slope < 0 else float("inf")


# ------------------------------------------------------------------- experiments

def measure(H, W, alpha, seed=0):
    """One (grid, alpha) measurement: returns a dict of metrics + the influence image."""
    A = build_grid(H, W)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(H * W)

    z_old, it_cold = solve(A, alpha, x)
    center = (H // 2) * W + (W // 2)
    z_new, it_del = solve(A, alpha, x, drop=center)               # exact (full) deletion
    z_warm, it_warm = solve(A, alpha, x, drop=center, z0=z_old)   # warm full deletion

    influence = np.abs(z_new - z_old)
    dist = chebyshev_from(H, W, center)
    rmax = min(H, W) // 2
    prof = decay_profile(influence, dist, rmax)
    R = truncation_radius(prof)
    xi = screening_length(prof)

    # truncated re-solve: only nodes within R of the deletion may move
    restrict = (dist <= R)
    z_trunc, it_trunc = solve(A, alpha, x, drop=center, z0=z_old, restrict=restrict)
    trunc_err = np.max(np.abs(z_trunc - z_new))                   # vs exact full deletion

    return {
        "H": H, "W": W, "N": H * W, "alpha": alpha,
        "it_cold": it_cold, "it_del": it_del, "it_warm": it_warm, "it_trunc": it_trunc,
        "R": R, "xi": xi, "trunc_err": trunc_err,
        "n_touched": int(restrict.sum()),
        "ratios": [prof[k + 1] / prof[k] for k in range(min(5, len(prof) - 1))],
        "influence_img": influence.reshape(H, W),
        "center_rc": (H // 2, W // 2),
    }


def heatmap_panel(results, fname):
    """Side-by-side log-influence heatmaps for a list of measurements."""
    os.makedirs(FIG_DIR, exist_ok=True)
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
    if n == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        img = np.log10(res["influence_img"] + 1e-12)
        im = ax.imshow(img, cmap="magma", vmin=-8, vmax=img.max())
        r, c = res["center_rc"]
        ax.plot(c, r, "c+", markersize=10, markeredgewidth=2)
        ax.set_title(f"alpha={res['alpha']:.2f}  xi~{res['xi']:.1f}  R={res['R']}")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, label="log10 |dz|")
    fig.suptitle("Deletion-influence field: delete center pixel, |z_new - z_old|")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    print("=" * 78)
    print("A) Fixed 61x61 grid -- contraction sets the screening length")
    print("=" * 78)
    print(f"{'alpha':>6} {'xi':>6} {'R(1e-4)':>8} {'touched/N':>12} "
          f"{'trunc_err':>10} {'it_cold':>8} {'it_warm':>8} {'ratios(d1..5)'}")
    panel = []
    for alpha in (0.80, 0.90, 0.95, 0.99):
        r = measure(61, 61, alpha)
        panel.append(r)
        ratios = " ".join(f"{x:.2f}" for x in r["ratios"])
        print(f"{alpha:6.2f} {r['xi']:6.1f} {r['R']:8d} "
              f"{r['n_touched']:6d}/{r['N']:<5d} {r['trunc_err']:10.2e} "
              f"{r['it_cold']:8d} {r['it_warm']:8d}  {ratios}")
    p = heatmap_panel(panel, "grid_deletion_screening.png")
    print(f"\nheatmap -> {p}")

    print("\n" + "=" * 78)
    print("B) alpha=0.90 (contractive): is R(eps) N-independent?  -> O(1) deletion")
    print("=" * 78)
    print(f"{'N':>6} {'R':>4} {'touched':>8} {'N/touched':>10} {'trunc_err':>10}")
    for s in (21, 31, 41, 61, 81, 101):
        r = measure(s, s, 0.90)
        print(f"{r['N']:6d} {r['R']:4d} {r['n_touched']:8d} "
              f"{r['N'] / r['n_touched']:10.1f} {r['trunc_err']:10.2e}")

    print("\n" + "=" * 78)
    print("C) alpha=0.99 (near-harmonic): R(eps) grows with N -> no scaling win")
    print("=" * 78)
    print(f"{'N':>6} {'R':>4} {'touched':>8} {'N/touched':>10} {'trunc_err':>10}")
    for s in (21, 31, 41, 61, 81, 101):
        r = measure(s, s, 0.99)
        print(f"{r['N']:6d} {r['R']:4d} {r['n_touched']:8d} "
              f"{r['N'] / r['n_touched']:10.1f} {r['trunc_err']:10.2e}")


if __name__ == "__main__":
    main()
