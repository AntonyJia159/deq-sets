"""Stage 0: well-posedness gate for local state-dependent set-equilibrium.

Does a local, state-dependent set-equilibrium converge to a near-unique fixed
point, and is there a usable radius window where it does?

Sweep: {fixed-graph vs state-dependent} x {local-mean vs local-attn}
       x radius x set-size x seed.  No training -- random weights.

Metrics per config:
  conv_rate  — fraction of sets reaching tol within max_iter
  mean_iter  — mean iters to converge (among converged)
  fp_gap     — max relative gap across random inits (uniqueness)
  oscillates — fraction where residual goes UP in the last 20% of iters
               (the specific failure mode for state-dependent graphs)

Run:  python -m experiments.local_wellposedness

Go/no-go thresholds (pre-registered):
  GREEN  — state-dependent converges >90%, fp_gap < 0.05, across N
  YELLOW — only fixed-graph converges, or only with tiny r
  RED    — state-dependent broadly oscillates/diverges
"""

import itertools
import sys

import numpy as np
import torch

from src.model import SetDEQ

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MAX_ITER = 200
TOL = 1e-5
N_INITS = 4
SEEDS = [0, 1]

RADII = [0.5, 1.0, 2.0, 4.0, 8.0, 100.0]
SET_SIZES = [16, 32, 64, 128]
AGGREGATORS = ["local_mean", "local_attn"]
GRAPH_SOURCES = ["input", "latent"]

D_IN = 2
D_LATENT = 32
HIDDEN = 64
N_CLASSES = 4
N_PROBE = 30


def make_data(N, n_sets, geometry, seed):
    """Generate synthetic point sets. Three geometries:
    - 'uniform': points iid in [-5, 5]^2
    - 'clustered': GMM with 3 tight clusters (exercises varying density)
    - 'chain': points on a 1D chain with small noise (multi-hop needed)
    """
    g = torch.Generator().manual_seed(seed)
    if geometry == "uniform":
        return torch.rand(n_sets, N, D_IN, generator=g) * 10.0 - 5.0
    elif geometry == "clustered":
        centers = torch.tensor([[-4.0, 0.0], [0.0, 4.0], [4.0, 0.0]])
        X = torch.randn(n_sets, N, D_IN, generator=g) * 0.5
        assign = torch.randint(0, 3, (n_sets, N), generator=g)
        for c in range(3):
            mask = assign == c
            X[mask] += centers[c]
        return X
    elif geometry == "chain":
        t = torch.linspace(0, 10, N).unsqueeze(0).expand(n_sets, -1)
        noise = torch.randn(n_sets, N, generator=g) * 0.1
        x_coord = t + noise
        y_coord = torch.randn(n_sets, N, generator=g) * 0.1
        return torch.stack([x_coord, y_coord], dim=-1)
    else:
        raise ValueError(geometry)


@torch.no_grad()
def probe_one(model, X, n_inits):
    """Run n_inits solves on each set in X (M, N, d).
    Returns per-set: converged, n_iter, fp_gap, oscillates."""
    M, N = X.shape[0], X.shape[1]

    all_Z = []
    all_conv = []
    all_niter = []
    all_osc = []

    for _ in range(n_inits):
        z0 = torch.randn(M, N, model.d_latent, device=X.device)
        z = z0.clone()
        residuals = []
        converged = torch.zeros(M, dtype=torch.bool, device=X.device)
        n_iter = torch.full((M,), MAX_ITER, device=X.device)

        for it in range(1, MAX_ITER + 1):
            fz = model.update(z, X)
            res = (fz - z).flatten(1).norm(dim=1) / (z.flatten(1).norm(dim=1) + 1e-8)
            residuals.append(res)
            z = fz

            newly = (~converged) & (res < TOL)
            n_iter[newly] = it
            converged = converged | newly

            if converged.all():
                break

        residuals = torch.stack(residuals, dim=0)
        T = residuals.shape[0]
        t80 = max(1, int(T * 0.8))
        tail = residuals[t80:]
        if tail.shape[0] > 1:
            diffs = tail[1:] - tail[:-1]
            osc = (diffs > 0).float().mean(dim=0) > 0.5
        else:
            osc = torch.zeros(M, dtype=torch.bool, device=X.device)

        all_Z.append(z.detach())
        all_conv.append(converged)
        all_niter.append(n_iter)
        all_osc.append(osc)

    all_Z = torch.stack(all_Z)
    fp_gap = torch.zeros(M, device=X.device)
    for i in range(n_inits):
        for j in range(i + 1, n_inits):
            d = ((all_Z[i] - all_Z[j]).flatten(1).norm(dim=1) /
                 (all_Z[j].flatten(1).norm(dim=1) + 1e-8))
            fp_gap = torch.maximum(fp_gap, d)

    conv_rate = torch.stack(all_conv).float().mean(dim=0)
    mean_niter = torch.stack(all_niter).float().mean(dim=0)
    osc_rate = torch.stack(all_osc).float().mean(dim=0)

    return {
        "conv_rate": conv_rate.mean().item(),
        "mean_iter": mean_niter[conv_rate > 0.5].mean().item()
                     if (conv_rate > 0.5).any() else float('nan'),
        "fp_gap": fp_gap.mean().item(),
        "fp_gap_max": fp_gap.max().item(),
        "osc_rate": osc_rate.mean().item(),
    }


def run_sweep():
    header = (f"{'agg':<12} {'graph':<8} {'r':>5} {'N':>5} "
              f"{'conv%':>6} {'iters':>6} {'fpgap':>7} {'fpg_mx':>7} "
              f"{'osc%':>6} {'geom':<10}")
    print(header)
    print("-" * len(header))

    results = []
    geometries = ["uniform", "clustered", "chain"]

    total = len(AGGREGATORS) * len(GRAPH_SOURCES) * len(RADII) * len(SET_SIZES)
    done = 0
    for agg, gsrc in itertools.product(AGGREGATORS, GRAPH_SOURCES):
        for radius in RADII:
            for N in SET_SIZES:
                done += 1
                sys.stdout.flush()
                seed_results = {k: [] for k in
                                ["conv_rate", "mean_iter", "fp_gap",
                                 "fp_gap_max", "osc_rate"]}
                for geom in geometries:
                    geom_results = {k: [] for k in seed_results}
                    for seed in SEEDS:
                        torch.manual_seed(seed)
                        model = SetDEQ(
                            d_in=D_IN, d_latent=D_LATENT, hidden=HIDDEN,
                            update=agg, n_classes=N_CLASSES,
                            max_iter=MAX_ITER, tol=TOL,
                            solver="damped", damping=1.0,
                            radius=radius, graph_source=gsrc,
                        ).to(DEV)
                        model.eval()

                        X = make_data(N, N_PROBE, geom, seed + 1000).to(DEV)
                        stats = probe_one(model, X, N_INITS)
                        for k in stats:
                            geom_results[k].append(stats[k])

                    for k in geom_results:
                        v = np.nanmean(geom_results[k])
                        seed_results[k].append(v)

                    row = {k: np.nanmean(geom_results[k]) for k in geom_results}
                    print(f"{agg:<12} {gsrc:<8} {radius:5.1f} {N:5d} "
                          f"{row['conv_rate']:6.1%} {row['mean_iter']:6.0f} "
                          f"{row['fp_gap']:7.3f} {row['fp_gap_max']:7.3f} "
                          f"{row['osc_rate']:6.1%} {geom:<10}")

                avg = {k: np.nanmean(seed_results[k]) for k in seed_results}
                results.append({
                    "agg": agg, "graph_source": gsrc, "radius": radius,
                    "N": N, **avg,
                })

    print("\n" + "=" * 80)
    print("AGGREGATE (averaged over geometries)")
    print(header.replace("geom", "").rstrip())
    print("-" * 70)
    for r in results:
        print(f"{r['agg']:<12} {r['graph_source']:<8} {r['radius']:5.1f} "
              f"{r['N']:5d} {r['conv_rate']:6.1%} {r['mean_iter']:6.0f} "
              f"{r['fp_gap']:7.3f} {r['fp_gap_max']:7.3f} "
              f"{r['osc_rate']:6.1%}")

    print("\n" + "=" * 80)
    print("GO/NO-GO ASSESSMENT")
    sd_results = [r for r in results if r["graph_source"] == "latent"]
    fx_results = [r for r in results if r["graph_source"] == "input"]

    if sd_results:
        sd_conv = np.mean([r["conv_rate"] for r in sd_results])
        sd_fpgap = np.mean([r["fp_gap"] for r in sd_results])
        sd_osc = np.mean([r["osc_rate"] for r in sd_results])
        print(f"\nState-dependent (latent graph):")
        print(f"  avg conv_rate = {sd_conv:.1%}")
        print(f"  avg fp_gap    = {sd_fpgap:.4f}")
        print(f"  avg osc_rate  = {sd_osc:.1%}")

        best_sd = [r for r in sd_results
                   if r["conv_rate"] > 0.9 and r["fp_gap"] < 0.05]
        if best_sd:
            radii_ok = sorted(set(r["radius"] for r in best_sd))
            sizes_ok = sorted(set(r["N"] for r in best_sd))
            print(f"  GREEN window: radii {radii_ok}, sizes {sizes_ok}")
        else:
            print("  No (radius, N) combo meets GREEN thresholds")

    if fx_results:
        fx_conv = np.mean([r["conv_rate"] for r in fx_results])
        fx_fpgap = np.mean([r["fp_gap"] for r in fx_results])
        print(f"\nFixed-graph (input graph) control:")
        print(f"  avg conv_rate = {fx_conv:.1%}")
        print(f"  avg fp_gap    = {fx_fpgap:.4f}")

    if sd_results:
        if best_sd:
            print("\n>>> VERDICT: GREEN — state-dependent local window exists")
        elif sd_conv > 0.5:
            print("\n>>> VERDICT: YELLOW — converges partially, "
                  "may need damping or smaller radius")
        else:
            print("\n>>> VERDICT: RED — state-dependent local broadly fails")


if __name__ == "__main__":
    run_sweep()
