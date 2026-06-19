"""Cardinality-shift experiment: train at N=24, probe at growing N.

Tests whether the equilibrium properties survive "train small, infer large".
Predictions: well-posedness (convergence) survives; path independence may degrade
with N; mean unlearning gap shrinks (each point more redundant) while the TAIL
(max gap) is where leakage lives; warm-start efficiency advantage grows.

Run from repo root:  python -m experiments.cardinality_shift
Writes experiments/results_cardinality.json for later plotting.
"""

import json
import torch

from src.data import GMMSetDataset
from src.model import SetDEQ
from src.train import train, evaluate
from src.metrics import path_independence_gap, unlearning_gap

K_RANGE = (1, 4)
N_TRAIN = 24
N_SWEEP = [24, 48, 96, 192]
PROBE_SOLVE = dict(max_iter=200, tol=1e-5)


def build_ds(n_points, n_samples, seed):
    return GMMSetDataset(n_samples=n_samples, k_range=K_RANGE, n_points=n_points,
                         d=2, sep=4.0, std=1.0, seed=seed)


def _mean(v):
    return sum(v) / len(v)


def probe_at_N(model, n_points, n_probe=30, seed=100):
    ds = build_ds(n_points, n_probe, seed)
    fp_mean, fp_max, agree, pi_conv = [], [], [], []
    ul_vals, match, warm_it, cold_it, warm_cv, cold_cv = [], [], [], [], [], []
    for i in range(len(ds)):
        x = ds.X[i]
        pi = path_independence_gap(model, x, n_inits=5, **PROBE_SOLVE)
        ridx = int(torch.randint(0, n_points, (1,)))
        ul = unlearning_gap(model, x, remove_idx=ridx, **PROBE_SOLVE)
        fp_mean.append(pi["fp_gap"]); agree.append(pi["pred_agreement"])
        pi_conv.append(pi["converged_frac"])
        ul_vals.append(ul["unlearn_gap"]); match.append(1.0 if ul["pred_match"] else 0.0)
        warm_it.append(ul["warm_iters"]); cold_it.append(ul["cold_iters"])
        warm_cv.append(1.0 if ul["warm_converged"] else 0.0)
        cold_cv.append(1.0 if ul["cold_converged"] else 0.0)
    acc = evaluate(model, build_ds(n_points, 200, seed + 1))
    return {
        "N": n_points, "acc": acc,
        "fp_gap_mean": _mean(fp_mean), "fp_gap_max": max(fp_mean),
        "pred_agree": _mean(agree), "pi_conv_frac": _mean(pi_conv),
        "unlearn_mean": _mean(ul_vals), "unlearn_max": max(ul_vals),
        "pred_match": _mean(match),
        "warm_iters": _mean(warm_it), "cold_iters": _mean(cold_it),
        "warm_conv": _mean(warm_cv), "cold_conv": _mean(cold_cv),
    }


def run(update, epochs=8):
    torch.manual_seed(0)
    train_ds = build_ds(N_TRAIN, 2000, seed=1)
    model = SetDEQ(d_in=2, d_latent=32, hidden=64, update=update,
                   n_classes=train_ds.n_classes, max_iter=30, tol=1e-4, damping=0.5)
    print(f"== training {update} at N={N_TRAIN} ==")
    train(model, train_ds, epochs=epochs, batch_size=64, lr=1e-3, log_every=0)
    rows = []
    for N in N_SWEEP:
        r = probe_at_N(model, N)
        rows.append(r)
        print(f"N={N:4d} acc={r['acc']:.3f} | fp_gap mean/max={r['fp_gap_mean']:.3f}/"
              f"{r['fp_gap_max']:.3f} agree={r['pred_agree']:.3f} conv={r['pi_conv_frac']:.2f} | "
              f"unlearn mean/max={r['unlearn_mean']:.3f}/{r['unlearn_max']:.3f} "
              f"match={r['pred_match']:.2f} | warm/cold={r['warm_iters']:.0f}/{r['cold_iters']:.0f}")
    return rows


def main():
    out = {}
    for upd in ["normdeepsets", "attn"]:
        print("\n" + "#" * 64)
        print(f"# {upd}")
        out[upd] = run(upd)
    with open("experiments/results_cardinality.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved experiments/results_cardinality.json")


if __name__ == "__main__":
    main()
