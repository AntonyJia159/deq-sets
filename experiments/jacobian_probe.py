"""Layer-3 Jacobian probe.

The load-bearing experiment under Reports #1/#2: it measures, rather than infers,
the spectral radius rho(J_f) at the trained fixed points and ties it to the
leakage we care about. Three questions:

  1. Are the contractive models actually contractive (rho < 1), and is attention
     actually near/over the bifurcation (rho -> 1 with a tail)? This is the direct
     evidence for the uniqueness claims we have only seen downstream via fp_gap.

  2. Does the linearized removal sensitivity ||dZ*/dw_k|| (mean-pool) track the
     measured unlearn_gap? Agreement = redundant point; large divergence = the
     pivotal/bifurcation case that is the MIA target.

  3. Does the IFT amplifier 1/(1-rho) rank-correlate with the measured unlearning
     tail across sets? That is the "spectral quantity governs the leak" statement
     the privacy story rests on.

Run:  & "D:\\deq-venv\\Scripts\\python.exe" -m experiments.jacobian_probe
"""

import json
import math
import os

import torch

from src.data import GMMSetDataset
from src.model import SetDEQ
from src.train import train
from src.metrics import unlearning_gap
from src.jacobian import jacobian_report

K_RANGE = (1, 4)
N = 24
N_PROBE = 30
SOLVE = dict(max_iter=200)


def _corr(a, b):
    a = torch.tensor(a, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64)
    if a.numel() < 2 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std()))


def run(update, epochs=8):
    torch.manual_seed(0)
    train_ds = GMMSetDataset(n_samples=2000, k_range=K_RANGE, n_points=N, d=2,
                             sep=4.0, std=1.0, seed=1)
    test_ds = GMMSetDataset(n_samples=500, k_range=K_RANGE, n_points=N, d=2,
                            sep=4.0, std=1.0, seed=2)
    model = SetDEQ(d_in=2, d_latent=32, hidden=64, update=update,
                   n_classes=train_ds.n_classes, max_iter=150, tol=1e-5)
    print(f"== training {update} at N={N} ==")
    train(model, train_ds, epochs=epochs, batch_size=64, lr=1e-3, log_every=0)

    rhos, amps, pred_rel, meas_gap, contractive = [], [], [], [], 0
    both_conv_gap = []  # gap only where BOTH solves converged: true multistability
    for i in range(N_PROBE):
        x = test_ds.X[i]
        xb = x.unsqueeze(0)
        z_star, info = model.solve(xb, **SOLVE)
        rep = jacobian_report(model, xb, z_star, remove_idx=0)
        ul = unlearning_gap(model, x, remove_idx=0, **SOLVE)
        rhos.append(rep["rho"])
        amps.append(rep["amplifier"] if math.isfinite(rep["amplifier"]) else float("nan"))
        contractive += int(rep["contractive"])
        meas_gap.append(ul["unlearn_gap"])
        pred_rel.append(rep["removal"]["pred_rel"] if rep["removal"] else float("nan"))
        if ul["warm_converged"] and ul["cold_converged"]:
            both_conv_gap.append(ul["unlearn_gap"])

    rt = torch.tensor(rhos)
    summary = {
        "update": update,
        "rho_mean": float(rt.mean()), "rho_median": float(rt.median()),
        "rho_max": float(rt.max()), "rho_min": float(rt.min()),
        "contractive_frac": contractive / N_PROBE,
        "unlearn_gap_mean": float(torch.tensor(meas_gap).mean()),
        "unlearn_gap_max": float(torch.tensor(meas_gap).max()),
        # disambiguates true multistability (two converged basins) from mere
        # under-convergence: gap restricted to sets where BOTH solves converged.
        "both_converged_n": len(both_conv_gap),
        "both_conv_gap_max": (float(torch.tensor(both_conv_gap).max())
                              if both_conv_gap else float("nan")),
        # the privacy-relevant claim: does the spectral amplifier predict the leak?
        "corr_amplifier_vs_gap": _corr(amps, meas_gap),
        "corr_predrel_vs_gap": _corr(pred_rel, meas_gap),
        "raw": {"rho": rhos, "amplifier": amps,
                "pred_rel": pred_rel, "unlearn_gap": meas_gap},
    }
    print(f"  rho  mean/median/max = {summary['rho_mean']:.3f} / "
          f"{summary['rho_median']:.3f} / {summary['rho_max']:.3f}  "
          f"(contractive {summary['contractive_frac']*100:.0f}%)")
    print(f"  measured unlearn_gap mean/max = {summary['unlearn_gap_mean']:.3f} / "
          f"{summary['unlearn_gap_max']:.3f}")
    print(f"  both-converged gap max = {summary['both_conv_gap_max']:.3f} "
          f"(n={summary['both_converged_n']}/{N_PROBE})")
    print(f"  corr(amplifier, gap)  = {summary['corr_amplifier_vs_gap']:.3f}")
    print(f"  corr(pred_rel, gap)   = {summary['corr_predrel_vs_gap']:.3f}")
    return summary


def main():
    results = {}
    for update in ("normdeepsets", "attn"):
        results[update] = run(update)
        print()
    out = os.path.join(os.path.dirname(__file__), "results_jacobian.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
