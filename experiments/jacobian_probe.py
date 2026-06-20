"""Layer-3 Jacobian probe + the decomposability pivot (linear attention).

Side-by-side over three update blocks that change ONE thing at a time:

  normdeepsets : mean-pool   -- decomposable, contractive, single basin (baseline)
  linattn      : linear attn -- DECOMPOSABLE but more expressive (the pivot)
  attn         : softmax attn-- NON-decomposable, expressive (multistable negative)

The decisive question: is exact unlearning governed by DECOMPOSABILITY or by raw
EXPRESSIVENESS? linattn isolates it -- if it stays globally unique (fp_gap ~ 0)
like mean-pool, decomposability is the lever; if it goes multistable like softmax
attention, expressiveness is.

Per architecture we report, at N=24:
  - test accuracy                              (the expressiveness axis)
  - rho(J_f) mean/max, contractive fraction    (LOCAL well-posedness)
  - fp_gap mean/max                            (GLOBAL uniqueness / multistability)
  - both-converged unlearn gap max             (true leakage, not under-convergence)
  - corr(pred_rel, gap)                        (does linearized dZ*/dw predict the leak)

Run:  & "D:\\deq-venv\\Scripts\\python.exe" -m experiments.jacobian_probe
"""

import json
import math
import os

import torch

from src.data import GMMSetDataset
from src.model import SetDEQ
from src.train import train, evaluate
from src.metrics import path_independence_gap, unlearning_gap
from src.jacobian import jacobian_report

K_RANGE = (1, 4)
N = 24
N_PROBE = 30
SOLVE = dict(max_iter=200)


def _corr(a, b):
    a = torch.tensor(a, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64)
    mask = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[mask], b[mask]
    if a.numel() < 2 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std()))


def run(update, epochs=15):
    torch.manual_seed(0)
    train_ds = GMMSetDataset(n_samples=2000, k_range=K_RANGE, n_points=N, d=2,
                             sep=4.0, std=1.0, seed=1)
    test_ds = GMMSetDataset(n_samples=500, k_range=K_RANGE, n_points=N, d=2,
                            sep=4.0, std=1.0, seed=2)
    model = SetDEQ(d_in=2, d_latent=32, hidden=64, update=update,
                   n_classes=train_ds.n_classes, max_iter=150, tol=1e-5)
    print(f"== training {update} at N={N} ==")
    train(model, train_ds, epochs=epochs, batch_size=64, lr=1e-3, log_every=0)
    acc = evaluate(model, test_ds)

    rhos, fp_gaps, pred_rel, meas_gap = [], [], [], []
    contractive = 0
    both_conv_gap = []
    for i in range(N_PROBE):
        x = test_ds.X[i]
        xb = x.unsqueeze(0)
        z_star, _ = model.solve(xb, **SOLVE)
        rep = jacobian_report(model, xb, z_star, remove_idx=0)
        pi = path_independence_gap(model, x, n_inits=5, **SOLVE)
        ul = unlearning_gap(model, x, remove_idx=0, **SOLVE)

        rhos.append(rep["rho"])
        contractive += int(rep["contractive"])
        fp_gaps.append(pi["fp_gap"])
        meas_gap.append(ul["unlearn_gap"])
        pred_rel.append(rep["removal"]["pred_rel"] if rep["removal"] else float("nan"))
        if ul["warm_converged"] and ul["cold_converged"]:
            both_conv_gap.append(ul["unlearn_gap"])

    rt, ft = torch.tensor(rhos), torch.tensor(fp_gaps)
    summary = {
        "update": update, "acc": acc,
        "rho_mean": float(rt.mean()), "rho_max": float(rt.max()),
        "contractive_frac": contractive / N_PROBE,
        "fp_gap_mean": float(ft.mean()), "fp_gap_max": float(ft.max()),
        "unlearn_gap_max": float(torch.tensor(meas_gap).max()),
        "both_converged_n": len(both_conv_gap),
        "both_conv_gap_max": (float(torch.tensor(both_conv_gap).max())
                              if both_conv_gap else float("nan")),
        "removal_supported": any(math.isfinite(p) for p in pred_rel),
        "corr_predrel_vs_gap": _corr(pred_rel, meas_gap),
        "raw": {"rho": rhos, "fp_gap": fp_gaps,
                "pred_rel": pred_rel, "unlearn_gap": meas_gap},
    }
    print(f"  acc={acc:.3f} | rho mean/max={summary['rho_mean']:.3f}/"
          f"{summary['rho_max']:.3f} (contractive {summary['contractive_frac']*100:.0f}%)")
    print(f"  fp_gap mean/max={summary['fp_gap_mean']:.3f}/{summary['fp_gap_max']:.3f} | "
          f"both-conv gap max={summary['both_conv_gap_max']:.3f} "
          f"(n={summary['both_converged_n']}/{N_PROBE})")
    print(f"  removal-knob={summary['removal_supported']} | "
          f"corr(pred_rel,gap)={summary['corr_predrel_vs_gap']:.3f}")
    return summary


def main():
    results = {}
    for update in ("normdeepsets", "linattn", "attn"):
        results[update] = run(update)
        print()
    print("=" * 70)
    print(f"{'model':<14}{'acc':>7}{'rho_max':>9}{'fp_gap_max':>12}"
          f"{'bothconv_max':>14}{'decomp':>8}")
    for u in ("normdeepsets", "linattn", "attn"):
        s = results[u]
        print(f"{u:<14}{s['acc']:>7.3f}{s['rho_max']:>9.3f}{s['fp_gap_max']:>12.3f}"
              f"{s['both_conv_gap_max']:>14.3f}{str(s['removal_supported']):>8}")

    out = os.path.join(os.path.dirname(__file__), "results_jacobian.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
