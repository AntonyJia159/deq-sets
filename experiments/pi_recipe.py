"""Does the Anil et al. (2022) recipe make ATTENTION path-independent without
contraction?

Their finding: path independence can be PROMOTED on unrestricted (non-contractive,
multistable-capable) architectures purely by training -- mixed zero/noise init +
randomized solver budget -- and it buys upward generalization and robustness to
adversarial (init-steering) attacks. For us the warm-start unlearning leak IS an
init-steering channel, so if this works on attention we get path independence
(single effective basin), test-time scaling, and a closed warm-start channel WITHOUT
paying the (hard, expressiveness-costing) contraction constraint.

We train attention two ways and probe both:
  baseline : fixed budget, random init each step (our previous setup)
  PI       : SetDEQ(pi_train=True) -- mixed init + randomized budget

Metrics (N=24): test acc; AA score (cosine of zero-init fp vs random-init fps, the
paper's metric, ->1 = path independent); fp_gap (->0 = unique); both-converged
unlearn gap (true multistability); rho(J_f).

Run:  & "D:\\deq-venv\\Scripts\\python.exe" -m experiments.pi_recipe
"""

import json
import os

import torch
import torch.nn.functional as F

from src.data import GMMSetDataset
from src.model import SetDEQ
from src.train import train, evaluate
from src.metrics import path_independence_gap, unlearning_gap
from src.jacobian import jacobian_report

K_RANGE = (1, 4)
N = 24
N_PROBE = 30
SOLVE = dict(max_iter=200)


@torch.no_grad()
def aa_score(model, x, n_rand=5, **solve_kw):
    """Asymptotic Alignment: mean cosine similarity between the zero-init fixed
    point and random-init fixed points (Anil et al. 2022). ->1 = path independent.
    """
    xb = x.unsqueeze(0)
    z_zero, _ = model.solve(xb, z0=torch.zeros(1, x.shape[0], model.d_latent), **solve_kw)
    zf = z_zero.flatten()
    sims = []
    for _ in range(n_rand):
        z0 = torch.randn(1, x.shape[0], model.d_latent)
        z, _ = model.solve(xb, z0=z0, **solve_kw)
        sims.append(F.cosine_similarity(zf, z.flatten(), dim=0).item())
    return sum(sims) / len(sims)


def run(pi_train, epochs=15):
    torch.manual_seed(0)
    train_ds = GMMSetDataset(n_samples=2000, k_range=K_RANGE, n_points=N, d=2,
                             sep=4.0, std=1.0, seed=1)
    test_ds = GMMSetDataset(n_samples=500, k_range=K_RANGE, n_points=N, d=2,
                            sep=4.0, std=1.0, seed=2)
    model = SetDEQ(d_in=2, d_latent=32, hidden=64, update="attn",
                   n_classes=train_ds.n_classes, max_iter=150, tol=1e-5,
                   pi_train=pi_train, pi_min_iter=10)
    tag = "PI-recipe" if pi_train else "baseline"
    print(f"== training attn ({tag}) ==")
    train(model, train_ds, epochs=epochs, batch_size=64, lr=1e-3, log_every=0)
    acc = evaluate(model, test_ds)

    fp_gaps, aas, rhos, both_conv_gap = [], [], [], []
    for i in range(N_PROBE):
        x = test_ds.X[i]
        xb = x.unsqueeze(0)
        z_star, _ = model.solve(xb, **SOLVE)
        rep = jacobian_report(model, xb, z_star)
        pi = path_independence_gap(model, x, n_inits=5, **SOLVE)
        ul = unlearning_gap(model, x, remove_idx=0, **SOLVE)
        fp_gaps.append(pi["fp_gap"])
        aas.append(aa_score(model, x, **SOLVE))
        rhos.append(rep["rho"])
        if ul["warm_converged"] and ul["cold_converged"]:
            both_conv_gap.append(ul["unlearn_gap"])

    ft, at, rt = (torch.tensor(fp_gaps), torch.tensor(aas), torch.tensor(rhos))
    summary = {
        "tag": tag, "acc": acc,
        "aa_mean": float(at.mean()), "aa_min": float(at.min()),
        "fp_gap_mean": float(ft.mean()), "fp_gap_max": float(ft.max()),
        "both_converged_n": len(both_conv_gap),
        "both_conv_gap_max": (float(torch.tensor(both_conv_gap).max())
                              if both_conv_gap else float("nan")),
        "rho_mean": float(rt.mean()), "rho_max": float(rt.max()),
    }
    print(f"  acc={acc:.3f} | AA mean/min={summary['aa_mean']:.3f}/{summary['aa_min']:.3f} | "
          f"fp_gap mean/max={summary['fp_gap_mean']:.3f}/{summary['fp_gap_max']:.3f}")
    print(f"  both-conv gap max={summary['both_conv_gap_max']:.3f} "
          f"(n={summary['both_converged_n']}/{N_PROBE}) | "
          f"rho mean/max={summary['rho_mean']:.3f}/{summary['rho_max']:.3f}")
    return summary


def main():
    results = {}
    for pi in (False, True):
        key = "pi" if pi else "baseline"
        results[key] = run(pi)
        print()
    print("=" * 64)
    print(f"{'attn':<12}{'acc':>7}{'AA_mean':>9}{'AA_min':>8}{'fp_gap_max':>12}"
          f"{'bothconv':>10}")
    for key in ("baseline", "pi"):
        s = results[key]
        print(f"{s['tag']:<12}{s['acc']:>7.3f}{s['aa_mean']:>9.3f}{s['aa_min']:>8.3f}"
              f"{s['fp_gap_max']:>12.3f}{s['both_conv_gap_max']:>10.3f}")
    out = os.path.join(os.path.dirname(__file__), "results_pi_recipe.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
