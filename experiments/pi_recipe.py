"""Does the Anil et al. (2022) recipe make ATTENTION path-independent without
contraction? -- SEED-AVERAGED (10 seeds, paired).

Their finding: path independence can be PROMOTED on unrestricted (non-contractive,
multistable-capable) architectures purely by training -- mixed zero/noise init +
randomized solver budget -- buying upward generalization and robustness to
adversarial (init-steering) attacks. For us the warm-start unlearning leak IS an
init-steering channel.

Single-seed run-3 suggested the recipe cuts the both-converged multistability gap
~7x with no accuracy cost, BUT single-seed variance is known-high here, so this
script repeats the paired comparison over 10 seeds and reports mean +/- std. Each
seed trains BOTH configs on the SAME data with the SAME init seed, so the only
difference is the recipe (a paired design). AA score dropped -- it was saturated /
uninformative; the both-converged relative-norm gap is the metric that discriminates.

  baseline : fixed budget, random init each step
  PI       : SetDEQ(pi_train=True) -- mixed init + randomized budget

Run:  & "D:\\deq-venv\\Scripts\\python.exe" -m experiments.pi_recipe
"""

import json
import os

import torch

from src.data import GMMSetDataset
from src.model import SetDEQ
from src.train import train, evaluate
from src.metrics import path_independence_gap, unlearning_gap

K_RANGE = (1, 4)
N = 24
N_PROBE = 30
SEEDS = list(range(10))
SOLVE = dict(max_iter=200)


def probe(model, test_ds):
    fp_gaps, both_conv_gap = [], []
    for i in range(N_PROBE):
        x = test_ds.X[i]
        pi = path_independence_gap(model, x, n_inits=5, **SOLVE)
        ul = unlearning_gap(model, x, remove_idx=0, **SOLVE)
        fp_gaps.append(pi["fp_gap"])
        if ul["warm_converged"] and ul["cold_converged"]:
            both_conv_gap.append(ul["unlearn_gap"])
    ft = torch.tensor(fp_gaps)
    bt = torch.tensor(both_conv_gap) if both_conv_gap else torch.tensor([float("nan")])
    return {
        "fp_gap_max": float(ft.max()),
        "both_conv_n": len(both_conv_gap),
        "both_conv_mean": float(bt.mean()),
        "both_conv_max": float(bt.max()),
    }


def run_one(pi_train, seed, train_ds, test_ds, epochs=15):
    torch.manual_seed(seed)
    model = SetDEQ(d_in=2, d_latent=32, hidden=64, update="attn",
                   n_classes=train_ds.n_classes, max_iter=150, tol=1e-5,
                   pi_train=pi_train, pi_min_iter=10)
    train(model, train_ds, epochs=epochs, batch_size=64, lr=1e-3, seed=seed, log_every=0)
    acc = evaluate(model, test_ds)
    r = probe(model, test_ds)
    r["acc"] = acc
    return r


def _stats(rows, key):
    t = torch.tensor([r[key] for r in rows], dtype=torch.float64)
    return float(t.mean()), float(t.std(unbiased=False))


def main():
    train_ds = GMMSetDataset(n_samples=2000, k_range=K_RANGE, n_points=N, d=2,
                             sep=4.0, std=1.0, seed=1)
    test_ds = GMMSetDataset(n_samples=500, k_range=K_RANGE, n_points=N, d=2,
                            sep=4.0, std=1.0, seed=2)

    per_seed = {"baseline": [], "pi": []}
    for seed in SEEDS:
        for pi in (False, True):
            key = "pi" if pi else "baseline"
            r = run_one(pi, seed, train_ds, test_ds)
            per_seed[key].append(r)
            print(f"seed {seed} {key:<9} acc={r['acc']:.3f} "
                  f"fp_gap_max={r['fp_gap_max']:.3f} "
                  f"both-conv mean/max={r['both_conv_mean']:.3f}/{r['both_conv_max']:.3f} "
                  f"(n={r['both_conv_n']}/{N_PROBE})")
        print()

    print("=" * 78)
    print(f"{'config':<10}{'acc':>14}{'fp_gap_max':>16}{'bothconv_mean':>18}"
          f"{'bothconv_max':>18}")
    summary = {}
    for key in ("baseline", "pi"):
        rows = per_seed[key]
        s = {m: _stats(rows, m) for m in
             ("acc", "fp_gap_max", "both_conv_mean", "both_conv_max", "both_conv_n")}
        summary[key] = s
        f = lambda m: f"{s[m][0]:.3f}+/-{s[m][1]:.3f}"
        print(f"{key:<10}{f('acc'):>14}{f('fp_gap_max'):>16}"
              f"{f('both_conv_mean'):>18}{f('both_conv_max'):>18}")

    out = os.path.join(os.path.dirname(__file__), "results_pi_recipe.json")
    with open(out, "w") as fh:
        json.dump({"per_seed": per_seed, "summary": summary, "seeds": SEEDS}, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
