"""Layer-1 smoke test: train a tiny set-DEQ and run the three probe metrics.

Run from the repo root:  python -m experiments.smoke_test
Everything here is sized to finish in a couple of minutes on CPU.
"""

import torch

from src.data import GMMSetDataset
from src.model import SetDEQ
from src.train import train, evaluate
from src.metrics import path_independence_gap, unlearning_gap


def main(update="deepsets", spectral=False, seed=0):
    torch.manual_seed(seed)

    k_range = (1, 4)
    train_ds = GMMSetDataset(n_samples=2000, k_range=k_range, n_points=24, d=2, seed=1)
    test_ds = GMMSetDataset(n_samples=500, k_range=k_range, n_points=24, d=2, seed=2)

    model = SetDEQ(
        d_in=2, d_latent=32, hidden=64, update=update,
        n_classes=train_ds.n_classes, max_iter=30, tol=1e-4, damping=0.5,
        spectral=spectral,
    )

    print(f"== training set-DEQ (update={update}, spectral={spectral}) ==")
    train(model, train_ds, epochs=8, batch_size=64, lr=1e-3)
    print(f"test accuracy: {evaluate(model, test_ds):.3f}\n")

    # Probe a handful of test sets with more solver iterations for clean fixed points.
    sol = dict(max_iter=200, tol=1e-6)
    fp_gaps, agrees, unlearn_gaps, matches, warm_it, cold_it = [], [], [], [], [], []
    for i in range(20):
        x = test_ds.X[i]
        pi = path_independence_gap(model, x, n_inits=5, **sol)
        ul = unlearning_gap(model, x, remove_idx=0, **sol)
        fp_gaps.append(pi["fp_gap"]); agrees.append(pi["pred_agreement"])
        unlearn_gaps.append(ul["unlearn_gap"]); matches.append(ul["pred_match"])
        warm_it.append(ul["warm_iters"]); cold_it.append(ul["cold_iters"])

    def mean(v):
        return sum(v) / len(v)

    print("== probe results (mean over 20 sets) ==")
    print(f"[1] path-independence  fp_gap={mean(fp_gaps):.4f}  "
          f"pred_agreement={mean(agrees):.3f}")
    print(f"[2] unlearning         gap={mean(unlearn_gaps):.4f}  "
          f"pred_match={mean(matches):.3f}")
    print(f"[3] efficiency         warm_iters={mean(warm_it):.1f}  "
          f"cold_iters={mean(cold_it):.1f}")
    print("\nReading: small fp_gap + agreement~1.0 => path independence holds; "
          "small unlearn_gap + match~1.0 => exact unlearning; "
          "warm_iters << cold_iters => efficiency bonus.")


if __name__ == "__main__":
    main()
