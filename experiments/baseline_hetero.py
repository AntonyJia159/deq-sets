"""How badly does linear-propagation + MLP (the InstantGNN expressiveness class) do on
heterophily? MLP (no graph, the floor) vs SGC and APPNP (low-pass linear propagation),
across the Platonov-filtered datasets, standard splits.

If MLP >= SGC/APPNP, the graph is actively HURTING the linear-propagation models -- the
low-pass-trap diagnosis -- which is the motivation for signed/nonlinear propagation.

Run:  D:\\deq-venv\\Scripts\\python.exe -m experiments.baseline_hetero
"""

import time

import numpy as np
import torch

from experiments.hetero_headtohead import load, MLP, SGC, APPNP, run_split
from experiments.cora_deletion import renorm_sparse

DATASETS = ["chameleon_filtered", "squirrel_filtered", "roman_empire", "amazon_ratings"]
N_SPLITS = 5
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print(f"device = {DEV}")
    for ds in DATASETS:
        X, y, A, masks, K = load(ds)
        X, y = X.to(DEV), y.to(DEV)
        Ahat = renorm_sparse(A).to(DEV)            # sparse adjacency on GPU
        d_in = X.shape[1]
        print(f"\n=== {ds}: {X.shape[0]} nodes, {d_in} feat, {K} classes, "
              f"{A.nnz // 2} edges ===", flush=True)
        builders = {
            "MLP (no graph)": lambda: MLP(d_in, 64, K),
            "SGC (linear)": lambda: SGC(d_in, K),
            "APPNP (linear)": lambda: APPNP(d_in, 64, K),
        }
        for name, build in builders.items():
            accs, t0 = [], time.time()
            for s in range(N_SPLITS):
                torch.manual_seed(s); np.random.seed(s)
                tr = torch.tensor(masks["train_masks"][s].astype(bool)).to(DEV)
                va = torch.tensor(masks["val_masks"][s].astype(bool)).to(DEV)
                te = torch.tensor(masks["test_masks"][s].astype(bool)).to(DEV)
                accs.append(run_split(build().to(DEV), X, y, Ahat, None, tr, va, te))
            print(f"  {name:<18} {np.mean(accs):.3f} +- {np.std(accs):.3f}"
                  f"   ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
