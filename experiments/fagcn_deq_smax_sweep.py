"""s_max sweep on the v3 nonlinear MLP cell (experiments/fagcn_deq_mlp.py).

Finding from v3: at s_max=0.5 the model pushed s->0.41 (against the cap) yet rho(J) stayed ~0.5 --
it is REACH-HUNGRY, not contraction-bound. Adding nonlinear MLP expressivity did not beat v2's
0.682. So the wall is reach (screening length), which only grows as rho(J) -> 1. This sweep raises
the cap s_max and measures the locality<->expressivity tax directly:
  - does accuracy climb as we allow more propagation reach?
  - where does the learned s saturate, and where does rho(J) cross 1 (-> INVALID: no longer an
    attracting fixed point, becomes a truncated deep net, voids the maintenance thesis)?

jac_gamma=1.0 is kept (still penalizes ||J||_F), so this finds the accuracy-optimal contraction
level UNDER that penalty. If acc stays flat while s saturates and rho stays low, jac_reg is the
limiter and we rerun with smaller gamma. 2 splits/value for a fast trend (variance ~0.005); the
winner gets a full 5-split confirm afterward.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.fagcn_deq_smax_sweep
"""

import os
import time

import numpy as np
import torch

from experiments.fagcn_deq_mlp import CFG, DEV, load, run_split

# 0.5 already measured (acc 0.687, s 0.41, rho 0.516); run the remaining caps.
S_MAX_GRID = [0.8, 1.2, 1.6]
N_SPLITS = 2
DS = "roman_empire"
RESULTS = os.path.join(os.path.dirname(__file__), "smax_sweep_results.txt")


def _done_values():
    """s_max values already recorded, so a re-run after a crash/sleep skips them."""
    if not os.path.exists(RESULTS):
        return set()
    vals = set()
    with open(RESULTS) as fh:
        for line in fh:
            if line.strip() and not line.startswith("#"):
                vals.add(float(line.split(",")[0]))
    return vals


def main():
    print(f"device = {DEV}   dataset = {DS}   splits/value = {N_SPLITS}")
    print(f"base config (s_max varies): {CFG}")
    print(f"appending results to {RESULTS}\n")
    if not os.path.exists(RESULTS):
        with open(RESULTS, "w") as fh:
            fh.write("# s_max, acc, std, learned_s, rho(J)\n0.5,0.687,0.003,0.41,0.516\n")
    done = _done_values()
    X, y, edges, deg, masks, K = load(DS)
    X, y, edges, deg = X.to(DEV), y.to(DEV), edges.to(DEV), deg.to(DEV)
    for s_max in S_MAX_GRID:
        if s_max in done:
            print(f"s_max {s_max:>4}: already done, skipping", flush=True)
            continue
        cfg = dict(CFG, s_max=s_max)
        accs, ss, rhos = [], [], []
        t0 = time.time()
        for sp in range(N_SPLITS):
            tr = torch.tensor(masks["train_masks"][sp].astype(bool)).to(DEV)
            va = torch.tensor(masks["val_masks"][sp].astype(bool)).to(DEV)
            te = torch.tensor(masks["test_masks"][sp].astype(bool)).to(DEV)
            a, d = run_split(X.shape[1], K, edges, deg, X, y, tr, va, te, cfg)
            accs.append(a); ss.append(d["s"]); rhos.append(d["rho"])
        macc, sacc, ms, mrho = np.mean(accs), np.std(accs), np.mean(ss), np.mean(rhos)
        with open(RESULTS, "a") as fh:                         # persist immediately
            fh.write(f"{s_max},{macc:.3f},{sacc:.3f},{ms:.2f},{mrho:.3f}\n")
        flag = "  <-- INVALID rho>=1 (not an equilibrium)" if mrho >= 1.0 else ""
        print(f"s_max {s_max:>4}: acc {macc:.3f} +- {sacc:.3f}  "
              f"[learned s {ms:.2f}  rho(J) {mrho:.3f}]  ({time.time()-t0:.0f}s){flag}", flush=True)
    print(f"\nfull curve in {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
