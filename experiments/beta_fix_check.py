"""Beta-fix: can a learnable-temperature log-sum-exp aggregator dial to the right semiring per task,
once beta is allowed to move? The naive version under-sharpened (beta stuck ~1.3, tropical 0.497 vs
hard-max 0.683) because (a) weight decay pulled beta_raw toward 0 and (b) beta had the same small lr
as everything else. Fix: beta_raw in its OWN param group -- NO weight decay, higher lr -- and init
beta high (~4) so it starts near max and can come DOWN on the spectral task if that is what helps.

Expectation if it works: tropical R^2 -> ~0.68 with large learned beta; spectral R^2 ~ sum-level with
beta pulled back down. Reference points: hard-max tropical 0.683 / spectral 0.553; naive logSExp
tropical 0.497 / spectral 0.599.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.beta_fix_check
"""

import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.broyden_synthetic import grid_graph
from experiments.aniso_teacher import AnisoTeacher
from experiments.mpnn_deq import MPNNDEQ, CFG
from experiments.semiring_compare import splits, r2

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, D_FEAT, R, N_SPLITS = 40, 16, 4, 3
BETA_INIT, BETA_LR = 4.0, 5e-2          # init beta~softplus(4)=4.02; let beta move fast, no wd
TASKS = [("laplacian", 2, "spectral"), ("maxreach", 1, "tropical")]


def train_betafix(edges, deg, X, t, tr, va, te):
    torch.manual_seed(0)
    m = MPNNDEQ(D_FEAT, 1, edges, deg, dict(CFG, agg="logsumexp")).to(DEV)
    m.beta_raw.data = torch.tensor(BETA_INIT, device=DEV)            # start near max
    other = [p for n, p in m.named_parameters() if n != "beta_raw"]
    opt = torch.optim.Adam([{"params": other, "weight_decay": CFG["wd"], "lr": CFG["lr"]},
                            {"params": [m.beta_raw], "weight_decay": 0.0, "lr": BETA_LR}])
    bv, bt, bb = -1e9, 0.0, 0.0
    for e in range(CFG["epochs"]):
        m.train(); opt.zero_grad()
        out, reg = m(X, jac=True)
        (F.mse_loss(out[tr].squeeze(-1), t[tr]) + CFG["jac_gamma"] * reg).backward()
        torch.nn.utils.clip_grad_norm_(other, 5.0); opt.step()
        if e % 10 == 0:
            m.eval()
            with torch.no_grad():
                out, _ = m(X)
            v = r2(out, t, va)
            if v > bv:
                bv, bt, bb = v, r2(out, t, te), F.softplus(m.beta_raw).item()
    return bt, bb


def main():
    print(f"device = {DEV}")
    edges, deg, N = grid_graph(L)
    edges, deg = edges.to(DEV), deg.to(DEV)
    print(f"grid {L}x{L}={N}, beta-fix (no-wd, lr {BETA_LR}, init beta {F.softplus(torch.tensor(BETA_INIT)).item():.1f})\n",
          flush=True)
    print(f"{'task':<10} {'logSExp R^2 (beta)':<24} {'ref: hard-max':<14} {'ref: naive logSExp'}",
          flush=True)
    refs = {"spectral": (0.553, 0.599), "tropical": (0.683, 0.497)}
    for tname, k, tlabel in TASKS:
        teacher = AnisoTeacher(edges, deg, N, d_feat=D_FEAT, R=R, k=k, seed=0, target=tname)
        teacher.generate()
        X, t = teacher.X, teacher.s
        t0 = time.time()
        res = [train_betafix(edges, deg, X, t, *splits(N, s)) for s in range(N_SPLITS)]
        r = np.mean([x[0] for x in res]); sd = np.std([x[0] for x in res])
        b = np.mean([x[1] for x in res])
        hm, naive = refs[tlabel]
        print(f"{tlabel:<10} {r:+.3f} +- {sd:.3f}  (beta {b:.1f})   {hm:<14.3f} {naive:.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
