"""BIDIRECTIONAL curriculum retrain — the edit-regime substrate for the proper Faber-face C2.

Same verified config as curriculum_checkpoints (bs64, f_tol 1e-4, 350 steps/stage) with ONE change:
sw.BIDIR=True, so position i attends to the two-sided band [i-W, i+W]. This is the edit/maintenance
regime the blueprint headlines (defs-after-uses, callers on both sides): J is banded but NOT triangular,
so the sigma_min/Faber (BVP/elliptic) certificate is the theoretically-proper one — unlike the causal
checkpoints where the kappa->xi formula was a category error and the product-Lyapunov form applied.

Note: degree doubles vs causal (21 vs 11 neighbors) and query-identity can propagate BACKWARD, so the
relay can meet in the middle — expect easier training / possibly different rho,sigma_min trajectory.

Saves to checkpoints/bidirXX.pt: state_dict + {stage_gap, recall, rho, sigma_min, resid, bidir=True}.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.curriculum_bidir
"""
import os
import time

import torch
import experiments.sliding_window_reach as sw

sw.BIDIR = True
sw.REL_BIAS = True     # bidir needs an explicit direction signal: without it, binding blends (probe round 1:
                       # recall stuck ~0.38 across init/steps/lr/s_max; value token's neighborhood is l/r-symmetric)
STAGES = [0, 8, 16, 24, 40]
STEPS_PER = 350
CKPT_DIR = "checkpoints"
sw.H, sw.dh = 4, sw.d // 4


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"device={sw.DEV}  BIDIRECTIONAL (band [i-{sw.W}, i+{sw.W}]) eq-softmax curriculum {STAGES}, "
          f"{STEPS_PER} steps/stage (bs64/ftol1e-4), checkpoint per stage -> {CKPT_DIR}/bidirXX.pt\n", flush=True)
    torch.manual_seed(0)
    m = sw.SeqDEQ("softmax", "deq").to(sw.DEV)
    for g in STAGES:
        sw.F_SWEEP = [g]
        t0 = time.time()
        sw.train(m, steps=STEPS_PER)
        m.eval()
        ge = torch.Generator().manual_seed(123)
        acc = sw.recall(m, g, ge)
        r, smin, rs = m.spectrum(sw.gen_mqar(1, g, torch.Generator().manual_seed(7))[0])
        path = os.path.join(CKPT_DIR, f"bidir{g:02d}.pt")
        torch.save({"state_dict": m.state_dict(), "stage_gap": g, "recall": acc,
                    "rho": r, "sigma_min": smin, "resid": rs, "H": sw.H, "W": sw.W, "bidir": True}, path)
        print(f"  gap {g:>2}: recall={acc:.3f}  rho={r:.3f}  sigma_min={smin:.3f}  resid={rs:.1e}  "
              f"-> {path}  ({time.time()-t0:.0f}s)", flush=True)
        m.train()
    print("\nDone. c2_bidir loads these for the Faber-face edit-locality measurement.", flush=True)


if __name__ == "__main__":
    main()
