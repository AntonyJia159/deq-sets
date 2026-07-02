"""Rerun the eq-softmax curriculum (EXACT verified config: bs64, f_tol 1e-4, 350 steps/stage — the run
that gave gap-24 recall 0.975) and SAVE A CHECKPOINT PER STAGE. These checkpoints span sigma_min 0.15->0.02
and rho 0.27->1.19, which is the conditioning range the C2 edit-locality measurement sweeps: measured xi vs
Faber-predicted reach across checkpoints = the sigma_min LAW, sequence version (incl. the rho>1 regime the
graph cells never entered).

Saves to checkpoints/currXX.pt: state_dict + {stage_gap, recall, rho, sigma_min, resid}.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.curriculum_checkpoints
"""
import os
import time

import torch
import experiments.sliding_window_reach as sw

STAGES = [0, 8, 16, 24, 40]          # gap-40 half-learned checkpoint is still a valid sigma_min point
STEPS_PER = 350
CKPT_DIR = "checkpoints"
sw.H, sw.dh = 4, sw.d // 4


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"device={sw.DEV}  eq-softmax curriculum {STAGES}, {STEPS_PER} steps/stage (verified config, "
          f"bs64/ftol1e-4), checkpoint per stage -> {CKPT_DIR}/\n", flush=True)
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
        path = os.path.join(CKPT_DIR, f"curr{g:02d}.pt")
        torch.save({"state_dict": m.state_dict(), "stage_gap": g, "recall": acc,
                    "rho": r, "sigma_min": smin, "resid": rs, "H": sw.H, "W": sw.W}, path)
        print(f"  gap {g:>2}: recall={acc:.3f}  rho={r:.3f}  sigma_min={smin:.3f}  resid={rs:.1e}  "
              f"-> {path}  ({time.time()-t0:.0f}s)", flush=True)
        m.train()
    print("\nDone. C2 (edit-locality) loads these and measures |dz| vs distance per checkpoint.", flush=True)


if __name__ == "__main__":
    main()
