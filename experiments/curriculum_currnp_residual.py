"""RESIDUAL smoke test — currnp recipe (CAUSAL + pure relative PE) but with a STATE SKIP in the cell.

The curr/currnp cell is residual around the INJECTION only (out = h0 + s*A(z)); J = s*dA/dz has NO identity
component. sw.RESIDUAL adds a learnable state skip (out += r*z) -> J gains +r*I -> I-J = (1-r)I - s*dA/dz.
QUESTION (minimal, comparable to currnp): does a residual connection move the conditioning ladder
(sigma_min / rho(J)), and does the model even USE the skip (learned r)?

Prediction sketch: the +r*I diagonal shift is structural (unlike the MLP's per-position nonlinear block). It
should show up in sigma_min directly; whether it HELPS the conditioning<->recall tension (lifts sigma_min at
matched recall) or just gets absorbed (r -> 0) is the open question this smoke test answers.

Trimmed ladder (smoke, not the full currnp run): PHASE_A forms the hop, PHASE_B at gaps {0,16,40} = well-cond,
mid, near-singular. Compare sigma_min/recall to currnp {0.197/1.0, 0.036/0.974, 0.024/0.810}.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.curriculum_currnp_residual
"""
import os
import time

import torch
import experiments.sliding_window_reach as sw

sw.BIDIR = False          # CAUSAL
sw.REL_BIAS = True        # relative position (with NO_POSW = the only position signal)
sw.READONLY_Q = True
sw.NO_POSW = True
sw.RESIDUAL = os.environ.get("DEQ_RES", "1") == "1"   # DEQ_RES=0 -> trimmed-ladder no-residual CONTROL
sw.H, sw.dh = 4, sw.d // 4

PHASE_A = [(2, 400), (4, 400), (10, 600)]
PHASE_B = [0, 16, 40]                            # trimmed: well-cond / mid / near-singular
STEPS_B = 350
CKPT_DIR = "checkpoints"
PREFIX = "currnpres" if sw.RESIDUAL else "currnptrim"   # trim = the same-ladder no-residual control

# currnp (no-residual) reference at matched gaps, for the side-by-side
CURRNP_REF = {0: (1.000, 0.197), 16: (0.974, 0.036), 40: (0.810, 0.024)}   # (recall, sigma_min)


def eval_stage(m, gap):
    m.eval()
    acc = sw.recall(m, gap, torch.Generator().manual_seed(123))
    rho, smin, rs = m.spectrum(sw.gen_mqar(1, gap, torch.Generator().manual_seed(7))[0])
    m.train()
    return acc, rho, smin, rs


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"device={sw.DEV}  RESIDUAL smoke (currnp recipe + state skip, R_MAX={sw.R_MAX}). "
          f"Does out+=r*z move the conditioning?\n", flush=True)
    torch.manual_seed(0)
    sw.W = 10
    m = sw.SeqDEQ("softmax", "deq").to(sw.DEV)

    sw.QUERY_FULL = True
    sw.F_SWEEP = [0]
    for w_stage, steps in PHASE_A:
        sw.W = w_stage
        t0 = time.time()
        sw.train(m, steps=steps)
        acc, rho, smin, rs = eval_stage(m, 0)
        print(f"  [A] w={w_stage:>2}: recall={acc:.3f}  rho={rho:.3f}  smin={smin:.3f}  r={(float(m.r.detach()) if sw.RESIDUAL else 0.0):.3f}  "
              f"resid={rs:.1e}  ({time.time()-t0:.0f}s)", flush=True)

    sw.QUERY_FULL = False
    sw.W = 10
    for g in PHASE_B:
        sw.F_SWEEP = [g]
        t0 = time.time()
        sw.train(m, steps=STEPS_B)
        acc, rho, smin, rs = eval_stage(m, g)
        path = os.path.join(CKPT_DIR, f"{PREFIX}{g:02d}.pt")
        torch.save({"state_dict": m.state_dict(), "stage_gap": g, "recall": acc,
                    "rho": rho, "sigma_min": smin, "resid": rs, "H": sw.H, "W": sw.W,
                    "bidir": False, "rel_bias": True, "readonly_q": True, "query_full": False,
                    "no_posw": True, "residual": sw.RESIDUAL, "r": (float(m.r.detach()) if sw.RESIDUAL else 0.0)}, path)
        ref = CURRNP_REF.get(g)
        cmp = (f"  | currnp: recall={ref[0]:.3f} smin={ref[1]:.3f}" if ref else "")
        print(f"  [B] gap {g:>2}: recall={acc:.3f}  rho={rho:.3f}  smin={smin:.3f}  r={(float(m.r.detach()) if sw.RESIDUAL else 0.0):.3f}  "
              f"resid={rs:.1e}{cmp}  ({time.time()-t0:.0f}s)  -> {path}", flush=True)
    print(f"\nDone. Read: r>0 kept = the model USES the skip; compare smin/recall vs currnp -> does the residual\n"
          f"lift sigma_min at matched recall (soften the tension) or get absorbed (r->0, conditioning unchanged)?",
          flush=True)


if __name__ == "__main__":
    main()
