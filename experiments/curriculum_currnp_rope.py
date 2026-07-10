"""currnp + RoPE — the un-ruled-out conditioning fix.

QK-norm (curriculum_currnp_qk) DOUBLED sigma_min but COLLAPSED recall 15-24pts: capping logit MAGNITUDE
starves the peaky attention long-gap recall needs (peaking<->contraction tension is tight). RoPE is the
orthogonal alternative flagged as un-ruled-out: a pure ROTATION of q,k -> norm-preserving, does NOT touch
attention sharpness, so it cannot cap peaking. It also SUPPLIES relative position (dot product depends on
j-i), so it cleanly REPLACES currnp's learned additive relative bias (REL_BIAS off here) -- a single-variable
swap of the positional mechanism (learned bias -> rotary), exactly how a modern RoPE transformer encodes
position. Hypothesis: a cleaner positional signal lets the cross-window relay form with LESS saturation ->
sigma_min UP without the recall cost QK-norm paid.

Same recipe as curriculum_currnp (causal + NO_POSW + window curriculum) with REL_BIAS off and sw.ROPE=True.
Saves checkpoints/currnpropeXX.pt (rope=True, rel_bias=False). Compare the sigma_min ladder + recall to
currnp* (relative bias) and currnpqk* (cosine attn).

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.curriculum_currnp_rope
"""
import os
import time

import torch
import experiments.sliding_window_reach as sw

sw.BIDIR = False
sw.REL_BIAS = False       # RoPE REPLACES the learned relative bias as the positional mechanism
sw.READONLY_Q = True
sw.NO_POSW = True
sw.ROPE = True            # the change vs curriculum_currnp
sw.H, sw.dh = 4, sw.d // 4

PHASE_A = [(2, 400), (4, 400), (10, 600)]
PHASE_B = [0, 8, 16, 24, 40]
STEPS_B = 350
CKPT_DIR = "checkpoints"
PREFIX = "currnprope"


def eval_stage(m, gap):
    m.eval()
    acc = sw.recall(m, gap, torch.Generator().manual_seed(123))
    r, smin, rs = m.spectrum(sw.gen_mqar(1, gap, torch.Generator().manual_seed(7))[0])
    m.train()
    return acc, r, smin, rs


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"device={sw.DEV}  currnp + RoPE (rotary q,k, REL_BIAS off): does a norm-preserving positional\n"
          f"  rotation lift sigma_min vs currnp (learned rel-bias) WITHOUT the recall cost QK-norm paid? "
          f"same recipe + RoPE.\n", flush=True)
    torch.manual_seed(0)
    sw.W = 10
    m = sw.SeqDEQ("softmax", "deq").to(sw.DEV)

    sw.QUERY_FULL = True
    sw.F_SWEEP = [0]
    for w_stage, steps in PHASE_A:
        sw.W = w_stage
        t0 = time.time()
        sw.train(m, steps=steps)
        acc, r, smin, rs = eval_stage(m, 0)
        print(f"  [A] w={w_stage:>2}: recall={acc:.3f}  rho={r:.3f}  smin={smin:.3f}  resid={rs:.1e}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    sw.QUERY_FULL = False
    sw.W = 10
    for g in PHASE_B:
        sw.F_SWEEP = [g]
        t0 = time.time()
        sw.train(m, steps=STEPS_B)
        acc, r, smin, rs = eval_stage(m, g)
        path = os.path.join(CKPT_DIR, f"{PREFIX}{g:02d}.pt")
        torch.save({"state_dict": m.state_dict(), "stage_gap": g, "recall": acc,
                    "rho": r, "sigma_min": smin, "resid": rs, "H": sw.H, "W": sw.W,
                    "bidir": False, "rel_bias": False, "readonly_q": True, "query_full": False,
                    "no_posw": True, "rope": True}, path)
        print(f"  [B] gap {g:>2}: recall={acc:.3f}  rho={r:.3f}  smin={smin:.3f}  resid={rs:.1e}  "
              f"-> {path}  ({time.time()-t0:.0f}s)", flush=True)
    print("\nDone. Compare currnprope sigma_min ladder + recall to currnp (rel-bias) and currnpqk (cosine) "
          "-> did RoPE lift conditioning without the recall cost?", flush=True)


if __name__ == "__main__":
    main()
