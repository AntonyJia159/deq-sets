"""currnp + RoPE ON TOP of the learned relative bias — the FAITHFUL conditioning parallel to QK-norm.

The first RoPE run (curriculum_currnp_rope) REPLACED the learned relative bias with rotary and the relay
never formed (recall plateaued 0.77 at w=10 vs currnp's 1.0, collapsed to ~guessing at long gaps) -- a
TRAINABILITY failure, so sigma_min was measured on a model not doing the task. That broke the QK-norm
parallel: QK-norm was added ON TOP of the working rel-bias recipe, isolating the conditioning effect at
PRESERVED recall. This run does the same for RoPE -- keep REL_BIAS on (relay still forms), ADD rotary q,k --
so sigma_min is comparable to currnp at matched recall. Question: does the orthogonal, norm-preserving
rotation shift conditioning at all when the task is still learned? (RoPE can't cap peaking, unlike QK-norm,
so the a-priori expectation is conditioning-neutral -> the honest test of "un-ruled-out" = "ruled neutral".)

Same recipe as curriculum_currnp (causal + NO_POSW + REL_BIAS + window curriculum) + sw.ROPE=True.
Saves checkpoints/currnproperbXX.pt (rope=True, rel_bias=True). Compare to currnp (no rope) and currnpqk.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.curriculum_currnp_rope_rb
"""
import os
import time

import torch
import experiments.sliding_window_reach as sw

sw.BIDIR = False
sw.REL_BIAS = True        # KEEP the learned relative bias (relay forms) -- RoPE is added on top
sw.READONLY_Q = True
sw.NO_POSW = True
sw.ROPE = True            # added on top of the working currnp recipe
sw.H, sw.dh = 4, sw.d // 4

PHASE_A = [(2, 400), (4, 400), (10, 600)]
PHASE_B = [0, 8, 16, 24, 40]
STEPS_B = 350
CKPT_DIR = "checkpoints"
PREFIX = "currnproperb"


def eval_stage(m, gap):
    m.eval()
    acc = sw.recall(m, gap, torch.Generator().manual_seed(123))
    r, smin, rs = m.spectrum(sw.gen_mqar(1, gap, torch.Generator().manual_seed(7))[0])
    m.train()
    return acc, r, smin, rs


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"device={sw.DEV}  currnp + RoPE ON TOP of rel-bias (relay preserved): does adding a norm-preserving\n"
          f"  rotation shift sigma_min vs currnp at MATCHED recall? same recipe + RoPE, rel-bias kept.\n",
          flush=True)
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
                    "bidir": False, "rel_bias": True, "readonly_q": True, "query_full": False,
                    "no_posw": True, "rope": True}, path)
        print(f"  [B] gap {g:>2}: recall={acc:.3f}  rho={r:.3f}  smin={smin:.3f}  resid={rs:.1e}  "
              f"-> {path}  ({time.time()-t0:.0f}s)", flush=True)
    print("\nDone. Compare currnproperb sigma_min ladder + recall to currnp (no rope) at matched recall "
          "-> did adding RoPE shift conditioning?", flush=True)


if __name__ == "__main__":
    main()
