"""currnp + QK-NORM (cosine attention) — the conditioning fix test.

currnp (causal + relative PE) was 2-8x MORE ill-conditioned than curr (absolute PE): the peaking<->contraction
tension (near-singular I-J <=> saturated attention) bites harder without an absolute anchor. QK-norm decouples
attention SHARPNESS (a learned per-head temperature tau) from logit MAGNITUDE (unbounded q.k): unit-normalize
q,k, scale by tau. Hypothesis: sigma_min UP (less saturation) with recall preserved (tau still peaks).

Identical recipe to curriculum_currnp (causal + NO_POSW + REL_BIAS + window curriculum) with sw.QK_NORM=True.
Saves checkpoints/currnpqkXX.pt (qk_norm=True). Compare the sigma_min ladder + recall to currnp* to see if it
helped.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.curriculum_currnp_qk
"""
import os
import time

import torch
import experiments.sliding_window_reach as sw

sw.BIDIR = False
sw.REL_BIAS = True
sw.READONLY_Q = True
sw.NO_POSW = True
sw.QK_NORM = True         # the one change vs curriculum_currnp
sw.H, sw.dh = 4, sw.d // 4

PHASE_A = [(2, 400), (4, 400), (10, 600)]
PHASE_B = [0, 8, 16, 24, 40]
STEPS_B = 350
CKPT_DIR = "checkpoints"
PREFIX = "currnpqk"


def eval_stage(m, gap):
    m.eval()
    acc = sw.recall(m, gap, torch.Generator().manual_seed(123))
    r, smin, rs = m.spectrum(sw.gen_mqar(1, gap, torch.Generator().manual_seed(7))[0])
    m.train()
    return acc, r, smin, rs


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"device={sw.DEV}  currnp + QK-NORM (cosine attention, learned per-head tau): does decoupling "
          f"sharpness\n  from magnitude lift sigma_min vs currnp (was 2-8x more ill-conditioned)? "
          f"same recipe + QK_NORM.\n", flush=True)
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
        tau = m.qk_tau.detach().cpu().tolist()
        print(f"  [A] w={w_stage:>2}: recall={acc:.3f}  rho={r:.3f}  smin={smin:.3f}  resid={rs:.1e}  "
              f"tau={[round(t, 1) for t in tau]}  ({time.time()-t0:.0f}s)", flush=True)

    sw.QUERY_FULL = False
    sw.W = 10
    for g in PHASE_B:
        sw.F_SWEEP = [g]
        t0 = time.time()
        sw.train(m, steps=STEPS_B)
        acc, r, smin, rs = eval_stage(m, g)
        tau = m.qk_tau.detach().cpu().tolist()
        path = os.path.join(CKPT_DIR, f"{PREFIX}{g:02d}.pt")
        torch.save({"state_dict": m.state_dict(), "stage_gap": g, "recall": acc,
                    "rho": r, "sigma_min": smin, "resid": rs, "H": sw.H, "W": sw.W,
                    "bidir": False, "rel_bias": True, "readonly_q": True, "query_full": False,
                    "no_posw": True, "qk_norm": True}, path)
        print(f"  [B] gap {g:>2}: recall={acc:.3f}  rho={r:.3f}  smin={smin:.3f}  resid={rs:.1e}  "
              f"tau={[round(t, 1) for t in tau]}  -> {path}  ({time.time()-t0:.0f}s)", flush=True)
    print("\nDone. Compare currnpqk sigma_min ladder + recall to currnp -> did QK-norm help conditioning?",
          flush=True)


if __name__ == "__main__":
    main()
