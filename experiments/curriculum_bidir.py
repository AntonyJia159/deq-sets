"""BIDIRECTIONAL curriculum retrain v2 — the edit-regime substrate for the proper Faber-face C2.

v1 (all knobs, rel-bias, roq, untied controls) stuck at recall ~0.38 = the ONE-LAYER CEILING: the
two-hop bind-then-retrieve circuit never forms under a bidirectional mask at full width (probe rounds
1-5, checkpoints/probe_bidir_round*.txt). Round 6 found the rescue = WINDOW CURRICULUM: at w=2 the
binding hop is forced by connectivity (a value's band is just its key + the next key; rel-bias picks
left) and recall snaps to 1.0, surviving widening to w=10.

TWO PHASES:
  A) window curriculum at gap 0, QUERY_FULL on (queries read the whole doc — at w=2 banded queries
     couldn't reach anything and no retrieval gradient would flow). Forms the circuit.
  B) queries RE-BANDED (QUERY_FULL off, w=10) + gap curriculum 0->8->16->24->40. With full-attention
     queries the gap sweep would never need the equilibrium relay and couldn't build the rho-up /
     sigma_min-down conditioning range that made the causal checkpoints informative; re-banding
     restores relay pressure. Architecture keeps REL_BIAS (binding direction) and READONLY_Q
     (queries inject nothing — the context equilibrium we certify stays query-clean).

Saves checkpoints/bidirXX.pt per gap stage: state_dict + recall/rho/sigma_min/resid + mask flags.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.curriculum_bidir
"""
import os
import time

import torch
import experiments.sliding_window_reach as sw

sw.BIDIR = True
sw.REL_BIAS = True
sw.READONLY_Q = True
sw.H, sw.dh = 4, sw.d // 4

PHASE_A = [(2, 400), (4, 400), (10, 600)]    # (window, steps) at gap 0, QUERY_FULL on
PHASE_B = [0, 8, 16, 24, 40]                 # gap stages at w=10, QUERY_FULL off (relay pressure)
STEPS_B = 350
CKPT_DIR = "checkpoints"
PREFIX = "bidir"                             # checkpoint name prefix (wrappers override, e.g. "bidirnp")


def eval_stage(m, gap):
    m.eval()
    acc = sw.recall(m, gap, torch.Generator().manual_seed(123))
    r, smin, rs = m.spectrum(sw.gen_mqar(1, gap, torch.Generator().manual_seed(7))[0])
    m.train()
    return acc, r, smin, rs


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"device={sw.DEV}  BIDIR curriculum v2: phase A window {PHASE_A} (query-full) -> "
          f"phase B gaps {PHASE_B} @ w=10 (queries re-banded), {STEPS_B} steps/stage\n", flush=True)
    torch.manual_seed(0)
    sw.W = 10                                              # relb table sized at full width
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
                    "bidir": True, "rel_bias": True, "readonly_q": sw.READONLY_Q, "query_full": False,
                    "no_posw": sw.NO_POSW}, path)
        print(f"  [B] gap {g:>2}: recall={acc:.3f}  rho={r:.3f}  smin={smin:.3f}  resid={rs:.1e}  "
              f"-> {path}  ({time.time()-t0:.0f}s)", flush=True)
    print("\nDone. c2_bidir loads these for the Faber-face edit-locality measurement.", flush=True)


if __name__ == "__main__":
    main()
