"""PURE-RELATIVE-PE CAUSAL curriculum (currnp) — the PE-decision experiment + the missing insert/delete cell.

curr* (causal) trained with ABSOLUTE PE via direct stages. This trains the CAUSAL relay with ONLY relative
position (NO_POSW + REL_BIAS), using the two-phase WINDOW curriculum proven necessary under relative PE for
the bidir relay (a full-width relative relay stalls at the one-layer ceiling ~0.38; w=2 forces the binding
hop, then widen). THE QUESTION: does causal+relative reach conditioning COMPARABLE to curr* (the sigma_min
ladder 0.18->0.03, recall down to ~0.83)?
  - yes  -> promote relative PE to the PRIMARY substrate; demote absolute to a pedagogical first experiment.
  - stalls at gap>0 -> the causal relay needs an absolute coordinate; keep absolute PE primary (honest,
    PE-agnostic-certificate framing + aligned-frame for structural edits).
Also fills the causal+relative cell missing from c2_insertdelete (curr=causal+abs, bidir/bidirnp done).

Mirrors curriculum_bidir's recipe with sw.BIDIR=False and CORRECT metadata (bidir=False). Saves currnpXX.pt.
Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.curriculum_currnp
"""
import os
import time

import torch
import experiments.sliding_window_reach as sw

sw.BIDIR = False          # CAUSAL (the one difference from bidirnp)
sw.REL_BIAS = True        # relative position -- the ONLY position signal once NO_POSW is on
sw.READONLY_Q = True      # query-clean context equilibrium (the proven relative recipe)
sw.NO_POSW = True         # no absolute embedding
sw.H, sw.dh = 4, sw.d // 4

PHASE_A = [(2, 400), (4, 400), (10, 600)]     # (window, steps) at gap 0, QUERY_FULL on -> forms the hop
PHASE_B = [0, 8, 16, 24, 40]                  # gap stages at w=10, queries re-banded (relay pressure)
STEPS_B = 350
CKPT_DIR = "checkpoints"
PREFIX = "currnp"


def eval_stage(m, gap):
    m.eval()
    acc = sw.recall(m, gap, torch.Generator().manual_seed(123))
    r, smin, rs = m.spectrum(sw.gen_mqar(1, gap, torch.Generator().manual_seed(7))[0])
    m.train()
    return acc, r, smin, rs


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"device={sw.DEV}  currnp (CAUSAL + pure relative PE) curriculum: phase A window {PHASE_A} "
          f"(query-full)\n  -> phase B gaps {PHASE_B} @ w=10 (queries re-banded), {STEPS_B} steps/stage. "
          f"Q: match curr* conditioning?\n", flush=True)
    torch.manual_seed(0)
    sw.W = 10                                              # relb table sized at full width before model init
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
                    "no_posw": True}, path)
        print(f"  [B] gap {g:>2}: recall={acc:.3f}  rho={r:.3f}  smin={smin:.3f}  resid={rs:.1e}  "
              f"-> {path}  ({time.time()-t0:.0f}s)", flush=True)
    print("\nDone. currnp = causal + relative PE. Compare the sigma_min ladder + recall to curr* -> PE decision.",
          flush=True)


if __name__ == "__main__":
    main()
