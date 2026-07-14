"""CANONICAL ANCHORLESS control -- the matched no-anchor twin of curriculum_anchor.

Isolates the ANCHOR effect. curriculum_anchor trains a global register FROM SCRATCH through the full
curriculum to gap 60; to attribute the gap-60 conditioning rescue (smin lifted off 0, recall up) to the
*anchor* and not to from-scratch-vs-graft or to the extra 60 stage, the control must be built the SAME way:
same two-phase window->gap recipe, same steps, same batch, PHASE_B extended to 60 -- only ANCHOR off.

Mirrors curriculum_anchor.py exactly with sw.ANCHOR=False. Writes currnpctl{00..60}.pt / bidirnpctl{00..60}.pt
under a distinct 'ctl' prefix so the established currnp*/bidirnp* (0..40) ladder is NOT clobbered. (The prior
currnp60/bidirnp60 from curriculum_resume60 were curriculum-continuations from *40.pt; this rebuilds the whole
ladder in one uninterrupted from-scratch run per substrate, so the anchor/anchorless pair is airtight.)

BATCH stays 64 to match the anchor canonical AND the rest of the paper's ladders (a bigger batch would confound
the anchor-vs-control comparison and island this pair off the bs=64 substrate tables). The GPU is heavily idle
on this workload (see experiments/PERF_NOTES.md), so the speed win is CONCURRENCY: pass a substrate arg and run
the two substrates as parallel processes.

Run BOTH (sequential):   D:\\deq-venv\\Scripts\\python.exe -u -m experiments.curriculum_anchorless
Run one (for parallel):  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.curriculum_anchorless currnp
                         D:\\deq-venv\\Scripts\\python.exe -u -m experiments.curriculum_anchorless bidir
"""
import os
import sys
import time

import torch
import experiments.sliding_window_reach as sw

sw.REL_BIAS = True
sw.READONLY_Q = True
sw.NO_POSW = True
sw.ANCHOR = False          # <-- the ONLY difference from curriculum_anchor
sw.H, sw.dh = 4, sw.d // 4

PHASE_A = [(2, 400), (4, 400), (10, 600)]          # identical recipe to curriculum_anchor
PHASE_B = [0, 8, 16, 24, 40, 60]
STEPS_B = 350
CKPT_DIR = "checkpoints"
ALL_SUBSTRATES = [("currnpctl", False), ("bidirnpctl", True)]


def eval_stage(m, gap):
    m.eval()
    acc = sw.recall(m, gap, torch.Generator().manual_seed(123))
    r, smin, rs = m.spectrum(sw.gen_mqar(1, gap, torch.Generator().manual_seed(7))[0])
    m.train()
    return acc, r, smin, rs


def train_one(prefix, bidir):
    sw.BIDIR = bidir
    print(f"\n=== [{prefix}] {'BIDIR' if bidir else 'CAUSAL'} + relative PE, NO anchor (control, from scratch) ===",
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
        path = os.path.join(CKPT_DIR, f"{prefix}{g:02d}.pt")
        torch.save({"state_dict": m.state_dict(), "stage_gap": g, "recall": acc,
                    "rho": r, "sigma_min": smin, "resid": rs, "H": sw.H, "W": sw.W,
                    "bidir": bidir, "rel_bias": True, "readonly_q": True, "query_full": False,
                    "no_posw": True, "anchor": False, "canonical": True}, path)
        print(f"  [B] gap {g:>2}: recall={acc:.3f}  rho={r:.3f}  smin={smin:.4f}  resid={rs:.1e}  "
              f"-> {path}  ({time.time()-t0:.0f}s)", flush=True)


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    sel = sys.argv[1].lower() if len(sys.argv) > 1 else None
    subs = ALL_SUBSTRATES if sel is None else [(p, b) for p, b in ALL_SUBSTRATES if sel in p]
    if not subs:
        print(f"no substrate matches '{sel}' (use currnp|bidir)"); return
    print(f"device={sw.DEV}  ANCHORLESS control (matched twin of curriculum_anchor): window {PHASE_A} -> "
          f"gaps {PHASE_B} @ w=10, {STEPS_B} steps/stage, bs=64, NO anchor.\n  substrates {[s[0] for s in subs]}",
          flush=True)
    t_all = time.time()
    for prefix, bidir in subs:
        train_one(prefix, bidir)
    print(f"\nDone ({(time.time()-t_all)/60:.1f} min). COMPARE gap-60 to curriculum_anchor: does the anchor lift "
          f"smin off ~0 and recall above these control values? That is the isolated anchor effect.", flush=True)


if __name__ == "__main__":
    main()
