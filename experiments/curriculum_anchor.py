"""CANONICAL anchor curriculum -- the load-bearing global-register object (replaces the *anchor60 GRAFT preview).

The graft (curriculum_anchor60) bolted a FRESH anchor onto a settled gap-40 body (strict=False, 600 steps): it
proved the hub rescues the near-singular long-gap relay (currnp 0.70->0.80 smin~0->0.016; bidirnp 0.46->0.75
smin~0.0001) but is path-dependent -- a channel wired into weights that already chose a windowed solution, not a
body co-adapted to the register. This trains the register FROM SCRATCH with sw.ANCHOR=True through the full
recipe, so the anchor is present from step 0 and the body learns to route through it.

Mirrors curriculum_currnp / curriculum_bidir EXACTLY (same two-phase window->gap curriculum, same steps), the
only changes: ANCHOR on from init, and PHASE_B extended 40->60 (the gap where pure-banded cratered). Saves the
full ladder {prefix}anchor{00,08,16,24,40,60}.pt -- a path-independent object safe to make load-bearing for the
far-field suite and note 11 folding. The graft 60s are preserved as *anchor60_graft.pt for a graft-vs-canonical
comparison.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.curriculum_anchor
"""
import os
import time

import torch
import experiments.sliding_window_reach as sw

sw.REL_BIAS = True         # relative position -- the proven relay recipe
sw.READONLY_Q = True       # query-clean context equilibrium
sw.NO_POSW = True          # no absolute embedding (shift-invariant, insert-ready)
sw.ANCHOR = True           # <-- the global register token, present from step 0
sw.H, sw.dh = 4, sw.d // 4

PHASE_A = [(2, 400), (4, 400), (10, 600)]          # (window, steps) at gap 0, QUERY_FULL on -> forms the hop
PHASE_B = [0, 8, 16, 24, 40, 60]                   # gap stages at w=10, queries re-banded (relay pressure); +60
STEPS_B = 350
CKPT_DIR = "checkpoints"
SUBSTRATES = [("currnpanchor", False), ("bidirnpanchor", True)]


def eval_stage(m, gap):
    m.eval()
    acc = sw.recall(m, gap, torch.Generator().manual_seed(123))
    r, smin, rs = m.spectrum(sw.gen_mqar(1, gap, torch.Generator().manual_seed(7))[0])
    m.train()
    return acc, r, smin, rs


def train_one(prefix, bidir):
    sw.BIDIR = bidir
    print(f"\n=== [{prefix}] {'BIDIR' if bidir else 'CAUSAL'} + relative PE + ANCHOR (from scratch) ===",
          flush=True)
    torch.manual_seed(0)
    sw.W = 10                                          # relb table sized at full width before model init
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
                    "no_posw": True, "anchor": True, "canonical": True}, path)
        print(f"  [B] gap {g:>2}: recall={acc:.3f}  rho={r:.3f}  smin={smin:.4f}  resid={rs:.1e}  "
              f"-> {path}  ({time.time()-t0:.0f}s)", flush=True)


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"device={sw.DEV}  CANONICAL ANCHOR curriculum: phase A window {PHASE_A} (query-full) -> "
          f"phase B gaps {PHASE_B} @ w=10 (re-banded), {STEPS_B} steps/stage, ANCHOR from scratch.\n"
          f"  substrates {[s[0] for s in SUBSTRATES]}. Full ladder = path-independent load-bearing object.",
          flush=True)
    t_all = time.time()
    for prefix, bidir in SUBSTRATES:
        train_one(prefix, bidir)
    print(f"\nDone ({(time.time()-t_all)/60:.1f} min). Canonical anchor ladder written. COMPARE the gap-60 to the "
          f"graft (*anchor60_graft.pt): currnp 0.80/smin~0.016, bidirnp 0.75/smin~1e-4. Canonical WINS if the "
          f"co-adapted body reaches >= graft recall AND lifts smin further (register load-bearing, not bolted on).",
          flush=True)


if __name__ == "__main__":
    main()
