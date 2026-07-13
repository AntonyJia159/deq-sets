"""FAST-PREVIEW resume stage: extend the currnp / bidirnp gap curriculum 40 -> 60 by REUSING the gap-40
checkpoint (skip re-deriving 0->40). This is a thin wrapper -- the canonical trainers (curriculum_currnp.py,
curriculum_bidir.py) are left untouched (PHASE_B still ends at 40). Because each PHASE_B stage is a fresh
sw.train call warm-continuing the previous weights, loading *40.pt and running ONE gap-60 stage reproduces
exactly what appending 60 to PHASE_B would do at that stage (modulo the RNG stream / optimizer re-init, which
is per-stage anyway). For the far-field re-run suite; NOT the canonical object -- rerun the full curriculum
with PHASE_B=[...,40,60] for that.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.curriculum_resume60
"""
import os
import time

import torch
import experiments.sliding_window_reach as sw

GAP = 60
STEPS = 350                         # canonical PHASE_B stage length (kept, so the stage matches the recipe)
CKPT_DIR = "checkpoints"

# (prefix, BIDIR) -- the two relative-PE substrates the far-field suite needs at gap 60
SUBSTRATES = [("currnp", False), ("bidirnp", True)]


def eval_stage(m, gap):
    m.eval()
    acc = sw.recall(m, gap, torch.Generator().manual_seed(123))
    r, smin, rs = m.spectrum(sw.gen_mqar(1, gap, torch.Generator().manual_seed(7))[0])
    m.train()
    return acc, r, smin, rs


def main():
    # shared relative-PE flags (identical across both trainers)
    sw.REL_BIAS = True
    sw.READONLY_Q = True
    sw.NO_POSW = True
    sw.H, sw.dh = 4, sw.d // 4
    print(f"device={sw.DEV}  RESUME-60 fast preview: reuse *40.pt -> one gap-{GAP} stage "
          f"({STEPS} steps), relative-PE substrates {[s[0] for s in SUBSTRATES]}\n", flush=True)
    for prefix, bidir in SUBSTRATES:
        src = os.path.join(CKPT_DIR, f"{prefix}40.pt")
        if not os.path.exists(src):
            print(f"[{prefix}] {src} missing, skip"); continue
        ck = torch.load(src, map_location=sw.DEV, weights_only=False)
        sw.BIDIR = bidir
        sw.W = 10                                   # relb table sized at full width BEFORE model init
        torch.manual_seed(0)
        m = sw.SeqDEQ("softmax", "deq").to(sw.DEV)
        m.load_state_dict(ck["state_dict"])         # warm start from the gap-40 weights
        m.train()
        sw.QUERY_FULL = False                       # phase-B regime (queries re-banded)
        sw.W = 10
        sw.F_SWEEP = [GAP]
        t0 = time.time()
        sw.train(m, steps=STEPS)
        acc, r, smin, rs = eval_stage(m, GAP)
        path = os.path.join(CKPT_DIR, f"{prefix}{GAP:02d}.pt")
        torch.save({"state_dict": m.state_dict(), "stage_gap": GAP, "recall": acc,
                    "rho": r, "sigma_min": smin, "resid": rs, "H": sw.H, "W": sw.W,
                    "bidir": bidir, "rel_bias": True, "readonly_q": True, "query_full": False,
                    "no_posw": True, "resumed_from": f"{prefix}40.pt"}, path)
        print(f"  [{prefix}] gap {GAP} (from {prefix}40, recall {ck['recall']:.3f}): recall={acc:.3f}  "
              f"rho={r:.3f}  smin={smin:.3f}  resid={rs:.1e}  -> {path}  ({time.time()-t0:.0f}s)", flush=True)
    print("\nDone. Preview gap-60 checkpoints (resumed). Watch recall vs the gap-40 values (sag => anchor "
          "token). For the canonical object, rerun the full curriculum with 60 appended to PHASE_B.", flush=True)


if __name__ == "__main__":
    main()
