"""CAUSAL GRAFTED matched-control -- the anchorless twin of the causal GRAFT anchor (currnpanchor60).

The causal graft anchor (curriculum_anchor60) warm-starts the trained currnp40 body (strict=False, fresh
anchor), then fine-tunes 600 steps with F_SWEEP=[40,60]. The existing anchorless currnp60 (curriculum_resume60)
is warm-from-40 too but used 350 steps / F_SWEEP=[60] -- NOT step/sweep-matched, so currnpanchor60-vs-currnp60
confounds the anchor with the differing fine-tune recipe.

This builds the matched control: IDENTICAL graft recipe (warm-start currnp40, 600 steps, F_SWEEP=[40,60]) with
ANCHOR OFF. Then currnpanchor60 (graft, +anchor) vs currnpgraftctl60 (graft, no anchor) differ ONLY in the
anchor -- the clean causal row for "graft the register onto a trained banded body." Causal only (bidir's pick
is the from-scratch canonical, not the graft).

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.curriculum_graftctl60
"""
import os
import time

import torch
import experiments.sliding_window_reach as sw

GAP = 60
STEPS = 600                        # matched to curriculum_anchor60
CKPT_DIR = "checkpoints"
SUBSTRATES = [("currnp", False)]   # causal only


def eval_stage(m, gap):
    m.eval()
    acc = sw.recall(m, gap, torch.Generator().manual_seed(123))
    r, smin, rs = m.spectrum(sw.gen_mqar(1, gap, torch.Generator().manual_seed(7))[0])
    m.train()
    return acc, r, smin, rs


def main():
    sw.REL_BIAS = True
    sw.READONLY_Q = True
    sw.NO_POSW = True
    sw.ANCHOR = False              # <-- the ONLY difference from curriculum_anchor60
    sw.H, sw.dh = 4, sw.d // 4
    print(f"device={sw.DEV}  CAUSAL grafted matched-control: warm currnp40 body, {STEPS} steps, "
          f"F_SWEEP=[40,{GAP}], NO anchor (twin of currnpanchor60).\n", flush=True)
    for prefix, bidir in SUBSTRATES:
        src = os.path.join(CKPT_DIR, f"{prefix}40.pt")
        if not os.path.exists(src):
            print(f"[{prefix}] {src} missing, skip"); continue
        ck = torch.load(src, map_location=sw.DEV, weights_only=False)
        sw.BIDIR = bidir
        sw.W = 10
        torch.manual_seed(0)
        m = sw.SeqDEQ("softmax", "deq").to(sw.DEV)
        m.load_state_dict(ck["state_dict"])          # strict: no anchor param -> clean full load
        print(f"  [{prefix}] loaded body from {prefix}40 (recall {ck['recall']:.3f})", flush=True)
        m.train()
        sw.QUERY_FULL = False
        sw.W = 10
        sw.F_SWEEP = [40, GAP]                        # matched ramp
        t0 = time.time()
        sw.train(m, steps=STEPS)
        acc, r, smin, rs = eval_stage(m, GAP)
        path = os.path.join(CKPT_DIR, f"{prefix}graftctl{GAP:02d}.pt")
        torch.save({"state_dict": m.state_dict(), "stage_gap": GAP, "recall": acc,
                    "rho": r, "sigma_min": smin, "resid": rs, "H": sw.H, "W": sw.W,
                    "bidir": bidir, "rel_bias": True, "readonly_q": True, "query_full": False,
                    "no_posw": True, "anchor": False, "resumed_from": f"{prefix}40.pt",
                    "graft_control": True}, path)
        print(f"  [{prefix}] gap {GAP} graft-control (no anchor): recall={acc:.3f}  rho={r:.3f}  "
              f"smin={smin:.4f}  resid={rs:.1e}  -> {path}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"\nCOMPARE to the causal graft anchor currnpanchor60: recall 0.801 / smin 0.016. If this control "
          f"(same recipe, no anchor) is lower on recall AND smin, the anchor is the cause -- the clean causal "
          f"graft row.", flush=True)


if __name__ == "__main__":
    main()
