"""ANCHOR gap-60 preview: add a global register token (sw.ANCHOR) and retrain the long-gap relay, to see
whether the hub-and-spoke shortcut keeps sigma_min off 0 and recall up where the pure-banded gap-60 cratered
(currnp 0.81->0.70 smin~0; bidirnp 0.82->0.46 smin~0). Practice-outruns-theory try: the anchor breaks strict
block-bandedness (rank-d border), so the far-field/early-stop machinery may degrade -- but maybe emergent
filtering keeps it usable. Warm-start the gap-40 BODY (strict=False -> fresh anchor), light 40/60 ramp so the
register forms without forgetting. Saves {prefix}anchor60.pt.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.curriculum_anchor60
"""
import os
import time

import torch
import experiments.sliding_window_reach as sw

GAP = 60
STEPS = 600
CKPT_DIR = "checkpoints"
SUBSTRATES = [("currnp", False), ("bidirnp", True)]


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
    sw.ANCHOR = True                                    # <-- the register token
    sw.H, sw.dh = 4, sw.d // 4
    print(f"device={sw.DEV}  ANCHOR gap-{GAP} preview: global register + relative-PE, warm from *40 body "
          f"(strict=False), {STEPS} steps, F_SWEEP=[40,{GAP}] ramp\n", flush=True)
    for prefix, bidir in SUBSTRATES:
        src = os.path.join(CKPT_DIR, f"{prefix}40.pt")
        if not os.path.exists(src):
            print(f"[{prefix}] {src} missing, skip"); continue
        ck = torch.load(src, map_location=sw.DEV, weights_only=False)
        sw.BIDIR = bidir
        sw.W = 10
        torch.manual_seed(0)
        m = sw.SeqDEQ("softmax", "deq").to(sw.DEV)
        missing, unexpected = m.load_state_dict(ck["state_dict"], strict=False)   # anchor = missing (fresh)
        print(f"  [{prefix}] loaded body from {prefix}40 (recall {ck['recall']:.3f}); "
              f"fresh params: {missing}", flush=True)
        m.train()
        sw.QUERY_FULL = False
        sw.W = 10
        sw.F_SWEEP = [40, GAP]                          # light ramp: keep gap-40 while forming the anchor for 60
        t0 = time.time()
        sw.train(m, steps=STEPS)
        acc, r, smin, rs = eval_stage(m, GAP)
        path = os.path.join(CKPT_DIR, f"{prefix}anchor{GAP:02d}.pt")
        torch.save({"state_dict": m.state_dict(), "stage_gap": GAP, "recall": acc,
                    "rho": r, "sigma_min": smin, "resid": rs, "H": sw.H, "W": sw.W,
                    "bidir": bidir, "rel_bias": True, "readonly_q": True, "query_full": False,
                    "no_posw": True, "anchor": True, "resumed_from": f"{prefix}40.pt"}, path)
        print(f"  [{prefix}] gap {GAP} + ANCHOR: recall={acc:.3f}  rho={r:.3f}  smin={smin:.4f}  resid={rs:.1e}"
              f"  -> {path}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"\nCOMPARE to no-anchor gap-60: currnp 0.704/smin~0.000, bidirnp 0.462/smin~0.001.\n"
          f"Anchor WINS if recall up AND smin lifted off 0 (hub carries the relay -> body better conditioned).\n"
          f"Then the far-field suite runs on the banded BODY + a certifiable rank-d anchor border.", flush=True)


if __name__ == "__main__":
    main()
