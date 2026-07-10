"""Pointer-chase capacity diagnostic — does WIDTH lift the depth-3 ceiling?

C6 pointer-chase capped ~0.68 (N=8) at d=64/single-tied-layer: iterated associative lookup compounds
per-hop error. The memory-flagged suspect is per-hop CAPACITY, not directionality (causal≈bidir) or
aliasing (both ruled out). Cheapest test of "just capacity": double the width d 64->128 (dh 16->32),
everything else identical to pointer_chase_train, and see if depth-3/5 recall lifts toward learnable
(~0.9). Stays in the dense-J oracle regime (L~30, d=128 -> N~3840); ground-truth lanes intact. If width
alone rescues it, pointer-chase becomes the reader-set validation substrate (true multi-hop lanes, hubs);
if not, the next lever is a value->key MLP in the cell (associative memory), reported as the finding.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.pointer_chase_train_wide
"""
import os
import time

import numpy as np
import torch

import experiments.sliding_window_reach as sw

sw.d = 128                 # the one change: double the width (was 64)
sw.H, sw.dh = 4, sw.d // 4  # dh 16 -> 32

import experiments.pointer_chase_train as pct   # imports AFTER sw.d is set (model builds from sw.d)

pct.CKPT_NAME = "pcchase_wide.pt"


def main():
    os.makedirs(pct.CKPT_DIR, exist_ok=True)
    print(f"device={sw.DEV}  pointer-chase WIDTH diagnostic: d={sw.d} (dh={sw.dh}), N={sw.NKEY} nodes, "
          f"n_roots={pct.N_ROOTS}.\n  Same recipe as pointer_chase_train; does width lift depth-3 off the "
          f"0.68 ceiling?\n", flush=True)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    sw.W = 10
    m = sw.SeqDEQ("softmax", "deq").to(sw.DEV)

    sw.QUERY_FULL = True
    for w_stage, steps in [(2, 400), (4, 400), (10, 500)]:
        sw.W = w_stage
        t0 = time.time()
        pct.pc_train(m, depth=1, steps=steps, gen=gen)
        acc = pct.pc_recall(m, depth=1, gen=gen)
        print(f"  [A] w={w_stage:>2} depth=1: recall={acc:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    sw.QUERY_FULL = False
    sw.W = 10
    from experiments.pointer_chase import gen_pointer_chase
    for depth in [1, 2, 3, 5]:
        t0 = time.time()
        pct.pc_train(m, depth=depth, steps=700, gen=gen)
        m.eval()
        acc = pct.pc_recall(m, depth=depth, gen=gen)
        r, smin, rs = m.spectrum(gen_pointer_chase(1, 8, torch.Generator().manual_seed(7),
                                                   depth=depth, n_roots=pct.N_ROOTS)[0])
        m.train()
        print(f"  [B] depth={depth}: recall={acc:.3f}  rho={r:.3f}  sigma_min={smin:.3f}  resid={rs:.1e}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    m.eval()
    accs = {d: pct.pc_recall(m, d, gen) for d in [1, 2, 3, 5]}
    torch.save({"state_dict": m.state_dict(), "recall_by_depth": accs, "H": sw.H, "W": sw.W, "d": sw.d,
                "bidir": False, "rel_bias": True, "readonly_q": True, "query_full": False,
                "no_posw": True, "n_roots": pct.N_ROOTS, "task": "pointer_chase_root"},
               os.path.join(pct.CKPT_DIR, pct.CKPT_NAME))
    print(f"\nFinal recall by depth (d={sw.d}): {', '.join(f'd{d}={a:.3f}' for d, a in accs.items())}", flush=True)
    print("READ: compare to the d=64 ceiling (depth-3 ~0.68). Lift toward 0.9 => width was the limiter and\n"
          "pointer-chase is a viable reader-set substrate; flat => capacity needs an MLP, not just width.",
          flush=True)


if __name__ == "__main__":
    main()
