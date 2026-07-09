"""Train a relative-PE causal DEQ on POINTER-CHASE-TO-ROOT -- the C6 viability check.

Question: can a windowed equilibrium relay learn fixed-point root-label propagation (chase pointers to the
terminal), and how does accuracy scale with chase DEPTH? This is the multi-hop generalization of the MQAR
relay -- each hop is one more round of "copy my pointer-target's root label", which the equilibrium iterates
for free (the C1 reach story). If depth>=3 holds, we have the C6 substrate; if it stalls, that is an honest
limit on equilibrium relay depth.

Recipe = currnp (relative PE: NO_POSW+REL_BIAS, causal, READONLY_Q) + window curriculum (relay formation at
depth 1) + DEPTH curriculum (1->2->3->5). Saves checkpoints/pcchase.pt (final) with substrate flags.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.pointer_chase_train
"""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import experiments.sliding_window_reach as sw
from experiments.pointer_chase import gen_pointer_chase

sw.BIDIR = False
sw.REL_BIAS = True
sw.READONLY_Q = True
sw.NO_POSW = True
sw.H, sw.dh = 4, sw.d // 4

N_ROOTS = 2
FILL_SWEEP = [0, 4, 8]
CKPT_DIR = "checkpoints"
CKPT_NAME = "pcchase.pt"        # a bidir wrapper overrides this + sets sw.BIDIR=True


def pc_recall(model, depth, gen, reps=4, n_roots=N_ROOTS):
    accs = []
    for _ in range(reps):
        toks, qmask, targ, _, _ = gen_pointer_chase(128, FILL_SWEEP[-1], gen, depth=depth, n_roots=n_roots)
        with torch.no_grad():
            logits = model.run(toks)
        accs.append((logits.argmax(-1)[qmask] == targ[qmask]).float().mean().item())
    return float(np.mean(accs))


def pc_train(model, depth, steps, gen, bs=64, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    for st in range(steps):
        Fill = FILL_SWEEP[torch.randint(len(FILL_SWEEP), (1,), generator=gen).item()]
        toks, qmask, targ, _, _ = gen_pointer_chase(bs, Fill, gen, depth=depth, n_roots=N_ROOTS)
        model.train(); opt.zero_grad()
        logits = model.run(toks)
        loss = F.cross_entropy(logits[qmask], targ[qmask])
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if torch.isfinite(loss) and torch.isfinite(gn):
            opt.step()
        else:
            opt.zero_grad()
        if st % 300 == 0:
            print(f"      step {st:>4} loss {loss.item():.3f}", flush=True)
    return loss.item()


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"device={sw.DEV}  pointer-chase-to-root, relative PE causal. Window curriculum (relay) at depth 1,\n"
          f"  then DEPTH curriculum 1->2->3->5 (n_roots={N_ROOTS}). Can the equilibrium chase to the root?\n",
          flush=True)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    sw.W = 10
    m = sw.SeqDEQ("softmax", "deq").to(sw.DEV)

    # Phase A: window curriculum at depth 1 (form the lookup relay); queries read full context.
    sw.QUERY_FULL = True
    for w_stage, steps in [(2, 400), (4, 400), (10, 500)]:
        sw.W = w_stage
        t0 = time.time()
        pc_train(m, depth=1, steps=steps, gen=gen)
        acc = pc_recall(m, depth=1, gen=gen)
        print(f"  [A] w={w_stage:>2} depth=1: recall={acc:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    # Phase B: depth curriculum, queries re-banded (relay pressure).
    sw.QUERY_FULL = False
    sw.W = 10
    for depth in [1, 2, 3, 5]:
        t0 = time.time()
        pc_train(m, depth=depth, steps=700, gen=gen)
        m.eval()
        acc = pc_recall(m, depth=depth, gen=gen)
        r, smin, rs = m.spectrum(gen_pointer_chase(1, 8, torch.Generator().manual_seed(7),
                                                    depth=depth, n_roots=N_ROOTS)[0])
        m.train()
        print(f"  [B] depth={depth}: recall={acc:.3f}  rho={r:.3f}  sigma_min={smin:.3f}  resid={rs:.1e}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    m.eval()
    accs = {d: pc_recall(m, d, gen) for d in [1, 2, 3, 5]}
    torch.save({"state_dict": m.state_dict(), "recall_by_depth": accs, "H": sw.H, "W": sw.W,
                "bidir": sw.BIDIR, "rel_bias": True, "readonly_q": True, "query_full": False,
                "no_posw": True, "n_roots": N_ROOTS, "task": "pointer_chase_root"},
               os.path.join(CKPT_DIR, CKPT_NAME))
    print(f"\nFinal recall by depth (same model): {', '.join(f'd{d}={a:.3f}' for d, a in accs.items())}", flush=True)
    print("READ: recall stays high as depth grows -> the equilibrium chases to the root at increasing depth (C6\n"
          "substrate viable). A cliff at some depth = the honest equilibrium relay-depth limit.", flush=True)


if __name__ == "__main__":
    main()
