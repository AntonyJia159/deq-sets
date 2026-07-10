"""Train a relative-PE causal DEQ on COLORED REGISTER RECALL -- learnability + length-gen check.

Repeated-key associative recall with a relabeled readout: select the latest same-color value (induction),
output T[value] (fixed permutation). Expected to learn on the SAME pure-attention substrate as MQAR (no MLP:
induction = attention, the permutation T = a relabeling the linear head absorbs). If so, the substrate stays
consistent with the rest of the paper and we get the striped-edit-response validation for free.

Recipe = currnp (relative PE: NO_POSW+REL_BIAS, causal, READONLY_Q) + window curriculum (form the induction
relay) + LENGTH curriculum (n_items grows past the window so the latest-write relay must cross windows).
Saves checkpoints/colored_recall.pt with substrate flags + task metadata.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.colored_recall_train
"""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import experiments.sliding_window_reach as sw
from experiments.colored_recall import gen_colored_recall, C, V

sw.BIDIR = False
sw.REL_BIAS = True
sw.READONLY_Q = True
sw.NO_POSW = True
sw.MLP = False                        # pure attention -- selection + relabel needs no per-position nonlinearity
sw.H, sw.dh = 4, sw.d // 4

FILL_SWEEP = [0, 4, 8]
CKPT_DIR = "checkpoints"
CKPT_NAME = "colored_recall.pt"


def cr_recall(model, n_items, gen, reps=4, fill=None):
    accs = []
    for _ in range(reps):
        fl = FILL_SWEEP[-1] if fill is None else fill
        toks, qmask, targ, _ = gen_colored_recall(128, n_items, gen, fill=fl)
        with torch.no_grad():
            pred = model.run(toks).argmax(-1)
        accs.append((pred[qmask] == targ[qmask]).float().mean().item())
    return float(np.mean(accs))


def cr_train(model, n_items, steps, gen, bs=64, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    for st in range(steps):
        Fill = FILL_SWEEP[torch.randint(len(FILL_SWEEP), (1,), generator=gen).item()]
        toks, qmask, targ, _ = gen_colored_recall(bs, n_items, gen, fill=Fill)
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
    print(f"device={sw.DEV}  colored register recall (C={C} colors, V={V} values, fixed transform), "
          f"relative-PE causal.\n  Window curriculum then LENGTH curriculum n_items=4->8->12->16 (latest-write "
          f"relay across windows).\n", flush=True)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    sw.W = 10
    m = sw.SeqDEQ("softmax", "deq").to(sw.DEV)

    sw.QUERY_FULL = True
    for w_stage, steps in [(2, 400), (4, 400), (10, 500)]:
        sw.W = w_stage
        t0 = time.time()
        cr_train(m, n_items=4, steps=steps, gen=gen)
        acc = cr_recall(m, n_items=4, gen=gen, fill=0)
        print(f"  [A] w={w_stage:>2} n=4: recall={acc:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    sw.QUERY_FULL = False
    sw.W = 10
    for n in [4, 8, 12, 16]:
        t0 = time.time()
        cr_train(m, n_items=n, steps=700, gen=gen)
        m.eval()
        acc = cr_recall(m, n_items=n, gen=gen)
        r, smin, rs = m.spectrum(gen_colored_recall(1, n, torch.Generator().manual_seed(7), fill=8)[0])
        m.train()
        print(f"  [B] n={n:>2}: recall={acc:.3f}  rho={r:.3f}  sigma_min={smin:.3f}  resid={rs:.1e}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    m.eval()
    print("\n  length generalization (trained n<=16):", flush=True)
    lg = {}
    for n in [8, 16, 24, 32]:
        acc = cr_recall(m, n_items=n, gen=gen)
        lg[n] = acc
        print(f"     n={n:>2}: recall={acc:.3f}", flush=True)

    torch.save({"state_dict": m.state_dict(), "lengen": lg, "H": sw.H, "W": sw.W, "C": C, "V": V,
                "bidir": False, "rel_bias": True, "readonly_q": True, "query_full": False,
                "no_posw": True, "mlp": False, "task": "colored_recall"},
               os.path.join(CKPT_DIR, CKPT_NAME))
    print(f"\nSaved {CKPT_NAME}. READ: recall high through n=16 => induction+relabel relay formed; holds at\n"
          "n=24/32 (untrained) => length generalization => substrate for the striped reader-set / reach demo.",
          flush=True)


if __name__ == "__main__":
    main()
