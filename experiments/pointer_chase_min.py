"""MINIMAL pointer-chase-to-root -- capacity test. Is the ~0.68 ceiling task difficulty or model capacity?

Shrink the task hard: N=4 nodes, single root, depth<=2, short curriculum. If recall jumps to ~0.9+, the
minimal DEQ CAN do content-based chase and the earlier ceiling was task difficulty (usable smaller C6 substrate).
If it ALSO caps ~0.7, the minimal cell genuinely can't chase -> C6-via-pointer-chase is a dead end at this scale.
Causal relative PE (direction shown not to matter).

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.pointer_chase_min
"""
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

NN, NR = 6, 2                      # 6 nodes, 2 roots (>=2 roots is essential: else the answer is always the
#   single root = "find the root", trivial, no chasing. 2 roots -> answer depends on WHICH tree start falls in.)
FILL = [0, 2, 4]


def rec(m, depth, gen, reps=6):
    a = []
    for _ in range(reps):
        toks, qmask, targ, _, _ = gen_pointer_chase(128, FILL[-1], gen, depth=depth, n_roots=NR, n_nodes=NN)
        with torch.no_grad():
            a.append((m.run(toks).argmax(-1)[qmask] == targ[qmask]).float().mean().item())
    return float(np.mean(a))


def tr(m, depth, steps, gen, bs=64):
    opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(steps):
        Fill = FILL[torch.randint(len(FILL), (1,), generator=gen).item()]
        toks, qmask, targ, _, _ = gen_pointer_chase(bs, Fill, gen, depth=depth, n_roots=NR, n_nodes=NN)
        m.train(); opt.zero_grad()
        loss = F.cross_entropy(m.run(toks)[qmask], targ[qmask])
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        if torch.isfinite(loss) and torch.isfinite(gn):
            opt.step()
    return loss.item()


def main():
    print(f"device={sw.DEV}  MINIMAL pointer-chase: N={NN} nodes, {NR} root, depth<=2. Capacity vs difficulty?\n",
          flush=True)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    sw.W = 10
    m = sw.SeqDEQ("softmax", "deq").to(sw.DEV)
    sw.QUERY_FULL = True
    for w, s in [(2, 300), (4, 300), (10, 400)]:
        sw.W = w
        t0 = time.time()
        tr(m, 1, s, gen)
        print(f"  [A] w={w:>2} depth=1: recall={rec(m, 1, gen):.3f}  ({time.time()-t0:.0f}s)", flush=True)
    sw.QUERY_FULL = False
    sw.W = 10
    for d in [1, 2, 3]:
        t0 = time.time()
        tr(m, d, 600, gen)
        print(f"  [B] depth={d}: recall={rec(m, d, gen):.3f}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"\nVERDICT: depth-2 recall high (~0.9+) => capacity was the ceiling, smaller substrate usable.\n"
          f"         still ~0.7 => minimal cell can't chase; C6-via-pointer-chase dead-ends at this scale.", flush=True)


if __name__ == "__main__":
    main()
