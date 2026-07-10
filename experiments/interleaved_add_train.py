"""Train a relative-PE causal DEQ on INTERLEAVED MODULAR ADDITION -- learnability + length-gen check.

Question 1 (learnability): can a windowed equilibrium relay learn to accumulate two parity-separated
running sums (mod P) and read them out? Summation is HARD for softmax attention (it averages, not sums),
so this is a genuine expressivity test -- the honest outcome is either "relay-accumulation forms" or "a
within-window averaging shortcut caps it."
Question 2 (length-gen): train on SHORT chains (k<=k_train), then evaluate on LONGER ones (k>k_train). If
the model learned relay-accumulation (not a global-average shortcut) it should generalize -- and that is
the mechanism the edit-locality-is-length-invariant demo needs.

Recipe = currnp (relative PE: NO_POSW+REL_BIAS, causal, READONLY_Q) + window curriculum (form the local
sum) + LENGTH curriculum (k grows past the window so accumulation must relay across windows). Saves
checkpoints/interleaved_add.pt with substrate flags + task metadata.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.interleaved_add_train
"""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import experiments.sliding_window_reach as sw
from experiments.interleaved_add import gen_interleaved, P

sw.BIDIR = False
sw.REL_BIAS = True
sw.READONLY_Q = True
sw.NO_POSW = True
sw.H, sw.dh = 4, sw.d // 4

FILL_SWEEP = [0, 4, 8]
CKPT_DIR = "checkpoints"
CKPT_NAME = "interleaved_add.pt"


def ia_recall(model, k, gen, reps=4, fill=None):
    """Per-reader (even/odd) and joint accuracy over reps batches."""
    ev, od = [], []
    for _ in range(reps):
        fl = FILL_SWEEP[-1] if fill is None else fill
        toks, qmask, targ, _ = gen_interleaved(128, k, gen, fill=fl)
        with torch.no_grad():
            pred = model.run(toks).argmax(-1)
        qpos = qmask[0].nonzero().flatten().tolist()      # [Qe, Qo] positions (same for the batch)
        ev.append((pred[:, qpos[0]] == targ[:, qpos[0]]).float().mean().item())
        od.append((pred[:, qpos[1]] == targ[:, qpos[1]]).float().mean().item())
    return float(np.mean(ev)), float(np.mean(od))


def ia_train(model, k, steps, gen, bs=64, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    for st in range(steps):
        Fill = FILL_SWEEP[torch.randint(len(FILL_SWEEP), (1,), generator=gen).item()]
        toks, qmask, targ, _ = gen_interleaved(bs, k, gen, fill=Fill)
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
    print(f"device={sw.DEV}  interleaved modular addition (P={P}), relative-PE causal. Window curriculum\n"
          f"  (form the local sum) then LENGTH curriculum k=4->8->12->16 (accumulation must relay across\n"
          f"  windows). Can the equilibrium accumulate two parity-separated running sums?\n", flush=True)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    sw.W = 10
    m = sw.SeqDEQ("softmax", "deq").to(sw.DEV)

    # Phase A: window curriculum at k=4 (form the local parity-sum), queries read full context.
    sw.QUERY_FULL = True
    for w_stage, steps in [(2, 500), (4, 500), (10, 600)]:
        sw.W = w_stage
        t0 = time.time()
        ia_train(m, k=4, steps=steps, gen=gen)
        ev, od = ia_recall(m, k=4, gen=gen, fill=0)
        print(f"  [A] w={w_stage:>2} k=4: even={ev:.3f} odd={od:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    # Phase B: length curriculum, queries re-banded (accumulation must relay).
    sw.QUERY_FULL = False
    sw.W = 10
    for k in [4, 8, 12, 16]:
        t0 = time.time()
        ia_train(m, k=k, steps=800, gen=gen)
        m.eval()
        ev, od = ia_recall(m, k=k, gen=gen)
        r, smin, rs = m.spectrum(gen_interleaved(1, k, torch.Generator().manual_seed(7), fill=8)[0])
        m.train()
        print(f"  [B] k={k:>2}: even={ev:.3f} odd={od:.3f}  rho={r:.3f}  sigma_min={smin:.3f}  "
              f"resid={rs:.1e}  ({time.time()-t0:.0f}s)", flush=True)

    # Length generalization: trained up to k=16, evaluate BEYOND.
    m.eval()
    print("\n  length generalization (trained k<=16):", flush=True)
    lg = {}
    for k in [8, 16, 24, 32]:
        ev, od = ia_recall(m, k=k, gen=gen)
        lg[k] = (ev, od)
        print(f"     k={k:>2}: even={ev:.3f} odd={od:.3f}", flush=True)

    torch.save({"state_dict": m.state_dict(), "lengen": lg, "H": sw.H, "W": sw.W, "P": P,
                "bidir": False, "rel_bias": True, "readonly_q": True, "query_full": False,
                "no_posw": True, "task": "interleaved_add"},
               os.path.join(CKPT_DIR, CKPT_NAME))
    print(f"\nSaved {CKPT_NAME}. READ: even/odd both high through k=16 => relay-accumulation formed;\n"
          "holds at k=24/32 (untrained) => length generalization (relay, not a global-average shortcut) =>\n"
          "the substrate for the edit-locality-is-length-invariant demo. A cliff at k>window => the honest\n"
          "accumulation-relay depth limit.", flush=True)


if __name__ == "__main__":
    main()
