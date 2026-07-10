"""Train a FACTORED bidirectional DEQ on SEGMENT AVERAGE -- learnability + conditioning + length-gen.

Real-valued factored substrate: h0 = [mode | value]; regression head over the value subspace; bidirectional
band + relative-position bias (the exp-decay locality kernel is a relative-position effect). Averaging is
CONTRACTIVE, so the expectation (vs the near-singular recall tasks) is WELL-CONDITIONED (sigma_min away from 0,
rho<1) -- the clean BVP/near-normal face. Piecewise-local -> should length-generalize (train short streams,
deploy long).

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.segment_average_train
"""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import experiments.sliding_window_reach as sw
from experiments.segment_average import gen_segment_average, DV

sw.FACTORED = True
sw.D_VALUE = DV
sw.BIDIR = True
sw.REL_BIAS = True
sw.READONLY_Q = False
sw.QUERY_FULL = False
sw.NO_POSW = True
sw.H, sw.dh = 4, sw.d // 4

CKPT_DIR = "checkpoints"
CKPT_NAME = "segment_average.pt"


def sa_eval(m, L, gen, reps=4, n_bnd=None):
    """MSE and relative error on value positions."""
    nb = max(2, L // 6) if n_bnd is None else n_bnd
    mses, rels = [], []
    for _ in range(reps):
        toks, values, target, tmask, _ = gen_segment_average(128, L, gen, n_bnd=nb)
        with torch.no_grad():
            pred = m.run(toks, values)
        d = (pred - target)[tmask]
        mses.append((d ** 2).mean().item())
        rels.append((d.norm() / (target[tmask].norm() + 1e-9)).item())
    return float(np.mean(mses)), float(np.mean(rels))


def sa_train(m, L, steps, gen, bs=64, lr=2e-3, n_bnd=None):
    nb = max(2, L // 6) if n_bnd is None else n_bnd
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    for st in range(steps):
        toks, values, target, tmask, _ = gen_segment_average(bs, L, gen, n_bnd=nb)
        m.train(); opt.zero_grad()
        pred = m.run(toks, values)
        loss = ((pred - target)[tmask] ** 2).mean()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        if torch.isfinite(loss) and torch.isfinite(gn):
            opt.step()
        else:
            opt.zero_grad()
        if st % 300 == 0:
            print(f"      step {st:>4} mse {loss.item():.4f}", flush=True)
    return loss.item()


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"device={sw.DEV}  segment average (factored [mode|value], d_value={DV}), bidirectional. Averaging\n"
          f"  is contractive -> expect WELL-conditioned. Length curriculum L=16->24->32, then generalize.\n",
          flush=True)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    sw.W = 10
    m = sw.SeqDEQ("softmax", "deq").to(sw.DEV)

    for L, steps in [(16, 700), (24, 700), (32, 800)]:
        t0 = time.time()
        sa_train(m, L, steps, gen)
        m.eval()
        mse, rel = sa_eval(m, L, gen)
        tk, vv = gen_segment_average(1, L, torch.Generator().manual_seed(7), n_bnd=max(2, L // 6))[:2]
        r, smin, rs = m.spectrum(tk, vv)
        m.train()
        print(f"  L={L:>2}: mse={mse:.4f} rel_err={rel:.3f}  rho={r:.3f} sigma_min={smin:.3f} resid={rs:.1e}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

    m.eval()
    print("\n  length generalization (trained L<=32):", flush=True)
    lg = {}
    for L in [24, 32, 48, 64]:
        mse, rel = sa_eval(m, L, gen)
        lg[L] = (mse, rel)
        print(f"     L={L:>2}: mse={mse:.4f} rel_err={rel:.3f}", flush=True)

    torch.save({"state_dict": m.state_dict(), "lengen": lg, "H": sw.H, "W": sw.W, "d_value": DV,
                "bidir": True, "rel_bias": True, "readonly_q": False, "query_full": False, "no_posw": True,
                "factored": True, "task": "segment_average"},
               os.path.join(CKPT_DIR, CKPT_NAME))
    print(f"\nSaved {CKPT_NAME}. READ: low rel_err + sigma_min away from 0 (rho<1) = learned + WELL-conditioned\n"
          "(the clean BVP face); holds at L=48/64 (untrained) = piecewise-local length generalization. Next:\n"
          "edit a value -> measured field response vs the segment tent AND the resolvent (I-J)^{-1}.", flush=True)


if __name__ == "__main__":
    main()
