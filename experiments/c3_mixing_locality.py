"""C3 — the mixing<->locality tradeoff. The window w is a single dial with opposite effects on two costs:
  - MIXING COST: to relay a binding across a fixed gap, one solve step propagates ~w positions, so the
    equilibrium needs ~gap/w iterations -> solve iterations FALL as w grows.
  - EDIT REACH (recompute-ball size): each hop of the edit-response covers ~w positions, so the screening
    length in POSITIONS, xi_pos = xi_hops * w, GROWS with w (a wider window = a wider blast radius).
So small w = cheap edits (tight ball) but slow solves; large w = fast solves but wide edits (dense limit =
every edit global). Expect an inverse Pareto: solve-iters ~ 1/w, xi_pos ~ w, product ~ const.

Reuses the C2 measurement machinery (Newton-polished tight solves, filler-edit hop-binned xi).
Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.c3_mixing_locality
"""
import time

import numpy as np
import torch
from torchdeq import get_deq

import experiments.sliding_window_reach as sw
import experiments.c2_edit_locality as c2      # imports with tf32 OFF (measurement-grade)

W_SWEEP = [5, 10, 20]
STAGES = [0, 8, 16]                              # curriculum; gap 16 relays for all w (16/5~3, 16/20~1)
STEPS = 300
TEST_GAP = 16
sw.H, sw.dh = 4, sw.d // 4


def inference_solve_iters(m, seqs, tol=1e-4):
    """MIXING cost: mean Anderson f-evals to converge at the INFERENCE tol (1e-4), not the tight
    measurement tol. Fresh solver so the count reflects what a deployment would pay per forward pass."""
    deq = get_deq(f_solver="anderson", f_max_iter=200, f_tol=tol)
    counts = []
    for toks in seqs:
        h0 = m.h0(toks); mask = sw.band_causal_mask(toks.shape[1], toks.device)
        maskp = m._maskp(mask); wn = m.wn()
        n = [0]
        def ff(z):
            n[0] += 1
            return m.f(z, h0, wn, maskp)
        with torch.no_grad():
            deq(ff, torch.zeros_like(h0))
        counts.append(n[0])
    return float(np.mean(counts))


def measure_xi_hops(m, toks, gen, n_edits=16):
    """LOCALITY: filler-edit screening length in HOPS (reuses C2's tight-solve + hop-binned fit)."""
    D, Z, noises = [], [], []
    for _ in range(n_edits):
        out = c2.edit_response_profile(m, toks, gen, "filler")
        if out is None:
            continue
        d_, z_, iw, ic, ok, weqc, noise = out
        if ok:
            D.append(d_); Z.append(z_); noises.append(noise)
    if not D:
        return np.nan
    xi_hops, _ = c2.fit_xi(np.concatenate(D), np.concatenate(Z), noise=float(np.max(noises)))
    return xi_hops


def main():
    print(f"device={sw.DEV}  C3 mixing<->locality: sweep window w, curriculum {STAGES}, measure at gap "
          f"{TEST_GAP}\n", flush=True)
    print(f"{'w':>3} {'recall':>7} {'solve_iters':>11} {'xi_hops':>8} {'xi_positions':>12} "
          f"{'iters*xi_pos':>12}", flush=True)
    rows = []
    for W in W_SWEEP:
        sw.W = W                                       # the dial; c2 reads sw.W too (band mask + /W)
        torch.manual_seed(0)
        m = sw.SeqDEQ("softmax", "deq").to(sw.DEV)
        t0 = time.time()
        for g in STAGES:
            sw.F_SWEEP = [g]
            sw.train(m, steps=STEPS)
        m.eval()
        ge = torch.Generator().manual_seed(123)
        acc = sw.recall(m, TEST_GAP, ge)
        # tight solver for the C2 xi machinery; separate 1e-4 solver for the mixing-cost count
        m.deq = get_deq(f_solver="anderson", f_max_iter=150, f_tol=1e-6,
                        ift=True, b_solver="anderson", b_max_iter=40)
        tgen = torch.Generator().manual_seed(7)
        seqs = [sw.gen_mqar(1, TEST_GAP, tgen)[0] for _ in range(4)]
        iters = inference_solve_iters(m, seqs)
        xi_hops = measure_xi_hops(m, seqs[0], tgen)
        xi_pos = xi_hops * W if np.isfinite(xi_hops) else np.nan
        prod = iters * xi_pos if np.isfinite(xi_pos) else np.nan
        rows.append((W, acc, iters, xi_hops, xi_pos, prod))
        print(f"{W:>3} {acc:>7.3f} {iters:>11.1f} {xi_hops:>8.2f} {xi_pos:>12.2f} {prod:>12.1f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    print("\nREAD: solve_iters should FALL and xi_positions should RISE as w grows (the Pareto). If"
          "\niters*xi_pos is roughly flat, the dial trades one cost for the other at ~constant product -"
          "\nthe mixing<->locality tension made quantitative on our own cell. (recall must stay high, else"
          "\nthe row is a broken model, not a tradeoff point.)", flush=True)


if __name__ == "__main__":
    main()
