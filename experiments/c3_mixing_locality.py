"""C3 (v2, reframed) — does the mixing<->locality tradeoff DISSOLVE at equilibrium?

v1 finding: solve-iters fall ~1/w (mixing cost, confirmed), but edit-reach in POSITIONS was ~flat (~3.6),
NOT rising with w as a naive Pareto predicts. Reframe: for a finite UNROLL, reach is capped at w*K, so w
trades mixing vs locality. At EQUILIBRIUM (C1) reach decouples from window*depth -> reach is set by sigma_min
(conditioning), not by w. So w is a SOLVE-SPEED dial, not a reach dial; the tradeoff dissolves. This run
tests that cleanly: fit xi in POSITIONS (v1's hop-binning is too coarse when xi < w -> w=20 gave nan), sweep
4 windows, and check whether xi_positions stays ~flat while solve-iters fall.

Prediction (reframe): solve_iters FALL with w; xi_positions ~CONSTANT across w (locality is sigma_min's job,
not the window's). recall must stay high (w=5 may under-train the gap-16 relay -> flagged, its xi is on a
half-formed cell).

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.c3_mixing_locality
"""
import time

import numpy as np
import torch
from torchdeq import get_deq

import experiments.sliding_window_reach as sw
import experiments.c2_edit_locality as c2      # imports with tf32 OFF (measurement-grade); reuse its solves

W_SWEEP = [5, 8, 12, 20]
STAGES = [0, 8, 16]
STEPS = 300
TEST_GAP = 16
sw.H, sw.dh = 4, sw.d // 4


def fit_xi_positions(dists, dz, noise):
    """Per-POSITION screening length (v1's hop-binning is too coarse when xi < w). log dz vs position-distance
    is a line with slope -1/xi; floor at the measured solver noise (pre-edit dz) x3."""
    dists, dz = np.asarray(dists), np.asarray(dz)
    ds = np.unique(dists)
    mean_dz = np.array([dz[dists == d].mean() for d in ds])
    floor = max(1e-8, 1e-5 * mean_dz.max(), 3.0 * noise)
    use = mean_dz > floor
    if use.sum() < 3:
        return np.nan
    slope, _ = np.polyfit(ds[use], np.log(mean_dz[use]), 1)
    return np.inf if slope >= 0 else -1.0 / slope     # in POSITIONS


def inference_solve_iters(m, seqs, tol=1e-4):
    """MIXING cost: mean Anderson f-evals to converge at the inference tol (what a forward pass pays)."""
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


def measure_xi_positions(m, seqs, gen, n_edits=6):
    """LOCALITY: filler-edit screening length in POSITIONS, pooled over several base sequences."""
    D, Z, noises = [], [], []
    for toks in seqs:
        for _ in range(n_edits):
            out = c2.edit_response_profile(m, toks, gen, "filler")
            if out is None:
                continue
            d_, z_, iw, ic, ok, weqc, noise = out
            if ok:
                D.append(d_); Z.append(z_); noises.append(noise)
    if not D:
        return np.nan
    return fit_xi_positions(np.concatenate(D), np.concatenate(Z), float(np.max(noises)))


def main():
    print(f"device={sw.DEV}  C3 v2: window sweep, xi fit in POSITIONS; does the tradeoff dissolve at "
          f"equilibrium?\n", flush=True)
    print(f"{'w':>3} {'recall':>7} {'solve_iters':>11} {'xi_positions':>12}   note", flush=True)
    for W in W_SWEEP:
        sw.W = W
        torch.manual_seed(0)
        m = sw.SeqDEQ("softmax", "deq").to(sw.DEV)
        t0 = time.time()
        for g in STAGES:
            sw.F_SWEEP = [g]
            sw.train(m, steps=STEPS)
        m.eval()
        ge = torch.Generator().manual_seed(123)
        acc = sw.recall(m, TEST_GAP, ge)
        m.deq = get_deq(f_solver="anderson", f_max_iter=150, f_tol=1e-6,
                        ift=True, b_solver="anderson", b_max_iter=40)
        tgen = torch.Generator().manual_seed(7)
        seqs = [sw.gen_mqar(1, TEST_GAP, tgen)[0] for _ in range(4)]
        iters = inference_solve_iters(m, seqs)
        xi_pos = measure_xi_positions(m, seqs, tgen)
        note = "under-trained (xi suspect)" if acc < 0.8 else ("xi < window (very local)" if xi_pos < W else "")
        print(f"{W:>3} {acc:>7.3f} {iters:>11.1f} {xi_pos:>12.2f}   {note}  ({time.time()-t0:.0f}s)",
              flush=True)

    print("\nREAD (reframe): if solve_iters FALL with w while xi_positions stays ~flat, the mixing<->locality"
          "\ntradeoff DISSOLVES at equilibrium -- w is a solve-speed dial, edit-reach is set by sigma_min"
          "\n(conditioning), not the window. That is C1's reach-decoupling seen from the maintenance side."
          "\nOnly rows with recall>0.8 are clean operating points.", flush=True)


if __name__ == "__main__":
    main()
