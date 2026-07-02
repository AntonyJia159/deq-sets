"""Lean pivot probe: does a CURRICULUM rescue the relay that cold-training couldn't find? reach_diag showed
eq-softmax trained COLD on gap 16 collapses to a trivial fixed point (rho->0, recall=chance). Here: eq-softmax
only, warm-started through 0->8->16->24, 350 steps/stage. If recall stays high (and rho stays non-trivial),
the relay IS learnable and the cold failure was an optimization/init issue -> run the full eq-vs-unroll
comparison. If it collapses at 16/24 even with curriculum, the relay is not being learned -> that's the C1
verdict and no need to spend on the full sweep.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.reach_probe_eq
"""
import time

import torch
import experiments.sliding_window_reach as sw

STAGES = [0, 8, 16, 24]
STEPS_PER = 350
sw.H, sw.dh = 4, sw.d // 4


def main():
    print(f"device={sw.DEV}  eq-softmax curriculum {STAGES}, {STEPS_PER} steps/stage (lean pivot probe)\n", flush=True)
    torch.manual_seed(0)
    m = sw.SeqDEQ("softmax", "deq").to(sw.DEV)
    for g in STAGES:
        sw.F_SWEEP = [g]
        t0 = time.time()
        sw.train(m, steps=STEPS_PER)
        m.eval()
        ge = torch.Generator().manual_seed(123)
        acc = sw.recall(m, g, ge)
        r, smin, rs = m.spectrum(sw.gen_mqar(1, g, torch.Generator().manual_seed(7))[0])
        print(f"  gap {g:>2}: recall={acc:.3f}  rho={r:.3f}  sigma_min(I-J)={smin:.3f}  resid={rs:.1e}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        m.train()
    print(f"\nREAD: recall high AND rho not ~0 through gap 16-24 => relay is learnable via curriculum "
          f"(cold-collapse was optimization) -> run full eq-vs-unroll. Chance recall with rho~0 => "
          f"trivial-collapse persists -> relay not learned = the C1 answer.", flush=True)


if __name__ == "__main__":
    main()
