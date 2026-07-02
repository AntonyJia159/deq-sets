"""Verify the speed knobs (bigger batch, looser training f_tol) don't break relay LEARNING before adopting
them. Baseline (bs64, f_tol1e-4) reached gap-24 recall 0.975. Re-run the eq curriculum 0->8->16->24 at the
fast settings and check recall holds. Larger batch = fewer updates (could miss the relay basin); looser tol
= IFT gradient taken at a less-converged fixed point (could bias the relay). If gap-24 recall stays ~0.9+,
adopt; else fall back.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.verify_speed_learning
"""
import time

import torch
import experiments.sliding_window_reach as sw

sw.H, sw.dh = 4, sw.d // 4
STAGES = [0, 8, 16, 24]


def run(bs, ftol, label):
    print(f"[{label}] bs={bs} train_ftol={ftol:.0e}", flush=True)
    torch.manual_seed(0)
    m = sw.SeqDEQ("softmax", "deq").to(sw.DEV)
    m.deq = sw.get_deq(f_solver="anderson", f_max_iter=60, f_tol=ftol,
                       ift=True, b_solver="anderson", b_max_iter=30)
    t0 = time.time()
    accs = {}
    for g in STAGES:
        sw.F_SWEEP = [g]
        sw.train(m, steps=350, bs=bs)
        m.eval()
        ge = torch.Generator().manual_seed(123)
        accs[g] = sw.recall(m, g, ge)
        m.train()
    dt = time.time() - t0
    print(f"    recall " + "  ".join(f"g{g}={accs[g]:.2f}" for g in STAGES) + f"   ({dt:.0f}s)\n", flush=True)
    return accs


def main():
    print(f"device={sw.DEV}  verify relay learning holds at fast settings (baseline gap24=0.975 @ bs64/1e-4)\n",
          flush=True)
    run(256, 1e-3, "bs256+ftol1e-3")
    run(512, 1e-3, "bs512+ftol1e-3")
    print("READ: adopt the fastest config whose gap-24 recall stays ~0.9+. If both hold, bs512+ftol1e-3 "
          "(~expected 10x+ throughput). Tighten f_tol back to 1e-4 for the C2 edit-locality measurement.",
          flush=True)


if __name__ == "__main__":
    main()
