"""Plot 1, decisive version. reach_diag showed: substrate fine (F=0 -> 1.0) but eq-softmax trained COLD on
gap 16 collapses to a trivial fixed point (rho->0, recall=chance) -- it never finds the multi-hop relay from
end-only supervision. Standard escape from a trivial-minimum collapse = CURRICULUM: grow the gap slowly,
warm-starting each stage from the last, so the relay is discovered incrementally.

Run the SAME curriculum (0->8->16->24->40) for eq-softmax vs finite K-unroll baselines. This is the reach
plot: one Picard/DEQ step carries info ~w=10, so a K-unroll has a HARD reach limit ~K*w (cannot represent a
longer relay at any training), while the equilibrium can relay as far as sigma_min allows -- IF the relay is
learnable at all. Prediction if C1 holds: unroll2 cliffs past gap~20, unroll4 past ~40, eq stays up.

Reports recall + rho + sigma_min(I-J) per stage (the last two certify the fixed point is non-trivial).

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.reach_curriculum
"""
import time

import torch
import experiments.sliding_window_reach as sw

STAGES = [0, 8, 16, 24, 40]          # gap ~ F+8; unroll reach ~K*w (w=10): unroll2~20, unroll4~40
STEPS_PER = 350                       # eq curriculum solved 0->24 at 350/stage (reach_probe_eq)
sw.H, sw.dh = 4, sw.d // 4           # multi-head (carry + read); F=0 works at H=4 per reach_diag


def curriculum(name, kind, mode, K):
    print(f"[{name}] curriculum {STAGES} (mode={mode}{'' if K is None else f' K={K}'}) ...", flush=True)
    torch.manual_seed(0)
    m = sw.SeqDEQ(kind, mode, K).to(sw.DEV)
    out = {}
    for g in STAGES:
        sw.F_SWEEP = [g]                                  # train (warm-started) on this gap only
        t0 = time.time()
        sw.train(m, steps=STEPS_PER)
        m.eval()
        ge = torch.Generator().manual_seed(123)
        acc = sw.recall(m, g, ge)
        try:
            r, smin, rs = m.spectrum(sw.gen_mqar(1, g, torch.Generator().manual_seed(7))[0])
        except Exception as e:
            r, smin, rs = float("nan"), float("nan"), float("nan"); print(f"    spectrum failed: {e}", flush=True)
        out[g] = (acc, r, smin, rs)
        print(f"    gap {g:>2}: recall={acc:.3f}  rho={r:.3f}  sigma_min(I-J)={smin:.3f}  resid={rs:.1e}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        m.train()
    print(flush=True)
    return out


def main():
    print(f"device={sw.DEV}  curriculum reach: same 0->8->16->24->40 schedule, eq vs finite unroll\n", flush=True)
    configs = [
        ("eq-softmax", "softmax", "deq", None),
        ("unroll2-softmax", "softmax", "unroll", 2),
        ("unroll4-softmax", "softmax", "unroll", 4),
    ]
    res = {name: curriculum(name, kind, mode, K) for name, kind, mode, K in configs}

    print("RECALL vs gap  (curriculum-trained; unroll hard reach ~K*w: unroll2~20, unroll4~40)")
    print("model              " + "  ".join(f"g={g:>2}" for g in STAGES), flush=True)
    for name, _, _, _ in configs:
        print(f"{name:<18} " + "  ".join(f"{res[name][g][0]:>4.2f}" for g in STAGES), flush=True)
    print("\nfixed-point health (rho, sigma_min(I-J)) per gap -- eq only (unroll is not a true fp):")
    for g in STAGES:
        a, r, sm, rs = res["eq-softmax"][g]
        print(f"  gap {g:>2}: recall={a:.2f}  rho={r:>6.3f}  sigma_min={sm:>6.3f}  resid={rs:.1e}", flush=True)
    print(f"\nREAD: if eq-softmax stays high through gap 24-40 while unroll2 cliffs after ~20 and unroll4 after "
          f"~40, then LOCAL attention + EQUILIBRIUM relays beyond ANY fixed finite depth (C1 holds). A "
          f"non-trivial fixed point requires rho not ~0 AND recall high -- rho~0 with chance recall = the "
          f"trivial-collapse failure mode from reach_diag.", flush=True)


if __name__ == "__main__":
    main()
