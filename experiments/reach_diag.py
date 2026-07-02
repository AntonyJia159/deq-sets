"""Diagnostic for the Plot-1 v2 null: eq-softmax was flat at 0.34 even at F=0 (the easy dense case that
stability_probe solved at 1.0). Isolate the regression: (a) multi-head config, (b) mixed-gap objective.
Train eq-softmax on a SINGLE fixed gap and check recall at H=1 vs H=4. If F=0-only recovers ~1.0, the
substrate is fine and mixed-gap training is the culprit; if not, the multi-head/dim config broke it.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.reach_diag
"""
import torch
import experiments.sliding_window_reach as sw


def run(H, Fill, steps=800):
    sw.H, sw.dh = H, sw.d // H
    sw.F_SWEEP = [Fill]                       # train (and sample) only this gap
    torch.manual_seed(0)
    m = sw.SeqDEQ("softmax", "deq").to(sw.DEV)
    sw.train(m, steps=steps)
    m.eval()
    ge = torch.Generator().manual_seed(123)
    acc = sw.recall(m, Fill, ge)
    r, smin, rs = m.spectrum(sw.gen_mqar(1, Fill, torch.Generator().manual_seed(7))[0])
    print(f">> H={H} Fill={Fill} steps={steps}: recall={acc:.3f}  rho={r:.3f}  "
          f"sigma_min(I-J)={smin:.3f}  resid={rs:.1e}\n", flush=True)


if __name__ == "__main__":
    print(f"device={sw.DEV}  diagnosing eq-softmax substrate (single fixed gap)\n", flush=True)
    run(1, 0)          # single head, easy gap -> should match stability_probe (~1.0) if substrate ok
    run(4, 0)          # multi head, easy gap -> isolates whether H=4 alone regressed it
    run(4, 16)         # multi head, mid gap (needs ~2-window relay) trained ALONE -> can eq relay at all?
