"""Diagnose the bidirectional training gap: causal gap-0 hit recall 1.0 in 350 steps; bidir v1 got 0.367
(loss still descending, rho jumped to 1.1 immediately — two-sided coupling = genuine feedback loops).
Question: is it STEPS (keep training), TEMPERATURE (lr too hot for a recurrent landscape), or
AMPLITUDE (s_max lets rho run away before the task is learned)?

Trains gap-0 ONLY under a few configs, printing recall/rho/sigma_min every 100 steps.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.bidir_train_probe
"""
import time

import torch
import experiments.sliding_window_reach as sw

sw.BIDIR = True
sw.H, sw.dh = 4, sw.d // 4
GAP = 0
CONFIGS = [
    ("causal-init", dict(lr=1e-3, s_max=0.9, steps=400, init="checkpoints/curr00.pt")),  # mask-swap transfer:
    #   binding circuitry exists in the causal ckpt (recall 1.0); only the two-sided flow is new. This is
    #   how dual-mode/FIM LMs are actually made (same weights, switched mask) — and the stuck bidir model
    #   retrieves-but-misbinds, so transfer attacks exactly the broken piece.
    ("base-long", dict(lr=3e-3, s_max=0.9, steps=700)),     # steps hypothesis
    ("cool-lr", dict(lr=1e-3, s_max=0.9, steps=700)),       # temperature hypothesis
    ("low-smax", dict(lr=3e-3, s_max=0.7, steps=700)),      # amplitude hypothesis
]


def train_tracked(m, steps, lr):
    import torch.nn.functional as F
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(0)
    for st in range(steps):
        toks, qmask, targ = sw.gen_mqar(64, GAP, gen)
        m.train(); opt.zero_grad()
        logits = m.run(toks)
        loss = F.cross_entropy(logits[qmask], targ[qmask])
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        if torch.isfinite(loss) and torch.isfinite(gn):
            opt.step()
        else:
            opt.zero_grad()
        if (st + 1) % 100 == 0:
            m.eval()
            acc = sw.recall(m, GAP, torch.Generator().manual_seed(123), reps=2)
            try:
                r, smin, rs = m.spectrum(sw.gen_mqar(1, GAP, torch.Generator().manual_seed(7))[0])
            except Exception:
                r, smin, rs = float("nan"), float("nan"), float("nan")
            print(f"    step {st+1:>4} loss {loss.item():.3f}  recall {acc:.3f}  rho {r:.3f}  "
                  f"smin {smin:.3f}  resid {rs:.1e}", flush=True)
            m.train()


def main():
    print(f"device={sw.DEV}  BIDIR gap-{GAP} training probe (causal reference: recall 1.0 @350 steps)\n",
          flush=True)
    for name, cfg in CONFIGS:
        print(f"[{name}] lr={cfg['lr']} s_max={cfg['s_max']} steps={cfg['steps']}"
              + (f" init={cfg['init']}" if "init" in cfg else ""), flush=True)
        sw.S_MAX = cfg["s_max"]
        torch.manual_seed(0)
        m = sw.SeqDEQ("softmax", "deq").to(sw.DEV)
        if "init" in cfg:
            cki = torch.load(cfg["init"], map_location=sw.DEV, weights_only=False)
            m.load_state_dict(cki["state_dict"])
            m.eval()
            acc0 = sw.recall(m, GAP, torch.Generator().manual_seed(123), reps=2)
            print(f"    step    0 (pre-finetune, causal weights under bidir mask)  recall {acc0:.3f}", flush=True)
        t0 = time.time()
        train_tracked(m, cfg["steps"], cfg["lr"])
        print(f"  ({time.time()-t0:.0f}s)\n", flush=True)
    sw.S_MAX = 0.9


if __name__ == "__main__":
    main()
