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
# ROUND 1 VERDICT (checkpoints/probe_bidir_log.txt): causal-init 0.38, base-long 0.39, cool-lr 0.41,
# low-smax 0.36 — plateau invariant to init/steps/lr/s_max.
# ROUND 2 VERDICT (probe_bidir_round2_log.txt): rel-bias 0.37 — explicit direction signal does NOT fix it,
# so the left/right-symmetry story is wrong or insufficient.
# ROUND 3 — split the hypothesis space: is it the EQUILIBRIUM or bidirectional ATTENTION per se?
#   unroll4-bidir learns  -> equilibrium-specific -> OVERSMOOTHING suspect: bidirectional row-stochastic
#     attention has a consensus/Perron mode; the resolvent amplifies it by 1/(1-s) while contracting
#     token-identity differences (= infinite-depth GNN on the band graph, the classic collapse; the
#     triangular/causal operator has no symmetric averaging spectrum, which is why causal never saw it).
#   unroll4-bidir also stuck -> representational/task interaction, rethink from scratch.
# The cossim probe reads the mechanism directly: oversmoothing = high mean pairwise cosine of z* rows.
# ROUND 3 VERDICT (probe_bidir_round3_log.txt): unroll4-bidir ALSO stuck (0.377 / 0.373 with relb) while
# unroll4-CAUSAL solved gap 0-16 at 1.0 in C1 -> NOT equilibrium-specific. cossim anchors: causal 0.033
# vs stuck-bidir 0.274 (collapse-pull real but secondary — finite depth fails identically).
# Surviving hypothesis = QUERY-IDENTITY LEAKAGE: query tokens are literal duplicates of key tokens;
# bidirectionally every context state absorbs them -> content matching poisoned. Causal masks protected
# the context for free. ROUND 4 = READ-ONLY QUERIES (sw.READONLY_Q): context keeps its bidirectional
# band, queries read out without injecting. Factorial: unroll4 (fast discriminator) + deq (the target).
CONFIGS = [
    ("roq-unroll4-relb", dict(lr=3e-3, s_max=0.9, steps=700, mode="unroll", K=4, rel_bias=True, roq=True)),
    ("roq-deq-relb", dict(lr=3e-3, s_max=0.9, steps=700, rel_bias=True, roq=True)),
]


def mean_cossim(m, gap=GAP):
    """Oversmoothing readout: mean off-diagonal pairwise cosine similarity of the final states z*
    across positions (averaged over 32 seqs). ~1 = consensus collapse (token identities blurred)."""
    toks = sw.gen_mqar(32, gap, torch.Generator().manual_seed(11))[0]
    with torch.no_grad():
        h0 = m.h0(toks)
        mask = sw.band_causal_mask(toks.shape[1], toks.device)
        z = m.solve(h0, mask)
        zn = torch.nn.functional.normalize(z, dim=-1)
        C = zn @ zn.transpose(1, 2)                          # (B,L,L)
        L = C.shape[-1]
        off = (C.sum((1, 2)) - L) / (L * (L - 1))
        return off.mean().item()


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
                  f"smin {smin:.3f}  resid {rs:.1e}  cossim {mean_cossim(m):.3f}", flush=True)
            m.train()


def main():
    print(f"device={sw.DEV}  BIDIR gap-{GAP} training probe (causal reference: recall 1.0 @350 steps)\n",
          flush=True)
    # cossim anchors: working-causal vs stuck-bidir fixed points (oversmoothing = stuck >> causal)
    import os
    for tag, path, bidir in [("causal curr00 (recall 1.0)", "checkpoints/curr00.pt", False),
                             ("stuck bidir00 (recall 0.37)", "checkpoints/bidir00.pt", True)]:
        if os.path.exists(path):
            sw.BIDIR, sw.REL_BIAS = bidir, False
            ck = torch.load(path, map_location=sw.DEV, weights_only=False)
            ma = sw.SeqDEQ("softmax", "deq").to(sw.DEV)
            ma.load_state_dict(ck["state_dict"]); ma.eval()
            print(f"  anchor {tag}: cossim={mean_cossim(ma):.3f}", flush=True)
            del ma
    sw.BIDIR = True
    print(flush=True)
    for name, cfg in CONFIGS:
        print(f"[{name}] lr={cfg['lr']} s_max={cfg['s_max']} steps={cfg['steps']}"
              + (f" init={cfg['init']}" if "init" in cfg else "")
              + (" rel_bias" if cfg.get("rel_bias") else "")
              + (" readonly-q" if cfg.get("roq") else "")
              + (f" mode={cfg.get('mode', 'deq')}" + (f" K={cfg['K']}" if "K" in cfg else "")), flush=True)
        sw.S_MAX = cfg["s_max"]
        sw.REL_BIAS = cfg.get("rel_bias", False)
        sw.READONLY_Q = cfg.get("roq", False)
        torch.manual_seed(0)
        m = sw.SeqDEQ("softmax", cfg.get("mode", "deq"), cfg.get("K")).to(sw.DEV)
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
