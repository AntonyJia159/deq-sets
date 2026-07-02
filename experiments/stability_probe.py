"""(b) Isolate the peaking<->contraction question: can a PEAKED softmax-attention DEQ be trained
STABLY to a genuine fixed point -- and is there an operating point that does BOTH recall and contraction?

Plot-1 v1 died because a 40-step UNROLLED softmax equilibrium went non-contractive as attention peaked,
and deep-unroll backprop gave garbage gradients. Here we remove BOTH confounds: short DENSE MQAR (reach
is trivial -> pure stability test, not a relay test), and train with torchdeq IMPLICIT DIFFERENTIATION +
an Anderson solver (the stable, proper way to train a DEQ -- no unrolled backprop). Then sweep the
contraction knob s_max and read, per setting:
  - recall acc      : did peaked attention solve the task?
  - solver residual : did the fixed-point iteration actually CONVERGE (else the "equilibrium" is garbage)?
  - rho(J)          : is the map contractive at the solution?
DECISION: if some s_max gives high recall AND small residual AND rho<~1, the tension is MANAGEABLE and the
unrolled-backprop was the real culprit -> proceed to the reach experiment with torchdeq. If peaking always
breaks convergence, the sequence direction has a real obstacle.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.stability_probe
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdeq import get_deq
from torchdeq.loss import power_method

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NKEY, NVAL, D_PAIR, NQ = 16, 16, 8, 4
L = 2 * D_PAIR + NQ
d = 64
S_SWEEP = [0.5, 0.7, 0.9, 1.1, 1.3]


def gen_mqar(batch, gen):
    keys = torch.rand(batch, NKEY, generator=gen).argsort(1)[:, :D_PAIR]
    vals = torch.randint(NVAL, (batch, D_PAIR), generator=gen)
    toks = torch.zeros(batch, L, dtype=torch.long)
    toks[:, 0:2 * D_PAIR:2] = keys
    toks[:, 1:2 * D_PAIR:2] = NKEY + vals
    qidx = torch.randint(D_PAIR, (batch, NQ), generator=gen)
    toks[:, 2 * D_PAIR:] = torch.gather(keys, 1, qidx)
    qmask = torch.zeros(batch, L, dtype=torch.bool); qmask[:, 2 * D_PAIR:] = True
    targ = torch.zeros(batch, L, dtype=torch.long); targ[:, 2 * D_PAIR:] = torch.gather(vals, 1, qidx)
    return toks.to(DEV), qmask.to(DEV), targ.to(DEV)


def _sn(W):
    try:
        s = torch.linalg.matrix_norm(W, ord=2)
    except (torch.linalg.LinAlgError, RuntimeError):
        s = torch.linalg.matrix_norm(W.cpu(), ord=2).to(W.device)
    return W / s


class Model(nn.Module):
    def __init__(self, s_max, **deq_kwargs):
        super().__init__()
        self.s_max = s_max
        self.emb = nn.Embedding(NKEY + NVAL, d)
        self.pos = nn.Parameter(0.02 * torch.randn(L, d))
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.s_raw = nn.Parameter(torch.tensor(0.4))
        self.head = nn.Linear(d, NVAL)
        cm = torch.tril(torch.ones(L, L))
        self.register_buffer("cmask", cm.bool())
        self.deq = get_deq(f_solver="anderson", f_max_iter=50, f_tol=1e-4, **deq_kwargs)

    @property
    def s(self):
        return self.s_max * torch.sigmoid(self.s_raw)

    def make_f(self, h0):
        Wq, Wk = self.Wq.weight, self.Wk.weight               # RAW -> peak
        Wv, Wo = _sn(self.Wv.weight), _sn(self.Wo.weight)     # normed -> contraction
        s = self.s

        def f(z):
            q, k, v = z @ Wq.t(), z @ Wk.t(), z @ Wv.t()
            sc = (q @ k.transpose(-1, -2)) / (d ** 0.5)
            sc = sc.masked_fill(~self.cmask, -1e30)
            a = torch.softmax(sc, -1)
            return h0 + s * (a @ v) @ Wo.t()
        return f

    def forward(self, toks):
        h0 = self.emb(toks) + self.pos
        f = self.make_f(h0)
        z = self.deq(f, torch.zeros_like(h0))[0][-1]
        return self.head(z)

    @torch.no_grad()
    def diagnose(self, toks):
        h0 = self.emb(toks) + self.pos
        f = self.make_f(h0)
        z = self.deq(f, torch.zeros_like(h0))[0][-1]
        resid = ((f(z) - z).norm() / (z.norm() + 1e-9)).item()
        with torch.enable_grad():
            z0 = z.detach().requires_grad_(True)
            _, rho = power_method(f(z0), z0, n_iters=30)
        return resid, rho.max().item()


def recall(model, gen, reps=4):
    accs = []
    for _ in range(reps):
        toks, qmask, targ = gen_mqar(128, gen)
        with torch.no_grad():
            logits = model(toks)
        accs.append((logits.argmax(-1)[qmask] == targ[qmask]).float().mean().item())
    return float(np.mean(accs))


def train(model, steps=800, bs=64):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(0)
    nan_steps = 0
    for st in range(steps):
        toks, qmask, targ = gen_mqar(bs, gen)
        model.train(); opt.zero_grad()
        logits = model(toks)
        loss = F.cross_entropy(logits[qmask], targ[qmask])
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if torch.isfinite(loss) and torch.isfinite(gn):
            opt.step()
        else:
            opt.zero_grad(); nan_steps += 1
    return loss.item(), nan_steps


def main():
    print(f"device = {DEV}  dense causal MQAR L={L} (reach trivial); torchdeq implicit-diff + Anderson; "
          f"sweeping s_max\n", flush=True)
    print(f"{'s_max':>6} {'recall':>7} {'resid':>9} {'rho(J)':>7} {'nan_steps':>10} {'verdict':>24}", flush=True)
    for sm in S_SWEEP:
        torch.manual_seed(0)
        m = Model(sm).to(DEV)
        t0 = time.time()
        _, nans = train(m)
        m.eval()
        ge = torch.Generator().manual_seed(123)
        acc = recall(m, ge)
        resid, rho = m.diagnose(*gen_mqar(64, torch.Generator().manual_seed(7))[:1])
        ok = acc > 0.9 and resid < 1e-2
        verdict = "STABLE + solves" if ok else ("solves but poor fp" if acc > 0.9 else
                  ("converged but no recall" if resid < 1e-2 else "unstable/degenerate"))
        print(f"{sm:>6.1f} {acc:>7.3f} {resid:>9.1e} {rho:>7.3f} {nans:>10} {verdict:>24}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    print(f"\nREAD: any row with recall>0.9 AND resid<1e-2 (a genuine, converged fixed point that ALSO does "
          f"peaked retrieval) => the peaking<->contraction tension is MANAGEABLE with implicit diff, and "
          f"Plot-1 v1 failed only because of unrolled backprop -> proceed with torchdeq. If none, the "
          f"tension is real and load-bearing.", flush=True)


if __name__ == "__main__":
    main()
