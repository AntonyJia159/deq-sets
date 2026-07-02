"""Plot 1 (blueprint C1): does LOCAL (sliding-window) attention + EQUILIBRIUM reach beyond a matched
finite unroll? Controlled-gap MQAR.

Setup: [k1 v1 .. kD vD][ F filler tokens ][ q1 .. qQ] -> predict each query's bound value. Window w:
position i attends to [i-w, i] (causal, banded). A query is separated from its value by a gap ~F, so
recall REQUIRES relaying the binding across ceil(gap/w) windows -- one Picard/DEQ step propagates info
~w positions, so a K-step UNROLL reaches only ~K*w, while the EQUILIBRIUM (many iters) reaches further
(until the sigma_min decay attenuates it). Prediction:
  - unroll-K softmax : recall cliffs when gap > K*w
  - equilibrium softmax : recall extends past K*w (local + equilibrium = long range)
  - equilibrium linear (Mamba/SSM analog) : fails throughout (no selection among the D keys)

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.sliding_window_reach
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NKEY, NVAL, NFILL, D_PAIR, NQ = 8, 8, 8, 4, 2
W = 10                                   # sliding-window width
d = 64
S_MAX = 0.9
F_SWEEP = [0, 8, 16, 24, 40]             # filler length -> query<->value gap ~ F (+ up to 2*D_PAIR)


def gen_mqar(batch, Fill, gen):
    L = 2 * D_PAIR + Fill + NQ
    keys = torch.rand(batch, NKEY, generator=gen).argsort(1)[:, :D_PAIR]
    vals = torch.randint(NVAL, (batch, D_PAIR), generator=gen)
    toks = torch.zeros(batch, L, dtype=torch.long)
    toks[:, 0:2 * D_PAIR:2] = keys
    toks[:, 1:2 * D_PAIR:2] = NKEY + vals
    if Fill > 0:
        toks[:, 2 * D_PAIR:2 * D_PAIR + Fill] = NKEY + NVAL + torch.randint(NFILL, (batch, Fill), generator=gen)
    qidx = torch.randint(D_PAIR, (batch, NQ), generator=gen)
    toks[:, 2 * D_PAIR + Fill:] = torch.gather(keys, 1, qidx)
    qmask = torch.zeros(batch, L, dtype=torch.bool); qmask[:, 2 * D_PAIR + Fill:] = True
    targ = torch.zeros(batch, L, dtype=torch.long); targ[:, 2 * D_PAIR + Fill:] = torch.gather(vals, 1, qidx)
    return toks.to(DEV), qmask.to(DEV), targ.to(DEV)


def _sn(Wm):
    try:
        s = torch.linalg.matrix_norm(Wm, ord=2)
    except (torch.linalg.LinAlgError, RuntimeError):
        s = torch.linalg.matrix_norm(Wm.cpu(), ord=2).to(Wm.device)
    return Wm / s


def band_causal_mask(L, device):
    i = torch.arange(L, device=device)[:, None]
    j = torch.arange(L, device=device)[None, :]
    return (j <= i) & (i - j <= W)                       # (L,L) True = allowed (causal + within window)


class SeqDEQ(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.emb = nn.Embedding(NKEY + NVAL + NFILL, d)
        self.posw = nn.Parameter(0.02 * torch.randn(256, d))     # absolute positions up to len 256
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.s_raw = nn.Parameter(torch.tensor(0.4))
        self.head = nn.Linear(d, NVAL)

    @property
    def s(self):
        return S_MAX * torch.sigmoid(self.s_raw)

    def h0(self, toks):
        return self.emb(toks) + self.posw[:toks.shape[1]]

    def wn(self):                                            # q/k RAW (must peak); v,o normed (contraction)
        return self.Wq.weight, self.Wk.weight, _sn(self.Wv.weight), _sn(self.Wo.weight)

    def f(self, z, h0, wn, mask):
        Wq, Wk, Wv, Wo = wn
        q, k, v = z @ Wq.t(), z @ Wk.t(), z @ Wv.t()
        if self.kind == "softmax":
            sc = (q @ k.transpose(-1, -2)) / (d ** 0.5)
            sc = sc.masked_fill(~mask, -1e30)
            a = torch.softmax(sc, -1)
        else:
            qf, kf = F.elu(q) + 1.0, F.elu(k) + 1.0
            sc = (qf @ kf.transpose(-1, -2)) * mask
            a = sc / (sc.sum(-1, keepdim=True) + 1e-6)
        agg = (a @ v) @ Wo.t()
        return h0 + self.s * agg

    def run(self, toks, iters):
        h0 = self.h0(toks)
        mask = band_causal_mask(toks.shape[1], toks.device)
        wn = self.wn()
        z = torch.zeros_like(h0)
        for _ in range(iters):
            z = self.f(z, h0, wn, mask)
        return self.head(z)


def recall(model, iters, Fill, gen, reps=4):
    accs = []
    for _ in range(reps):
        toks, qmask, targ = gen_mqar(128, Fill, gen)
        with torch.no_grad():
            logits = model.run(toks, iters)
        accs.append((logits.argmax(-1)[qmask] == targ[qmask]).float().mean().item())
    return float(np.mean(accs))


def train(model, iters, steps=800, bs=64):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(0)
    for st in range(steps):
        Fill = F_SWEEP[torch.randint(len(F_SWEEP), (1,), generator=gen).item()]     # sample a gap
        toks, qmask, targ = gen_mqar(bs, Fill, gen)
        model.train(); opt.zero_grad()
        logits = model.run(toks, iters)
        loss = F.cross_entropy(logits[qmask], targ[qmask])
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if torch.isfinite(loss) and torch.isfinite(gn):
            opt.step()
        else:
            opt.zero_grad()
        if st % 200 == 0:
            print(f"      step {st:>4} loss {loss.item():.3f}", flush=True)
    return loss.item()


def main():
    print(f"device = {DEV}  window w={W}, {D_PAIR} pairs + filler + {NQ} queries; unroll reaches ~K*w, "
          f"equilibrium reaches further\n", flush=True)
    configs = [                                             # name, kind, train_iters, eval_iters
        ("eq-softmax", "softmax", 40, 80),
        ("unroll2-softmax", "softmax", 2, 2),
        ("unroll4-softmax", "softmax", 4, 4),
        ("eq-linear", "linear", 40, 80),
    ]
    rows = {}
    for name, kind, tit, eit in configs:
        print(f"[{name}] training (iters {tit}) ...", flush=True)
        torch.manual_seed(0)
        m = SeqDEQ(kind).to(DEV)
        t0 = time.time()
        train(m, tit)
        m.eval()
        ge = torch.Generator().manual_seed(123)
        rows[name] = [recall(m, eit, Fl, ge) for Fl in F_SWEEP]
        print(f"  done ({time.time()-t0:.0f}s)\n", flush=True)

    print("RECALL vs gap  (reach ~K*w: unroll2~20, unroll4~40; column = filler F, gap~F)")
    print("model              " + "  ".join(f"F={f:>2}" for f in F_SWEEP), flush=True)
    for name, _, _, _ in configs:
        print(f"{name:<18} " + "  ".join(f"{a:>4.2f}" for a in rows[name]), flush=True)
    print(f"\nREAD: if eq-softmax stays high as F grows while unroll-K collapses past gap~K*w, then LOCAL "
          f"attention + EQUILIBRIUM = long-range reach (equilibrium earns expressivity beyond finite "
          f"depth); eq-linear low throughout confirms the selection gap is the softmax's doing.", flush=True)


if __name__ == "__main__":
    main()
