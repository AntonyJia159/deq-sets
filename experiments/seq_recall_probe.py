"""Mamba : us  ::  InstantGNN : our early idea.  The sequence-transplant of maintenance_compare.py.

Mamba / linear-attention / SSMs get cheap parallel-train + serial-infer from LINEARITY (associative
scan; O(1) recurrent state), exactly as InstantGNN got cheap editing from a LINEAR resolvent. Their
edit-decay is rho(A) -- the linear special case of our sigma_min. We ask the transplanted questions:

  (1) EXPRESSIVITY (does the nonlinear member earn its keep?). On MQAR (multi-query associative
      recall -- the standard discriminator that separates SSMs/linear-attention from softmax
      attention), softmax = an argmax/SELECTION over keys, the sequence analog of the tropical/
      order-statistic gain that was our ONE categorical win on graphs. Prediction: softmax-DEQ >
      linear-DEQ on recall.

  (2) EDIT-LOCALITY / MAINTAINABILITY. Edit a value token, warm-start re-solve, and ask WHERE the
      equilibrium moves. Crux: softmax recall is powerful because a query attends to a FAR key, so the
      edit-response may be NON-local in sequence distance but LOCAL over the ATTENTION GRAPH (only
      positions attending to the edit move). If so, maintainability hinges on attention SPARSITY, not
      sequence proximity -> the case for sliding-window. We measure spatial-vs-content structure.

Toy scale (fits a 6GB GPU): dense attention with a causal mask, two cells differing only in
softmax-vs-linear-kernel mixing, solved to equilibrium by Anderson.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.seq_recall_probe
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NKEY, NVAL, D_PAIR, NQ = 16, 16, 8, 4        # 8 key->value pairs, 4 queries
L = 2 * D_PAIR + NQ                          # sequence length
d = 64
S_MAX = 0.9


# ---- MQAR data: [k1 v1 k2 v2 ... kD vD  q1 q2 ... qQ], predict the value bound to each query key ----
# token ids: keys 0..NKEY-1 ; values NKEY..NKEY+NVAL-1 ; readout is over the NVAL value classes.
def gen_mqar(batch, gen):
    keys = torch.rand(batch, NKEY, generator=gen).argsort(1)[:, :D_PAIR]    # (B,D_PAIR) distinct key ids
    vals = torch.randint(NVAL, (batch, D_PAIR), generator=gen)
    toks = torch.zeros(batch, L, dtype=torch.long)
    toks[:, 0:2 * D_PAIR:2] = keys                                          # even positions: keys
    toks[:, 1:2 * D_PAIR:2] = NKEY + vals                                   # odd positions: values
    qidx = torch.randint(D_PAIR, (batch, NQ), generator=gen)               # which pairs are queried
    toks[:, 2 * D_PAIR:2 * D_PAIR + NQ] = torch.gather(keys, 1, qidx)
    qmask = torch.zeros(batch, L, dtype=torch.bool)
    qmask[:, 2 * D_PAIR:2 * D_PAIR + NQ] = True
    targ = torch.zeros(batch, L, dtype=torch.long)
    targ[:, 2 * D_PAIR:2 * D_PAIR + NQ] = torch.gather(vals, 1, qidx)
    return toks.to(DEV), qmask.to(DEV), targ.to(DEV)


def _sn(W):
    try:
        s = torch.linalg.matrix_norm(W, ord=2)
    except (torch.linalg.LinAlgError, RuntimeError):
        s = torch.linalg.matrix_norm(W.cpu(), ord=2).to(W.device)
    return W / s


class SeqDEQ(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind                                              # "softmax" | "linear"
        self.emb = nn.Embedding(NKEY + NVAL, d)
        self.pos = nn.Parameter(0.02 * torch.randn(L, d))
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.s_raw = nn.Parameter(torch.tensor(0.4))
        self.head = nn.Linear(d, NVAL)
        cm = torch.tril(torch.ones(L, L))                             # causal mask (dense; toy L)
        self.register_buffer("cmask", cm)

    @property
    def s(self):
        return S_MAX * torch.sigmoid(self.s_raw)

    def h0(self, toks):
        return self.emb(toks) + self.pos                             # (B, L, d), graph-independent inject

    def wn(self):
        """Value path spectral-normed (bounds the map -> contraction); query/key left RAW so the
        softmax can PEAK -- spectral-norming q/k caps logits and kills sharp retrieval. Computed once."""
        return self.Wq.weight, self.Wk.weight, _sn(self.Wv.weight)

    def _attn(self, z, wn):
        Wq, Wk, Wv = wn
        q, k, v = z @ Wq.t(), z @ Wk.t(), z @ Wv.t()                 # (B,L,d)
        if self.kind == "softmax":
            sc = (q @ k.transpose(-1, -2)) / (d ** 0.5)             # (B,L,L)
            sc = sc.masked_fill(self.cmask == 0, -1e30)
            a = torch.softmax(sc, dim=-1)                            # nonlinear SELECTION (argmax-ish)
        else:                                                        # linear-kernel attention (Performer-form)
            qf, kf = F.elu(q) + 1.0, F.elu(k) + 1.0
            sc = (qf @ kf.transpose(-1, -2)) * self.cmask           # (B,L,L) nonneg, no softmax
            a = sc / (sc.sum(-1, keepdim=True) + 1e-6)              # convex weights -> bounded, contractive
        return a @ v, a

    def f(self, z, h0, wn):
        agg, _ = self._attn(z, wn)
        return h0 + self.s * agg

    @torch.no_grad()
    def solve(self, h0, z=None, tol=1e-7, maxit=300, m=5):
        z = torch.zeros_like(h0) if z is None else z.clone()
        shape = z.shape
        wn = self.wn()
        Fh, Xh = [], []
        gx = self.f(z, h0, wn)
        for it in range(1, maxit + 1):
            res = gx - z
            rn = (res.norm() / (gx.norm() + 1e-9)).item()
            if rn < tol or not np.isfinite(rn):
                break
            Fh.append(res.reshape(-1)); Xh.append(gx.reshape(-1))
            if len(Fh) > m:
                Fh.pop(0); Xh.pop(0)
            if len(Fh) == 1:
                z = gx
            else:
                Fm = torch.stack(Fh, 1)
                A = Fm.t() @ Fm
                A = A + 1e-8 * A.diag().mean() * torch.eye(A.shape[0], device=A.device)
                al = torch.linalg.solve(A, torch.ones(A.shape[0], 1, device=A.device))
                al = al / al.sum()
                z = (torch.stack(Xh, 1) @ al).reshape(shape)
            gx = self.f(z, h0, wn)
        return z, it

    def forward(self, toks):
        h0 = self.h0(toks)
        z = self.deq_solve_train(h0)
        return self.head(z)                                          # (B,L,NVAL)

    def deq_solve_train(self, h0, iters=30):
        """Unrolled Picard for training (cheap, differentiable); contractive so it converges."""
        wn = self.wn()                                              # spectral-norm ONCE, not per iter
        z = torch.zeros_like(h0)
        for _ in range(iters):
            z = self.f(z, h0, wn)
        return z


def recall_acc(model, toks, qmask, targ):
    logits = model(toks)
    pred = logits.argmax(-1)
    correct = (pred[qmask] == targ[qmask]).float().mean().item()
    return correct


def train(model, gen, steps=1000, bs=64):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
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
            opt.zero_grad()
        if st % 150 == 0:
            print(f"    step {st:>4}  loss {loss.item():.3f}", flush=True)
    return loss.item()


@torch.no_grad()
def edit_probe(model, gen):
    """Edit ONE value token; measure where the equilibrium moves: spatial distance vs attention graph."""
    toks, qmask, targ = gen_mqar(1, gen)
    h0 = model.h0(toks)
    z0, _ = model.solve(h0)
    # pick a value token (odd position in the pair block) whose key IS queried, and its retrieving query
    val_pos = None; q_pos = None
    keys = toks[0, 0:2 * D_PAIR:2]
    for j in range(NQ):
        p = 2 * D_PAIR + j
        kj = toks[0, p]
        matches = (keys == kj).nonzero()
        if len(matches):
            val_pos = 2 * int(matches[0]) + 1; q_pos = p; break
    if val_pos is None:
        return None
    new = toks.clone()
    new[0, val_pos] = NKEY + ((int(toks[0, val_pos]) - NKEY + 1) % NVAL)     # flip the bound value
    h0e = model.h0(new)
    zw, itw = model.solve(h0e, z0)                                            # warm-start (attack=edit)
    zc, itc = model.solve(h0e)                                                # cold
    warm_cold = ((zw - zc).norm() / (zw.norm() + 1e-9)).item()
    dz = (zw - z0)[0].norm(dim=-1).cpu().numpy()                              # (L,) per-position move
    # attention weight each position pays to the edited value token (softmax cell)
    _, a = model._attn(z0, model.wn())
    att_to_edit = a[0, :, val_pos].cpu().numpy()                             # (L,) how much i attends to edit
    dist = np.abs(np.arange(L) - val_pos)
    return dict(val_pos=val_pos, q_pos=q_pos, warm_cold=warm_cold, itw=itw, itc=itc,
                dz=dz, att=att_to_edit, dist=dist)


def main():
    print(f"device = {DEV}  seq_len {L} ({D_PAIR} pairs + {NQ} queries), d={d}\n", flush=True)
    res = {}
    for kind in ["linear", "softmax"]:
        torch.manual_seed(0)
        gen = torch.Generator().manual_seed(0)
        model = SeqDEQ(kind).to(DEV)
        t0 = time.time()
        train(model, gen)
        model.eval()
        ge = torch.Generator().manual_seed(123)
        accs = [recall_acc(model, *gen_mqar(128, ge)) for _ in range(5)]
        acc = float(np.mean(accs))
        print(f"[{kind:>8}] MQAR recall acc {acc:.3f}   s {model.s.item():.2f}   ({time.time()-t0:.0f}s)",
              flush=True)
        res[kind] = (model, acc)

    print(f"\nEXPRESSIVITY: softmax {res['softmax'][1]:.3f} vs linear {res['linear'][1]:.3f}  -> "
          f"{'softmax WINS recall (nonlinear earns its keep, the selection/argmax gain)' if res['softmax'][1] > res['linear'][1] + 0.05 else 'TIE/inconclusive'}",
          flush=True)

    print("\nEDIT-LOCALITY (softmax cell): change one value token, warm-start re-solve.", flush=True)
    ge = torch.Generator().manual_seed(7)
    for trial in range(3):
        r = edit_probe(res["softmax"][0], ge)
        if r is None:
            continue
        dz, att, dist = r["dz"], r["att"], r["dist"]
        # spatial locality: corr(|dz|, -dist) ; content locality: corr(|dz|, attention-to-edit)
        cs = np.corrcoef(dz, -dist)[0, 1] if np.std(dz) > 0 else float("nan")
        cc = np.corrcoef(dz, att)[0, 1] if np.std(dz) > 0 and np.std(att) > 0 else float("nan")
        print(f"  trial {trial}: warm==cold {r['warm_cold']:.1e} (warm {r['itw']}/cold {r['itc']} it) | "
              f"|dz| at edited {dz[r['val_pos']]:.3f}, at retrieving-query {dz[r['q_pos']]:.3f}, "
              f"mean-elsewhere {np.mean(np.delete(dz, [r['val_pos'], r['q_pos']])):.3f} | "
              f"corr(|dz|,-dist) {cs:+.2f}  corr(|dz|,attn) {cc:+.2f}", flush=True)
    print("\nREAD: if corr(|dz|,attn) >> corr(|dz|,-dist), the edit-response is CONTENT-local over the "
          "attention graph, NOT spatial -> maintainability needs attention SPARSITY (sliding-window), "
          "not sequence proximity. That reframes 'screening length in hops' as 'attention-graph reach'.",
          flush=True)


if __name__ == "__main__":
    main()
