"""Plot 1 (blueprint C1): does LOCAL (sliding-window) attention + EQUILIBRIUM reach beyond a matched
finite unroll? Controlled-gap MQAR.  [v2: torchdeq implicit-diff + Anderson, multi-head, sigma_min readout]

Setup: [k1 v1 .. kD vD][ F filler tokens ][ q1 .. qQ] -> predict each query's bound value. Window w:
position i attends to [i-w, i] (causal, banded). A query is separated from its value by a gap ~F, so
recall REQUIRES relaying the binding across ceil(gap/w) windows -- one Picard/DEQ step propagates info
~w positions, so a K-step UNROLL reaches only ~K*w, while the EQUILIBRIUM (solve to tol) reaches further
(until the sigma_min decay attenuates it). Prediction:
  - unroll-K softmax : recall cliffs when gap > K*w
  - equilibrium softmax : recall extends past K*w (local + equilibrium = long range)
  - equilibrium linear (Mamba/SSM analog) : fails throughout (no selection among the D keys)

v1 died because a 40-step UNROLLED softmax equilibrium went non-contractive as attention peaked and
deep-unroll backprop gave garbage gradients (stability_probe, committed b62740f, resolved this). v2 trains
the equilibrium models with torchdeq IMPLICIT DIFFERENTIATION (ift=True) + Anderson -- the stable, proper
way -- and leaves the K-unroll baselines as the finite-depth control. Reports rho(J) and sigma_min(I-J)
at the solution (sigma_min is the quantity that certifies edit-locality at rho>1).

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.sliding_window_reach
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdeq import get_deq
from torchdeq.loss import power_method

torch.backends.cuda.matmul.allow_tf32 = True         # Ada: near-free on the projection matmuls
torch.backends.cudnn.allow_tf32 = True

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NKEY, NVAL, NFILL, D_PAIR, NQ = 8, 8, 8, 4, 2
W = 10                                   # sliding-window width
d = 64
H = 4                                    # heads (carry + read; single head can't relay across windows)
dh = d // H
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


BIDIR = False                            # module flag: True -> two-sided band [i-W, i+W] (edit regime; C2-bidir)
REL_BIAS = False                         # module flag: per-head learned relative-position bias b[h, j-i] on the
                                         # logits (T5-style). Needed bidirectionally: a value token's neighborhood
                                         # is left/right SYMMETRIC (own key at -1, next pair's key at +1) and
                                         # content+weak-absolute-PE can't break the tie -> binding blends (probe:
                                         # recall stuck ~0.38 across init/steps/lr/s_max). Causal mask gave
                                         # direction for free, so causal checkpoints don't have this parameter.


READONLY_Q = False                       # module flag (BIDIR only): context tokens cannot attend TO query
                                         # positions — queries read out without injecting. Round-3 finding:
                                         # bidirectionally, query tokens (literal duplicates of key tokens)
                                         # leak their identity into every context state, poisoning the
                                         # content-matching recall needs; causal masks got this protection
                                         # for free (nothing before the queries could see them).
QUERY_FULL = False                       # module flag (BIDIR only): query rows attend to the FULL context
                                         # regardless of band — probes read the whole document; the banded
                                         # object whose edit-locality we certify is the CONTEXT-CONTEXT
                                         # block, and queries are readout appendages hanging off it.
NO_POSW = False                          # module flag: drop the learned ABSOLUTE positional embedding from
                                         # h0 (position enters only via the relative bias relb). posw_ablation
                                         # showed posw is load-bearing for the cross-window relay when trained
                                         # with it; this flag tests whether a PURE-relative substrate can relay
                                         # at all — the viability premise of the insert/delete (aligned-frame)
                                         # application story.
QK_NORM = False                          # module flag: cosine attention. L2-normalize q,k over the head dim and
                                         # scale logits by a LEARNED temperature (replacing 1/sqrt(dh)). Decouples
                                         # attention SHARPNESS (the learned tau) from logit MAGNITUDE (unbounded
                                         # q.k) -> caps saturation, so I-J stays off singular (sigma_min up) while
                                         # tau still lets attention peak for recall. Tests the conditioning fix on
                                         # the relative substrate (currnp was 2-8x more ill-conditioned than curr).
ROPE = False                             # module flag: rotary position embedding on q,k (head dim). The
                                         # UN-RULED-OUT conditioning option after QK_NORM (cosine attn) failed:
                                         # QK_NORM capped logit MAGNITUDE and collapsed recall (peaking<->contraction
                                         # tension). RoPE is a pure ROTATION -- norm-preserving, ORTHOGONAL, doesn't
                                         # touch attention sharpness -- so it can't cap peaking. It supplies
                                         # relative position via q,k rotation (dot product depends on j-i), a clean
                                         # single-var swap for currnp's learned additive rel-bias. Hypothesis: a
                                         # cleaner positional signal lets the relay form with LESS saturation ->
                                         # sigma_min up WITHOUT the recall cost QK_NORM paid. RoPE base = 1e4.
MLP = False                              # module flag: add a per-position MLP (GELU, 4x) to the cell, making it
                                         # a full transformer block (attention + MLP) instead of pure attention.
                                         # Needed for ARITHMETIC tasks: pure attention AVERAGES value embeddings
                                         # and a linear head can't recover a content-dependent sum (mod P) from
                                         # an average -> the modular-addition nonlinearity (grokking needs the
                                         # MLP). Opt-in; curr/currnp/bidir substrates leave it off (pure
                                         # attention, as studied). Output-normed + s-scaled so DEQ contraction
                                         # control is unchanged; enters J as an extra per-position block.
FACTORED = False                         # module flag: FACTORED embedding for real-valued tasks. h0 becomes
                                         # concat(mode_emb(toks), values): the first (d - D_VALUE) dims are a
                                         # MODE lookup (role/boundary, from token ids), the last D_VALUE dims are
                                         # a real VALUE payload passed in alongside toks. Readout uses a linear
                                         # REGRESSION head (d -> D_VALUE) with MSE, not the classifier. f() and
                                         # solve() are unchanged (they act on the d-dim state); only h0 build +
                                         # readout differ, so spectrum()/resolvent probes still apply. Opt-in;
                                         # off for every token-classification substrate.
D_VALUE = 0                              # value-subspace width when FACTORED (d_mode = d - D_VALUE)
N_MODES = 4                              # number of mode tokens (filler / value-carrier / boundary / spare)
RESIDUAL = False                         # module flag: add a STATE skip path to the cell -- out += r*z (r
                                         # learnable in [0,R_MAX)). The curr/currnp/bidir cell is residual around
                                         # the INJECTION (out = h0 + s*A(z)) but has NO identity path on z, so
                                         # J = s*dA/dz has no I component. A residual connection adds +r*I to J
                                         # -> I-J = (1-r)I - s*dA/dz: shifts the diagonal directly (structural,
                                         # not just a nonlinear per-position block like MLP). Fixed point stays
                                         # well-posed for r<1 (z* = (h0 + s*A(z*))/(1-r)). Tests how a residual
                                         # moves sigma_min / rho(J) / rho(G) at matched task. Opt-in; off elsewhere.
R_MAX = 0.8                              # cap on the residual coefficient (keeps 1-r away from 0)


def band_causal_mask(L, device):
    i = torch.arange(L, device=device)[:, None]
    j = torch.arange(L, device=device)[None, :]
    if BIDIR:
        m = (i - j).abs() <= W                           # bidirectional band (Faber/BVP face)
        if READONLY_Q:
            m = m & ((j < L - NQ) | (j == i))            # queries attendable by nobody (self excepted)
        if QUERY_FULL:
            m = m | ((i >= L - NQ) & ((j < L - NQ) | (j == i)))   # query rows see all context
        return m
    return (j <= i) & (i - j <= W)                       # (L,L) True = allowed (causal + within window)


class SeqDEQ(nn.Module):
    def __init__(self, kind, mode, K=None):
        super().__init__()
        self.kind, self.mode, self.K = kind, mode, K
        self.emb = nn.Embedding(NKEY + NVAL + NFILL, d)
        self.posw = nn.Parameter(0.02 * torch.randn(256, d))     # absolute positions (substitution edits)
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.s_raw = nn.Parameter(torch.tensor(0.4))
        if QK_NORM:                                          # learned cosine-attention temperature (per head);
            self.qk_tau = nn.Parameter(torch.full((H,), 2.0 * dh ** 0.5))   # init gives peaking headroom
        if REL_BIAS:
            self.relb = nn.Parameter(0.01 * torch.randn(H, 2 * W + 1))   # b[h, (j-i)+W]
        if MLP:                                                  # per-position transformer MLP (arithmetic cell)
            self.mlp_in = nn.Linear(d, 4 * d, bias=False)
            self.mlp_out = nn.Linear(4 * d, d, bias=False)
        if RESIDUAL:                                             # learnable state-skip coefficient (r ~ 0.08 init)
            self.r_raw = nn.Parameter(torch.tensor(-2.2))
        self.head = nn.Linear(d, NVAL)
        if FACTORED:                                             # real-valued factored substrate
            self.mode_emb = nn.Embedding(N_MODES, d - D_VALUE)   # mode/role/boundary (d_mode dims)
            self.head_reg = nn.Linear(d, D_VALUE)                # regression readout over the value subspace
        if mode == "deq":
            self.deq = get_deq(f_solver="anderson", f_max_iter=60, f_tol=1e-4,
                               ift=True, b_solver="anderson", b_max_iter=30)

    @property
    def s(self):
        return S_MAX * torch.sigmoid(self.s_raw)

    @property
    def r(self):
        return R_MAX * torch.sigmoid(self.r_raw)

    def h0(self, toks, values=None):
        if FACTORED:                                             # [ mode(d-D_VALUE) | value(D_VALUE) ]
            return torch.cat([self.mode_emb(toks), values], dim=-1)
        if NO_POSW:
            return self.emb(toks)
        return self.emb(toks) + self.posw[:toks.shape[1]]

    def wn(self):                                            # q/k RAW (must peak); v,o normed (contraction)
        return self.Wq.weight, self.Wk.weight, _sn(self.Wv.weight), _sn(self.Wo.weight)

    def _maskp(self, mask):                                  # precompute once per solve (hoist out of f)
        if self.kind != "softmax":
            return mask.float()
        base = torch.where(mask, 0.0, -1e30)
        if hasattr(self, "relb"):                            # (H,L,L) additive bias broadcast over batch
            L = mask.shape[0]
            i = torch.arange(L, device=mask.device)
            Wt = (self.relb.shape[1] - 1) // 2               # table half-width (fixed at construction;
            delta = (i[None, :] - i[:, None]).clamp(-Wt, Wt) + Wt   # runtime W may be smaller in a w-curriculum)
            return base + self.relb[:, delta]
        return base

    def _rope_cs(self, L, device):                           # rotary cos/sin, cached per (L, device);
        cache = getattr(self, "_rope_cache", None)           # constant across solve iters and across z
        if cache is None:
            cache = {}; self._rope_cache = cache
        key = (L, str(device))
        if key not in cache:
            half = dh // 2
            inv_freq = 10000.0 ** (-torch.arange(half, device=device).float() / half)   # (half,)
            ang = torch.arange(L, device=device).float()[:, None] * inv_freq[None, :]    # (L, half)
            cache[key] = (torch.cos(ang)[None, None], torch.sin(ang)[None, None])        # (1,1,L,half)
        return cache[key]

    def f(self, z, h0, wn, maskp):
        Wq, Wk, Wv, Wo = wn
        B, L, _ = z.shape
        q = (z @ Wq.t()).view(B, L, H, dh).transpose(1, 2)   # B,H,L,dh
        k = (z @ Wk.t()).view(B, L, H, dh).transpose(1, 2)
        v = (z @ Wv.t()).view(B, L, H, dh).transpose(1, 2)
        if ROPE:                                             # rotate q,k -> dot product carries relative pos
            cos, sin = self._rope_cs(L, z.device)
            half = dh // 2
            def _rot(x):
                x1, x2 = x[..., :half], x[..., half:]
                return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
            q, k = _rot(q), _rot(k)
        if self.kind == "softmax":
            if QK_NORM:                                       # cosine attention: unit q,k * learned per-head tau
                q = F.normalize(q, dim=-1)
                k = F.normalize(k, dim=-1)
                sc = self.qk_tau.view(1, H, 1, 1) * (q @ k.transpose(-1, -2)) + maskp
            else:
                sc = (q @ k.transpose(-1, -2)) / (dh ** 0.5) + maskp   # additive bias (0 / -1e30)
            a = torch.softmax(sc, -1)
        else:                                                # linear-kernel attention (Mamba/SSM analog)
            qf, kf = F.elu(q) + 1.0, F.elu(k) + 1.0
            sc = (qf @ kf.transpose(-1, -2)) * maskp          # 0/1 multiplicative mask
            a = sc / (sc.sum(-1, keepdim=True) + 1e-6)
        o = (a @ v).transpose(1, 2).reshape(B, L, d)
        out = h0 + self.s * (o @ Wo.t())
        if RESIDUAL:                                             # state skip -> J gains a +r*I block (I-J -> (1-r)I - sA')
            out = out + self.r * z
        if MLP:                                                  # transformer MLP sub-block (normed + s-scaled)
            hm = F.gelu(out @ _sn(self.mlp_in.weight).t()) @ _sn(self.mlp_out.weight).t()
            out = out + self.s * hm
        return out

    def solve(self, h0, mask):
        wn = self.wn()
        maskp = self._maskp(mask)
        z0 = torch.zeros_like(h0)
        ff = lambda z: self.f(z, h0, wn, maskp)
        if self.mode == "deq":
            return self.deq(ff, z0)[0][-1]
        z = z0
        for _ in range(self.K):
            z = ff(z)
        return z

    def run(self, toks, values=None):
        h0 = self.h0(toks, values)
        mask = band_causal_mask(toks.shape[1], toks.device)
        z = self.solve(h0, mask)
        return self.head_reg(z) if FACTORED else self.head(z)

    def spectrum(self, toks1, values=None):
        """Exact rho(J), sigma_min(I-J), and residual at the fixed point on a single small example."""
        h0 = self.h0(toks1, values)
        mask = band_causal_mask(toks1.shape[1], toks1.device)
        wn = self.wn()
        maskp = self._maskp(mask)
        with torch.no_grad():
            z = self.solve(h0, mask)
            resid = ((self.f(z, h0, wn, maskp) - z).norm() / (z.norm() + 1e-9)).item()
        shape, N = z.shape, z.numel()
        zf = z.reshape(-1).detach()
        ff = lambda zv: self.f(zv.view(shape), h0, wn, maskp).reshape(-1)
        J = torch.func.jacrev(ff)(zf)                        # (N,N) dense -- small example only
        ImJ = torch.eye(N, device=J.device) - J
        smin = torch.linalg.svdvals(ImJ).min().item()
        rho = torch.linalg.eigvals(J).abs().max().item()
        return rho, smin, resid


def recall(model, Fill, gen, reps=4):
    accs = []
    for _ in range(reps):
        toks, qmask, targ = gen_mqar(128, Fill, gen)
        with torch.no_grad():
            logits = model.run(toks)
        accs.append((logits.argmax(-1)[qmask] == targ[qmask]).float().mean().item())
    return float(np.mean(accs))


def train(model, steps=1000, bs=64):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(0)
    for st in range(steps):
        Fill = F_SWEEP[torch.randint(len(F_SWEEP), (1,), generator=gen).item()]     # sample a gap
        toks, qmask, targ = gen_mqar(bs, Fill, gen)
        model.train(); opt.zero_grad()
        logits = model.run(toks)
        loss = F.cross_entropy(logits[qmask], targ[qmask])
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if torch.isfinite(loss) and torch.isfinite(gn):
            opt.step()
        else:
            opt.zero_grad()
        if st % 250 == 0:
            print(f"      step {st:>4} loss {loss.item():.3f}", flush=True)
    return loss.item()


def main():
    print(f"device = {DEV}  window w={W}, H={H} heads, {D_PAIR} pairs + filler + {NQ} queries; "
          f"unroll reaches ~K*w, equilibrium (torchdeq ift+Anderson) reaches further\n", flush=True)
    configs = [                                             # name, kind, mode, K
        ("eq-softmax", "softmax", "deq", None),
        ("unroll2-softmax", "softmax", "unroll", 2),
        ("unroll4-softmax", "softmax", "unroll", 4),
        ("eq-linear", "linear", "deq", None),
    ]
    rows, spec = {}, {}
    for name, kind, mode, K in configs:
        print(f"[{name}] training ({mode}{'' if K is None else f' K={K}'}) ...", flush=True)
        torch.manual_seed(0)
        m = SeqDEQ(kind, mode, K).to(DEV)
        t0 = time.time()
        train(m)
        m.eval()
        ge = torch.Generator().manual_seed(123)
        rows[name] = [recall(m, Fl, ge) for Fl in F_SWEEP]
        try:
            spec[name] = m.spectrum(gen_mqar(1, 0, torch.Generator().manual_seed(7))[0])
        except Exception as e:
            spec[name] = (float("nan"), float("nan"), float("nan")); print(f"  spectrum failed: {e}", flush=True)
        print(f"  done ({time.time()-t0:.0f}s)  rho={spec[name][0]:.3f} sigma_min(I-J)={spec[name][1]:.3f} "
              f"resid={spec[name][2]:.1e}\n", flush=True)

    print("RECALL vs gap  (reach ~K*w: unroll2~20, unroll4~40; column = filler F, gap~F+8)")
    print("model              " + "  ".join(f"F={f:>2}" for f in F_SWEEP), flush=True)
    for name, _, _, _ in configs:
        print(f"{name:<18} " + "  ".join(f"{a:>4.2f}" for a in rows[name]), flush=True)
    print("\nspectrum @ fixed point (F=0):")
    for name, _, mode, _ in configs:
        r, sm, rs = spec[name]
        print(f"  {name:<18} rho={r:>6.3f}  sigma_min(I-J)={sm:>6.3f}  resid={rs:.1e}"
              f"{'' if mode=='deq' else '  (unroll: not a true fixed point)'}", flush=True)
    print(f"\nREAD: if eq-softmax stays high as F grows while unroll-K collapses past gap~K*w, then LOCAL "
          f"attention + EQUILIBRIUM = long-range reach (equilibrium earns expressivity beyond finite "
          f"depth); eq-linear low throughout confirms the selection gap is the softmax's doing. sigma_min>0 "
          f"at rho>1 = the map is invertible around the fixed point (edit-locality certifiable via Faber).", flush=True)


if __name__ == "__main__":
    main()
