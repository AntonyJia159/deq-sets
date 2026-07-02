"""Speed benchmark for the DEQ training step (we run this a lot now, so it's worth tuning). The workload is
LAUNCH-BOUND, not FLOP-bound: tiny model (d=64, H=4), tiny batch, but an iterative Anderson solve calls the
cell f() ~30-60x per forward and ~30x per backward -> dozens of tiny GPU kernels dominated by launch/python
overhead. Levers tested (throughput = examples/sec of a full fwd+bwd training step, higher=better):

  baseline      : current settings (bs=64, f_tol=1e-4, fp32 matmul)
  tf32          : allow_tf32 (Ada supports it) -- near-free
  bs256 / bs512 : amortize fixed per-step overhead over more examples (GPU is under-utilized at bs=64)
  ftol1e-3      : looser solve tol -> fewer solver iterations during training
  fused-qkv     : one (d->3d) projection instead of 3 separate matmuls (fewer kernel launches)
  compile       : torch.compile(f) -- fuse the cell's tiny kernels across solver iterations (biggest
                  expected win for a launch-bound iterative solver)

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.bench_deq_step
"""
import time

import torch
import torch.nn.functional as F

import experiments.sliding_window_reach as sw

DEV = sw.DEV
L_TEST = 26          # ~ gap-16 stage (the slow, near-rho=1 regime)


def make_batch(bs, gen):
    toks, qmask, targ = sw.gen_mqar(bs, L_TEST - 2 * sw.D_PAIR - sw.NQ, gen)
    return toks, qmask, targ


def time_step(model, bs, steps=40, warmup=8):
    gen = torch.Generator().manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for i in range(warmup + steps):
        if i == warmup:
            torch.cuda.synchronize(); t0 = time.time()
        toks, qmask, targ = make_batch(bs, gen)
        opt.zero_grad()
        logits = model.run(toks)
        loss = F.cross_entropy(logits[qmask], targ[qmask])
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    return dt, bs / dt


def fresh(bs=64, ftol=1e-4, fused=False, compile_f=False):
    torch.manual_seed(0)
    m = sw.SeqDEQ("softmax", "deq").to(DEV)
    m.deq = sw.get_deq(f_solver="anderson", f_max_iter=60, f_tol=ftol,
                       ift=True, b_solver="anderson", b_max_iter=30)
    if fused:                                    # fuse Wq,Wk,Wv into one d->3d projection
        Wqkv = torch.nn.Parameter(torch.cat([m.Wq.weight, m.Wk.weight, m.Wv.weight], 0).detach())
        m.register_parameter("Wqkv", Wqkv)

        def f_fused(z, h0, wn, maskp):
            _, _, _, Wo = wn
            B, Lc, _ = z.shape
            qkv = z @ m.Wqkv.t()
            q, k, v = qkv.split(sw.d, dim=-1)
            q = q.view(B, Lc, sw.H, sw.dh).transpose(1, 2)
            k = k.view(B, Lc, sw.H, sw.dh).transpose(1, 2)
            v = v.view(B, Lc, sw.H, sw.dh).transpose(1, 2)
            sc = (q @ k.transpose(-1, -2)) / (sw.dh ** 0.5) + maskp
            a = torch.softmax(sc, -1)
            o = (a @ v).transpose(1, 2).reshape(B, Lc, sw.d)
            return h0 + m.s * (o @ Wo.t())
        m.f = f_fused
    if compile_f:
        m.f = torch.compile(m.f)
    return m


def main():
    print(f"device={DEV}  L={L_TEST} (near-rho=1 regime), fwd+bwd training step throughput\n", flush=True)
    print(f"{'config':<14} {'sec/step':>9} {'examples/s':>11} {'speedup':>8}", flush=True)
    base = None
    runs = [
        ("baseline",   dict(bs=64)),
        ("bs256",      dict(bs=256)),
        ("bs512",      dict(bs=512)),
        ("ftol1e-3",   dict(bs=256, ftol=1e-3)),
        ("fused-qkv",  dict(bs=256, fused=True)),
        ("compile",    dict(bs=256, compile_f=True)),
        ("all-opt",    dict(bs=256, ftol=1e-3, fused=True, compile_f=True)),
    ]
    for name, kw in runs:
        bs = kw.pop("bs")
        try:
            if name == "tf32" or name == "all-opt":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            m = fresh(bs=bs, **kw)
            dt, thru = time_step(m, bs)
            if base is None and name == "baseline":
                base = thru
            sp = thru / base if base else 1.0
            print(f"{name:<14} {dt*1000:>8.1f}m {thru:>11.0f} {sp:>7.2f}x", flush=True)
        except Exception as e:
            print(f"{name:<14} FAILED: {repr(e)[:80]}", flush=True)
    # tf32 as a separate toggle on baseline bs
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        m = fresh(bs=256)
        dt, thru = time_step(m, 256)
        print(f"{'tf32(bs256)':<14} {dt*1000:>8.1f}m {thru:>11.0f} {thru/base:>7.2f}x", flush=True)
    except Exception as e:
        print(f"{'tf32':<14} FAILED: {repr(e)[:80]}", flush=True)
    print("\nREAD: examples/s is throughput; pick the knobs that stack. compile+bs are the expected big "
          "wins for a launch-bound iterative solver; tf32/fused are near-free adds. Verify learning still "
          "holds (short curriculum) before adopting bs/ftol changes.", flush=True)


if __name__ == "__main__":
    main()
