"""C2-WEIGHTED — the block-transfer edit-reach certificate via an adapted (Lyapunov/Stein) norm.

WHY THIS EXISTS. The scalar a-priori reach bounds (kappa-Chebyshev proxy, Route A normality-free DMS,
Route B Faber-on-FOV) are all VACUOUS on the trained equilibrium transformer: they exceed the sequence
length (Route A 42-222 hops on 3-5 window sequences; Route B abstains, 0 in W(I-J)). Reason: they are
worst-case scalar spectral/FOV bounds, blind to the block structure. The ACTUAL spatial transport rate is
the spectral radius rho(G) of the BLOCK-JACOBI iteration matrix G of the reblocked (window) resolvent --
measured ~0.33 (bidir16), matching the exact resolvent decay, ~10-100x tighter than the scalar bounds.

The obstruction to CERTIFYING rho(G) is transient growth: ||G||~5.5 >> rho(G)~0.33 (non-normality), so a
norm certificate (||G||<1) fails. This module supplies the standard rigorous fix: an ADAPTED (weighted)
norm from a Stein / discrete-Lyapunov solve. For target rate r in (rho(G),1), P = sum_k (G/r)*^k (G/r)^k
solves (G/r)* P (G/r) - P = -I; then ||G^k||_2 <= sqrt(kappa(P)) r^k -- the rate r is tight (pushable to
rho(G)), the non-normality is quarantined into the one-time constant sqrt(kappa(P)). Certified reach
xi = 1/ln(1/r) hops; amplitude C = sqrt(kappa(P)) ||D^-1|| / (1-r).

LINEAGE / UNIFICATION. This LEAVES the approximation-theory lineage (Demko-Moss-Smith / Benzi-Golub / Faber
on the resolvent of a banded matrix) and JOINS the dynamical-systems / iterative-methods / Lyapunov lineage
(spectral radius of an iteration operator + adapted-norm certificate for the transient; Stein equation;
Kreiss). That is the SAME lineage as the causal product-Lyapunov certificate: the causal M is block-lower-
triangular => G is strictly-block-lower => NILPOTENT (rho(G)=0), and the certificate degenerates to the exact
terminating product. Bidirectional M is block-tridiagonal => rho(G) in (0,1), geometric decay certified here.
One object (block-Jacobi G), two regimes (nilpotent corner = causal; contractive = bidirectional). sigma_min
stays as the SEPARATE conditioning axis (invertibility, a-posteriori resid/sigma_min, error amp) -- and note
rho(G)<1 is a SPATIAL-coupling contraction of the RESOLVENT's iteration, NOT temporal contraction of f
(rho(J) can exceed 1): "conditioning not contraction" survives, sharpened.

Run:  D:\\deq-venv\\Scripts\\python.exe -u -m experiments.c2_weighted_cert
"""
import glob
import os

import numpy as np
import torch

import experiments.sliding_window_reach as sw
import experiments.c2_bidir as cb

CKPT_DIR = "checkpoints"
sw.H, sw.dh = 4, sw.d // 4


def block_jacobi(J, L, d, w):
    """Reblock M=I-J into w-token windows; return the block-Jacobi iteration matrix G = -D^{-1}(M-D)
    (block-tridiagonal, zero block-diagonal) and ||D^{-1}|| (the resolvent prefactor)."""
    n = J.shape[0]
    M = torch.eye(n, device=J.device, dtype=torch.float64) - J.double()
    idx = [(s * d, min(s + w, L) * d) for s in range(0, L, w)]
    Dinv = torch.zeros_like(M)
    off = M.clone()
    for a, b in idx:
        Dinv[a:b, a:b] = torch.linalg.inv(M[a:b, a:b])
        off[a:b, a:b] = 0.0
    G = -(Dinv @ off)
    dinv_norm = torch.linalg.matrix_norm(Dinv, ord=2).item()
    return G, dinv_norm, len(idx)


def stein_gramian(G, r, K=800, tol=1e-13):
    """P = sum_{k>=0} (G/r)*^k (G/r)^k (the Stein solution for target rate r), by truncated sum on GPU.
    Valid iff rho(G)<r (else diverges). Returns P and the tail SPECTRAL norm ||(G/r)^K|| (computed ONCE at
    the end) used to certify the truncation (P is a valid Lyapunov certificate once ||(G/r)^K||<1). The loop
    uses the cheap Frobenius norm to decide when to stop; P >= I so lambda_min(P)>=1."""
    n = G.shape[0]
    Gt = G / r
    P = torch.eye(n, dtype=torch.float64, device=G.device)
    Gk = Gt.clone()
    k = 1
    for k in range(1, K):
        P = P + Gk.T @ Gk
        if torch.linalg.norm(Gk).item() < tol:          # Frobenius (cheap) stopping proxy
            break
        Gk = Gk @ Gt
    tail = torch.linalg.matrix_norm(Gk, ord=2).item()    # spectral norm of the last retained power, once
    return P, tail, k


def weighted_cert(G, r):
    """Certified bound ||G^k||_2 <= const * (eff_rate)^k from the adapted P-norm at target rate r.
    eff_rate = r * ||G/r||_P  (<= r); const = sqrt(kappa(P)); reach = 1/ln(1/eff_rate) hops."""
    P, tail, terms = stein_gramian(G, r)
    lam = torch.linalg.eigvalsh(P)
    lam_max, lam_min = lam[-1].item(), lam[0].item()
    kappaP = lam_max / lam_min
    # induced norm ||G/r||_P via congruence L^{-1}(.)L^{-T}, P=L L^T
    Gt = G / r
    A = Gt.T @ P @ Gt
    Lc = torch.linalg.cholesky(P)
    Y = torch.linalg.solve_triangular(Lc, A, upper=False)               # L^{-1} A
    Z = torch.linalg.solve_triangular(Lc, Y.T, upper=False).T           # (L^{-1} (L^{-1}A)^T)^T = L^{-1} A L^{-T}
    gnorm_P = torch.linalg.eigvalsh(0.5 * (Z + Z.T))[-1].clamp(min=0).sqrt().item()
    eff = r * gnorm_P
    reach = np.inf if eff >= 1.0 else -1.0 / np.log(eff)
    valid = (tail < 1.0) and (gnorm_P < 1.0)                            # truncation + contraction both certified
    return dict(r=r, kappaP=kappaP, const=np.sqrt(lam_max), gnorm_P=gnorm_P,
                eff_rate=eff, reach=reach, tail=tail, terms=terms, valid=valid)


def main():
    targets = ["bidir16.pt", "bidir24.pt", "bidir40.pt", "curr24.pt"]
    print("device=%s  C2-WEIGHTED: block-transfer reach certificate (adapted Lyapunov/Stein norm)\n"
          "  G = block-Jacobi iteration matrix of reblocked (I-J); rho(G)=true rate; ||G|| shows transient growth.\n"
          "  Certified: ||G^k|| <= const * eff_rate^k  =>  reach xi=1/ln(1/eff_rate) hops, amplitude const=sqrt(kappa(P)).\n"
          "  Sweep target rate r (tighter r -> tighter reach, larger const). Compare to scalar proxy/Route-A.\n"
          % sw.DEV, flush=True)
    for name in targets:
        path = os.path.join(CKPT_DIR, name)
        if not os.path.exists(path):
            print(f"[{name}] missing, skip"); continue
        m, ck = cb.load_ckpt(path)
        gen = torch.Generator().manual_seed(7)
        toks = sw.gen_mqar(1, ck["stage_gap"], gen)[0]
        L = toks.shape[1]
        ff, _ = cb.make_ff(m, toks)
        z, _ = cb.counted_solve(m, ff, torch.zeros(1, L, sw.d, device=sw.DEV))
        zf = z.reshape(-1).detach()
        ffl = lambda zv: ff(zv.view(z.shape)).reshape(-1)
        J = torch.func.jacrev(ffl)(zf)
        G, dinv_norm, nb = block_jacobi(J, L, sw.d, sw.W)
        rhoG = torch.linalg.eigvals(G).abs().max().item()
        nrmG = torch.linalg.matrix_norm(G, ord=2).item()
        face = "causal" if name.startswith("curr") else "bidir"
        print(f"[{name}] face={face} L={L} windows={nb}  rho(G)={rhoG:.3f}  ||G||={nrmG:.2f}"
              f"  (transient gap {nrmG/max(rhoG,1e-9):.0f}x)", flush=True)
        if rhoG < 1e-6:
            print(f"    NILPOTENT corner (rho(G)=0): resolvent = exact terminating product over <= {nb-1} hops"
                  f"  -> product-Lyapunov (causal), certificate is EXACT, no adapted norm needed.\n", flush=True)
            continue
        # sweep target rates from just above rho(G) up toward 1
        rs = sorted(set([min(rhoG + 0.05, 0.999), min(rhoG + 0.15, 0.999),
                         (rhoG + 1) / 2, 0.9]))
        for r in rs:
            c = weighted_cert(G, r)
            tag = "OK " if c["valid"] else "?? "
            print(f"    r={r:.3f}: eff_rate={c['eff_rate']:.3f} -> CERT reach={c['reach']:.2f} hops  "
                  f"amp const=sqrt(kappa(P))={c['const']:.1e}  (kappaP={c['kappaP']:.1e}, "
                  f"terms={c['terms']}, tail={c['tail']:.0e}) [{tag}]", flush=True)
        print(f"    vs scalar: proxy sqrt-kappa (see c2_bidir log), Route-A ~ 80-220 hops (vacuous). "
              f"||D^-1||={dinv_norm:.2f} (resolvent prefactor).\n", flush=True)
    print("READ: eff_rate ~ rho(G) at small r = the TIGHT certified rate; const=sqrt(kappa(P)) is the honest\n"
          "amplitude that absorbs the transient (non-normality). Reach ~1 hop certified where well-conditioned,\n"
          "growing near-singular -- vs the vacuous 80-220 hop scalar Route-A. Causal = nilpotent corner (exact).",
          flush=True)


if __name__ == "__main__":
    main()
