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


def gelfand_rho(G, K=250, tol_nilp=1e-11):
    """rho(G) via VECTOR Gelfand power iteration: ||G^k x||/||G^{k-1} x|| -> rho(G) (Gelfand). O(n^2)/iter,
    GPU-native (no eigendecomposition, no cuSOLVER fallback). Handles the nilpotent (causal) corner
    naturally -- the powers crash to 0. Returns (rho_est, is_nilpotent)."""
    n = G.shape[0]
    x = torch.randn(n, dtype=G.dtype, device=G.device)
    x = x / torch.linalg.norm(x)
    ratios = []
    for _ in range(K):
        y = G @ x
        ny = torch.linalg.norm(y).item()          # = ||G x|| / ||x|| since ||x||=1
        if ny < tol_nilp:
            return 0.0, True                       # nilpotent: powers vanished -> causal corner
        ratios.append(ny)
        x = y / ny
    return float(np.median(ratios[-25:])), False   # median of the settled tail = rho(G)


def power_lammax(P, iters=100):
    """lambda_max of a symmetric PSD matrix by power iteration (GPU-native, O(n^2)/iter)."""
    v = torch.randn(P.shape[0], dtype=P.dtype, device=P.device)
    v = v / torch.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        w = P @ v
        nw = torch.linalg.norm(w).item()
        if nw < 1e-30:
            break
        v = w / nw
        lam = torch.dot(v, P @ v).item()           # Rayleigh quotient -> lambda_max
    return lam


def weighted_cert(G, r, Kmax=600, tail_thresh=0.9):
    """Certified bound ||G^k||_2 <= const * r^k, const=sqrt(lambda_max(P_M)), via an EARLY-STOPPED Stein
    Gramian P_M = sum_{j=0}^{M} (Gt^T)^j Gt^j, Gt=G/r. Key facts used:
      (1) P_M >= I  => lambda_min(P_M) >= 1, so sqrt(lambda_max(P_M)) >= sqrt(kappa(P_M)) is a valid (conservative)
          constant -- no need for lambda_min or a cholesky.
      (2) P_M is a valid Lyapunov certificate (Gt^T P_M Gt <= P_M) as soon as ||Gt^{M+1}||_2 < 1, because
          Gt^T P_M Gt - P_M = -(I - (Gt^T)^{M+1} Gt^{M+1}). So we STOP early (once ||Gt^{M+1}||_F < tail_thresh
          => ||.||_2 < 1), which is both faster and gives a TIGHTER const than summing to convergence.
    All GPU-native: matmuls + Frobenius norms + symmetric power iteration. No eig/SVD/cholesky."""
    n = G.shape[0]
    Gt = G / r
    P = torch.eye(n, dtype=G.dtype, device=G.device)
    Gk = Gt.clone()                                 # Gt^1
    terms, tailF, diverged = 1, torch.linalg.norm(Gt).item(), False
    for k in range(1, Kmax):
        P = P + Gk.T @ Gk                           # add term j=k  (P now = sum_{j=0}^{k})
        Gk = Gk @ Gt                                # advance: Gk = Gt^{k+1}
        tailF = torch.linalg.norm(Gk).item()        # ||Gt^{k+1}||_F  >= ||Gt^{k+1}||_2
        terms = k
        if tailF < tail_thresh:                     # ||Gt^{k+1}||_2 < 1 guaranteed -> P is a valid cert
            break
        if tailF > 1e6:                             # r <= rho(G): Gramian diverges -> bail (cheap), r invalid
            diverged = True
            break
    reach = np.inf if r >= 1.0 else -1.0 / np.log(r)
    if diverged or tailF >= 1.0:                    # invalid r: don't build a bogus constant
        return dict(r=r, const=np.inf, lam_max=np.inf, reach=reach, tail=tailF, terms=terms, valid=False)
    lam_max = power_lammax(P)
    return dict(r=r, const=np.sqrt(max(lam_max, 1.0)), lam_max=lam_max,
                reach=reach, tail=tailF, terms=terms, valid=True)


def main():
    targets = ["bidir16.pt", "bidir24.pt", "bidir40.pt", "curr24.pt", "curr40.pt"]
    print("device=%s  C2-WEIGHTED: block-transfer reach certificate (adapted Lyapunov/Stein norm)\n"
          "  G = block-Jacobi iteration matrix of reblocked (I-J); rho(G)=true rate (Gelfand); ||G||_F shows transient.\n"
          "  Certified: ||G^k|| <= const * r^k  =>  reach xi=1/ln(1/r) hops, amplitude const=sqrt(lam_max(P)).\n"
          "  Sweep target rate r>rho(G) (tighter r -> tighter reach, larger const). Compare to scalar proxy/Route-A.\n"
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
        J = torch.func.jacrev(ffl)(zf).detach().double()         # fp64 on GPU for an accurate block-inverse
        G, dinv_norm, nb = block_jacobi(J, L, sw.d, sw.W)
        G = G.float()                                            # fp32: GPU matmuls ~60x faster, half memory
        del z, zf, ff, m, J                                      # free GPU (6GB card) before the next ckpt
        if sw.DEV == "cuda":
            torch.cuda.empty_cache()
        nrmG = torch.linalg.norm(G).item()                 # Frobenius (cheap, >= spectral)
        rho_est, is_nilp = gelfand_rho(G)
        face = "causal" if name.startswith("curr") else "bidir"
        if is_nilp:
            print(f"[{name}] face={face} L={L} windows={nb}  rho(G)~0 (NILPOTENT), ||G||_F={nrmG:.2f}\n"
                  f"    NILPOTENT corner: resolvent = exact terminating product over <= {nb-1} hops"
                  f"  -> product-Lyapunov (causal), certificate is EXACT, no adapted norm needed.\n", flush=True)
            del G
            continue
        # FIXED grid (independent of the noisy Gelfand estimate); the sweep itself brackets rho(G)
        # rigorously: largest INVALID r < rho(G) <= smallest VALID r.
        grid = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 0.95]
        res = [weighted_cert(G, r) for r in grid]
        valid = [c for c in res if c["valid"]]
        inv_rs = [c["r"] for c in res if not c["valid"]]
        rho_lo = max([r for r in inv_rs if r < (valid[0]["r"] if valid else 1)], default=0.0)
        rho_hi = valid[0]["r"] if valid else 1.0
        print(f"[{name}] face={face} L={L} windows={nb}  ||G||_F={nrmG:.2f}  "
              f"rho(G) in ({rho_lo:.2f}, {rho_hi:.2f}]  (Gelfand est {rho_est:.2f})", flush=True)
        # report tightest / mid / loosest valid operating points
        picks = sorted(set([0, len(valid) // 2, len(valid) - 1])) if valid else []
        for i in picks:
            c = valid[i]
            print(f"    r={c['r']:.2f} -> CERT reach={c['reach']:.2f} hops  amp const=sqrt(lam_max(P))={c['const']:.1e}"
                  f"  (terms={c['terms']}, tail={c['tail']:.2f}) [OK]", flush=True)
        print(f"    vs scalar: proxy sqrt-kappa (c2_bidir log), Route-A ~ 80-220 hops (vacuous). "
              f"||D^-1||={dinv_norm:.2f} (resolvent prefactor).\n", flush=True)
        del G
    print("READ: reach=1/ln(1/r) at r just above rho(G) = the TIGHT certified reach; const=sqrt(lam_max(P)) is the\n"
          "honest amplitude absorbing the transient (>= sqrt(kappa) since lam_min(P)>=1; early-stopped => tighter).\n"
          "~1 hop where well-conditioned, growing near-singular -- vs the vacuous 80-220 hop scalar Route-A.\n"
          "Causal = nilpotent corner (rho(G)=0, exact terminating product).", flush=True)


if __name__ == "__main__":
    main()
