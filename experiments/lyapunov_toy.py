"""Toy: 5x5 causal banded J (window w=2, scalar 'blocks' d=1 for readability).
Three certificates on the same matrix: eigenvalues/rho, kappa-Chebyshev, FOV-Faber, product-Lyapunov.
Three cases: (A) mild coupling; (B) pure Mamba (bidiagonal, no self-coupling); (C) 'rho lies'
(strong coupling, small diagonal -- spectrum says stable, influence actually GROWS).
"""
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=120)


def build_J(delta, a, b):
    """Causal banded: J[i,i]=delta_i (self), J[i,i-1]=a_i (1-hop), J[i,i-2]=b_i (2-hop)."""
    L = len(delta)
    J = np.zeros((L, L))
    for i in range(L):
        J[i, i] = delta[i]
        if i >= 1: J[i, i - 1] = a[i]
        if i >= 2: J[i, i - 2] = b[i]
    return J


def fov_support(A, n=720):
    """h(theta) = lambda_max(Herm(e^{i theta} A)). 0 in W(A) iff h(theta) >= 0 for ALL theta."""
    hs = []
    for th in np.linspace(0, 2 * np.pi, n, endpoint=False):
        H = (np.exp(1j * th) * A + (np.exp(1j * th) * A).conj().T) / 2
        hs.append(np.linalg.eigvalsh(H)[-1])
    return np.array(hs)


def report(name, delta, a, b):
    J = build_J(delta, a, b)
    L = len(delta)
    ImJ = np.eye(L) - J
    R = np.linalg.inv(ImJ)                                   # exact resolvent (L1, the oracle)

    eig = np.linalg.eigvals(J)
    sv = np.linalg.svd(ImJ, compute_uv=False)
    smin, kappa = sv[-1], sv[0] / sv[-1]
    rk = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)         # old kappa-Chebyshev rate (SPD-only theorem)

    # product-Lyapunov certificate: one-hop coupling alpha, self-coupling delta -> g = alpha/(1-delta)
    alpha = max(abs(a[i]) + abs(b[i]) for i in range(1, L))  # max row coupling norm (band row-sum)
    dmax = max(abs(d) for d in delta)
    g = alpha / (1 - dmax)

    # measured decay: mean |R| on each sub-diagonal, and the empirical per-step ratio
    diag_means = [np.mean([abs(R[i, i - k]) for i in range(k, L)]) for k in range(L)]
    ratios = [diag_means[k + 1] / diag_means[k] for k in range(L - 1)]

    hs = fov_support(ImJ)
    zero_in_W = hs.min() >= 0                                # 0 inside W(I-J) => Faber bound VACUOUS

    print(f"--- {name} ---")
    print(f"J =\n{build_J(delta, a, b)}")
    print(f"eig(J)              : {np.sort_complex(eig)}   rho={abs(eig).max():.3f}"
          f"   <- ONLY the diagonal (blind to coupling a,b)")
    print(f"sigma_min(I-J)      : {smin:.4f}   kappa: {kappa:.1f}   kappa-Chebyshev rate r={rk:.3f}")
    print(f"FOV of (I-J)        : 0 {'INSIDE  -> Faber bound VACUOUS' if zero_in_W else 'outside -> Faber usable'}"
          f"   (min support h = {hs.min():.3f})")
    print(f"product-Lyapunov    : alpha={alpha:.2f}, delta={dmax:.2f} -> g = alpha/(1-delta) = {g:.3f}"
          f"   ({'decay' if g < 1 else 'GROWTH'} certificate)")
    print(f"exact resolvent |R| mean per sub-diagonal k=0..{L-1}: {np.array(diag_means)}")
    print(f"measured per-step ratio (should track g, NOT r or rho): {np.array(ratios)}\n")
    return R


# (A) mild coupling: everything sane, all certificates agree qualitatively
report("A: mild  (delta=0.3, a~0.5, b=0.15)",
       delta=[0.3] * 5, a=[0, .5, .55, .45, .5], b=[0, 0, .15, .15, .15])

# (B) pure Mamba: bidiagonal, no self-coupling -> resolvent sub-diagonal = EXACT products of a_i
R = report("B: Mamba (delta=0, b=0, a=[.9,.5,.99,.7])",
           delta=[0] * 5, a=[0, .9, .5, .99, .7], b=[0] * 5)
print("   check B: R[4,0] =", f"{R[4,0]:.4f}", "  vs  a4*a3*a2*a1 =", f"{.7*.99*.5*.9:.4f}",
      "  <- influence of x0 on x4 is LITERALLY the product\n")

# (C) rho lies: strong coupling (a=2), small diagonal -> spectrum says 'stable', influence GROWS
report("C: rho lies (delta=0.3, a=2.0, b=0)",
       delta=[0.3] * 5, a=[0, 2, 2, 2, 2], b=[0] * 5)

print("COST: product-Lyapunov needs only the BAND's block norms: O(L * w) blocks, each d x d")
print("      -> O(L * w * d^2) work, LINEAR in sequence length, no inverse, no eigensolve.")
print("      sigma_min / FOV need the dense (Ld)x(Ld) matrix: O((Ld)^3). At L=100k, w=10, d=64:")
print("      product ~ 4e10 flops (seconds); dense sigma_min ~ 2.6e20 flops (impossible).")
