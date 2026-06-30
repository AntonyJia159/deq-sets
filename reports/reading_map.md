# Reading map — the three threads, the bridge, and which result rests on which fact

A single index for the eclectic pieces. The project sits at the confluence of three lineages;
the contribution is the **bridge** plus a **conditioning theory of editability**. Structured to
ZJ's own synthesis (2026-06-29).

---

## One-sentence frame

> A Graph-DEQ is simultaneously a **nonlinear PageRank** and a **contractive NCA run to infinite
> time**. That dual identity buys expressivity (recurrentize any off-the-shelf operator, *including
> across semirings*) and a clean editability theory — local maintenance is governed by the
> **well-conditioning of `(I−J)`** (Demko–Moss–Smith / resolvent decay), which *subsumes* the
> linear-PageRank editing trick as the special case where the resolvent is linear.

---

## Thread A — PageRank / generalized-linear graph methods (the LINEAR ancestor)

- **What it is.** Linear graph interaction (a PPR-type resolvent) sandwiched by nonlinear
  encode/decode layers. Gives *infinite **linear** reach* in one shot, and — crucially — **local
  incremental editing** by exploiting linearity (maintain residuals, push only the delta).
- **Canonical works.** PageRank; APPNP / "Predict then Propagate" (Gasteiger/Klicpera et al., ICLR
  2019, arXiv:1810.05997); SGC (Wu et al., ICML 2019, arXiv:1902.07153); SIGN (Frasca et al. 2020,
  arXiv:2004.11198); **InstantGNN** (Zheng et al., KDD 2022, arXiv:2206.01379) and the dynamic-PPR
  incremental line.
- **What we take / where we depart.** Their cheap exact updates *require* linearity (push invariants
  need superposition). We show the enabler is **well-conditioning, not linearity** → their methods
  are the **linear-resolvent special case** of ours.
  - *Our result:* `maintenance_compare.py` — the "InstantGNN-linear" baseline is literally our cell
    with the message made linear (`z=αh₀+(1−α)Âz`); both are exactly maintainable, expressivity
    differs only by the (nonlinear) message.

## Thread B — Neural Cellular Automata (the NONLINEAR self-organisation ancestor)

- **What it is.** Nonlinear *local* update rules; emergent self-organisation; **local
  destruction → regeneration** (damage a region, the rule heals it).
- **Canonical works.** Growing NCA (Mordvintsev et al., Distill 2020); Graph NCA / Learning Graph
  Cellular Automata (Grattarola et al., NeurIPS 2021); **ZJ's own NCA↔DEQ paper** (arXiv:2501.03573)
  — self-cite, stakes the equivalence.
- **What we take / where we depart.** A DEQ **is** a *contractive NCA run to infinite time* (the
  equilibrium is the `t→∞` limit of the cellular rule), but with a **more concise and stable
  training theory** than NCAs — implicit differentiation + `σ_min` conditioning instead of
  unrolled-BPTT heuristics. NCA "regeneration-after-damage" becomes **warm-start local re-solve**,
  now with *exactness / contraction guarantees* NCAs lack.
  - *Our result:* `maintenance_demo.py` / `maintenance_tropical.py` — delete a node, warm-start
    re-solve; warm == cold to 1e-7 (path-independent), response decays with distance (`ξ`).

## Thread C — Graph-DEQs / IGNNs (the synthesis point)

- **What it is.** Prediction = fixed point `z = f(z)`; a *nonlinear extension of PageRank* and a
  *limiting case of NCA* at once.
- **Canonical works.** DEQ (Bai, Kolter, Koltun, NeurIPS 2019, arXiv:1909.01377); IGNN (Gu et al.,
  NeurIPS 2020, arXiv:2009.06211); Monotone Operator Equilibrium Nets (Winston & Kolter, 2020);
  TorchDEQ (Geng & Kolter, arXiv:2310.18605).
- **What we add.** The **editability/maintenance theory** (below) and the **semiring expansion** of
  the operator design space.

---

## The bridge — recurrentize the broader DL ecosystem, *especially across semirings*

Casting any off-the-shelf operator as an equilibrium opens a **more liberal design space**: signed
attention (FAGCN), per-edge message MLPs (GatedGCN/GINE), and — the categorical one —
**different semirings**.

- **Sum-product = linear algebra = all of spectral/manifold graph theory.** Any `g(L)` is a linear
  functional, so spectral tasks are `φ(linear aggregate)` and ties the linear incumbents (Report #8 §5).
- **Tropical (max-plus) is categorically out of reach of linear-push methods** — `max` is an order
  statistic, not a function of any linear aggregate. This is the **neural-algorithmic-reasoning**
  connection: the aggregator must match the algorithm's semiring.
  - *Refs:* Veličković et al., *Neural Execution of Graph Algorithms* (ICLR 2020, arXiv:1910.10593,
    max-agg ↔ Bellman–Ford); Xu et al., *What Can Neural Networks Reason About?* (ICLR 2020,
    arXiv:1905.13211, algorithmic alignment); Dudzik & Veličković, *GNNs are Dynamic Programmers*
    (NeurIPS 2022, arXiv:2203.15544); Mensch & Blondel, *Differentiable Dynamic Programming* (ICML
    2018, arXiv:1802.03676, smoothed/log-sum-exp DP).
  - *Our result:* `semiring_compare.py` 2×2 (sum→max +0.12 controlled); `beta_fix_check.py`
    (log-sum-exp generalist, trails the hard-max specialist by ~0.09).

---

## The theoretical backbone — which result rests on which classical fact

**Relocate local editability from *linearity* to *operator well-conditioning*.** Poke the fixed
point → `(I−J)δu=δf` → the response is the **resolvent** `(I−J)⁻¹`. Edit-locality = the resolvent's
entries decay with graph distance.

| Claim | Classical fact | Governing quantity | Our experiment |
|---|---|---|---|
| Resolvent of a graph-sparse, well-conditioned `(I−J)` decays with distance | **Demko–Moss–Smith** (1984, banded inverses); **Benzi–Golub** (1999, matrix functions in graph distance) — via **Chebyshev/Faber** best-poly approximation excluding the pole `z=1` | `dist(1, W(J))`; `σ_min(I−J)` as the L2 proxy — **not** `ρ(J)` | `broyden_conditioning.py` (σ_min dissociation); `sigma_min_law.py` — the **quantitative** test: Faber `ξ_pred(κ)` upper-bounds measured `ξ` on 22/22 configs and is tight (`r≈0.9`) in the edit-local regime `κ≲8`; loose near singularity (worst-case ≠ typical reach). *Caveat:* a single-`s_max` sweep is `ρ`/`σ_min`-collinear (cell self-limits to `ρ≤0.97`), so it confirms the magnitude law but does **not** by itself beat `ρ` — that needs the `ρ>1`-yet-local runs |
| `ρ>1` is fine; locality holds if the spectrum avoids `+1` | Faber poly-approx needs analyticity on a region around `spec(J)`, *not* `ρ<1` (Neumann **not** required) | distance from `+1` to the field of values | `fagcn_deq_locality.py` (roman, ρ≈1.3, local) |
| Sharply local even near the contraction boundary | same | `ρ=0.955` yet `ξ≈1` hop | `maintenance_demo.py` |
| The same condition covers the **tropical** semiring | frozen-routing linearisation + Faber (non-smoothness confined to the edit core: ~1% far-field argmax switches) | `σ_min(I−J)` on the subgradient | `maxplus_conditioning.py` |

**Two regimes (both subsume the linear theory):**
- **Contractive** (`ρ<1`, Picard converges) — the *strong* regime: warm ≡ cold to machine precision.
- **Non-contractive** (`ρ>1`, `σ_min` healthy, Broyden) — the *weaker* regime; a possible **soft mode**
  exists. New territory the linear theory can't describe; this is the novel use of DMS.

**Failure mode — "local in name only."** A low-degree Faber polynomial in `J` can still have *hundreds*
of effective degrees: the screening length `ξ ≈ √κ(I−J)/2` blows up when `σ_min→0` (κ huge). Locality
is only *practical* when `ξ ≪ diameter`; just-barely-nonsingular `(I−J)` is technically local but
useless. (Two thresholds: `σ_min>0` = existence/Broyden-findable; `√κ/2 ≪ diameter` = actually local.)

---

## Honest ledgers

**Expressivity.**
- *Within a semiring:* ties the strongest maintainable incumbent — a near-theorem (edit-local ⇒
  low-dim neighbourhood ⇒ a wide linear readout is approximately universal). `unroll_vs_eq.py`:
  going to equilibrium ≈ a matched finite unroll (reach = edit-length = ξ).
- *Across semirings:* real and categorical vs linear-push (tropical). Soft generalist (log-sum-exp)
  vs hard specialist (max) — ~0.09 gap.
- *Not* beyond-finite-depth.

**Trade-off axes (the practical design space).**
- **Contractivity control** (`κ`/`σ_min`): accuracy ↔ locality.
- **Ring-truncation size**: speed ↔ exactness *at edit time* — **the** big lever for large-diameter
  streaming graphs (prediction ring ≪ diameter; freeze the rest).
- **Semiring choice**: which operator/algebra the task needs.

---

## Reports, in order

#1–5 (set-DEQ properties, unlearning channels) · **#6** contraction-as-the-door · **#7** the
`σ_min` susceptibility theory (the locality proof spine) · **#8** the editable-class framework +
the semiring axis + the honest expressivity ledger. Lit record: `reports/editability_literature.txt`.
