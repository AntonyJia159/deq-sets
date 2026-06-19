# Results log

## Layer 1, run 1 — are path-independence & exact unlearning free? (2026-06-19)

Task: GMM cluster-count classification, 2000 train / 500 test, N=24 points, d=2,
k in {1..4}. SetDEQ d_latent=32, 8 epochs, unrolled-solver gradient.
Probes: 20 test sets, solver max_iter=200, tol=1e-6, damping=0.5.

| Variant | Test acc | fp_gap | pred_agree | unlearn_gap | warm/cold iters |
|---|---|---|---|---|---|
| DeepSets (unconstrained) | 0.71 | 0.08 | 1.00 | 123,507 | 188 / 195 (capped) |
| DeepSets + spectral norm | 0.71 | 0.11 | 1.00 | 11,416,646 | 185 / 188 (capped) |
| Attention (LayerNorm)    | 0.83 | 0.033 | 0.98 | 0.029 | 88 / 116 (converged) |

### Takeaways
1. **Not free.** Unconstrained DeepSets update has no well-posed fixed point: the
   iteration never converges (hits the 200 cap), the latent diverges, and the
   unlearning gap is divergence noise, not a real measurement.
2. **Naive spectral norm is insufficient (worse here).** Bounding each weight's
   spectral radius to 1 does not make the update a contraction in z, because it
   consumes both z and pooled-z (Lipschitz ~2x) and ReLU/concat compound it.
3. **LayerNorm attention update gives the properties approximately for free.**
   Warm and cold solves both converge to nearly the same fixed point
   (fp_gap 0.033), unlearning is near-exact (gap 0.029, preds match 100%), and
   the warm-start efficiency bonus is real (~24% fewer iters). Mechanism is
   boundedness (LayerNorm), not spectral contractivity.
4. Attention's residual non-uniqueness (fp_gap 0.033, pred_agree 0.98) is exactly
   the regime where the Layer-2 MIA / DP-noise analysis becomes the right tool.

### Verification notes
- The huge DeepSets unlearn_gap is genuine divergence, not a metric bug: the
  attention run uses identical metric code and returns sane finite values, and
  the DeepSets solver iters sit at the 200 cap (non-convergence).
- Gradient is unrolled through the solver (Layer-1 simplification); swap in
  torchdeq phantom/implicit gradients before trusting train-time behavior at depth.

## Layer 1, run 2 — controlled normalization test (2026-06-19)

Added NormDeepSetsUpdate: identical wrapper to AttnUpdate (input injection +
residual + 2 LayerNorms + FF) but mean-pool aggregator instead of attention.

| Variant | Mixing | Norm | Test acc | fp_gap | pred_agree | unlearn_gap | warm/cold |
|---|---|---|---|---|---|---|---|
| DeepSets (raw)      | mean-pool | no  | 0.71 | 0.08 | 1.00 | 123,507    | 188/195 (capped) |
| DeepSets + spectral | mean-pool | no  | 0.71 | 0.11 | 1.00 | 11,416,646 | 185/188 (capped) |
| NormDeepSets        | mean-pool | yes | 0.75 | 0.039 | 1.00 | 0.025 | 130/152 |
| Attention           | attention | yes | 0.83 | 0.033 | 0.98 | 0.029 | 88/116 |

### Clean dissociation
- **Normalization -> well-posedness.** Swapping only the norm (raw -> NormDeepSets,
  same aggregator) flips divergence (gap 1e5) to convergence + near-exact
  unlearning (gap 0.025). Aggregator was never the cause of divergence.
- **Attention -> accuracy.** Holding norm fixed (NormDeepSets -> Attention), acc
  rises 0.75 -> 0.83. Aggregator models the interactive cluster dynamics.

### Hypothesis to test (not asserted)
NormDeepSets is *cleaner* on equilibrium properties (pred_agree 1.00 vs 0.98,
slightly smaller unlearn_gap) despite lower accuracy. Possible expressiveness <->
uniqueness tradeoff: attention's richer fixed points may be marginally less
unique / more leaky. Stress-test under cardinality shift and harder configs.

## Layer 1, run 3 — cardinality shift (2026-06-19)

Train at N=24, probe at N in {24,48,96,192} (k fixed at 1..4; only density grows).
Probes: 30 sets, 5 inits, solver max_iter=200 tol=1e-5. Random single-point removal.

NormDeepSets:
| N | acc | fp_gap mean/max | agree | conv | unlearn mean/max | warm/cold |
|---|---|---|---|---|---|---|
| 24  | 0.685 | 0.051/0.322 | 1.00 | 0.51 | 0.039/0.221 | 132/156 |
| 48  | 0.730 | 0.033/0.339 | 1.00 | 0.64 | 0.044/0.587 | 107/137 |
| 96  | 0.715 | 0.037/0.258 | 1.00 | 0.73 | 0.049/0.353 | 95/134 |
| 192 | 0.675 | 0.020/0.226 | 1.00 | 0.77 | 0.025/0.357 | 78/131 |

Attention:
| N | acc | fp_gap mean/max | agree | conv | unlearn mean/max | match | warm/cold |
|---|---|---|---|---|---|---|---|
| 24  | 0.790 | 0.027/0.317 | 0.99 | 0.71 | 0.052/0.479 | 1.00 | 95/125 |
| 48  | 0.815 | 0.034/0.404 | 1.00 | 0.66 | 0.029/0.395 | 1.00 | 97/131 |
| 96  | 0.840 | 0.039/0.323 | 1.00 | 0.69 | 0.043/0.290 | 1.00 | 89/132 |
| 192 | 0.860 | 0.032/0.289 | 1.00 | 0.64 | 0.020/0.121 | 0.97 | 73/143 |

### Findings vs predictions
1. **Efficiency widens (confirmed).** warm iters fall with N, cold flat -> warm-start
   advantage grows (NormDeepSets 24->53, attn 30->70).
2. **Path independence does NOT crack with cardinality (prediction refuted).** fp_gap
   flat/improving, pred_agree ~1.0 at all N. Lesson: bifurcations live on the
   cluster-SEPARATION axis (geometric ambiguity), not the density axis. Sharper
   hypothesis for the next experiment.
3. **Unlearning: mean small/shrinking, the TAIL is the story (confirmed two-level).**
   NormDeepSets keeps a latent tail (max ~0.3-0.6, ~10x mean); attention's tail shrinks
   in magnitude but surfaces as a rare DECISION FLIP (pred_match 0.97 at N=192 = one
   pivotal removal changed the answer).
4. **Well-posedness: boundedness survives, strict convergence does NOT certify.** No
   divergence at any N (guard never fired, fp_gap small), BUT only ~50-77% of solves reach
   tol=1e-5 in 200 iters. This is solver slowness (damping 0.5, contraction ~0.96 needs
   ~280 iters), NOT non-uniqueness. NormDeepSets conv RATE improves with N (mean-field
   stabilization). DO NOT claim 100% converged.
5. **Bonus:** attention accuracy IMPROVES with N (0.79->0.86) = real "infer large = more
   power"; NormDeepSets plateaus (~0.70, mean-pool saturates).

### Next
- **Swap damped iteration for Anderson / TorchDEQ** — top priority; needed to certify
  convergence at scale before any of this becomes a figure.
- **Cluster-separation sweep** (the real bifurcation knob): vary inter-cluster distance,
  watch path independence / unlearning break as clusters merge. This is where the
  pivotal-removal tail and the MIA story should concentrate.
- Then the adversarial phase (MIA, DP-noise) in the bifurcation regime.
